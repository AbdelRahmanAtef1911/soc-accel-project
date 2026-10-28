/*
 * stonne_backend.cpp - runs one GEMM on STONNE (see stonne_backend.h).
 * Needs STONNE with patches/stonne-accel.patch applied (adds getNCycles()).
 */
#include "stonne_backend.h"

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "STONNEModel.h"
#include "utility.h"

uint64_t stonne_gemm(const StonneParams &p, const char *job_name,
                     const int8_t *A8, const int8_t *B8, int32_t *C32,
                     unsigned m, unsigned k, unsigned n) {
    std::vector<float> A(m * k), B(n * k), C(m * n, 0.0f);
    for (unsigned i = 0; i < m * k; i++) A[i] = A8[i];
    for (unsigned i = 0; i < n * k; i++) B[i] = B8[i];

    Config sc;  /* STONNE hardware configuration, defaults from Config.cpp */
    sc.print_stats_enabled = p.stats_files ? 1 : 0;
    std::string lname = job_name;

    std::ostringstream sink;  /* STONNE prints progress messages; hide them unless asked */
    std::streambuf *old = nullptr;
    if (!p.verbose) old = std::cout.rdbuf(sink.rdbuf());

    Stonne *s = nullptr;
    float *mk_sp = nullptr, *kn_sp = nullptr;
    unsigned *mk_bm = nullptr, *kn_bm = nullptr, *out_bm = nullptr;
    std::vector<float> KN;

    if (p.arch == 0) {  /* rigid TPU-like mesh: rows x cols PEs, output stationary */
        sc.m_MSNetworkCfg.multiplier_network_type = OS_MESH;
        sc.m_MSNetworkCfg.ms_rows = p.tpu_rows;
        sc.m_MSNetworkCfg.ms_cols = p.tpu_cols;
        sc.m_MSNetworkCfg.ms_size = p.tpu_rows * p.tpu_cols;
        sc.m_SDMemoryCfg.n_read_ports = p.tpu_rows + p.tpu_cols;   /* one feed per row/column */
        sc.m_SDMemoryCfg.n_write_ports = p.tpu_rows * p.tpu_cols;
        sc.m_SDMemoryCfg.mem_controller_type = TPU_OS_DENSE;
        sc.m_ASNetworkCfg.reduce_network_type = TEMPORALRN;
        sc.m_ASNetworkCfg.accumulation_buffer_enabled = 1;
        s = new Stonne(sc);
        s->loadDenseGEMM(lname, n, k, m, A.data(), B.data(), C.data(), CNN_DATAFLOW);
        s->loadGEMMTile(p.tpu_cols, 1, p.tpu_rows);  /* T_N, T_K, T_M: fill the whole mesh */
    } else if (p.arch == 1) {  /* MAERI: flexible dense, tile chosen by STONNE Mapper */
        sc.m_MSNetworkCfg.multiplier_network_type = LINEAR;
        sc.m_MSNetworkCfg.ms_size = p.num_ms;
        sc.m_SDMemoryCfg.n_read_ports = p.dn_bw;
        sc.m_SDMemoryCfg.n_write_ports = p.rn_bw;
        sc.m_SDMemoryCfg.mem_controller_type = MAERI_DENSE_WORKLOAD;
        sc.m_ASNetworkCfg.reduce_network_type = ASNETWORK;
        sc.m_ASNetworkCfg.accumulation_buffer_enabled = 1;
        s = new Stonne(sc);
        s->loadDenseGEMM(lname, n, k, m, A.data(), B.data(), C.data(), CNN_DATAFLOW);
        s->generateTile(TileGenerator::Generator::CHOOSE_AUTOMATICALLY,
                        TileGenerator::Target::PERFORMANCE);
    } else {  /* SIGMA: flexible, only non-zero pairs are multiplied */
        sc.m_MSNetworkCfg.multiplier_network_type = LINEAR;
        sc.m_MSNetworkCfg.ms_size = p.num_ms;
        sc.m_SDMemoryCfg.n_read_ports = p.dn_bw;
        sc.m_SDMemoryCfg.n_write_ports = p.rn_bw;
        sc.m_SDMemoryCfg.mem_controller_type = SIGMA_SPARSE_GEMM;
        KN.resize(k * n);  /* SIGMA takes the second operand as K x N */
        for (unsigned r = 0; r < n; r++)
            for (unsigned q = 0; q < k; q++) KN[q * n + r] = B[r * k + q];
        /* bitmap compression; in hardware this is done once, offline, for weights */
        mk_bm = generateBitMapFromDense(A.data(), m, k, GEN_BY_ROWS);
        kn_bm = generateBitMapFromDense(KN.data(), k, n, GEN_BY_COLS);
        mk_sp = generateMatrixSparseFromDense(A.data(), mk_bm, m, k, GEN_BY_ROWS);
        kn_sp = generateMatrixSparseFromDense(KN.data(), kn_bm, k, n, GEN_BY_COLS);
        out_bm = new unsigned[m * n];
        s = new Stonne(sc);
        s->loadGEMM(lname, n, k, m, mk_sp, kn_sp, mk_bm, kn_bm, C.data(), out_bm,
                    p.sigma_wst ? MK_STR_KN_STA : MK_STA_KN_STR);
    }

    s->run();
    uint64_t cycles = s->getNCycles();
    delete s;
    delete[] mk_bm;
    delete[] kn_bm;
    delete[] mk_sp;
    delete[] kn_sp;
    delete[] out_bm;
    if (old) std::cout.rdbuf(old);

    /* all partial sums are integers below 2^24, so float holds them exactly */
    for (unsigned i = 0; i < m * n; i++) C32[i] = (int32_t)std::lrint(C[i]);
    return cycles;
}

uint64_t stonne_conv(const StonneParams &p, const char *job_name,
                     const int8_t *X8, const int8_t *W8, int32_t *Y32,
                     unsigned H, unsigned W, unsigned C, unsigned R, unsigned S,
                     unsigned K, unsigned G, unsigned stride, unsigned pt, unsigned pl,
                     unsigned Ho, unsigned Wo) {
    const bool depthwise = (G == C && K == C && G > 1);
    if (p.arch != 1 || !(G == 1 || depthwise) || stride == 0) return 0;

    /* padded input, sized so that STONNE's valid convolution gives Ho x Wo */
    const unsigned Hp = (Ho - 1) * stride + R, Wp = (Wo - 1) * stride + S;
    std::vector<float> Xp((size_t)Hp * Wp * C, 0.0f);
    for (unsigned y = 0; y < Hp; y++) {
        int iy = (int)y - (int)pt;
        if (iy < 0 || iy >= (int)H) continue;
        for (unsigned x = 0; x < Wp; x++) {
            int ix = (int)x - (int)pl;
            if (ix < 0 || ix >= (int)W) continue;
            for (unsigned c = 0; c < C; c++) Xp[((size_t)y * Wp + x) * C + c] = X8[((size_t)iy * W + ix) * C + c];
        }
    }
    /* STONNE filter layout: [g][k][r][s][c] with per-group K and C */
    const unsigned wsz = depthwise ? R * S * C : K * R * S * C;
    std::vector<float> Wf(wsz);
    if (depthwise) { /* TFLite [r][s][c] -> [c][r][s] */
        for (unsigned r = 0; r < R; r++)
            for (unsigned q = 0; q < S; q++)
                for (unsigned c = 0; c < C; c++) Wf[(c * R + r) * S + q] = W8[(r * S + q) * C + c];
    } else {
        for (unsigned i = 0; i < wsz; i++) Wf[i] = W8[i];
    }
    std::vector<float> Yf((size_t)Ho * Wo * K, 0.0f);

    Config sc;
    sc.print_stats_enabled = p.stats_files ? 1 : 0;
    sc.m_MSNetworkCfg.multiplier_network_type = LINEAR;
    sc.m_MSNetworkCfg.ms_size = p.num_ms;
    sc.m_SDMemoryCfg.n_read_ports = p.dn_bw;
    sc.m_SDMemoryCfg.n_write_ports = p.rn_bw;
    sc.m_SDMemoryCfg.mem_controller_type = MAERI_DENSE_WORKLOAD;
    sc.m_ASNetworkCfg.reduce_network_type = ASNETWORK;
    sc.m_ASNetworkCfg.accumulation_buffer_enabled = 1;

    std::ostringstream sink;
    std::streambuf *old = nullptr;
    if (!p.verbose) old = std::cout.rdbuf(sink.rdbuf());

    Stonne *s = new Stonne(sc);
    /* STONNE names: X/Y are the input rows/columns, K the filters, N the batch */
    s->loadCONVLayer(job_name, R, S, C, K, G, 1, Hp, Wp, stride, Xp.data(), Wf.data(), Yf.data());
    if (depthwise) {
        /* mRNA maps one channel at a time (R*S multipliers busy). Instead, give
         * each channel R*S multipliers and run as many channels (T_G) and output
         * columns (T_Y') side by side as fit; tile sizes must divide G and Wo. */
        unsigned tg = 1, ty = 1;
        for (unsigned d = 1; d <= G; d++)
            if (G % d == 0 && d * R * S <= p.num_ms) tg = d;
        for (unsigned d = 1; d <= Wo; d++)
            if (Wo % d == 0 && tg * R * S * d <= p.num_ms) ty = d;
        s->loadTile(R, S, 1, 1, tg, 1, 1, ty);  /* T_R T_S T_C T_K T_G T_N T_X' T_Y' */
    } else {
        s->generateTile(TileGenerator::Generator::CHOOSE_AUTOMATICALLY, TileGenerator::Target::PERFORMANCE);
    }
    s->run();
    uint64_t cycles = s->getNCycles();
    delete s;
    if (old) std::cout.rdbuf(old);

    for (size_t i = 0; i < Yf.size(); i++) Y32[i] = (int32_t)std::lrint(Yf[i]);
    return cycles;
}
