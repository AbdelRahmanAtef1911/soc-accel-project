/*
 * tb_accel.cpp - step 3: test the accelerator model alone, on the course
 * SystemC 2.3.3 (C++20), before putting it into the RISC-V platform.
 *
 *   Driver --(TLM)--> StonneAccel --(TLM, DMA)--> Memory
 *
 * The driver plays the role of the CPU: it writes the registers, moves the
 * data (CPU-copy mode) or lets the DMA do it, starts the job, polls STATUS and
 * compares the result with plain C++ code. GEMM jobs run on every
 * architecture; CONV jobs (standard and depthwise, with padding and stride)
 * run on MAERI and must be rejected with STATUS.ERROR on TPU and SIGMA.
 * Every job prints one line; the run ends with "ALL TESTS PASSED" or
 * "SOME TESTS FAILED".
 *
 * usage: ./tb_accel [tpu|maeri|sigma] [num_pe] [dn_bw]
 *   tpu   : num_pe must be a square (64, 256, 1024); dn_bw is fixed = 2*sqrt(num_pe)
 *   maeri : num_pe and dn_bw powers of two (defaults 256 and 64)
 *   sigma : same as maeri
 */
#include <systemc>
#include <tlm>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/simple_target_socket.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "stonne_accel.h"

using namespace sc_core;

/* ---------------- main memory seen by the DMA ---------------- */
struct Memory : sc_module {
    tlm_utils::simple_target_socket<Memory> tsock;
    std::vector<uint8_t> data;
    Memory(sc_module_name n, size_t bytes) : sc_module(n), tsock("tsock"), data(bytes, 0) {
        tsock.register_b_transport(this, &Memory::b_transport);
    }
    void b_transport(tlm::tlm_generic_payload &t, sc_time &) {
        uint64_t a = t.get_address();
        unsigned n = t.get_data_length();
        if (a + n > data.size()) { t.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE); return; }
        if (t.is_read()) memcpy(t.get_data_ptr(), &data[a], n);
        else memcpy(&data[a], t.get_data_ptr(), n);
        t.set_response_status(tlm::TLM_OK_RESPONSE);
    }
};

/* ---------------- the "CPU" ---------------- */
struct Job { const char *name; unsigned M, K, N; int weight_zero_pct; };
struct ConvJob { const char *name; unsigned H, W, C, R, S, KOUT, G, stride; };

struct Driver : sc_module {
    tlm_utils::simple_initiator_socket<Driver> isock;
    Memory *mem = nullptr;
    sc_time clk = sc_time(10, SC_NS);
    int failures = 0;
    std::vector<Job> jobs;
    std::vector<ConvJob> conv_jobs;
    bool conv_supported = false;

#if SC_VERSION_MAJOR < 3
    SC_HAS_PROCESS(Driver);
#endif
    Driver(sc_module_name n) : sc_module(n), isock("isock") { SC_THREAD(test); }

    void xfer(tlm::tlm_command cmd, uint32_t addr, void *p, unsigned len) {
        tlm::tlm_generic_payload t;
        sc_time d = SC_ZERO_TIME;
        t.set_command(cmd);
        t.set_address(addr);
        t.set_data_ptr((unsigned char *)p);
        t.set_data_length(len);
        t.set_streaming_width(len);
        t.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
        isock->b_transport(t, d);
        if (t.get_response_status() != tlm::TLM_OK_RESPONSE) {
            printf("  bus error at 0x%08x\n", addr);
            failures++;
        }
        wait(d);  /* each access costs the latency the target reports */
    }
    void wr(uint32_t a, uint32_t v) { xfer(tlm::TLM_WRITE_COMMAND, a, &v, 4); }
    uint32_t rd(uint32_t a) { uint32_t v = 0; xfer(tlm::TLM_READ_COMMAND, a, &v, 4); return v; }
    uint64_t now() { return (uint64_t)(sc_time_stamp() / clk); }

    void run_job(const Job &j, bool dma, unsigned seed) {
        const unsigned M = j.M, K = j.K, N = j.N;
        std::vector<int8_t> A(M * K), B(N * K);
        std::vector<int32_t> ref(M * N, 0), C(M * N, 0);
        srand(seed);
        for (auto &x : A) x = (int8_t)(rand() % 256 - 128);
        for (auto &x : B) x = (rand() % 100 < j.weight_zero_pct) ? 0 : (int8_t)(rand() % 256 - 128);
        for (unsigned m = 0; m < M; m++)
            for (unsigned n = 0; n < N; n++) {
                int32_t s = 0;
                for (unsigned k = 0; k < K; k++) s += A[m * K + k] * B[n * K + k];
                ref[m * N + n] = s;
            }

        const uint32_t a_addr = 0x0000, b_addr = 0x40000, c_addr = 0x80000;
        uint64_t t0 = now();
        wr(ACCEL_CTRL, CTRL_CLEAR_DONE);
        wr(ACCEL_M, M); wr(ACCEL_K, K); wr(ACCEL_N, N);
        if (dma) {
            memcpy(&mem->data[a_addr], A.data(), A.size());
            memcpy(&mem->data[b_addr], B.data(), B.size());
            wr(ACCEL_A_ADDR, a_addr); wr(ACCEL_B_ADDR, b_addr); wr(ACCEL_C_ADDR, c_addr);
        } else { /* CPU-copy: one 32-bit write per 4 bytes */
            for (unsigned i = 0; i < A.size(); i += 4) {
                uint32_t w = 0; memcpy(&w, &A[i], std::min<size_t>(4, A.size() - i));
                wr(ACCEL_SPM_A + i, w);
            }
            for (unsigned i = 0; i < B.size(); i += 4) {
                uint32_t w = 0; memcpy(&w, &B[i], std::min<size_t>(4, B.size() - i));
                wr(ACCEL_SPM_B + i, w);
            }
        }
        uint64_t t1 = now();
        wr(ACCEL_CTRL, CTRL_START | (dma ? CTRL_USE_DMA : 0));
        while (!(rd(ACCEL_STATUS) & STATUS_DONE)) wait(10 * clk);  /* poll every 10 cycles */
        uint64_t t2 = now();
        if (dma) memcpy(C.data(), &mem->data[c_addr], C.size() * 4);
        else for (unsigned i = 0; i < C.size(); i++) C[i] = (int32_t)rd(ACCEL_SPM_C + 4 * i);
        uint64_t t3 = now();

        bool ok = (rd(ACCEL_STATUS) & STATUS_ERROR) == 0;
        for (unsigned i = 0; i < C.size() && ok; i++) ok = (C[i] == ref[i]);
        if (!ok) failures++;
        uint32_t ld = rd(ACCEL_CYC_LOAD), cp = rd(ACCEL_CYC_COMPUTE), st = rd(ACCEL_CYC_STORE);
        double util = 100.0 * M * K * N / ((double)rd(ACCEL_NUM_PE) * cp);
        char shape[48];
        snprintf(shape, sizeof shape, "GEMM %ux%ux%u%s", M, K, N, j.weight_zero_pct ? " w0=50%" : "");
        printf("%-13s %-3s %-22s| in %6llu run %6llu out %6llu | ld %6u cmp %6u st %6u | %5.1f%% %s\n",
               j.name, dma ? "DMA" : "CPU", shape,
               (unsigned long long)(t1 - t0), (unsigned long long)(t2 - t1), (unsigned long long)(t3 - t2),
               ld, cp, st, util, ok ? "PASS" : "FAIL");
    }

    /* TFLite SAME padding */
    static void same(unsigned in, unsigned k, unsigned s, unsigned &out, unsigned &pad) {
        out = (in + s - 1) / s;
        int total = (int)((out - 1) * s + k) - (int)in;
        pad = total > 0 ? total / 2 : 0;
    }

    void run_conv(const ConvJob &j, bool dma, unsigned seed) {
        unsigned Ho, Wo, pt, pl;
        same(j.H, j.R, j.stride, Ho, pt);
        same(j.W, j.S, j.stride, Wo, pl);
        const bool dw = j.G > 1;
        std::vector<int8_t> X(j.H * j.W * j.C), Wt(dw ? j.R * j.S * j.C : j.KOUT * j.R * j.S * j.C);
        std::vector<int32_t> ref(Ho * Wo * j.KOUT, 0), Y(Ho * Wo * j.KOUT, 0);
        srand(seed);
        for (auto &x : X) x = (int8_t)(rand() % 256 - 128);
        for (auto &x : Wt) x = (int8_t)(rand() % 256 - 128);
        for (unsigned oy = 0; oy < Ho; oy++)
            for (unsigned ox = 0; ox < Wo; ox++)
                for (unsigned oc = 0; oc < j.KOUT; oc++) {
                    int32_t acc = 0;
                    for (unsigned r = 0; r < j.R; r++)
                        for (unsigned q = 0; q < j.S; q++) {
                            int iy = (int)(oy * j.stride + r) - (int)pt, ix = (int)(ox * j.stride + q) - (int)pl;
                            if (iy < 0 || ix < 0 || iy >= (int)j.H || ix >= (int)j.W) continue;
                            if (dw) acc += X[(iy * j.W + ix) * j.C + oc] * Wt[(r * j.S + q) * j.C + oc];
                            else for (unsigned c = 0; c < j.C; c++)
                                acc += X[(iy * j.W + ix) * j.C + c] * Wt[((oc * j.R + r) * j.S + q) * j.C + c];
                        }
                    ref[(oy * Wo + ox) * j.KOUT + oc] = acc;
                }

        const uint32_t x_addr = 0x0000, w_addr = 0x40000, y_addr = 0x80000;
        uint64_t t0 = now();
        wr(ACCEL_CTRL, CTRL_CLEAR_DONE);
        wr(ACCEL_OP, OP_CONV);
        wr(ACCEL_CONV_H, j.H); wr(ACCEL_CONV_W, j.W); wr(ACCEL_CONV_C, j.C);
        wr(ACCEL_CONV_R, j.R); wr(ACCEL_CONV_S, j.S); wr(ACCEL_CONV_KOUT, j.KOUT);
        wr(ACCEL_CONV_G, j.G); wr(ACCEL_CONV_STRIDE, j.stride);
        wr(ACCEL_CONV_PAD_T, pt); wr(ACCEL_CONV_PAD_L, pl);
        wr(ACCEL_CONV_HO, Ho); wr(ACCEL_CONV_WO, Wo);
        if (dma) {
            memcpy(&mem->data[x_addr], X.data(), X.size());
            memcpy(&mem->data[w_addr], Wt.data(), Wt.size());
            wr(ACCEL_A_ADDR, x_addr); wr(ACCEL_B_ADDR, w_addr); wr(ACCEL_C_ADDR, y_addr);
        } else {
            for (unsigned i = 0; i < X.size(); i += 4) {
                uint32_t w = 0; memcpy(&w, &X[i], std::min<size_t>(4, X.size() - i));
                wr(ACCEL_SPM_A + i, w);
            }
            for (unsigned i = 0; i < Wt.size(); i += 4) {
                uint32_t w = 0; memcpy(&w, &Wt[i], std::min<size_t>(4, Wt.size() - i));
                wr(ACCEL_SPM_B + i, w);
            }
        }
        uint64_t t1 = now();
        wr(ACCEL_CTRL, CTRL_START | (dma ? CTRL_USE_DMA : 0));
        while (!(rd(ACCEL_STATUS) & STATUS_DONE)) wait(10 * clk);
        uint64_t t2 = now();
        bool err = rd(ACCEL_STATUS) & STATUS_ERROR;
        if (!conv_supported) { /* rigid / sparse-GEMM accelerators must refuse CONV */
            printf("%-13s %-3s CONV on this architecture -> ERROR flag %s\n", j.name, dma ? "DMA" : "CPU",
                   err ? "set (expected)" : "NOT set");
            if (!err) failures++;
            wr(ACCEL_OP, OP_GEMM);
            return;
        }
        if (dma) memcpy(Y.data(), &mem->data[y_addr], Y.size() * 4);
        else for (unsigned i = 0; i < Y.size(); i++) Y[i] = (int32_t)rd(ACCEL_SPM_C + 4 * i);
        uint64_t t3 = now();
        bool ok = !err;
        for (unsigned i = 0; i < Y.size() && ok; i++) ok = (Y[i] == ref[i]);
        if (!ok) failures++;
        uint32_t ld = rd(ACCEL_CYC_LOAD), cp = rd(ACCEL_CYC_COMPUTE), st = rd(ACCEL_CYC_STORE);
        uint64_t macs = (uint64_t)Ho * Wo * j.KOUT * j.R * j.S * (dw ? 1 : j.C);
        char shape[48];
        snprintf(shape, sizeof shape, "%s %ux%ux%u k%u s%u", dw ? "DW" : "CONV", j.H, j.W, j.C, j.R, j.stride);
        printf("%-13s %-3s %-22s| in %6llu run %6llu out %6llu | ld %6u cmp %6u st %6u | %5.1f%% %s\n",
               j.name, dma ? "DMA" : "CPU", shape,
               (unsigned long long)(t1 - t0), (unsigned long long)(t2 - t1), (unsigned long long)(t3 - t2),
               ld, cp, st, 100.0 * macs / ((double)rd(ACCEL_NUM_PE) * cp), ok ? "PASS" : "FAIL");
        wr(ACCEL_OP, OP_GEMM);
    }

    void test() {
        printf("ID register: 0x%08x (%s)\n", rd(ACCEL_ID), rd(ACCEL_ID) == ACCEL_ID_VALUE ? "ok" : "WRONG");
        if (rd(ACCEL_ID) != ACCEL_ID_VALUE) failures++;
        unsigned seed = 1;
        for (const Job &j : jobs) {
            run_job(j, false, seed);
            run_job(j, true, seed);
            seed++;
        }
        conv_supported = (rd(ACCEL_CAPS) & CAPS_CONV) != 0;
        printf("CAPS register: GEMM %s, CONV %s\n", (rd(ACCEL_CAPS) & CAPS_GEMM) ? "yes" : "no",
               conv_supported ? "yes" : "no");
        for (const ConvJob &j : conv_jobs) {
            run_conv(j, false, seed);
            run_conv(j, true, seed);
            seed++;
            if (!conv_supported) break;  /* one refusal check is enough */
        }
        /* a job too big for the scratchpads must raise ERROR, not crash */
        wr(ACCEL_CTRL, CTRL_CLEAR_DONE);
        wr(ACCEL_M, 4096); wr(ACCEL_K, 4096); wr(ACCEL_N, 1);
        wr(ACCEL_CTRL, CTRL_START);
        while (!(rd(ACCEL_STATUS) & STATUS_DONE)) wait(10 * clk);
        bool err = rd(ACCEL_STATUS) & STATUS_ERROR;
        printf("oversized job -> ERROR flag %s\n", err ? "set (expected)" : "NOT set");
        if (!err) failures++;
        printf("%s (%d failure%s), simulated time %s\n", failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED",
               failures, failures == 1 ? "" : "s", sc_time_stamp().to_string().c_str());
        sc_stop();
    }
};

int sc_main(int argc, char **argv) {
    AccelConfig cfg;
    std::string arch = argc > 1 ? argv[1] : "tpu";
    unsigned pe = argc > 2 ? atoi(argv[2]) : 256;
    if (arch == "tpu") {
        cfg.arch = ARCH_TPU;
        unsigned r = (unsigned)std::lround(std::sqrt((double)pe));
        cfg.tpu_rows = cfg.tpu_cols = r;
    } else {
        cfg.arch = (arch == "maeri") ? ARCH_MAERI : ARCH_SIGMA;
        cfg.num_ms = pe;
        cfg.dn_bw = cfg.rn_bw = argc > 3 ? atoi(argv[3]) : 64;
    }

    Driver drv("cpu");
    StonneAccel acc("accel", cfg);
    Memory mem("mem", 1 << 20);
    drv.mem = &mem;
    drv.isock.bind(acc.tsock);
    acc.isock.bind(mem.tsock);

    drv.jobs = {
        {"tiny", 4, 16, 4, 0},
        {"resnet8_conv0", 1024, 27, 16, 0},
        {"dscnn_pw2", 125, 64, 64, 0},
        {"dscnn_pw2_50z", 125, 64, 64, 50},
        {"resnet8_fc9", 1, 64, 10, 0},
    };
    drv.conv_jobs = {
        /* name            H   W   C  R  S KOUT G  stride */
        {"conv3x3",         8,  8,  4, 3, 3, 8,   1, 1},
        {"dscnn_conv0",    49, 10,  1, 10, 4, 64, 1, 2},
        {"resnet8_conv3",  32, 32, 16, 3, 3, 32,  1, 2},
        {"mbnet_dw1",      48, 48,  8, 3, 3, 8,   8, 1},
        {"dscnn_dw1",      25,  5, 64, 3, 3, 64, 64, 1},
        {"mbnet_pw4",      24, 24, 16, 1, 1, 32,  1, 1},
    };
    sc_start();
    return drv.failures ? 1 : 0;
}
