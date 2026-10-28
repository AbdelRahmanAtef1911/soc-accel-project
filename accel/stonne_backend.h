/*
 * stonne_backend.h - the only interface between the SystemC model and STONNE.
 *
 * STONNE's headers define many global enum names (CONV, FC, LEFT, NOP, ...).
 * Keeping them inside stonne_backend.cpp stops them from clashing with the
 * SystemC model or the RISC-V VP headers. This header has no dependencies.
 */
#ifndef STONNE_BACKEND_H
#define STONNE_BACKEND_H

#include <cstdint>

struct StonneParams {
    int arch;            /* 0 = TPU-like (rigid output-stationary mesh)   */
                         /* 1 = MAERI (flexible, dense, STONNE Mapper)    */
                         /* 2 = SIGMA (flexible, skips zeros)             */
    unsigned tpu_rows;   /* TPU: mesh rows                                */
    unsigned tpu_cols;   /* TPU: mesh columns                             */
    unsigned num_ms;     /* MAERI/SIGMA: multiplier switches (power of 2) */
    unsigned dn_bw;      /* MAERI/SIGMA: distribution network bandwidth   */
    unsigned rn_bw;      /* MAERI/SIGMA: reduction network bandwidth      */
    bool sigma_wst;      /* SIGMA: weights (B) stationary instead of A    */
    bool verbose;        /* show STONNE's own console output              */
    bool stats_files;    /* let STONNE write its JSON/counter files       */
};

/*
 * C[m][n] = sum_k A[m][k] * B[n][k]   (A, B int8; C int32)
 * Runs one GEMM on STONNE and returns the number of accelerator cycles.
 * job_name labels STONNE's stats files when stats_files is set.
 */
uint64_t stonne_gemm(const StonneParams &p, const char *job_name,
                     const int8_t *A, const int8_t *B, int32_t *C,
                     unsigned m, unsigned k, unsigned n);

/*
 * Y = conv(X, Wt) on MAERI with STONNE's own CONV layer (tile from mRNA).
 *   X  int8 [H][W][C], Wt int8 [K][R][S][C] (G == 1) or [R][S][C] (G == C == K)
 *   Y  int32 [Ho][Wo][K]; zero padding pt/pl added here (by the "loader")
 * Returns cycles, or 0 if the shape is not supported.
 */
uint64_t stonne_conv(const StonneParams &p, const char *job_name,
                     const int8_t *X, const int8_t *Wt, int32_t *Y,
                     unsigned H, unsigned W, unsigned C, unsigned R, unsigned S,
                     unsigned K, unsigned G, unsigned stride, unsigned pt, unsigned pl,
                     unsigned Ho, unsigned Wo);

#endif
