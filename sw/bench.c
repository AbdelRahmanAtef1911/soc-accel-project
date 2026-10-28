/*
 * bench.c - runs every MLPerf Tiny layer (sw/layers.h) on the RISC-V ISS and
 * prints one CSV line per layer and per mode.
 *
 *   mode SW  : step 1, the whole layer in C on the CPU
 *   mode CPU : steps 2-3, GEMM on the accelerator, CPU copies the data
 *   mode DMA : steps 2-3, GEMM on the accelerator, accelerator DMA moves the data
 * Two HW/SW partitions (build option OFFLOAD_CONV):
 *   0 (default) "gemm" partition: the CPU lowers CONV layers with im2col and
 *     the accelerator runs a GEMM; depthwise layers stay on the CPU
 *     (place = CPU). Works on every architecture.
 *   1 "conv" partition: the accelerator runs CONV and depthwise layers
 *     natively (OP_CONV, MAERI only), so the CPU does no im2col; PW and FC
 *     layers need no im2col anyway and keep using OP_GEMM.
 *
 * Every accelerated result is compared with the CPU result (check = PASS/FAIL).
 *
 * Build-time options (see the Makefile):
 *   RUN_SW, RUN_CPU, RUN_DMA  1/0  which modes to run
 *   ONLY_MODEL "resnet8"           run one model only
 *   ONLY_LAYER n                   run one layer index only
 *   WZERO p                        percent of weights forced to zero (pruning)
 *   OFFLOAD_CONV 1                 use the "conv" partition (MAERI only)
 *   CHECK 0                        skip the CPU reference (faster sweeps;
 *                                  check = SKIP). Only with RUN_SW=0.
 */
#include <stdint.h>

#include "accel_driver.h"
#include "kernels.h"
#include "layers.h"
#include "runtime.h"

#ifndef RUN_SW
#define RUN_SW 1
#endif
#ifndef RUN_CPU
#define RUN_CPU 0
#endif
#ifndef RUN_DMA
#define RUN_DMA 0
#endif
#ifndef WZERO
#define WZERO 0
#endif
#ifndef OFFLOAD_CONV
#define OFFLOAD_CONV 0
#endif
#ifndef CHECK
#define CHECK 1
#endif
#if !CHECK && RUN_SW
#error "CHECK=0 only makes sense with RUN_SW=0"
#endif

/* largest MLPerf Tiny layer sizes (checked at run time) */
#define X_MAX (40 * 1024)
#define W_MAX (64 * 1024)
#define A_MAX (160 * 1024)
#define Y_MAX (40 * 1024)

static int8_t X[X_MAX] __attribute__((aligned(16)));
static int8_t Wt[W_MAX] __attribute__((aligned(16)));
static int8_t A[A_MAX] __attribute__((aligned(16)));
static int32_t Yref[Y_MAX] __attribute__((aligned(16)));
static int32_t Yacc[Y_MAX] __attribute__((aligned(16)));

static uint32_t rng;
static int8_t rand8(void) { /* xorshift32 */
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return (int8_t)(rng >> 24);
}

__attribute__((unused)) static int streq(const char *a, const char *b) {
    while (*a && *a == *b) a++, b++;
    return *a == *b;
}

static const char *TYPE[] = {"CONV", "PW", "DW", "FC"};

static void csv(const layer_t *L, uint32_t M, uint32_t K, uint32_t N, const char *mode, const char *place,
                uint64_t prep, uint64_t in, uint64_t run, uint64_t out, const accel_times_t *t,
                const char *check) {
    uint64_t macs = (uint64_t)M * K * N;
    print_str("CSV,");
    print_str(L->model); print_char(',');
    print_str(L->name); print_char(',');
    print_str(TYPE[L->type]); print_char(',');
    print_u64(M); print_char(',');
    print_u64(K); print_char(',');
    print_u64(N); print_char(',');
    print_u64(macs); print_char(',');
    print_str(mode); print_char(',');
    print_str(place); print_char(',');
    print_u64(prep); print_char(',');
    print_u64(in); print_char(',');
    print_u64(run); print_char(',');
    print_u64(out); print_char(',');
    print_u64(prep + in + run + out); print_char(',');
    print_u64(t ? t->acc_load : 0); print_char(',');
    print_u64(t ? t->acc_compute : 0); print_char(',');
    print_u64(t ? t->acc_store : 0); print_char(',');
    print_str(check); print_char('\n');
}

static const char *check(int err, uint32_t n) {
    if (err) return "FAIL";
    if (!CHECK) return "SKIP";
    for (uint32_t i = 0; i < n; i++)
        if (Yacc[i] != Yref[i]) return "FAIL";
    return "PASS";
}

/* "conv" partition: CONV / PW / DW layers go to the accelerator as they are */
static void run_accel_conv(const layer_t *L, int mode) {
    const char *mname = mode == ACCEL_DMA ? "DMA" : "CPU";
    accel_conv_t cv = {L->H, L->W, L->C, L->R, L->S, L->K, (L->type == L_DW) ? L->C : 1,
                       L->sh, L->pt, L->pl, L->Ho, L->Wo};
    uint32_t n = L->Ho * L->Wo * L->K;
    for (uint32_t i = 0; i < n; i++) Yacc[i] = 0x55555555;
    accel_times_t t;
    int err = accel_conv(X, Wt, Yacc, &cv, mode, &t);
    uint32_t M = L->Ho * L->Wo, K = (L->type == L_DW) ? L->R * L->S : L->R * L->S * L->C;
    uint32_t N = (L->type == L_DW) ? L->C : L->K;
    csv(L, M, K, N, mname, "ACC", 0, t.in, t.run, t.out, &t, check(err, n));
}

static void run_accel(const layer_t *L, int mode, uint64_t t_sw) {
    const char *mname = mode == ACCEL_DMA ? "DMA" : "CPU";
    uint32_t M, K, N;
    if (OFFLOAD_CONV && (L->type == L_CONV || L->type == L_DW)) {
        run_accel_conv(L, mode);
        return;
    }
    if (L->type == L_DW) { /* stays on the CPU */
        csv(L, L->Ho * L->Wo, L->R * L->S, L->C, mname, "CPU", 0, 0, t_sw, 0, 0, "PASS");
        return;
    }
    const int8_t *Aptr = X;
    uint64_t t0 = now_cycles();
    if (L->type == L_CONV) {
        M = L->Ho * L->Wo; K = L->R * L->S * L->C; N = L->K;
        im2col(L, X, A);  /* software part of the lowering */
        Aptr = A;
    } else if (L->type == L_PW) {
        M = L->Ho * L->Wo; K = L->C; N = L->K;  /* NHWC input is already A */
    } else { /* FC */
        M = 1; K = L->C; N = L->K;
    }
    uint64_t prep = now_cycles() - t0;

    for (uint32_t i = 0; i < M * N; i++) Yacc[i] = 0x55555555;  /* catch missing writes */
    accel_times_t t;
    int err = accel_gemm(Aptr, Wt, Yacc, M, K, N, mode, &t);

    csv(L, M, K, N, mname, "ACC", prep, t.in, t.run, t.out, &t, check(err, M * N));
}

int main(void) {
    if (!accel_present()) {
        print_str("ERROR: accelerator not found at ACCEL_BASE\n");
        return 1;
    }
    static const char *ARCH[] = {"TPU", "MAERI", "SIGMA"};
    print_str("INFO,arch,");
    print_str(ARCH[accel_arch() % 3]);
    print_str(",pe,"); print_u64(accel_num_pe());
    print_str(",dn_bw,"); print_u64(accel_dn_bw());
    print_str(",wzero,"); print_u64(WZERO);
    print_str(",partition,"); print_str(OFFLOAD_CONV ? "conv" : "gemm");
    print_char('\n');
    if (OFFLOAD_CONV && !(accel_caps() & CAPS_CONV)) {
        print_str("ERROR: this program uses OP_CONV, run it with --accel-arch maeri\n");
        return 1;
    }
    print_str("CSV,model,layer,type,M,K,N,macs,mode,place,prep,in,run,out,total,acc_load,acc_compute,acc_store,check\n");

    int failures = 0;
    for (unsigned li = 0; li < NUM_LAYERS; li++) {
        const layer_t *L = &LAYERS[li];
#ifdef ONLY_MODEL
        if (!streq(L->model, ONLY_MODEL)) continue;
#endif
#ifdef ONLY_LAYER
        if (li != ONLY_LAYER) continue;
#endif
        uint32_t xs = L->H * L->W * L->C;
        uint32_t ws = (L->type == L_DW) ? L->R * L->S * L->C
                    : (L->type == L_FC) ? L->K * L->C : L->K * L->R * L->S * L->C;
        uint32_t ys = L->Ho * L->Wo * (L->type == L_DW ? L->C : L->K);
        uint32_t as = (L->type == L_CONV) ? L->Ho * L->Wo * L->R * L->S * L->C : 0;
        if (xs > X_MAX || ws > W_MAX || ys > Y_MAX || as > A_MAX) {
            print_str("ERROR: buffers too small for "); print_str(L->name); print_char('\n');
            return 1;
        }

        rng = 0x12345u + li * 7919u;  /* same data in every run */
        for (uint32_t i = 0; i < xs; i++) X[i] = rand8();
        for (uint32_t i = 0; i < ws; i++) {
            int8_t w = rand8();
            Wt[i] = ((rng % 100) < WZERO) ? 0 : w;
        }

        /* CPU result: the step-1 time and the reference for checking.
         * With CHECK=0 only depthwise layers are run on the CPU, because in
         * the "gemm" partition their CPU time is part of the measurement. */
        uint64_t t0 = now_cycles(), t_sw = 0;
        if (CHECK || (L->type == L_DW && !OFFLOAD_CONV)) {
            if (L->type == L_DW) dwconv_sw(L, X, Wt, Yref);
            else if (L->type == L_FC) fc_sw(L, X, Wt, Yref);
            else conv_sw(L, X, Wt, Yref);
            t_sw = now_cycles() - t0;
        }

        uint32_t M = L->Ho * L->Wo, K = L->R * L->S * L->C, N = L->K;
        if (L->type == L_FC) { M = 1; K = L->C; }
        if (L->type == L_DW) { K = L->R * L->S; N = L->C; }
        if (RUN_SW) csv(L, M, K, N, "SW", "CPU", 0, 0, t_sw, 0, 0, "PASS");

        if (RUN_CPU) run_accel(L, ACCEL_CPU_COPY, t_sw);
        if (RUN_DMA) run_accel(L, ACCEL_DMA, t_sw);
    }
    (void)failures;
    print_str("DONE\n");
    return 0;
}
