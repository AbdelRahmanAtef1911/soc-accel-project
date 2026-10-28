/*
 * bench_dma.c - step 5: how should the DMA work once the computation is on
 * the accelerator? Runs every MLPerf Tiny layer (sw/layers.h) in DMA mode and
 * prints one CSV line per layer and per mode:
 *
 *   mode DMA   the whole layer as one job: load -> compute -> store (step 3b)
 *   mode TILE  the job cut into tiles, still one phase after the other
 *   mode DBUF  tiles + double buffering: DMA overlaps the computation
 *
 * How the CPU waits for each job is a build option (WAIT):
 *   0 POLL  busy-wait on STATUS (as in steps 2-4)
 *   1 WFI   completion interrupt, the CPU sleeps (WFI) meanwhile
 *   2 BG    completion interrupt, the CPU runs background work meanwhile
 *           (a stand-in for other application work: a checksum over a buffer
 *           in main memory, so it also uses the bus)
 * The bus itself is chosen when the VP starts (--accel-bus-model,
 * --accel-dma-mode, --accel-dma-burst); the INFO line records it.
 *
 * With WAIT=0 the DMA rows use the unchanged step 2-4 driver, so they must
 * give exactly the step 3b numbers. Every result is checked against the CPU
 * reference (check = PASS/FAIL) unless CHECK=0 (fast sweeps, check = SKIP).
 *
 * Build options (see the Makefile): WAIT, NTILES (default 4), RUN_ONE,
 * RUN_TILE, RUN_DBUF (1/0), OFFLOAD_CONV (default 1), CHECK, ONLY_MODEL.
 */
#include <stdint.h>

#include "accel_driver.h"
#include "accel_ext.h"
#include "kernels.h"
#include "layers.h"
#include "runtime.h"

#ifndef WAIT
#define WAIT 0
#endif
#ifndef NTILES
#define NTILES 4
#endif
#ifndef RUN_ONE
#define RUN_ONE 1
#endif
#ifndef RUN_TILE
#define RUN_TILE 1
#endif
#ifndef RUN_DBUF
#define RUN_DBUF 1
#endif
#ifndef OFFLOAD_CONV
#define OFFLOAD_CONV 1
#endif
#ifndef CHECK
#define CHECK 1
#endif

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
static int8_t rand8(void) { /* xorshift32, same data as bench.c */
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return (int8_t)(rng >> 24);
}

__attribute__((unused)) static int streq(const char *a, const char *b) {
    while (*a && *a == *b) a++, b++;
    return *a == *b;
}

/* ---- background work for WAIT=2: one unit = 16 words of a checksum ---- */
static uint32_t bg_data[256];
static uint32_t bg_sum, bg_pos;
static void background_unit(void) {
    uint32_t s = bg_sum;
    for (int i = 0; i < 16; i++) s = s * 31u + bg_data[(bg_pos + i) & 255];
    bg_pos += 16;
    bg_sum = s;
}

static const char *TYPE[] = {"CONV", "PW", "DW", "FC"};
static const char *WAITNAME[] = {"POLL", "WFI", "BG"};

static void csv(const layer_t *L, uint32_t M, uint32_t K, uint32_t N, const char *mode, const char *place,
                uint64_t prep, const accel_xtimes_t *t, uint64_t cpu_time, const char *check) {
    print_str("CSV,");
    print_str(L->model); print_char(',');
    print_str(L->name); print_char(',');
    print_str(TYPE[L->type]); print_char(',');
    print_u64(M); print_char(',');
    print_u64(K); print_char(',');
    print_u64(N); print_char(',');
    print_u64((uint64_t)M * K * N); print_char(',');
    print_str(mode); print_char(',');
    print_str(WAITNAME[WAIT]); print_char(',');
    print_str(place); print_char(',');
    if (t) {
        print_u64(prep); print_char(',');
        print_u64(t->in); print_char(',');
        print_u64(t->run); print_char(',');
        print_u64(prep + t->in + t->run); print_char(',');
        print_u64(t->tiles); print_char(',');
        print_u64(t->acc_load); print_char(',');
        print_u64(t->acc_compute); print_char(',');
        print_u64(t->acc_store); print_char(',');
        print_u64(t->acc_total); print_char(',');
        print_u64(t->bytes_in); print_char(',');
        print_u64(t->polls); print_char(',');
        print_u64(t->sleep); print_char(',');
        print_u64(t->bg_units); print_char(',');
        print_u64(t->resp); print_char(',');
        print_u64(t->cpu_txn); print_char(',');
        print_u64(t->cpu_wait); print_char(',');
        print_u64(t->cpu_maxw); print_char(',');
        print_u64(t->dma_wait); print_char(',');
    } else { /* layer stays on the CPU */
        print_str("0,0,"); print_u64(cpu_time); print_char(','); print_u64(cpu_time);
        print_str(",0,0,0,0,0,0,0,0,0,0,0,0,0,0,");
    }
    print_str(check); print_char('\n');
}

static const char *check(int err, uint32_t n) {
    if (err) return "FAIL";
    if (!CHECK) return "SKIP";
    for (uint32_t i = 0; i < n; i++)
        if (Yacc[i] != Yref[i]) return "FAIL";
    return "PASS";
}

/* one accelerated run of layer L in the given mode (0 one piece, 1 tiled, 2 dbuf) */
static void run_mode(const layer_t *L, int m, uint64_t t_sw) {
    static const char *MODE[] = {"DMA", "TILE", "DBUF"};
    accel_policy_t pol = {m == 0 ? 0 : NTILES, m == 2, WAIT, background_unit};
    accel_xtimes_t t = {0};
    uint32_t M, K, N, n;
    uint64_t prep = 0;
    int err;
    const int native = OFFLOAD_CONV && (L->type == L_CONV || L->type == L_DW);

    if (L->type == L_DW && !native) { /* "gemm" partition: depthwise stays on the CPU */
        csv(L, L->Ho * L->Wo, L->R * L->S, L->C, MODE[m], "CPU", 0, 0, t_sw, "PASS");
        return;
    }
    if (native) {
        accel_conv_t cv = {L->H, L->W, L->C, L->R, L->S, L->K, (L->type == L_DW) ? L->C : 1,
                           L->sh, L->pt, L->pl, L->Ho, L->Wo};
        M = L->Ho * L->Wo;
        K = (L->type == L_DW) ? L->R * L->S : L->R * L->S * L->C;
        N = (L->type == L_DW) ? L->C : L->K;
        n = L->Ho * L->Wo * L->K;
        for (uint32_t i = 0; i < n; i++) Yacc[i] = 0x55555555;
        if (m == 0 && WAIT == ACCEL_WAIT_POLL) { /* the unchanged step 2-4 driver */
            accel_times_t t0;
            err = accel_conv(X, Wt, Yacc, &cv, ACCEL_DMA, &t0);
            accel_read_counters(&t);
            t.in = t0.in;
            t.run = t0.run + t0.out;
        } else {
            err = accel_conv_x(X, Wt, Yacc, &cv, &pol, &t);
        }
    } else {
        const int8_t *Aptr = X;
        uint64_t t0 = now_cycles();
        if (L->type == L_CONV) {
            M = L->Ho * L->Wo; K = L->R * L->S * L->C; N = L->K;
            im2col(L, X, A);
            Aptr = A;
        } else if (L->type == L_PW) {
            M = L->Ho * L->Wo; K = L->C; N = L->K;
        } else {
            M = 1; K = L->C; N = L->K;
        }
        prep = now_cycles() - t0;
        n = M * N;
        for (uint32_t i = 0; i < n; i++) Yacc[i] = 0x55555555;
        if (m == 0 && WAIT == ACCEL_WAIT_POLL) {
            accel_times_t tt;
            err = accel_gemm(Aptr, Wt, Yacc, M, K, N, ACCEL_DMA, &tt);
            accel_read_counters(&t);
            t.in = tt.in;
            t.run = tt.run + tt.out;
        } else {
            err = accel_gemm_x(Aptr, Wt, Yacc, M, K, N, &pol, &t);
        }
    }
    csv(L, M, K, N, MODE[m], "ACC", prep, &t, 0, check(err, n));
}

int main(void) {
    if (!accel_present()) {
        print_str("ERROR: accelerator not found at ACCEL_BASE\n");
        return 1;
    }
    if (OFFLOAD_CONV && !(accel_caps() & CAPS_CONV)) {
        print_str("ERROR: this program uses OP_CONV, run it with --accel-arch maeri\n");
        return 1;
    }
    if (!(accel_caps() & CAPS_TILE)) {
        print_str("ERROR: the accelerator has no step-5 features (rebuild accel32-vp)\n");
        return 1;
    }
    for (int i = 0; i < 256; i++) bg_data[i] = i * 2654435761u;
    if (WAIT != ACCEL_WAIT_POLL) accel_irq_setup();

    /* cost of one background unit when nothing else runs */
    uint64_t c0 = now_cycles();
    for (int i = 0; i < 100; i++) background_unit();
    uint64_t bg_cycles = (now_cycles() - c0) / 100;

    static const char *ARCH[] = {"TPU", "MAERI", "SIGMA"};
    static const char *DMODE[] = {"block", "steal", "transparent"};
    print_str("INFO,arch,"); print_str(ARCH[accel_arch() % 3]);
    print_str(",pe,"); print_u64(accel_num_pe());
    print_str(",dn_bw,"); print_u64(accel_dn_bw());
    print_str(",partition,"); print_str(OFFLOAD_CONV ? "conv" : "gemm");
    print_str(",wait,"); print_str(WAITNAME[WAIT]);
    print_str(",ntiles,"); print_u64(NTILES);
    print_str(",bus,"); print_str((accel_caps() & CAPS_BUS) ? "shared" : "ideal");
    print_str(",dma_mode,"); print_str(DMODE[accel_dma_mode() % 3]);
    print_str(",burst,"); print_u64(accel_dma_burst());
    print_str(",bg_unit_cycles,"); print_u64(bg_cycles);
    print_char('\n');
    print_str("CSV,model,layer,type,M,K,N,macs,mode,wait,place,prep,in,run,total,tiles,acc_load,acc_compute,"
              "acc_store,acc_total,bytes_in,polls,sleep,bg_units,resp,cpu_txn,cpu_wait,cpu_maxw,dma_wait,check\n");

    for (unsigned li = 0; li < NUM_LAYERS; li++) {
        const layer_t *L = &LAYERS[li];
#ifdef ONLY_MODEL
        if (!streq(L->model, ONLY_MODEL)) continue;
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
        rng = 0x12345u + li * 7919u;  /* same data as bench.c */
        for (uint32_t i = 0; i < xs; i++) X[i] = rand8();
        for (uint32_t i = 0; i < ws; i++) Wt[i] = rand8();  /* no pruning (WZERO = 0) */

        uint64_t t0 = now_cycles(), t_sw = 0;
        if (CHECK || (L->type == L_DW && !OFFLOAD_CONV)) {
            if (L->type == L_DW) dwconv_sw(L, X, Wt, Yref);
            else if (L->type == L_FC) fc_sw(L, X, Wt, Yref);
            else conv_sw(L, X, Wt, Yref);
            t_sw = now_cycles() - t0;
        }
        if (RUN_ONE) run_mode(L, 0, t_sw);
        if (RUN_TILE) run_mode(L, 1, t_sw);
        if (RUN_DBUF) run_mode(L, 2, t_sw);
    }
    print_str("DONE\n");
    return 0;
}
