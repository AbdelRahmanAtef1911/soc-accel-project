/*
 * bench_rt.c - gap experiment: can the CPU still meet a real-time deadline
 * while the accelerator runs inference?
 *
 * A TinyML SoC does not only run the network. In keyword spotting (the DS-CNN
 * model) the CPU must take one audio sample every 62.5 us (16 kHz) and filter
 * it while the accelerator works. This program adds that task to step 5:
 *
 *   - a periodic task released by the CLINT timer every RT_PERIOD_US (62 us,
 *     about 16 kHz; mtime counts microseconds of simulated time). The task
 *     reads one sample, runs a FIR_TAPS-tap FIR filter and stores the result;
 *   - its deadline is the next release (the sample is overwritten then);
 *   - meanwhile every MLPerf Tiny layer runs on the accelerator (step 3b
 *     accelerator, conv partition, DMA, completion interrupt + WFI), in one
 *     piece (mode DMA) and with double buffering (mode DBUF, NTILES tiles);
 *   - while it waits, the CPU sleeps (WFI) or, with RT_BG=1, runs background
 *     application work (a checksum over main memory, as in step 5c).
 *
 * The bus is chosen when the VP starts (ideal, or --accel-bus-model with
 * --accel-dma-mode block | steal | transparent and --accel-dma-burst).
 * For each layer and mode it prints the layer time and the real-time task's
 * statistics while that layer ran on the accelerator:
 *   rt_rel   releases, rt_miss  responses later than the deadline,
 *   rt_skip  releases lost completely (the task was still late at the next one),
 *   rt_max / rt_avg   response time (release -> task done), in cycles,
 *   rt_lat   longest release -> handler entry latency.
 * Line "ISO" gives the task's response time with the accelerator idle.
 *
 * Build: make rt [RT_PERIOD_US=62] [FIR_TAPS=16] [RT_BG=0|1] [NTILES=4] [RT_CHECK=0|1] [ONLY_MODEL=...]
 */
#include <stdint.h>

#include "accel_driver.h"
#include "accel_ext.h"
#include "irq.h"
#include "kernels.h"
#include "layers.h"
#include "runtime.h"

#ifndef RT_PERIOD_US
#define RT_PERIOD_US 62
#endif
#ifndef FIR_TAPS
#define FIR_TAPS 16
#endif
#ifndef NTILES
#define NTILES 4
#endif
#ifndef CHECK
#define CHECK 0
#endif
#ifndef RT_BG       /* 1: the CPU also runs background work while the accelerator works */
#define RT_BG 0
#endif

#define CYC_PER_US 100u                  /* 100 MHz                          */
#define PERIOD_CYC ((uint64_t)RT_PERIOD_US * CYC_PER_US)
#define CLINT_MTIMECMP_LO 0x02004000u    /* hart 0                           */
#define CLINT_MTIMECMP_HI 0x02004004u
#define CLINT_MTIME_LO    0x0200BFF8u

/* ------------------------------------------------------------------ */
/* the periodic real-time task                                         */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t rel, miss, skip;
    uint64_t max_resp, sum_resp, max_lat;
} rt_stats_t;

static volatile rt_stats_t rt;
static volatile int rt_on;              /* count statistics only while 1     */
static volatile uint32_t rt_next;       /* next release, in us (mtime units) */
static int32_t adc[64], coef[FIR_TAPS], hist[FIR_TAPS], rt_out[256];
static uint32_t hpos, nsamp;

void rt_timer_isr(void) {
    const uint64_t entry = now_cycles();
    const uint64_t release = (uint64_t)rt_next * CYC_PER_US;

    /* the task: one sample through a FIR filter */
    hist[hpos] = adc[nsamp & 63];
    int32_t acc = 0;
    uint32_t j = hpos;
    for (int i = 0; i < FIR_TAPS; i++) {
        acc += coef[i] * hist[j];
        j = (j == 0) ? FIR_TAPS - 1 : j - 1;
    }
    hpos = (hpos + 1 == FIR_TAPS) ? 0 : hpos + 1;
    rt_out[nsamp & 255] = acc;
    nsamp++;

    const uint64_t done = now_cycles();
    if (rt_on) {
        uint64_t r = done - release, l = entry - release;
        rt.rel++;
        rt.sum_resp += r;
        if (r > rt.max_resp) rt.max_resp = r;
        if (l > rt.max_lat) rt.max_lat = l;
        if (r > PERIOD_CYC) rt.miss++;
    }
    /* next release; releases that already passed are lost */
    rt_next += RT_PERIOD_US;
    uint32_t now = REG32(CLINT_MTIME_LO);
    while (rt_next <= now) {
        rt_next += RT_PERIOD_US;
        if (rt_on) rt.skip++;
    }
    REG32(CLINT_MTIMECMP_LO) = rt_next;
}

static void rt_start(void) {
    for (int i = 0; i < 64; i++) adc[i] = (int32_t)(i * 2654435761u) >> 20;
    for (int i = 0; i < FIR_TAPS; i++) coef[i] = (i * 37 + 11) & 255;
    REG32(CLINT_MTIMECMP_HI) = 0;
    rt_next = REG32(CLINT_MTIME_LO) + RT_PERIOD_US;
    REG32(CLINT_MTIMECMP_LO) = rt_next;
    csr_set(mie, MIE_MTIE);
}

static void rt_reset(void) {
    rt.rel = rt.miss = rt.skip = 0;
    rt.max_resp = rt.sum_resp = rt.max_lat = 0;
}

/* ------------------------------------------------------------------ */
/* the inference work (as bench_dma.c, WAIT = WFI, conv partition)      */
/* ------------------------------------------------------------------ */
#define X_MAX (40 * 1024)
#define W_MAX (64 * 1024)
#define Y_MAX (40 * 1024)
static int8_t X[X_MAX] __attribute__((aligned(16)));
static int8_t Wt[W_MAX] __attribute__((aligned(16)));
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
/* background work (RT_BG=1), as in bench_dma.c: one unit = 16 words of a
 * checksum over a buffer in main memory, so it also uses the bus */
static uint32_t bg_data[256];
static uint32_t bg_sum, bg_pos;
static void background_unit(void) {
    uint32_t s = bg_sum;
    for (int i = 0; i < 16; i++) s = s * 31u + bg_data[(bg_pos + i) & 255];
    bg_pos += 16;
    bg_sum = s;
}

static const char *TYPE[] = {"CONV", "PW", "DW", "FC"};

static void print_rt(void) {
    print_u64(rt.rel); print_char(',');
    print_u64(rt.miss); print_char(',');
    print_u64(rt.skip); print_char(',');
    print_u64(rt.max_resp); print_char(',');
    print_u64(rt.rel ? rt.sum_resp / rt.rel : 0); print_char(',');
    print_u64(rt.max_lat);
}

static void run_mode(const layer_t *L, int dbuf) {
    accel_policy_t pol = {dbuf ? NTILES : 0, dbuf, RT_BG ? ACCEL_WAIT_BG : ACCEL_WAIT_WFI, background_unit};
    accel_xtimes_t t = {0};
    uint32_t n;
    int err;
    for (uint32_t i = 0; i < Y_MAX; i++) Yacc[i] = 0x55555555;
    rt_reset();
    rt_on = 1;
    if (L->type == L_CONV || L->type == L_DW) {
        accel_conv_t cv = {L->H, L->W, L->C, L->R, L->S, L->K, (L->type == L_DW) ? L->C : 1,
                           L->sh, L->pt, L->pl, L->Ho, L->Wo};
        n = L->Ho * L->Wo * L->K;
        err = accel_conv_x(X, Wt, Yacc, &cv, &pol, &t);
    } else {
        uint32_t M = (L->type == L_PW) ? L->Ho * L->Wo : 1;
        n = M * L->K;
        err = accel_gemm_x(X, Wt, Yacc, M, L->C, L->K, &pol, &t);
    }
    rt_on = 0;
    const char *chk = err ? "FAIL" : "SKIP";
    if (!err && CHECK) {
        chk = "PASS";
        for (uint32_t i = 0; i < n; i++)
            if (Yacc[i] != Yref[i]) { chk = "FAIL"; break; }
    }
    print_str("CSV,");
    print_str(L->model); print_char(',');
    print_str(L->name); print_char(',');
    print_str(TYPE[L->type]); print_char(',');
    print_str(dbuf ? "DBUF" : "DMA"); print_char(',');
    print_u64(t.in + t.run); print_char(',');
    print_u64(t.acc_load); print_char(',');
    print_u64(t.acc_compute); print_char(',');
    print_u64(t.acc_store); print_char(',');
    print_u64(t.acc_total); print_char(',');
    print_u64(t.bytes_in); print_char(',');
    print_rt(); print_char(',');
    print_u64(t.cpu_maxw); print_char(',');
    print_u64(t.dma_wait); print_char(',');
    print_u64(t.bg_units); print_char(',');
    print_str(chk); print_char('\n');
}

int main(void) {
    if (!accel_present() || !(accel_caps() & CAPS_CONV) || !(accel_caps() & CAPS_TILE)) {
        print_str("ERROR: needs the step-5 accelerator with --accel-arch maeri\n");
        return 1;
    }
    for (int i = 0; i < 256; i++) bg_data[i] = i * 2654435761u;
    accel_irq_setup();  /* trap handler, PLIC, MEIE, MIE */
    rt_start();         /* CLINT timer, MTIE: the task runs from now on */

    static const char *DMODE[] = {"block", "steal", "transparent"};
    print_str("INFO,period_us,"); print_u64(RT_PERIOD_US);
    print_str(",fir_taps,"); print_u64(FIR_TAPS);
    print_str(",ntiles,"); print_u64(NTILES);
    print_str(",bus,"); print_str((accel_caps() & CAPS_BUS) ? "shared" : "ideal");
    print_str(",dma_mode,"); print_str(DMODE[accel_dma_mode() % 3]);
    print_str(",burst,"); print_u64(accel_dma_burst());
    print_str(",cpu_waits_by,"); print_str(RT_BG ? "background_work" : "wfi");
    print_char('\n');

    /* the task alone: accelerator idle, CPU asleep between releases */
    rt_reset();
    rt_on = 1;
    while (rt.rel < 64) asm volatile("wfi");
    rt_on = 0;
    print_str("ISO,"); print_rt(); print_char('\n');

    print_str("CSV,model,layer,type,mode,total,acc_load,acc_compute,acc_store,acc_total,bytes_in,"
              "rt_rel,rt_miss,rt_skip,rt_max,rt_avg,rt_lat,cpu_maxw,dma_wait,bg_units,check\n");
    for (unsigned li = 0; li < NUM_LAYERS; li++) {
        const layer_t *L = &LAYERS[li];
#ifdef ONLY_MODEL
        if (!streq(L->model, ONLY_MODEL)) continue;
#endif
        uint32_t xs = L->H * L->W * L->C;
        uint32_t ws = (L->type == L_DW) ? L->R * L->S * L->C
                    : (L->type == L_FC) ? L->K * L->C : L->K * L->R * L->S * L->C;
        uint32_t ys = L->Ho * L->Wo * (L->type == L_DW ? L->C : L->K);
        if (xs > X_MAX || ws > W_MAX || ys > Y_MAX) {
            print_str("ERROR: buffers too small for "); print_str(L->name); print_char('\n');
            return 1;
        }
        rng = 0x12345u + li * 7919u;  /* same data as bench.c */
        for (uint32_t i = 0; i < xs; i++) X[i] = rand8();
        for (uint32_t i = 0; i < ws; i++) Wt[i] = rand8();
        if (CHECK) {
            if (L->type == L_DW) dwconv_sw(L, X, Wt, Yref);
            else if (L->type == L_FC) fc_sw(L, X, Wt, Yref);
            else conv_sw(L, X, Wt, Yref);
        }
        run_mode(L, 0);
        run_mode(L, 1);
    }
    print_str("DONE\n");
    return 0;
}
