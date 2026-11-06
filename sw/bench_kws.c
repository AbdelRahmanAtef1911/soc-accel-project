/*
 * bench_kws.c - course project, stage A: one DS-CNN inference released at a
 * known time, while
 *   - the CPU runs its 62 us control task (CLINT timer, 16-tap FIR; deadline =
 *     next release), sleeping (WFI) in between, and
 *   - the streaming sensor (accel/sensor_dma.h, if the platform has one) keeps
 *     filling its FIFO and draining it over the shared request/grant bus.
 *
 * Measurement (project plan, Sections 4.2 and 4.6):
 *   warm-up    WARMUP control-task releases, so the sensor is in steady state;
 *   release    t = 0 of the inference: the sensor window is opened (MARK) and
 *              the 10 DS-CNN layers are run back to back on the accelerator
 *              (conv partition, completion interrupt + WFI, DMA one piece or
 *              double buffered with NTILES tiles; timing-only data);
 *   T_acc      release -> end of the last layer, including any wait before the
 *              first DMA; DNN deadline met if T_acc <= T_inf;
 *   window     [release, release + T_inf): control-task releases, late and lost
 *              ones, worst response; sensor overflows, peak FIFO occupancy and
 *              longest sensor bus wait. If the DNN is late, the program waits
 *              for it, so T_acc is still measured.
 *
 * Build: make kws [KWS_TINF_US=5000] [KWS_DBUF=1] [NTILES=4] [WARMUP=16]
 */
#include <stdint.h>

#include "accel_driver.h"
#include "accel_ext.h"
#include "irq.h"
#include "kernels.h"
#include "layers.h"
#include "runtime.h"
#include "sensor_regs.h"

#ifndef RT_PERIOD_US
#define RT_PERIOD_US 62
#endif
#ifndef FIR_TAPS
#define FIR_TAPS 16
#endif
#ifndef NTILES
#define NTILES 4
#endif
#ifndef KWS_TINF_US
#define KWS_TINF_US 5000
#endif
#ifndef KWS_DBUF
#define KWS_DBUF 1
#endif
#ifndef WARMUP
#define WARMUP 16
#endif

#define CYC_PER_US 100u
#define PERIOD_CYC ((uint64_t)RT_PERIOD_US * CYC_PER_US)
#define CLINT_MTIMECMP_LO 0x02004000u
#define CLINT_MTIMECMP_HI 0x02004004u
#define CLINT_MTIME_LO    0x0200BFF8u
#define SEN(r) REG32(SENSOR_BASE + (r))

/* ---------------- the periodic control task ---------------- */
typedef struct {
    uint32_t rel, miss, skip;
    uint64_t max_resp, max_lat;
} rt_stats_t;

static volatile rt_stats_t rt;
static volatile uint64_t win_t0, win_t1;  /* count releases in [win_t0, win_t1) */
static volatile uint32_t rt_next;
static volatile uint32_t warm;            /* releases so far (warm-up) */
static int32_t adc[64], coef[FIR_TAPS], hist[FIR_TAPS], rt_out[256];
static uint32_t hpos, nsamp;

static int in_win(uint64_t t) { return t >= win_t0 && t < win_t1; }

void rt_timer_isr(void) {
    const uint64_t entry = now_cycles();
    const uint64_t release = (uint64_t)rt_next * CYC_PER_US;

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
    warm++;

    const uint64_t done = now_cycles();
    if (in_win(release)) {
        uint64_t r = done - release, l = entry - release;
        rt.rel++;
        if (r > rt.max_resp) rt.max_resp = r;
        if (l > rt.max_lat) rt.max_lat = l;
        if (r > PERIOD_CYC) rt.miss++;
    }
    rt_next += RT_PERIOD_US;
    uint32_t now = REG32(CLINT_MTIME_LO);
    while (rt_next <= now) {  /* releases that already passed are lost */
        if (in_win((uint64_t)rt_next * CYC_PER_US)) rt.skip++;
        rt_next += RT_PERIOD_US;
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

/* ---------------- the inference (timing-only data) ---------------- */
#define X_MAX (8 * 1024)
#define W_MAX (8 * 1024)
#define Y_MAX (8 * 1024)
static int8_t X[X_MAX] __attribute__((aligned(16)));
static int8_t Wt[W_MAX] __attribute__((aligned(16)));
static int32_t Y[Y_MAX] __attribute__((aligned(16)));

static int streq(const char *a, const char *b) {
    while (*a && *a == *b) a++, b++;
    return *a == *b;
}

static int run_layer(const layer_t *L, accel_xtimes_t *t) {
    accel_policy_t pol = {KWS_DBUF ? NTILES : 0, KWS_DBUF, ACCEL_WAIT_WFI, 0};
    if (L->type == L_CONV || L->type == L_DW) {
        accel_conv_t cv = {L->H, L->W, L->C, L->R, L->S, L->K, (L->type == L_DW) ? L->C : 1,
                           L->sh, L->pt, L->pl, L->Ho, L->Wo};
        return accel_conv_x(X, Wt, Y, &cv, &pol, t);
    }
    uint32_t M = (L->type == L_PW) ? L->Ho * L->Wo : 1;
    return accel_gemm_x(X, Wt, Y, M, L->C, L->K, &pol, t);
}

int main(void) {
    if (!accel_present() || !(accel_caps() & CAPS_CONV) || !(accel_caps() & CAPS_TILE)) {
        print_str("ERROR: needs the step-5 accelerator with --accel-arch maeri\n");
        return 1;
    }
    const int has_sensor = SEN(SENSOR_ID) == SENSOR_ID_VALUE;
    uint32_t seed = 0x12345u;
    for (uint32_t i = 0; i < X_MAX; i++) { seed = seed * 1664525u + 1013904223u; X[i] = (int8_t)(seed >> 24); }
    for (uint32_t i = 0; i < W_MAX; i++) { seed = seed * 1664525u + 1013904223u; Wt[i] = (int8_t)(seed >> 24); }

    accel_irq_setup();
    win_t0 = win_t1 = 0;
    rt_start();
    while (warm < WARMUP) asm volatile("wfi");  /* sensor and control task reach steady state */

    /* release: open the measurement window, then start the inference */
    SEN(SENSOR_WINDOW_US) = KWS_TINF_US;
    SEN(SENSOR_MARK) = 1;
    const uint64_t t_rel = now_cycles();
    rt.rel = rt.miss = rt.skip = 0;
    rt.max_resp = rt.max_lat = 0;
    win_t1 = t_rel + (uint64_t)KWS_TINF_US * CYC_PER_US;
    win_t0 = t_rel;

    uint64_t lay_end[16];
    int nl = 0, err = 0;
    for (unsigned li = 0; li < NUM_LAYERS && nl < 16; li++) {
        const layer_t *L = &LAYERS[li];
        if (!streq(L->model, "dscnn")) continue;
        accel_xtimes_t t = {0};
        err |= run_layer(L, &t);
        lay_end[nl++] = now_cycles();
    }
    const uint64_t t_done = now_cycles();
    const uint64_t t_acc = t_done - t_rel;

    /* let the window close (the control task keeps waking the CPU) */
    while (now_cycles() < win_t1) asm volatile("wfi");

    print_str("CSV,tinf_us,t_acc,dnn_met,layers,rt_rel,rt_miss,rt_skip,rt_max,rt_lat,"
              "sen_rate_ksps,sen_theta,sen_depth,sen_ovf_win,sen_peak_win,sen_maxwait,burst,dbuf,check\n");
    print_str("CSV,"); print_u64(KWS_TINF_US); print_char(',');
    print_u64(t_acc); print_char(',');
    print_u64(t_acc <= (uint64_t)KWS_TINF_US * CYC_PER_US); print_char(',');
    print_u64(nl); print_char(',');
    print_u64(rt.rel); print_char(',');
    print_u64(rt.miss); print_char(',');
    print_u64(rt.skip); print_char(',');
    print_u64(rt.max_resp); print_char(',');
    print_u64(rt.max_lat); print_char(',');
    print_u64(has_sensor ? SEN(SENSOR_RATE_KSPS) : 0); print_char(',');
    print_u64(has_sensor ? SEN(SENSOR_THETA) : 0); print_char(',');
    print_u64(has_sensor ? SEN(SENSOR_DEPTH) : 0); print_char(',');
    print_u64(has_sensor ? SEN(SENSOR_OVF_WINDOW) : 0); print_char(',');
    print_u64(has_sensor ? SEN(SENSOR_PEAK_WINDOW) : 0); print_char(',');
    print_u64(has_sensor ? SEN(SENSOR_MAXWAIT) : 0); print_char(',');
    print_u64(accel_dma_burst()); print_char(',');
    print_u64(KWS_DBUF); print_char(',');
    print_str(err ? "FAIL" : "SKIP"); print_char('\n');
    print_str("LAYERS");
    for (int i = 0; i < nl; i++) { print_char(','); print_u64(lay_end[i] - t_rel); }
    print_char('\n');
    print_str("DONE\n");
    return 0;
}
