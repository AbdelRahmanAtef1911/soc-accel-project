#include "accel_ext.h"

#include "accel_regs.h"
#include "irq.h"
#include "runtime.h"

#define AREG(off) REG32(ACCEL_BASE + (off))

uint32_t accel_dma_mode(void) { return AREG(ACCEL_DMA_MODE); }
uint32_t accel_dma_burst(void) { return AREG(ACCEL_DMA_BURST); }

void accel_irq_setup(void) { irq_init(); }

void accel_read_counters(accel_xtimes_t *t) {
    t->acc_load = AREG(ACCEL_CYC_LOAD);
    t->acc_compute = AREG(ACCEL_CYC_COMPUTE);
    t->acc_store = AREG(ACCEL_CYC_STORE);
    t->acc_total = AREG(ACCEL_CYC_TOTAL);
    t->tiles = AREG(ACCEL_TILES);
    t->bytes_in = AREG(ACCEL_BYTES_IN);
    t->polls = AREG(ACCEL_POLLS);
    t->cpu_txn = AREG(ACCEL_BUS_CPU_TXN);
    t->cpu_wait = AREG(ACCEL_BUS_CPU_WAIT);
    t->cpu_maxw = AREG(ACCEL_BUS_CPU_MAXW);
    t->dma_wait = AREG(ACCEL_BUS_DMA_WAIT);
    t->error = (AREG(ACCEL_STATUS) & STATUS_ERROR) != 0;
}

static uint64_t done_time(void) {
    return ((uint64_t)AREG(ACCEL_DONE_HI) << 32) | AREG(ACCEL_DONE_LO);
}

/* Rows per tile: about rows/tiles, but every tile must fit the scratchpad
 * (half of it with double buffering). Tile input bytes = a0 + rows * a1
 * (a0 covers the halo of a convolution), output bytes = rows * c1. */
static uint32_t pick_tile_rows(uint32_t rows, uint32_t tiles, int dbuf, int64_t a0, int64_t a1, int64_t c1) {
    const int64_t cap_a = dbuf ? ACCEL_SPM_A_SIZE / 2 : ACCEL_SPM_A_SIZE;
    const int64_t cap_c = dbuf ? ACCEL_SPM_C_SIZE / 2 : ACCEL_SPM_C_SIZE;
    uint32_t tr = (rows + tiles - 1) / tiles;
    while (tr > 1 && (a0 + (int64_t)tr * a1 > cap_a || (int64_t)tr * c1 > cap_c)) tr--;
    return tr;
}

static int start_and_wait(uint32_t ctrl, const accel_policy_t *pol, uint64_t t0, accel_xtimes_t *t) {
    if (pol->wait != ACCEL_WAIT_POLL) {
        accel_irq_flag = 0;
        ctrl |= CTRL_IRQ;
    }
    uint64_t t1 = now_cycles();
    AREG(ACCEL_CTRL) = ctrl;
    uint64_t seen;
    if (pol->wait == ACCEL_WAIT_POLL) {
        while (!(AREG(ACCEL_STATUS) & STATUS_DONE)) {
        }
        seen = now_cycles();
    } else if (pol->wait == ACCEL_WAIT_WFI) {
        t->sleep = irq_wait_wfi();
        seen = accel_irq_entry;
    } else {
        uint64_t units = 0;
        while (!accel_irq_flag) {
            pol->background();
            units++;
        }
        t->bg_units = units;
        seen = accel_irq_entry;
    }
    uint64_t t2 = now_cycles();
    t->in = t1 - t0;
    t->run = t2 - t1;
    t->resp = seen - done_time();
    accel_read_counters(t);
    return t->error ? -1 : 0;
}

int accel_gemm_x(const int8_t *A, const int8_t *B, int32_t *C, uint32_t M, uint32_t K, uint32_t N,
                 const accel_policy_t *pol, accel_xtimes_t *t) {
    uint32_t ctrl = CTRL_START | CTRL_USE_DMA, tr = 0;
    *t = (accel_xtimes_t){0};
    if (pol->tiles) {
        tr = pick_tile_rows(M, pol->tiles, pol->dbuf, 0, K, (int64_t)N * 4);
        ctrl |= pol->dbuf ? CTRL_DBUF : CTRL_TILE;
    }
    uint64_t t0 = now_cycles();
    AREG(ACCEL_CTRL) = CTRL_CLEAR_DONE;
    AREG(ACCEL_OP) = OP_GEMM;
    AREG(ACCEL_M) = M;
    AREG(ACCEL_K) = K;
    AREG(ACCEL_N) = N;
    AREG(ACCEL_A_ADDR) = (uint32_t)(uintptr_t)A;
    AREG(ACCEL_B_ADDR) = (uint32_t)(uintptr_t)B;
    AREG(ACCEL_C_ADDR) = (uint32_t)(uintptr_t)C;
    if (tr) AREG(ACCEL_TILE_ROWS) = tr;
    return start_and_wait(ctrl, pol, t0, t);
}

int accel_conv_x(const int8_t *X, const int8_t *Wt, int32_t *Y, const accel_conv_t *cv,
                 const accel_policy_t *pol, accel_xtimes_t *t) {
    uint32_t ctrl = CTRL_START | CTRL_USE_DMA, tr = 0;
    *t = (accel_xtimes_t){0};
    if (pol->tiles) {
        int64_t row = (int64_t)cv->W * cv->C;  /* one input row in bytes */
        tr = pick_tile_rows(cv->Ho, pol->tiles, pol->dbuf, ((int64_t)cv->R - cv->stride) * row,
                            (int64_t)cv->stride * row, (int64_t)cv->Wo * cv->KOUT * 4);
        ctrl |= pol->dbuf ? CTRL_DBUF : CTRL_TILE;
    }
    uint64_t t0 = now_cycles();
    AREG(ACCEL_CTRL) = CTRL_CLEAR_DONE;
    AREG(ACCEL_OP) = OP_CONV;
    AREG(ACCEL_CONV_H) = cv->H;
    AREG(ACCEL_CONV_W) = cv->W;
    AREG(ACCEL_CONV_C) = cv->C;
    AREG(ACCEL_CONV_R) = cv->R;
    AREG(ACCEL_CONV_S) = cv->S;
    AREG(ACCEL_CONV_KOUT) = cv->KOUT;
    AREG(ACCEL_CONV_G) = cv->G;
    AREG(ACCEL_CONV_STRIDE) = cv->stride;
    AREG(ACCEL_CONV_PAD_T) = cv->pad_top;
    AREG(ACCEL_CONV_PAD_L) = cv->pad_left;
    AREG(ACCEL_CONV_HO) = cv->Ho;
    AREG(ACCEL_CONV_WO) = cv->Wo;
    AREG(ACCEL_A_ADDR) = (uint32_t)(uintptr_t)X;
    AREG(ACCEL_B_ADDR) = (uint32_t)(uintptr_t)Wt;
    AREG(ACCEL_C_ADDR) = (uint32_t)(uintptr_t)Y;
    if (tr) AREG(ACCEL_TILE_ROWS) = tr;
    return start_and_wait(ctrl, pol, t0, t);
}
