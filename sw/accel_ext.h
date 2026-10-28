/*
 * accel_ext.h - step 5 driver: the same GEMM / CONV jobs as accel_driver.h,
 * DMA mode only, with three new choices per job:
 *
 *   tiles  0   the whole job in one piece (as in steps 2-4)
 *          n   cut it into about n tiles (TILE_ROWS = ceil(rows / n), made
 *              smaller if a tile would not fit the scratchpad / half of it)
 *   dbuf   1   double buffering: the accelerator's DMA loads tile i+1 and
 *              stores tile i-1 while tile i is computed (needs tiles >= 1)
 *   wait       how the CPU waits for the job:
 *          ACCEL_WAIT_POLL  read STATUS in a loop (every read is a bus access)
 *          ACCEL_WAIT_WFI   completion interrupt, sleep with WFI meanwhile
 *          ACCEL_WAIT_BG    completion interrupt, run background work meanwhile
 *
 * accel_driver.c is left exactly as it was, so steps 2-4 do not change.
 */
#ifndef ACCEL_EXT_H
#define ACCEL_EXT_H

#include <stdint.h>

#include "accel_driver.h"

enum { ACCEL_WAIT_POLL = 0, ACCEL_WAIT_WFI = 1, ACCEL_WAIT_BG = 2 };

typedef struct {
    uint32_t tiles;
    int dbuf;
    int wait;
    void (*background)(void);  /* one unit of background work (ACCEL_WAIT_BG) */
} accel_policy_t;

typedef struct {
    uint64_t in;           /* CPU: configure the job (cycles)                    */
    uint64_t run;          /* CPU: START until the CPU carries on               */
    uint32_t acc_load, acc_compute, acc_store;  /* accelerator: per-phase sums   */
    uint32_t acc_total;    /* accelerator: START -> DONE                         */
    uint32_t tiles;        /* tiles the accelerator used                         */
    uint32_t bytes_in;     /* bytes the DMA read (tile halos are read twice)     */
    uint32_t polls;        /* STATUS reads while busy                            */
    uint64_t sleep;        /* ACCEL_WAIT_WFI: cycles asleep                      */
    uint64_t bg_units;     /* ACCEL_WAIT_BG: background units done               */
    uint64_t resp;         /* DONE -> CPU reacts (poll loop exit / handler entry) */
    uint32_t cpu_txn, cpu_wait, cpu_maxw, dma_wait;  /* shared-bus counters      */
    int error;
} accel_xtimes_t;

void accel_irq_setup(void);  /* once, before using ACCEL_WAIT_WFI / ACCEL_WAIT_BG */

int accel_gemm_x(const int8_t *A, const int8_t *B, int32_t *C, uint32_t M, uint32_t K, uint32_t N,
                 const accel_policy_t *pol, accel_xtimes_t *t);
int accel_conv_x(const int8_t *X, const int8_t *Wt, int32_t *Y, const accel_conv_t *cv,
                 const accel_policy_t *pol, accel_xtimes_t *t);

/* after a job run with the original driver (accel_driver.c): read the
 * step-5 counters of that job into t (in/run are not touched) */
void accel_read_counters(accel_xtimes_t *t);

uint32_t accel_dma_mode(void);
uint32_t accel_dma_burst(void);

#endif
