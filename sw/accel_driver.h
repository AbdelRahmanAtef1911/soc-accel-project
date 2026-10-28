/*
 * accel_driver.h - bare-metal driver for the GEMM accelerator.
 *
 *   C[M][N] = A[M][K] x B[N][K]^T   (A, B int8 in main memory, C int32)
 *
 * Two ways to move the data, chosen per call:
 *   ACCEL_CPU_COPY : the CPU writes A and B into the accelerator's scratchpads
 *                    and reads C back, one 32-bit bus access per word.
 *   ACCEL_DMA      : the CPU only writes the addresses; the accelerator's DMA
 *                    reads A/B from main memory and writes C back.
 */
#ifndef ACCEL_DRIVER_H
#define ACCEL_DRIVER_H

#include <stdint.h>

enum { ACCEL_CPU_COPY = 0, ACCEL_DMA = 1 };

typedef struct {
    uint64_t in;          /* CPU view: configure + move A/B in (cycles)   */
    uint64_t run;         /* CPU view: START until DONE seen              */
    uint64_t out;         /* CPU view: move C out                         */
    uint32_t acc_load;    /* accelerator view: DMA load cycles            */
    uint32_t acc_compute; /* accelerator view: STONNE compute cycles      */
    uint32_t acc_store;   /* accelerator view: DMA store cycles           */
    int error;            /* STATUS.ERROR after the job                   */
} accel_times_t;

int accel_present(void);
uint32_t accel_arch(void);   /* 0 TPU, 1 MAERI, 2 SIGMA */
uint32_t accel_num_pe(void);
uint32_t accel_dn_bw(void);

uint32_t accel_caps(void);   /* CAPS_GEMM | CAPS_CONV */

int accel_gemm(const int8_t *A, const int8_t *B, int32_t *C, uint32_t M, uint32_t K, uint32_t N,
               int mode, accel_times_t *t);

/* Native convolution (OP_CONV, MAERI only): the accelerator reads the NHWC
 * input and the TFLite weights directly and adds the zero padding itself, so
 * the CPU does no im2col. Depthwise: pass G = C = KOUT. */
typedef struct {
    uint32_t H, W, C, R, S, KOUT, G, stride, pad_top, pad_left, Ho, Wo;
} accel_conv_t;

int accel_conv(const int8_t *X, const int8_t *Wt, int32_t *Y, const accel_conv_t *cv,
               int mode, accel_times_t *t);

#endif
