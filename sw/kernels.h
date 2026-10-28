/*
 * kernels.h - the software (CPU) side of the partition.
 *
 * Tensors use TFLite layouts:
 *   input  X : [H][W][C]          int8
 *   conv   Wt: [K][R][S][C]       int8   (one filter = R*S*C contiguous bytes)
 *   dw     Wt: [R][S][C]          int8
 *   output Y : [Ho][Wo][K]        int32  (before requantisation)
 */
#ifndef KERNELS_H
#define KERNELS_H

#include <stdint.h>

#include "layers.h"

/* Step 1 baseline: direct loops, everything on the CPU */
void conv_sw(const layer_t *L, const int8_t *X, const int8_t *Wt, int32_t *Y);
void dwconv_sw(const layer_t *L, const int8_t *X, const int8_t *Wt, int32_t *Y);
void fc_sw(const layer_t *L, const int8_t *X, const int8_t *Wt, int32_t *Y);

/* Lowering a convolution to a GEMM for the accelerator:
 * A[Ho*Wo][R*S*C], zero-filled where the window hits the padding. */
void im2col(const layer_t *L, const int8_t *X, int8_t *A);

#endif
