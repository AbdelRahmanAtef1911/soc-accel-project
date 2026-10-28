#include "kernels.h"

void conv_sw(const layer_t *L, const int8_t *X, const int8_t *Wt, int32_t *Y) {
    const int H = L->H, W = L->W, C = L->C, R = L->R, S = L->S, K = L->K;
    for (int oy = 0; oy < L->Ho; oy++)
        for (int ox = 0; ox < L->Wo; ox++)
            for (int oc = 0; oc < K; oc++) {
                int32_t acc = 0;
                for (int ky = 0; ky < R; ky++) {
                    int iy = oy * L->sh - L->pt + ky;
                    if (iy < 0 || iy >= H) continue;
                    for (int kx = 0; kx < S; kx++) {
                        int ix = ox * L->sw - L->pl + kx;
                        if (ix < 0 || ix >= W) continue;
                        const int8_t *xp = &X[(iy * W + ix) * C];
                        const int8_t *wp = &Wt[((oc * R + ky) * S + kx) * C];
                        for (int c = 0; c < C; c++) acc += xp[c] * wp[c];
                    }
                }
                Y[(oy * L->Wo + ox) * K + oc] = acc;
            }
}

void dwconv_sw(const layer_t *L, const int8_t *X, const int8_t *Wt, int32_t *Y) {
    const int H = L->H, W = L->W, C = L->C, R = L->R, S = L->S;
    for (int oy = 0; oy < L->Ho; oy++)
        for (int ox = 0; ox < L->Wo; ox++)
            for (int c = 0; c < C; c++) {
                int32_t acc = 0;
                for (int ky = 0; ky < R; ky++) {
                    int iy = oy * L->sh - L->pt + ky;
                    if (iy < 0 || iy >= H) continue;
                    for (int kx = 0; kx < S; kx++) {
                        int ix = ox * L->sw - L->pl + kx;
                        if (ix < 0 || ix >= W) continue;
                        acc += X[(iy * W + ix) * C + c] * Wt[(ky * S + kx) * C + c];
                    }
                }
                Y[(oy * L->Wo + ox) * C + c] = acc;
            }
}

void fc_sw(const layer_t *L, const int8_t *X, const int8_t *Wt, int32_t *Y) {
    const int Kin = L->C, N = L->K;
    for (int n = 0; n < N; n++) {
        int32_t acc = 0;
        for (int k = 0; k < Kin; k++) acc += X[k] * Wt[n * Kin + k];
        Y[n] = acc;
    }
}

void im2col(const layer_t *L, const int8_t *X, int8_t *A) {
    const int H = L->H, W = L->W, C = L->C, R = L->R, S = L->S;
    const int row = R * S * C;
    for (int oy = 0; oy < L->Ho; oy++)
        for (int ox = 0; ox < L->Wo; ox++) {
            int8_t *a = &A[(oy * L->Wo + ox) * row];
            for (int ky = 0; ky < R; ky++) {
                int iy = oy * L->sh - L->pt + ky;
                for (int kx = 0; kx < S; kx++) {
                    int ix = ox * L->sw - L->pl + kx;
                    int8_t *dst = &a[(ky * S + kx) * C];
                    if (iy < 0 || iy >= H || ix < 0 || ix >= W) {
                        for (int c = 0; c < C; c++) dst[c] = 0;
                    } else {
                        const int8_t *src = &X[(iy * W + ix) * C];
                        for (int c = 0; c < C; c++) dst[c] = src[c];
                    }
                }
            }
        }
}
