#include "accel_driver.h"

#include "accel_regs.h"
#include "runtime.h"

#define AREG(off) REG32(ACCEL_BASE + (off))

int accel_present(void) { return AREG(ACCEL_ID) == ACCEL_ID_VALUE; }
uint32_t accel_arch(void) { return AREG(ACCEL_ARCH); }
uint32_t accel_num_pe(void) { return AREG(ACCEL_NUM_PE); }
uint32_t accel_dn_bw(void) { return AREG(ACCEL_DN_BW); }
uint32_t accel_caps(void) { return AREG(ACCEL_CAPS); }

/* copy n bytes into a scratchpad window with 32-bit stores */
static void copy_to_spm(uint32_t window, const int8_t *src, uint32_t n) {
    volatile uint32_t *dst = (volatile uint32_t *)(uintptr_t)(ACCEL_BASE + window);
    uint32_t words = n / 4, i;
    if (((uintptr_t)src & 3) == 0) {
        const uint32_t *s = (const uint32_t *)src;
        for (i = 0; i < words; i++) dst[i] = s[i];
    } else {
        for (i = 0; i < words; i++) {
            const uint8_t *b = (const uint8_t *)src + 4 * i;
            dst[i] = b[0] | (b[1] << 8) | (b[2] << 16) | ((uint32_t)b[3] << 24);
        }
    }
    if (n & 3) { /* last partial word */
        uint32_t w = 0;
        for (uint32_t j = 0; j < (n & 3); j++) w |= (uint32_t)(uint8_t)src[4 * words + j] << (8 * j);
        dst[words] = w;
    }
}

/* start the job, wait for DONE, read C/Y back (CPU-copy), fill t */
static int finish(int32_t *C, uint32_t out_words, int mode, uint64_t t0, accel_times_t *t) {
    uint64_t t1 = now_cycles();
    AREG(ACCEL_CTRL) = CTRL_START | (mode == ACCEL_DMA ? CTRL_USE_DMA : 0);
    while (!(AREG(ACCEL_STATUS) & STATUS_DONE)) {
        /* busy-wait: every poll is one bus access to the accelerator */
    }
    uint64_t t2 = now_cycles();

    if (mode != ACCEL_DMA) {
        volatile uint32_t *src = (volatile uint32_t *)(uintptr_t)(ACCEL_BASE + ACCEL_SPM_C);
        for (uint32_t i = 0; i < out_words; i++) C[i] = (int32_t)src[i];
    }
    uint64_t t3 = now_cycles();

    t->in = t1 - t0;
    t->run = t2 - t1;
    t->out = t3 - t2;
    t->acc_load = AREG(ACCEL_CYC_LOAD);
    t->acc_compute = AREG(ACCEL_CYC_COMPUTE);
    t->acc_store = AREG(ACCEL_CYC_STORE);
    t->error = (AREG(ACCEL_STATUS) & STATUS_ERROR) != 0;
    return t->error ? -1 : 0;
}

int accel_gemm(const int8_t *A, const int8_t *B, int32_t *C, uint32_t M, uint32_t K, uint32_t N,
               int mode, accel_times_t *t) {
    uint64_t t0 = now_cycles();

    AREG(ACCEL_CTRL) = CTRL_CLEAR_DONE;
    AREG(ACCEL_OP) = OP_GEMM;
    AREG(ACCEL_M) = M;
    AREG(ACCEL_K) = K;
    AREG(ACCEL_N) = N;
    if (mode == ACCEL_DMA) {
        AREG(ACCEL_A_ADDR) = (uint32_t)(uintptr_t)A;
        AREG(ACCEL_B_ADDR) = (uint32_t)(uintptr_t)B;
        AREG(ACCEL_C_ADDR) = (uint32_t)(uintptr_t)C;
    } else {
        copy_to_spm(ACCEL_SPM_A, A, M * K);
        copy_to_spm(ACCEL_SPM_B, B, N * K);
    }
    return finish(C, M * N, mode, t0, t);
}

int accel_conv(const int8_t *X, const int8_t *Wt, int32_t *Y, const accel_conv_t *cv,
               int mode, accel_times_t *t) {
    uint64_t t0 = now_cycles();
    int dw = cv->G > 1;

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
    if (mode == ACCEL_DMA) {
        AREG(ACCEL_A_ADDR) = (uint32_t)(uintptr_t)X;
        AREG(ACCEL_B_ADDR) = (uint32_t)(uintptr_t)Wt;
        AREG(ACCEL_C_ADDR) = (uint32_t)(uintptr_t)Y;
    } else {
        copy_to_spm(ACCEL_SPM_A, X, cv->H * cv->W * cv->C);
        copy_to_spm(ACCEL_SPM_B, Wt, cv->R * cv->S * cv->C * (dw ? 1 : cv->KOUT));
    }
    return finish(Y, cv->Ho * cv->Wo * cv->KOUT, mode, t0, t);
}
