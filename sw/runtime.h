/*
 * runtime.h - the few things a bare-metal program needs: printing through
 * the VP's syscall interface (--intercept-syscalls), a cycle timer, and the
 * memory functions GCC may call on its own.
 */
#ifndef RUNTIME_H
#define RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#include "accel_regs.h"

#define REG32(addr) (*(volatile uint32_t *)(uintptr_t)(addr))

void print_str(const char *s);
void print_u64(uint64_t v);
void print_i32(int32_t v);
void print_char(char c);

/* Simulated time in cycles (100 MHz), read from the accelerator's timer
 * registers. This is the platform's global clock, so it includes bus and
 * memory delays as well as instruction time. */
static inline uint64_t now_cycles(void) {
    uint32_t hi, lo, hi2;
    do {
        hi = REG32(ACCEL_BASE + ACCEL_NOW_HI);
        lo = REG32(ACCEL_BASE + ACCEL_NOW_LO);
        hi2 = REG32(ACCEL_BASE + ACCEL_NOW_HI);
    } while (hi != hi2);
    return ((uint64_t)hi << 32) | lo;
}

/* Instructions-based cycle counter of the ISS (for comparison only). */
static inline uint32_t rdcycle(void) {
    uint32_t c;
    asm volatile("rdcycle %0" : "=r"(c));
    return c;
}

void *memcpy(void *dst, const void *src, size_t n);
void *memset(void *dst, int c, size_t n);
size_t strlen(const char *s);

#endif
