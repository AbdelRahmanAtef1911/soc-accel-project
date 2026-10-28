#include "runtime.h"

static long sys_write(int fd, const void *buf, long n) {
    register long a0 asm("a0") = fd;
    register long a1 asm("a1") = (long)buf;
    register long a2 asm("a2") = n;
    register long a7 asm("a7") = 64; /* SYS_WRITE */
    asm volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7) : "memory");
    return a0;
}

void print_char(char c) { sys_write(1, &c, 1); }

void print_str(const char *s) { sys_write(1, s, (long)strlen(s)); }

void print_u64(uint64_t v) {
    char b[24];
    int i = 23;
    b[i] = 0;
    do {
        b[--i] = (char)('0' + v % 10);
        v /= 10;
    } while (v);
    print_str(b + i);
}

void print_i32(int32_t v) {
    if (v < 0) {
        print_char('-');
        print_u64((uint64_t)(-(int64_t)v));
    } else {
        print_u64((uint64_t)v);
    }
}

void *memcpy(void *dst, const void *src, size_t n) {
    uint8_t *d = (uint8_t *)dst;
    const uint8_t *s = (const uint8_t *)src;
    while (n--) *d++ = *s++;
    return dst;
}

void *memset(void *dst, int c, size_t n) {
    uint8_t *d = (uint8_t *)dst;
    while (n--) *d++ = (uint8_t)c;
    return dst;
}

size_t strlen(const char *s) {
    size_t n = 0;
    while (s[n]) n++;
    return n;
}
