/* hello.c - first program on the platform: prints, reads the accelerator ID
 * and times a small loop with both clocks. */
#include "accel_regs.h"
#include "runtime.h"

static volatile int32_t buf[1000];

int main(void) {
    print_str("hello from RV32 on accel32-vp\n");
    uint32_t id = REG32(ACCEL_BASE + ACCEL_ID);
    print_str("accelerator ID register = ");
    print_u64(id);
    print_str(id == ACCEL_ID_VALUE ? " (STNE, ok)\n" : " (WRONG)\n");

    uint64_t t0 = now_cycles();
    uint32_t c0 = rdcycle();
    for (int i = 0; i < 1000; i++) buf[i] = i;
    int32_t s = 0;
    for (int i = 0; i < 1000; i++) s += buf[i];
    uint32_t c1 = rdcycle();
    uint64_t t1 = now_cycles();
    print_str("sum 0..999 = "); print_i32(s); print_str(s == 499500 ? " (ok)\n" : " (WRONG)\n");
    print_str("loop: simulated time "); print_u64(t1 - t0);
    print_str(" cycles, ISS rdcycle "); print_u64(c1 - c0); print_str(" cycles\n");
    return id == ACCEL_ID_VALUE && s == 499500 ? 0 : 1;
}
