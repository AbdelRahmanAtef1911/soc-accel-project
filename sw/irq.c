#include "irq.h"

#include "accel_regs.h"
#include "runtime.h"

volatile uint32_t accel_irq_flag;
volatile uint32_t accel_irq_count;
volatile uint64_t accel_irq_entry;

void trap_entry(void);  /* trap.S */

static void die(const char *msg, uint32_t v) {
    print_str(msg);
    print_u64(v);
    print_char('\n');
    asm volatile("li a0, 1\n li a7, 93\n ecall");  /* exit(1) through the VP */
    for (;;) {}
}

/* called from trap.S with the caller-saved registers already saved */
void trap_handler(void) {
    uint32_t cause = csr_read(mcause);
    if (cause != MCAUSE_MEI) die("ERROR: unexpected trap, mcause=", cause);

    uint64_t t = now_cycles();
    uint32_t id = REG32(PLIC_CLAIM);                    /* claim: which source?        */
    if (id == ACCEL_IRQ_ID) {
        REG32(ACCEL_BASE + ACCEL_IRQ_STATUS) = 1;       /* acknowledge at the device   */
        accel_irq_entry = t;
        accel_irq_count++;
        accel_irq_flag = 1;
    }
    REG32(PLIC_CLAIM) = id;                             /* complete                     */
}

void irq_init(void) {
    csr_write(mtvec, (uint32_t)(uintptr_t)trap_entry);  /* direct mode */
    REG32(PLIC_PRIORITY(ACCEL_IRQ_ID)) = 1;
    REG32(PLIC_THRESHOLD) = 0;
    REG32(PLIC_ENABLE) = 1u << ACCEL_IRQ_ID;
    csr_set(mie, MIE_MEIE);
    csr_set(mstatus, MSTATUS_MIE);
}

uint64_t irq_wait_wfi(void) {
    uint64_t slept = 0;
    for (;;) {
        csr_clear(mstatus, MSTATUS_MIE);
        if (accel_irq_flag) break;
        uint64_t a = now_cycles();
        asm volatile("wfi");
        slept += now_cycles() - a;
        csr_set(mstatus, MSTATUS_MIE);  /* the pending interrupt is taken here */
    }
    csr_set(mstatus, MSTATUS_MIE);
    return slept;
}
