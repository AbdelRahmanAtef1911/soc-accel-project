/*
 * irq.h - step 5: the accelerator's completion interrupt on the CPU side.
 *
 *   accelerator --(irq line)--> PLIC source 1 --(MEIP)--> RV32 core
 *
 * irq_init() installs the trap handler (trap.S + irq.c), programs the PLIC
 * (priority, enable, threshold) and enables the machine external interrupt.
 * The handler claims the interrupt at the PLIC, acknowledges it at the
 * accelerator (IRQ_STATUS), completes it at the PLIC and sets accel_irq_flag.
 */
#ifndef IRQ_H
#define IRQ_H

#include <stdint.h>

#define MSTATUS_MIE (1u << 3)
#define MIE_MEIE    (1u << 11)
#define MCAUSE_MEI  0x8000000Bu   /* machine external interrupt */

#define csr_read(reg) ({ uint32_t v_; asm volatile("csrr %0, " #reg : "=r"(v_)); v_; })
#define csr_write(reg, val) asm volatile("csrw " #reg ", %0" ::"r"((uint32_t)(val)))
#define csr_set(reg, val) asm volatile("csrs " #reg ", %0" ::"r"((uint32_t)(val)))
#define csr_clear(reg, val) asm volatile("csrc " #reg ", %0" ::"r"((uint32_t)(val)))

extern volatile uint32_t accel_irq_flag;   /* set by the handler, cleared by the driver */
extern volatile uint32_t accel_irq_count;  /* interrupts handled so far                 */
extern volatile uint64_t accel_irq_entry;  /* NOW (cycles) at handler entry, last one   */

void irq_init(void);

/* Sleep with WFI until the handler has set accel_irq_flag. Interrupts are
 * masked while the flag is checked, so an interrupt that arrives just before
 * WFI cannot be lost (WFI still wakes up on a pending, enabled interrupt).
 * Returns the cycles spent asleep. */
uint64_t irq_wait_wfi(void);

#endif
