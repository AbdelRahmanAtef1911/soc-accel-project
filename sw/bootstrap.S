# Entry point for the bare-metal benchmark. The VP sets the stack pointer
# to the top of memory before jumping here.
.globl _start
.globl main

_start:
    jal main
    # exit(main's return value): SYS_EXIT = 93, code already in a0
    li a7, 93
    ecall
1:  j 1b
