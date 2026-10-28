/*
 * accel_regs.h - register map of the accelerator.
 *
 * Shared by the SystemC model (accel/stonne_accel.h) and the bare-metal
 * driver that runs on the RISC-V ISS (sw/accel_driver.c), so both sides
 * always agree on the addresses. Plain C, no SystemC includes.
 *
 * Two job types (register OP):
 *
 *  OP_GEMM  C[M][N] = A[M][K] x B[N][K]^T            (all three architectures)
 *    A : int8, row-major, one row per output pixel (im2col rows)
 *    B : int8, row-major, one row per filter (TFLite OHWI weight layout)
 *    C : int32, row-major (NHWC output before requantisation)
 *
 *  OP_CONV  Y = conv(X, Wt), padding made by the accelerator  (MAERI only)
 *    X  : int8 [H][W][C]            (A_ADDR / scratchpad A)
 *    Wt : int8 [KOUT][R][S][C]      standard conv, G = 1
 *         int8 [R][S][C]            depthwise conv, G = C = KOUT
 *    Y  : int32 [HO][WO][KOUT]      (C_ADDR / scratchpad C)
 *
 * Step 5 (DMA/bus extension) adds tiling + double buffering (CTRL_TILE,
 * CTRL_DBUF, TILE_ROWS), a completion interrupt (CTRL_IRQ, IRQ_STATUS, PLIC
 * source 1), the DMA bus-ownership mode and burst size (DMA_MODE, DMA_BURST)
 * and per-job counters (CYC_TOTAL, POLLS, DONE time, shared-bus waits).
 * Software that does not touch them sees the accelerator of steps 2-4.
 */
#ifndef ACCEL_REGS_H
#define ACCEL_REGS_H

#define ACCEL_BASE        0x10000000u   /* where the platform maps the accelerator */
#define ACCEL_SIZE        0x00400000u   /* registers + three scratchpad windows    */

/* ---- registers (offsets from ACCEL_BASE, 32-bit each) ---- */
#define ACCEL_ID          0x00  /* RO  0x53544E45 = "STNE"                          */
#define ACCEL_CTRL        0x04  /* WO  bit0 START, bit1 USE_DMA, bit2 CLEAR_DONE,
                                         bit3 TILE, bit4 DBUF, bit5 IRQ (step 5)     */
#define ACCEL_STATUS      0x08  /* RO  bit0 BUSY,  bit1 DONE,    bit2 ERROR          */
#define ACCEL_M           0x0C  /* RW  GEMM: rows of A / C                           */
#define ACCEL_K           0x10  /* RW  GEMM: reduction length                        */
#define ACCEL_N           0x14  /* RW  GEMM: rows of B / columns of C                */
#define ACCEL_A_ADDR      0x18  /* RW  main-memory address of A or X (DMA mode)      */
#define ACCEL_B_ADDR      0x1C  /* RW  main-memory address of B or Wt (DMA mode)     */
#define ACCEL_C_ADDR      0x20  /* RW  main-memory address of C or Y (DMA mode)      */
#define ACCEL_ARCH        0x24  /* RO  0 = TPU-like, 1 = MAERI, 2 = SIGMA            */
#define ACCEL_NUM_PE      0x28  /* RO  number of multipliers                         */
#define ACCEL_DN_BW       0x2C  /* RO  distribution-network bandwidth (elems/cycle)  */
#define ACCEL_CYC_LOAD    0x30  /* RO  cycles spent by the DMA loading the inputs    */
#define ACCEL_CYC_COMPUTE 0x34  /* RO  cycles reported by STONNE for the job         */
#define ACCEL_CYC_STORE   0x38  /* RO  cycles spent by the DMA storing the output    */
#define ACCEL_BYTES_IN    0x3C  /* RO  bytes moved into the accelerator (last job)   */
#define ACCEL_BYTES_OUT   0x40  /* RO  bytes moved out of the accelerator (last job) */
#define ACCEL_NOW_LO      0x44  /* RO  simulated time in cycles, low word            */
#define ACCEL_NOW_HI      0x48  /* RO  simulated time in cycles, high word           */
#define ACCEL_JOBS        0x4C  /* RO  number of finished jobs                       */
#define ACCEL_OP          0x50  /* RW  0 = OP_GEMM, 1 = OP_CONV                      */
#define ACCEL_CAPS        0x54  /* RO  bit0 GEMM supported, bit1 CONV supported      */
#define ACCEL_CONV_H      0x58  /* RW  CONV: input height                            */
#define ACCEL_CONV_W      0x5C  /* RW  CONV: input width                             */
#define ACCEL_CONV_C      0x60  /* RW  CONV: input channels                          */
#define ACCEL_CONV_R      0x64  /* RW  CONV: kernel height                           */
#define ACCEL_CONV_S      0x68  /* RW  CONV: kernel width                            */
#define ACCEL_CONV_KOUT   0x6C  /* RW  CONV: output channels                         */
#define ACCEL_CONV_G      0x70  /* RW  CONV: groups (1 or C for depthwise)           */
#define ACCEL_CONV_STRIDE 0x74  /* RW  CONV: stride (same in both directions)        */
#define ACCEL_CONV_PAD_T  0x78  /* RW  CONV: zero rows added on top                  */
#define ACCEL_CONV_PAD_L  0x7C  /* RW  CONV: zero columns added on the left          */
#define ACCEL_CONV_HO     0x80  /* RW  CONV: output height                           */
#define ACCEL_CONV_WO     0x84  /* RW  CONV: output width                            */

/* ---- step 5: DMA/bus extension (all zero / unused by steps 1-4) ---- */
#define ACCEL_TILE_ROWS   0x88  /* RW  rows per tile when CTRL_TILE is set: GEMM rows
                                         of A/C, CONV output rows (input rows + halo
                                         are fetched automatically)                  */
#define ACCEL_IRQ_STATUS  0x8C  /* RW  bit0 = completion interrupt pending;
                                         write 1 to acknowledge (clears the line)    */
#define ACCEL_CYC_TOTAL   0x90  /* RO  cycles from START to DONE (last job)          */
#define ACCEL_TILES       0x94  /* RO  number of tiles of the last job              */
#define ACCEL_POLLS       0x98  /* RO  STATUS reads while the last job was busy      */
#define ACCEL_DONE_LO     0x9C  /* RO  time (cycles) the last job finished, low      */
#define ACCEL_DONE_HI     0xA0  /* RO  ... high word                                 */
#define ACCEL_DMA_MODE    0xA4  /* RW  bus ownership: DMA_MODE_BLOCK/STEAL/TRANSPARENT
                                         (only matters with a shared-bus platform)   */
#define ACCEL_DMA_BURST   0xA8  /* RW  DMA burst size in bytes (default: platform)   */
#define ACCEL_BUS_CPU_TXN 0xAC  /* RO  last job: CPU bus transactions while busy     */
#define ACCEL_BUS_CPU_WAIT 0xB0 /* RO  last job: cycles the CPU waited for the bus   */
#define ACCEL_BUS_CPU_MAXW 0xB4 /* RO  last job: longest single CPU wait (cycles)    */
#define ACCEL_BUS_DMA_WAIT 0xB8 /* RO  last job: cycles the DMA waited for the bus   */

#define ACCEL_ID_VALUE    0x53544E45u

#define OP_GEMM           0u
#define OP_CONV           1u
#define CAPS_GEMM         (1u << 0)
#define CAPS_CONV         (1u << 1)
#define CAPS_TILE         (1u << 2)   /* tiling + double buffering (step 5)      */
#define CAPS_IRQ          (1u << 3)   /* completion interrupt (step 5)           */
#define CAPS_BUS          (1u << 4)   /* platform has the shared-bus model       */

#define CTRL_START        (1u << 0)
#define CTRL_USE_DMA      (1u << 1)
#define CTRL_CLEAR_DONE   (1u << 2)
#define CTRL_TILE         (1u << 3)   /* split the job into tiles of TILE_ROWS rows */
#define CTRL_DBUF         (1u << 4)   /* + double buffering: DMA overlaps compute   */
#define CTRL_IRQ          (1u << 5)   /* raise the interrupt when this job is done  */

#define STATUS_BUSY       (1u << 0)
#define STATUS_DONE       (1u << 1)
#define STATUS_ERROR      (1u << 2)

#define DMA_MODE_BLOCK       0u  /* burst/block: hold the bus for the whole transfer */
#define DMA_MODE_STEAL       1u  /* cycle stealing: give the bus back after every burst */
#define DMA_MODE_TRANSPARENT 2u  /* only use bus cycles the CPU leaves idle          */

/* ---- interrupt controller of the accel32-vp platform (FE310-style PLIC) ---- */
#define PLIC_BASE         0x0C000000u
#define PLIC_PRIORITY(id) (PLIC_BASE + 4u * (id))
#define PLIC_ENABLE       (PLIC_BASE + 0x2000u)    /* hart 0, sources 0..31  */
#define PLIC_THRESHOLD    (PLIC_BASE + 0x200000u)  /* hart 0                 */
#define PLIC_CLAIM        (PLIC_BASE + 0x200004u)  /* read = claim, write = complete */
#define ACCEL_IRQ_ID      1u                       /* PLIC source of the accelerator */

/* ---- scratchpad windows inside the accelerator ---- */
#define ACCEL_SPM_A       0x00100000u   /* int8  A or X,  up to ACCEL_SPM_A_SIZE bytes */
#define ACCEL_SPM_B       0x00200000u   /* int8  B or Wt, up to ACCEL_SPM_B_SIZE bytes */
#define ACCEL_SPM_C       0x00300000u   /* int32 C or Y,  up to ACCEL_SPM_C_SIZE bytes */
#define ACCEL_SPM_A_SIZE  (256u * 1024u)
#define ACCEL_SPM_B_SIZE  (128u * 1024u)
#define ACCEL_SPM_C_SIZE  (256u * 1024u)

#endif /* ACCEL_REGS_H */
