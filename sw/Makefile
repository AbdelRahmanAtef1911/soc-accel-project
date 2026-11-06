# Bare-metal software for the RISC-V ISS (RV32IM, 100 MHz model).
#   make                      -> bench_sw.elf   (step 1: CPU only)
#   make acc                  -> bench_acc.elf  (steps 2-3: SW + CPU-copy + DMA)
#   make acc WZERO=50         -> pruned weights (50 % zeros), file bench_acc_wz50.elf
#   make acc ONLY_MODEL=dscnn -> one model only, file bench_acc_dscnn.elf
#   make conv                 -> bench_conv.elf (step 3b: "conv" partition, MAERI only)
#   make dse                  -> bench_dse.elf / bench_dse_conv.elf: DMA mode only, no CPU
#                                reference (fast design-space sweeps, check = SKIP)
#   make hello                -> hello.elf (toolchain check)
#
# Step 5 (DMA/bus extension, conv partition, DMA mode only):
#   make dma                  -> bench_dma_poll.elf: modes DMA, TILE, DBUF, CPU polls
#   make dma WAIT=wfi         -> bench_dma_wfi.elf:  completion interrupt + WFI
#   make dma WAIT=bg          -> bench_dma_bg.elf:   completion interrupt + background work
#   options: NTILES=8 (default 4), MODES=dbuf (or one / tile / one,dbuf ...),
#            CHECK=0 (no CPU reference, fast sweeps), ONLY_MODEL=dscnn,
#            PARTITION=gemm (im2col on the CPU, depthwise on the CPU)
#   e.g. make dma WAIT=bg MODES=dbuf CHECK=0 -> bench_dma_bg_dbuf_fast.elf

CC     := riscv64-unknown-elf-gcc
ARCH   := -march=rv32im_zicsr -mabi=ilp32
CFLAGS := $(ARCH) -O2 -Wall -ffreestanding -fno-builtin -fno-tree-loop-distribute-patterns \
          -I. -I../accel
LDFLAGS := $(ARCH) -nostdlib -nostartfiles -Wl,--no-relax
LIBS   := -lgcc

WZERO      ?= 0
ONLY_MODEL ?=
SUFFIX     := $(if $(filter-out 0,$(WZERO)),_wz$(WZERO))$(if $(ONLY_MODEL),_$(ONLY_MODEL))
DEFS       := -DWZERO=$(WZERO) $(if $(ONLY_MODEL),-DONLY_MODEL=\"$(ONLY_MODEL)\")

SRC := bootstrap.S runtime.c kernels.c accel_driver.c bench.c
HDR := runtime.h kernels.h accel_driver.h layers.h ../accel/accel_regs.h

all: bench_sw$(SUFFIX).elf
acc: bench_acc$(SUFFIX).elf
conv: bench_conv$(SUFFIX).elf
dse: bench_dse$(SUFFIX).elf bench_dse_conv$(SUFFIX).elf

bench_sw$(SUFFIX).elf: $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(DEFS) -DRUN_SW=1 -DRUN_CPU=0 -DRUN_DMA=0 $(LDFLAGS) $(SRC) $(LIBS) -o $@

bench_acc$(SUFFIX).elf: $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(DEFS) -DRUN_SW=1 -DRUN_CPU=1 -DRUN_DMA=1 $(LDFLAGS) $(SRC) $(LIBS) -o $@

bench_conv$(SUFFIX).elf: $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(DEFS) -DRUN_SW=1 -DRUN_CPU=1 -DRUN_DMA=1 -DOFFLOAD_CONV=1 $(LDFLAGS) $(SRC) $(LIBS) -o $@

bench_dse$(SUFFIX).elf: $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(DEFS) -DRUN_SW=0 -DRUN_CPU=0 -DRUN_DMA=1 -DCHECK=0 $(LDFLAGS) $(SRC) $(LIBS) -o $@

bench_dse_conv$(SUFFIX).elf: $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(DEFS) -DRUN_SW=0 -DRUN_CPU=0 -DRUN_DMA=1 -DCHECK=0 -DOFFLOAD_CONV=1 $(LDFLAGS) $(SRC) $(LIBS) -o $@

hello.elf: bootstrap.S runtime.c hello.c runtime.h
	$(CC) $(CFLAGS) $(LDFLAGS) bootstrap.S runtime.c hello.c $(LIBS) -o $@
hello: hello.elf

clean:
	rm -f *.elf

# ---------------- step 5 ----------------
WAIT      ?= poll
NTILES    ?= 4
MODES     ?= one,tile,dbuf
CHECK     ?= 1
PARTITION ?= conv
WAIT_NUM  := $(if $(filter wfi,$(WAIT)),1,$(if $(filter bg,$(WAIT)),2,0))
comma     := ,
SUFFIX5   := _$(WAIT)$(if $(filter-out 4,$(NTILES)),_t$(NTILES))$(if $(filter-out one$(comma)tile$(comma)dbuf,$(MODES)),_$(subst $(comma),-,$(MODES)))$(if $(filter 0,$(CHECK)),_fast)$(if $(filter gemm,$(PARTITION)),_gemm)$(if $(ONLY_MODEL),_$(ONLY_MODEL))
DEFS5     := -DWAIT=$(WAIT_NUM) -DNTILES=$(NTILES) -DCHECK=$(CHECK) \
             -DRUN_ONE=$(if $(findstring one,$(MODES)),1,0) -DRUN_TILE=$(if $(findstring tile,$(MODES)),1,0) \
             -DRUN_DBUF=$(if $(findstring dbuf,$(MODES)),1,0) -DOFFLOAD_CONV=$(if $(filter gemm,$(PARTITION)),0,1) \
             $(if $(ONLY_MODEL),-DONLY_MODEL=\"$(ONLY_MODEL)\")
SRC5 := bootstrap.S trap.S runtime.c kernels.c accel_driver.c accel_ext.c irq.c bench_dma.c
HDR5 := $(HDR) accel_ext.h irq.h

dma: bench_dma$(SUFFIX5).elf

bench_dma$(SUFFIX5).elf: $(SRC5) $(HDR5)
	$(CC) $(CFLAGS) $(DEFS5) $(LDFLAGS) $(SRC5) $(LIBS) -o $@

.PHONY: all acc conv dse hello clean dma

# ---------------- gap experiment: periodic real-time task during inference ----------------
#   make rt                   -> bench_rt.elf: 62 us timer task (16-tap FIR) + every layer on the
#                                accelerator (conv partition, WFI), modes DMA and DBUF
#   options: RT_PERIOD_US=62, FIR_TAPS=16, RT_BG=1 (CPU busy with background work instead of WFI),
#            NTILES=4, RT_CHECK=0 (1: compare with the CPU), ONLY_MODEL=dscnn
RT_PERIOD_US ?= 62
FIR_TAPS     ?= 16
RT_CHECK     ?= 0
RT_BG        ?= 0
SUFFIXRT := $(if $(filter 1,$(RT_BG)),_bg)$(if $(filter-out 62,$(RT_PERIOD_US)),_p$(RT_PERIOD_US))$(if $(filter-out 16,$(FIR_TAPS)),_f$(FIR_TAPS))$(if $(filter-out 4,$(NTILES)),_t$(NTILES))$(if $(filter 1,$(RT_CHECK)),_check)$(if $(ONLY_MODEL),_$(ONLY_MODEL))
DEFSRT   := -DRT_TIMER -DRT_BG=$(RT_BG) -DRT_PERIOD_US=$(RT_PERIOD_US) -DFIR_TAPS=$(FIR_TAPS) -DNTILES=$(NTILES) -DCHECK=$(RT_CHECK) \
            $(if $(ONLY_MODEL),-DONLY_MODEL=\"$(ONLY_MODEL)\")
SRCRT := bootstrap.S trap.S runtime.c kernels.c accel_driver.c accel_ext.c irq.c bench_rt.c

rt: bench_rt$(SUFFIXRT).elf

bench_rt$(SUFFIXRT).elf: $(SRCRT) $(HDR5)
	$(CC) $(CFLAGS) $(DEFSRT) $(LDFLAGS) $(SRCRT) $(LIBS) -o $@

.PHONY: rt
