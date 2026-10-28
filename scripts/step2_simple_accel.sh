#!/usr/bin/env bash
# STEP 2 - offload GEMMs to a simple rigid accelerator: a 16x16 TPU-like
# output-stationary systolic mesh (256 multipliers). The CPU still does
# im2col and every depthwise layer ("gemm" partition).
# Each layer runs three times: SW, CPU-copy and DMA; every accelerated
# result is checked against the CPU result.
#   bash scripts/step2_simple_accel.sh            (~4-6 min)
#   MODEL=dscnn bash scripts/step2_simple_accel.sh
set -e
cd "$(dirname "$0")/.."
make -s -C sw acc ONLY_MODEL=$MODEL
sfx=${MODEL:+_$MODEL}
bash scripts/run_vp.sh bench_acc$sfx.elf step2_tpu$sfx.csv --accel-arch tpu --accel-rows 16 --accel-cols 16
python3 scripts/summarize.py results/step2_tpu$sfx.csv --layers
