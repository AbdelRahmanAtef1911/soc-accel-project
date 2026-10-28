#!/usr/bin/env bash
# STEP 3 - a better accelerator, in two moves so their effects can be separated:
#   3a  same partition as step 2, flexible MAERI fabric (256 multipliers,
#       256-wide distribution network): does a better compute engine help?
#   3b  MAERI + native convolution (OP_CONV): the accelerator also takes the
#       im2col work and the depthwise layers off the CPU ("conv" partition).
#   bash scripts/step3_better_accel.sh            (~8-12 min)
#   MODEL=dscnn bash scripts/step3_better_accel.sh
set -e
cd "$(dirname "$0")/.."
make -s -C sw acc conv ONLY_MODEL=$MODEL
sfx=${MODEL:+_$MODEL}
bash scripts/run_vp.sh bench_acc$sfx.elf  step3a_maeri_gemm$sfx.csv --accel-arch maeri --accel-ms 256 --accel-dnbw 256
bash scripts/run_vp.sh bench_conv$sfx.elf step3b_maeri_conv$sfx.csv --accel-arch maeri --accel-ms 256 --accel-dnbw 256
python3 scripts/summarize.py results/step3a_maeri_gemm$sfx.csv results/step3b_maeri_conv$sfx.csv
