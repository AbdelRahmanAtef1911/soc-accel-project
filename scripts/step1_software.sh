#!/usr/bin/env bash
# STEP 1 - the whole workload in software on the RISC-V CPU (the baseline).
#   bash scripts/step1_software.sh            all three models (~2-3 min)
#   MODEL=dscnn bash scripts/step1_software.sh one model only
set -e
cd "$(dirname "$0")/.."
make -s -C sw all ONLY_MODEL=$MODEL
elf=bench_sw${MODEL:+_$MODEL}.elf
bash scripts/run_vp.sh $elf step1_sw${MODEL:+_$MODEL}.csv --accel-arch tpu
python3 scripts/summarize.py results/step1_sw${MODEL:+_$MODEL}.csv
