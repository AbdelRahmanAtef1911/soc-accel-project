#!/usr/bin/env bash
# regress_all.sh - rerun every step (0-5, both sweeps, the gap sweep) and check that each
# result CSV is byte-identical to reference_results/. Use it after changing
# the platform, the accelerator model or STONNE.
#
#   bash scripts/regress_all.sh          (about 2-2.5 h with 2 cores)
#   SKIP_RUNS=1 bash scripts/regress_all.sh   (only compare results/ with reference_results/)
#
# Exit code 0 = every file identical.
set -e
cd "$(dirname "$0")/.."
source setup/env.sh

if [ -z "$SKIP_RUNS" ]; then
    (cd unit_test && make -s && make -s test && make -s test5) | grep -E "PASSED|FAIL" || true
    (cd unit_test && make -s test_rg) | grep -E "PASSED|FAIL|test 8" || true   # project bus, tests 1-8
    bash scripts/step0_reproduce_stonne.sh
    bash scripts/step1_software.sh
    bash scripts/step2_simple_accel.sh
    bash scripts/step3_better_accel.sh
    python3 scripts/dse.py
    bash scripts/step5_dma_extension.sh
    python3 scripts/gap_sweep.py
fi

same=0; diff=0; missing=0
for ref in reference_results/*.csv reference_results/dse/*.csv reference_results/dse5/*.csv \
           reference_results/gap/*.csv; do
    new="results/${ref#reference_results/}"
    if [ ! -f "$new" ]; then echo "MISSING   $new"; missing=$((missing + 1))
    elif cmp -s "$ref" "$new"; then same=$((same + 1))
    else echo "DIFFERENT $new"; diff=$((diff + 1)); fi
done
echo "regress_all: $same identical, $diff different, $missing missing"
[ "$diff" = 0 ] && [ "$missing" = 0 ]
