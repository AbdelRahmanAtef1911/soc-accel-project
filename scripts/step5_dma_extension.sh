#!/usr/bin/env bash
# STEP 5 - once the computation is on the accelerator, how should the data
# movement work? Same accelerator as step 3b (MAERI, 256 multipliers,
# 256-wide network, "conv" partition, DMA mode), three SoC-level changes:
#   5a  tiling + double buffering: the DMA overlaps the computation
#       (ideal bus, the CPU polls; checked against the CPU; the one-piece
#       DMA rows must equal step 3b exactly)
#   5b  completion interrupt: the CPU sleeps in WFI instead of polling
#       (ideal bus; checked)
#   5c  shared bus: CPU and DMA compete for one bus (cycle stealing) while
#       the CPU runs background work between interrupts (checked)
#   5d  the sweeps (scripts/dse5.py, 28 runs, check = SKIP) and figure 4
#
#   bash scripts/step5_dma_extension.sh                (~30-45 min with 2 cores)
#   SKIP_SWEEPS=1 bash scripts/step5_dma_extension.sh  (5a-5c only, ~10 min)
#   MODEL=dscnn bash scripts/step5_dma_extension.sh    (5a-5c for one model, no sweeps)
set -e
cd "$(dirname "$0")/.."
MAERI="--accel-arch maeri --accel-ms 256 --accel-dnbw 256"
sfx=${MODEL:+_$MODEL}
make -s -C sw dma ONLY_MODEL=$MODEL
make -s -C sw dma WAIT=wfi ONLY_MODEL=$MODEL
make -s -C sw dma WAIT=bg ONLY_MODEL=$MODEL

echo "--- 5a: tiling and double buffering (ideal bus, the CPU polls)"
bash scripts/run_vp.sh bench_dma_poll$sfx.elf step5a_dbuf$sfx.csv $MAERI
ref=results/step3b_maeri_conv$sfx.csv
[ -f "$ref" ] || ref=reference_results/step3b_maeri_conv.csv
python3 scripts/summarize5.py results/step5a_dbuf$sfx.csv --compare "$ref"

echo "--- 5b: completion interrupt, the CPU sleeps in WFI (ideal bus)"
bash scripts/run_vp.sh bench_dma_wfi$sfx.elf step5b_irq_wfi$sfx.csv $MAERI

echo "--- 5c: shared bus, cycle stealing, the CPU runs background work while it waits"
bash scripts/run_vp.sh bench_dma_bg$sfx.elf step5c_bus_steal_bg$sfx.csv $MAERI --accel-bus-model --accel-dma-mode steal
python3 scripts/summarize5.py results/step5b_irq_wfi$sfx.csv results/step5c_bus_steal_bg$sfx.csv

if [ -z "$MODEL" ] && [ -z "$SKIP_SWEEPS" ]; then
    echo "--- 5d: sweeps"
    python3 scripts/dse5.py
    python3 scripts/plot_step5.py
fi
echo "step5: OK"
