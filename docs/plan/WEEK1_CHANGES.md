# Week 1: request/grant bus — what changed

New or changed files (copy them over the same paths in your project):

| File | Change |
|---|---|
| `accel/rg_bus.h` | **new**: request/grant arbiter, 5 policy variants, handover on ownership change, CPU port |
| `accel/stonne_accel.h` | DMA requests the RG bus per burst/fragment when `rgbus` is set; job bus counters; idle/active flag |
| `vp_platform/accel/main.cpp` | options `--bus-arbiter`, `--bus-handover`, `--bus-frag-bytes`, `--bus-window-cycles`, `--bus-budget-bytes`, `--bus-fast-path`; `RGBUS,...` summary lines |
| `unit_test/tb_rg.cpp` | **new**: plan tests 1-7 |
| `unit_test/tb_dma.cpp` | `-DRG_BUS` build (`tb_dma_rg`) for test 8; default build unchanged |
| `unit_test/Makefile` | targets `tb_rg`, `tb_dma_rg`, `test_rg` |
| `unit_test/check_rg_regression.py` | **new**: test-8 checker |
| `scripts/regress_all.sh` | also runs `make test_rg` |
| `reference_results/rg/` | **new**: reference outputs of the tests and runs below |
| `README.md` | section "Course project: request/grant bus (week 1)" |

After copying, in the container:

```bash
cd /workspace/soc-accel-project && source setup/env.sh
bash setup/04_build_vp.sh                      # rebuild accel32-vp (about 20 s)
(cd unit_test && make test_rg)                 # tests 1-8: ALL TESTS PASSED, test 8 PASS x3
bash scripts/run_vp.sh bench_sw.elf rg/step1_sw_rr.csv --accel-arch tpu --bus-arbiter rr --bus-fast-path
cmp results/rg/step1_sw_rr.csv reference_results/step1_sw.csv && echo IDENTICAL
```

Results of the week-1 checks:

- Tests 1-7 (`tb_rg`): all pass. Sensor-first max wait 267 cycles = bound beta + a (1024-B DNN bursts);
  round-robin 268 <= 270; RR grant counts equal within 1; budget 960 B per aligned 1000-cycle window
  (X = 1000 B, sliding window up to 1024 B); 13,793 handovers = 13,793 ownership changes.
- Test 8: `tb_dma` part 1 on the RG bus is identical to the step-5 reference with a = 0, and exactly
  +1 cycle per job with a = 1 (the DMA's first request follows the CPU's START write: one ownership
  change per job). The optional fast path gives identical testbench output.
- Platform: CPU-only step 1 on the RG bus (`--bus-fast-path`) is identical to `step1_sw.csv`
  (48 rows); the step-5 model is unchanged (`step5c_bus_steal_bg.csv` and `tb_dma` identical).
- The fast path is off by default: in a timer-driven DS-CNN run it gave the same arbitration but
  shifted interrupt recognition by a few cycles, so project runs always synchronise the CPU
  (DS-CNN real-time run: 12 s host time vs 8 s with the fast path).
