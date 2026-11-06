# Week 2: stage A — first coarse failure map

## What was added

| File | Change |
|---|---|
| `accel/sensor_regs.h` | **new**: sensor register map (shared by platform and software) |
| `accel/sensor_dma.h` | **new**: streaming sensor + FIFO + sensor DMA (third bus master), measurement window |
| `vp_platform/accel/main.cpp` | sensor at `0x1100_0000`; options `--sensor-rate-ksps --sensor-theta --sensor-depth --sensor-bytes`, `--cpu-fetch-private`; `SENSOR,...` and `WINDOW,...` summary lines |
| `sw/bench_kws.c`, `sw/Makefile` | **new** `make kws`: warm-up, release, 10 DS-CNN layers back to back (double buffered, 4 tiles), T_acc, window statistics |
| `scripts/stageA.py` | **new**: 124 runs, matched no-sensor baselines, overload indicator, bound check, failure map |
| `reference_results/stageA/`, `stageA_summary.csv`, `stageA_baselines.csv`, `figures/fig6_stageA.png` | results |

Run: `make -C sw kws && python3 scripts/stageA.py --jobs 2 --plot` (about 4 minutes with 2 cores).

## Model used (stated in the paper)

- Request/grant bus, handover a = 1 on ownership change, 100 MHz, burst = 10 + ceil(b/4) cycles.
- **CPU instruction fetches from a private 1-cycle memory** (`--cpu-fetch-private`, an instruction TCM);
  the CPU's **loads and stores** (stack, FIR state, peripheral registers) use the shared bus.
- Sensor: S = 4 B, theta fixed per (rate, B) = smallest power of two meeting C2 under round-robin blocking.
- DNN: DS-CNN, release after 16 control-task periods of warm-up, T_inf = 5 ms; the window is [release, release + 5 ms).

## Results (116 points + 8 baselines)

1. **The bound holds and is tight.** All 19 sensor-first points that satisfy C1 and C2 have zero overflows.
   At 100 MB/s, B = 1024 B, F = 230 (the bound): measured peak occupancy 229, sensor worst wait 267 cycles = beta + a.

2. **At this resolution the arbitration policy hardly matters.** All four policies give the same pass/fail at every
   one of the 29 (rate, B, depth) points, and their dT_acc differ by at most 3.4 percentage points. Three reasons:
   the bus is work-conserving, so total DNN delay depends on how much the others use the bus, not on order;
   DS-CNN here is bus-bound; and every tenure is non-preemptive. Policy differences can only show in the
   minimum FIFO depth between the coarse depths, which stage B resolves.

3. **The CPU deadline is set by the DNN burst length, not by the policy.** With 1024-B bursts the control task
   misses 17 of 80 releases **even without a sensor, under every policy, including CPU-first**: each CPU
   load or store can wait for an in-flight 266-cycle burst, and the interrupt routine makes ~100 of them
   (worst response 10,134 cycles vs a 6,200-cycle deadline). With 64-B bursts the worst response is about 1,200 cycles.

4. **The DNN delay is set by the sensor's share of the bus.** Because the DNN is bus-bound, dT_acc is close to
   the sensor's bus share, (R/theta)(10 + ceil(theta*S/4) + handover)/f_clk. At 3.2 MB/s with theta = 1, the
   10-cycle setup per 4-byte burst costs about 10 % of the bus, so dT_acc = +11 % (just over the 10 % margin).
   At 100 MB/s the raw data alone is 25 % of the bus, so **no** policy or FIFO can meet a 10 % margin.

5. **Burst length costs inference time.** With no sensor, B = 64 B gives T_acc = 205,163 cycles vs 146,389 for
   B = 1024 B (+40 %). This gap is not in dT_acc (matched baseline) and is reported separately.

6. **Strict FIFO (8 samples):** all limits are met only up to 0.32 MB/s, and only with B = 64 B.
   Relaxed FIFO: same, because the binding limits at higher rates are C and D, not the FIFO.

## What this changes for stage B

- **theta must be a sweep axis.** The rule "smallest theta that meets C2" minimises the FIFO but maximises the
  sensor's burst overhead. At 3.2 MB/s, theta = 4 with B = 64 B needs only F >= 5 (within the strict limit)
  and about 3 % of the bus, so the D failure there is caused by the theta rule, not by the policies.
  Stage B sweeps theta in powers of two from the C2 minimum up to the FIFO limit (still fixed across depths,
  so depth bisection stays valid).
- **Separate "impossible for any arbiter" from "policy failure".** For a bus-bound DNN, dT_acc is close to
  s/(1 - s), where s is the sensor's bus share (3.2 MB/s, theta = 1: estimate 11.6 %, measured 10.6 %). It is an
  estimate, not a bound, because double buffering lets part of the DMA overlap compute. Points where the estimate
  is far above the margin are labelled "margin infeasible by bandwidth", not policy failures; only the remaining
  points can show a research gap.
- **Bisect the minimum FIFO depth per policy** between the coarse depths (policy differences live there).
- **Burst length is the main CPU lever**: add B = 256 B and variant 5 (DNN fragmentation + budget), which keeps
  long transfers but bounds each CPU wait.
- **Model variant (stage C):** CPU data in a private TCM. Then only peripheral-register accesses compete with
  the DMA, and the C failures above would largely disappear. The base model keeps CPU data on the shared bus
  because many low-end MCUs have no data TCM; the paper must state this choice.
