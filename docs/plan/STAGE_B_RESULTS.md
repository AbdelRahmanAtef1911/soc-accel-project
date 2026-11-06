# Stage B — exact minimum FIFO, threshold sweep, burst length and budget policy

Run: `python3 scripts/stageB.py --jobs 2 --plot` (240 points, 912 runs, about 30 min with 2 cores; runs are
cached, so a second call only re-analyses). Results: `results/stageB_summary.csv`, `stageB_baselines.csv`,
`figures/fig7_stageB_minfifo.png`, `figures/fig8_stageB_tradeoff.png`.

## What stage B adds to stage A

- **theta is a sweep axis**: powers of two from the C2 minimum to 128 samples (rates >= 3.2 MB/s).
- **Exact minimum FIFO depth** F_min for every (policy, B, rate, theta), by integer bisection on [theta, 512].
  Valid because theta is fixed and no rule reads the depth: a run with no overflow is identical at every larger
  depth (test 9 checks this), so the CPU and DNN results of the passing run hold for every F >= F_min.
- **B = 256 B** and **variant 5** (budget + 64-B fragments, window 1000 cycles, rho = 0.25 and 0.5, with B = 1024).
- **Delay is judged against the unthrottled no-sensor run at the same B.** Policies 1-4 have identical no-sensor
  runs, so this changes nothing for them; for variant 5 its own throttling is a protection cost and is counted
  (dT against its own no-sensor run is also reported as `dT_own_pct`).
- **Overhead estimate** to separate "impossible for any arbiter" from "policy failure": the smallest sensor bus
  share reachable within the FIFO limit, s_min = (R/theta_max)(10 + ceil(theta_max*S/4) + a)/f_clk with
  theta_max = min(F_limit, 128). Points with s_min >= margin are excluded from the gap test. Checked against the
  simulations: no configuration within the FIFO limit passes the delay margin where s_min >= margin (0 violations
  in all 6 cases).

## Checks

| Check | Result |
|---|---|
| Bound, sensor-first: F_min <= F_suf where C2 holds | 50 points, 0 violations; F_min/F_suf = 0.50-1.00 |
| Bound, round-robin (beta_RR) | 50 points, 0 violations; F_min/F_suf = 0.50-1.00 |
| Test 9, depth monotonicity (window results at F_min = at F = 512) | 6/6 identical |
| Overhead estimate (no pass where s_min >= margin) | 0 violations |

The bound is tight: at 100 MB/s, B = 1024 B, F_min = 229 against F_suf = 230.

## Results

1. **The arbitration policy does not change the minimum FIFO.** Across all 50 (B, rate, theta) points, the four
   policies give F_min within **one sample** of each other (at most 3.3 %). dT_acc differs by at most 2.6
   percentage points, and the CPU outcome differs at only 3 of 50 points. Reason: the sensor's worst wait is one
   in-flight, non-preemptive DNN burst under every policy; the CPU's 1-cycle accesses are the only thing the
   order changes.
2. **The DNN burst length B is the main design lever.** It sets the CPU deadline (B = 1024: 17 late/lost releases
   with no sensor, under every policy; B = 256: worst response 3,212 cycles, no misses; B = 64: 1,207 cycles) and
   the sensor blocking term. B = 256 B is the sweet spot: CPU safe, only +8 % DNN time over B = 1024 without a sensor.
3. **theta trades FIFO against bus overhead.** At 3.2 MB/s, theta = 8 with B = 256 needs F_min = 8 (strict limit)
   and costs +2.4 %; theta = 1 needs F = 1 but costs about +10 %.
4. **Variant 5 (budget + fragmentation) protects the CPU at B = 1024 (worst response 722-1,182 cycles) but
   throttles the DNN by +60 % (rho = 0.5) to +200 % (rho = 0.25)** with these parameters, so it never meets the
   delay margin. Fragmenting to 64 B alone (variant 1-4 with B = 64) gives the same CPU protection at +40 %.
5. **What can be met** (10 % margin):
   - up to 3.2 MB/s: all limits with the strict 8-sample FIFO (B = 256, theta = 8 at 3.2 MB/s);
   - 32 MB/s: only with the relaxed FIFO (F about 141-145, theta = 128, dT about 10 %);
   - 100 MB/s: nothing; the sensor's bus share alone (>= 27 % with theta = 128) exceeds the margin.

## Research-gap test (plan Section 6)

Question: is there a feasible point where **no** established policy meets the FIFO, CPU and delay limits together?

| FIFO limit | 5 % margin | 10 % margin | 25 % margin |
|---|---|---|---|
| strict (8 samples) | none | none | **32 MB/s** |
| relaxed (1280 samples) | none | none | none |

Excluded because bus overhead alone exceeds the margin: strict FIFO, 32 and 100 MB/s at 5 % and 10 %, and
100 MB/s at 25 %; relaxed FIFO, 100 MB/s at every margin and 32 MB/s at 5 %.

The one candidate, **32 MB/s with an 8-sample FIFO and a 25 % margin**, is not an arbitration failure. theta = 8
would meet the 25 % margin (dT 22-25 %) but needs F_min = 11-15, because samples keep arriving during the sensor's
own 18-cycle burst and the handover (the removal rule frees the FIFO only when the burst ends), even before any
DNN blocking. theta = 4 fits in F = 7 but its 10-cycle setup per 16 bytes costs +43 %. No policy can change the
sensor's own burst time, so this corner is a FIFO-design question (word-by-word removal, ping-pong buffers), not
an arbitration one.

**Conclusion (decision rule of Section 6):** under this bus model there is no feasible point where established
arbitration policies fail together. The result is a sound course case study with a clear message: on a
non-preemptive shared bus with a light CPU, arbitration order is almost irrelevant; the DNN burst length, the
sensor's drain threshold and the bus overhead per burst decide the outcome. A PhD-level gap would need a harder
setting where order matters: heavier CPU traffic, several sensors or several DNN masters, an AXI-style
interconnect with outstanding transactions or split bursts, or a preemptive DMA.

## Limits of this stage

- One workload (DS-CNN, double buffered), one sensor, one CPU task, T_inf = 5 ms.
- CPU data on the shared bus, instruction fetch private (a data-TCM variant would remove most CPU failures).
- Variant 5 with one window size and two budgets; other parameters could reduce its throttling.
- theta restricted to powers of two up to 128; depth capped at 512.
