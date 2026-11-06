#!/usr/bin/env python3
"""
gap_sweep.py - gap experiment: a periodic real-time task on the CPU (CLINT
timer, 62 us period ~ 16 kHz audio, 16-tap FIR) while every MLPerf Tiny layer
runs on the step-3b accelerator (MAERI 256, conv partition, completion
interrupt + WFI). For each bus / DMA setting it records the inference time
and how many task deadlines are missed (sw/bench_rt.c).

    python3 scripts/gap_sweep.py              all runs
    python3 scripts/gap_sweep.py --jobs 2     parallel runs (default: CPU cores / 2)
    python3 scripts/gap_sweep.py --list

Settings
  ideal         no bus contention (what an accelerator-only or ideal-bus model sees)
  block         DMA holds the bus for a whole transfer
  steal         DMA gives the bus back after every burst
  transparent   DMA only uses bus cycles the CPU leaves idle
  each with burst sizes 16 ... 1024 B, in three scenarios:
    16-tap task, CPU sleeps (WFI) between samples
    64-tap task, CPU sleeps between samples
    16-tap task, CPU busy with background work (as in step 5c) between samples

Results: results/gap/<run>.csv (+ .log) and results/gap_summary.csv
(one row per run, model and mode).
"""
import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "results")
MAERI = ["--accel-arch", "maeri", "--accel-ms", "256", "--accel-dnbw", "256"]
BURSTS = (16, 64, 256, 1024)


def elf(taps=16, cpu="wfi"):
    """program file and make arguments, same naming rule as sw/Makefile"""
    sfx = ("_bg" if cpu == "bg" else "") + (f"_f{taps}" if taps != 16 else "")
    args = (("RT_BG=1",) if cpu == "bg" else ()) + ((f"FIR_TAPS={taps}",) if taps != 16 else ())
    return (f"bench_rt{sfx}.elf", args)


# three scenarios: (task taps, what the CPU does while it waits, bursts per bus mode)
SCENARIOS = [
    (16, "wfi", {"ideal": BURSTS, "block": BURSTS, "steal": (16, 64, 128, 256, 512, 1024),
                 "transparent": BURSTS}),
    (64, "wfi", {"ideal": (64, 1024), "block": (64, 1024), "steal": (16, 64, 128, 256, 512, 1024),
                 "transparent": BURSTS}),
    (16, "bg", {"ideal": (64, 1024), "block": (64, 1024), "steal": (16, 64, 128, 256, 1024),
                "transparent": (64, 1024)}),
]
RUNS = []  # (run name, task taps, cpu, (elf, make args), vp options, bus, burst)
for taps, cpu, plan in SCENARIOS:
    sfx = (f"_f{taps}" if taps != 16 else "") + ("_bg" if cpu == "bg" else "")
    for mode, bursts in plan.items():
        for b in bursts:
            opts = MAERI + ["--accel-dma-burst", str(b)]
            if mode != "ideal":
                opts += ["--accel-bus-model", "--accel-dma-mode", mode]
            RUNS.append((f"{mode}_{b}{sfx}", taps, cpu, elf(taps, cpu), opts, mode, b))


def run(r):
    name, taps, cpu, (e, margs), opts, *_ = r
    out = f"gap/{name}.csv"
    p = subprocess.run(["bash", os.path.join(PROJ, "scripts", "run_vp.sh"), e, out] + opts,
                       capture_output=True, text=True)
    print(p.stdout.strip() or p.stderr.strip(), flush=True)
    if p.returncode:
        raise SystemExit(f"run {name} failed")


def iso(log):
    for line in open(log):
        if line.startswith("ISO,"):
            v = [int(x) for x in line.strip().split(",")[1:]]
            return v[3], v[4]  # max, avg response
    return 0, 0


def summarize():
    rows = []
    for name, taps, cpu, _, _, mode, b in RUNS:
        path = os.path.join(RES, "gap", name + ".csv")
        if not os.path.exists(path):
            continue
        iso_max, iso_avg = iso(path[:-4] + ".log")
        data = list(csv.DictReader(open(path)))
        for model in ("resnet8", "dscnn", "mbnet"):
            for m in ("DMA", "DBUF"):
                d = [x for x in data if x["model"] == model and x["mode"] == m]
                if not d:
                    continue
                S = lambda k: sum(int(x[k]) for x in d)
                rel, miss, skip = S("rt_rel"), S("rt_miss"), S("rt_skip")
                rows.append(dict(run=name, taps=taps, cpu=cpu, bus=mode, burst=b, model=model, mode=m,
                                 inference_cycles=S("total"), rt_releases=rel + skip, rt_late=miss,
                                 rt_lost=skip, rt_missed_pct=round(100.0 * (miss + skip) / max(1, rel + skip), 2),
                                 rt_max_resp=max(int(x["rt_max"]) for x in d), rt_iso_max=iso_max,
                                 layers_with_miss=sum(1 for x in d if int(x["rt_miss"]) + int(x["rt_skip"]) > 0),
                                 layers=len(d), cpu_max_bus_wait=max(int(x["cpu_maxw"]) for x in d),
                                 bg_units=S("bg_units")))
    out = os.path.join(RES, "gap_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", help="do not rerun runs whose CSV exists")
    a = ap.parse_args()
    if a.list:
        for r in RUNS:
            print(r[0], r[3][0], " ".join(r[4]))
        return
    if not a.summary_only:
        for e, margs in sorted({r[3] for r in RUNS}, key=str):
            subprocess.run(["make", "-s", "-C", os.path.join(PROJ, "sw"), "rt"] + list(margs), check=True)
        todo = [r for r in RUNS if not (a.skip_existing and
                                        os.path.exists(os.path.join(RES, "gap", r[0] + ".csv")))]
        with ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(run, todo))
    summarize()


if __name__ == "__main__":
    sys.exit(main())
