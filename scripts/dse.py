#!/usr/bin/env python3
"""
dse.py - step 4: design-space exploration. Runs many accelerator
configurations on all MLPerf Tiny layers (DMA mode, no CPU reference, so it
is fast) and writes one summary table.

    python3 scripts/dse.py                  run every sweep (about 20 runs)
    python3 scripts/dse.py --only tpu_size  run one sweep
    python3 scripts/dse.py --jobs 8         parallel runs (default: CPU cores / 2)
    python3 scripts/dse.py --list           show the sweeps and exit

Results: results/dse/<run>.csv (+ .log) and results/dse_summary.csv with one
row per run and model: total cycles and where they go (prep = im2col on the
CPU, cpu = depthwise layers left on the CPU, load/compute/store = DMA-in,
STONNE compute, DMA-out).

Before trusting a sweep, the same configuration must have passed the checked
runs of steps 2-3 (check = PASS); here check = SKIP by design.
"""
import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "results")

# (sweep, run name, program, accel32-vp options, weight zeros %)
TPU = ["--accel-arch", "tpu"]
MAERI = ["--accel-arch", "maeri", "--accel-ms", "256"]
SIGMA = ["--accel-arch", "sigma", "--accel-ms", "256"]
RUNS = []
# 1. size of the rigid mesh
for n in (8, 16, 32):
    RUNS.append(("tpu_size", f"tpu{n}x{n}", "bench_dse.elf",
                 TPU + ["--accel-rows", str(n), "--accel-cols", str(n)], 0))
# 2. distribution bandwidth of the flexible fabric (same partition as the TPU)
for bw in (32, 64, 128, 256):
    RUNS.append(("maeri_bw", f"maeri256_bw{bw}", "bench_dse.elf", MAERI + ["--accel-dnbw", str(bw)], 0))
# 3. HW/SW partition: native conv + depthwise on the accelerator
for bw in (64, 256):
    RUNS.append(("partition", f"maeri256_bw{bw}_conv", "bench_dse_conv.elf", MAERI + ["--accel-dnbw", str(bw)], 0))
# 4. data movement: DMA bytes per cycle
for w in (4, 8, 16):
    RUNS.append(("dma_width", f"tpu16_dma{w}", "bench_dse.elf",
                 TPU + ["--accel-rows", "16", "--accel-cols", "16", "--accel-dma-width", str(w)], 0))
    RUNS.append(("dma_width", f"maeri256_bw256_conv_dma{w}", "bench_dse_conv.elf",
                 MAERI + ["--accel-dnbw", "256", "--accel-dma-width", str(w)], 0))
# 5. pruned weights: SIGMA skips zeros, the TPU cannot
for z in (0, 50, 80):
    sfx = f"_wz{z}" if z else ""
    RUNS.append(("sparsity", f"sigma256_bw256_wz{z}", f"bench_dse{sfx}.elf", SIGMA + ["--accel-dnbw", "256"], z))
    RUNS.append(("sparsity", f"tpu16_wz{z}", f"bench_dse{sfx}.elf",
                 TPU + ["--accel-rows", "16", "--accel-cols", "16"], z))


def build(wzeros):
    for z in sorted(wzeros):
        subprocess.run(["make", "-s", "-C", os.path.join(PROJ, "sw"), "dse", f"WZERO={z}"], check=True)


def run_one(r):
    sweep, name, elf, opts, _ = r
    out = os.path.join(RES, "dse", name + ".csv")
    p = subprocess.run(["bash", os.path.join(PROJ, "scripts", "run_vp.sh"), elf, out] + opts,
                       capture_output=True, text=True)
    print(p.stdout.strip() or p.stderr.strip(), flush=True)
    return r, out, p.returncode


def summarize(results):
    rows = []
    for (sweep, name, elf, opts, wz), out, rc in results:
        if rc != 0 or not os.path.exists(out):
            print(f"skipping {name}: run failed")
            continue
        info = {}
        for line in open(out[:-4] + ".log"):
            if line.startswith("INFO,"):
                f = line.strip().split(",")[1:]
                info = {f[i]: f[i + 1] for i in range(0, len(f) - 1, 2)}
        opt = dict(zip(opts[::2], opts[1::2]))
        per = {}
        for x in csv.DictReader(open(out)):
            m = per.setdefault(x["model"], dict(total=0, prep=0, cpu=0, load=0, compute=0, store=0,
                                                macs=0, in_=0, out=0))
            m["total"] += int(x["total"])
            m["macs"] += int(x["macs"])
            if x["place"] == "CPU":
                m["cpu"] += int(x["total"])
            else:
                m["prep"] += int(x["prep"])
                m["load"] += int(x["acc_load"])
                m["compute"] += int(x["acc_compute"])
                m["store"] += int(x["acc_store"])
        for model, m in per.items():
            rows.append(dict(sweep=sweep, run=name, model=model, arch=info.get("arch"), pe=info.get("pe"),
                             dn_bw=info.get("dn_bw"), partition=info.get("partition"),
                             dma_width=opt.get("--accel-dma-width", "4"), wzero=wz,
                             total_cycles=m["total"], prep_cycles=m["prep"], cpu_cycles=m["cpu"],
                             load_cycles=m["load"], compute_cycles=m["compute"], store_cycles=m["store"],
                             macs=m["macs"]))
    path = os.path.join(RES, "dse_summary.csv")
    keep = []
    if os.path.exists(path):  # keep rows of runs not repeated this time
        names = {r["run"] for r in rows}
        keep = [r for r in csv.DictReader(open(path)) if r["run"] not in names]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else list(keep[0].keys()))
        w.writeheader()
        for r in keep + rows:
            w.writerow(r)
    print(f"wrote {path} ({len(keep) + len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run one sweep: " + ", ".join(sorted({r[0] for r in RUNS})))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    runs = [r for r in RUNS if not a.only or r[0] == a.only]
    if a.list or not runs:
        for r in RUNS:
            print(f"{r[0]:10s} {r[1]:28s} {r[2]:22s} {' '.join(r[3])}")
        sys.exit(0 if runs else 1)
    os.makedirs(os.path.join(RES, "dse"), exist_ok=True)
    build({r[4] for r in runs})
    print(f"{len(runs)} runs, {a.jobs} at a time")
    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(run_one, runs))
    summarize(results)


if __name__ == "__main__":
    main()
