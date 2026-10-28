#!/usr/bin/env python3
"""
dse5.py - step 5 sweeps: how should the DMA work? All runs use the step-3b
accelerator (MAERI, 256 multipliers, 256-wide network, "conv" partition),
DMA mode, no CPU reference (check = SKIP; the checked runs are in
step5_dma_extension.sh).

    python3 scripts/dse5.py                 all sweeps (28 runs)
    python3 scripts/dse5.py --only bus      one sweep
    python3 scripts/dse5.py --jobs 2        parallel runs (default: CPU cores / 2)
    python3 scripts/dse5.py --list

Sweeps
  overlap   one piece / tiled / double-buffered, DMA width 4, 8, 16 B/cycle
  tiles     double buffering with 2, 4, 8, 16 tiles (DMA width 4 and 16)
  wait      polling vs interrupt + WFI vs interrupt + background work (ideal bus)
  bus       shared bus: block / cycle stealing / transparent x poll / WFI / BG
  burst     shared bus, CPU busy (BG): burst size 4 ... 1024 B, block vs steal

Results: results/dse5/<run>.csv (+ .log) and results/dse5_summary.csv, one
row per run, model and mode (see summarize5.py for the columns' meaning).
"""
import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "results")
MAERI = ["--accel-arch", "maeri", "--accel-ms", "256", "--accel-dnbw", "256"]


def elf(wait="poll", ntiles=4, modes="one,tile,dbuf"):
    """make arguments and file name, same naming rule as sw/Makefile"""
    sfx = f"_{wait}" + (f"_t{ntiles}" if ntiles != 4 else "") + \
          (f"_{modes.replace(',', '-')}" if modes != "one,tile,dbuf" else "") + "_fast"
    args = [f"WAIT={wait}", f"NTILES={ntiles}", f"MODES={modes}", "CHECK=0"]
    return f"bench_dma{sfx}.elf", args


def bus(mode, burst=64):
    return ["--accel-bus-model", "--accel-dma-mode", mode, "--accel-dma-burst", str(burst)]


# (sweep, run name, (elf, make args), accel32-vp options)
RUNS = []
for w in (4, 8, 16):
    RUNS.append(("overlap", f"overlap_dma{w}", elf(), MAERI + ["--accel-dma-width", str(w)]))
for w in (4, 16):
    for t in (2, 8, 16):
        RUNS.append(("tiles", f"dbuf_t{t}_dma{w}", elf(ntiles=t, modes="dbuf"),
                     MAERI + ["--accel-dma-width", str(w)]))
for wait in ("wfi", "bg"):
    RUNS.append(("wait", f"ideal_{wait}", elf(wait, modes="one,dbuf"), MAERI))
for mode in ("block", "steal", "transparent"):
    for wait in ("poll", "wfi", "bg"):
        RUNS.append(("bus", f"bus_{mode}_{wait}", elf(wait, modes="one,dbuf"), MAERI + bus(mode)))
for mode in ("block", "steal"):
    for b in (4, 16, 256, 1024):
        RUNS.append(("burst", f"burst_{mode}_{b}", elf("bg", modes="dbuf"), MAERI + bus(mode, b)))


def build(runs):
    done = set()
    for _, _, (name, args), _ in runs:
        if name not in done:
            subprocess.run(["make", "-s", "-C", os.path.join(PROJ, "sw"), "dma"] + args, check=True)
            done.add(name)


def run_one(r):
    sweep, name, (prog, _), opts = r
    out = os.path.join(RES, "dse5", name + ".csv")
    p = subprocess.run(["bash", os.path.join(PROJ, "scripts", "run_vp.sh"), prog, out] + opts,
                       capture_output=True, text=True)
    print(p.stdout.strip() or p.stderr.strip(), flush=True)
    return r, out, p.returncode


def summarize(results):
    rows = []
    for (sweep, name, _, opts), out, rc in results:
        if rc != 0 or not os.path.exists(out):
            print(f"skipping {name}: run failed")
            continue
        info = {}
        for line in open(out[:-4] + ".log"):
            if line.startswith("INFO,"):
                f = line.strip().split(",")[1:]
                info = {f[i]: f[i + 1] for i in range(0, len(f) - 1, 2)}
        opt = dict(zip(opts[::2], opts[1::2]))
        bg_unit = int(info.get("bg_unit_cycles", "0"))
        per = {}
        for x in csv.DictReader(open(out)):
            m = per.setdefault((x["model"], x["mode"]), defaultdict(int))
            for k in ("total", "run", "acc_total", "acc_load", "acc_compute", "acc_store", "bytes_in", "polls",
                      "sleep", "bg_units", "resp", "cpu_txn", "cpu_wait", "dma_wait"):
                m[k] += int(x[k])
            m["cpu_maxw"] = max(m["cpu_maxw"], int(x["cpu_maxw"]))
            m["acc_layers"] += x["place"] == "ACC"
        for (model, mode), m in per.items():
            free = m["sleep"] + m["bg_units"] * bg_unit
            rows.append(dict(
                sweep=sweep, run=name, model=model, mode=mode, wait=info.get("wait"), bus=info.get("bus"),
                dma_mode=info.get("dma_mode") if info.get("bus") == "shared" else "-",
                burst=info.get("burst"), dma_width=opt.get("--accel-dma-width", "4"),
                ntiles=info.get("ntiles") if mode != "DMA" else "1",
                total_cycles=m["total"], job_cycles=m["acc_total"], load_cycles=m["acc_load"],
                compute_cycles=m["acc_compute"], store_cycles=m["acc_store"], bytes_in=m["bytes_in"],
                wait_cycles=m["run"], cpu_free_cycles=free,
                cpu_free_pct=round(100.0 * free / m["run"], 1) if m["run"] else 0,
                bg_units=m["bg_units"], polls=m["polls"], resp_avg=m["resp"] // max(1, m["acc_layers"]),
                cpu_bus_txn=m["cpu_txn"], cpu_bus_wait=m["cpu_wait"], cpu_bus_maxwait=m["cpu_maxw"],
                dma_bus_wait=m["dma_wait"]))
    path = os.path.join(RES, "dse5_summary.csv")
    keep = []
    if os.path.exists(path):  # keep rows of runs not repeated this time
        names = {r["run"] for r in rows}
        keep = [r for r in csv.DictReader(open(path)) if r["run"] not in names]
    order = {r[1]: i for i, r in enumerate(RUNS)}
    allrows = sorted(keep + rows, key=lambda r: (order.get(r["run"], 999), r["model"], r["mode"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
        w.writeheader()
        for r in allrows:
            w.writerow(r)
    print(f"wrote {path} ({len(allrows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run one sweep: " + ", ".join(sorted({r[0] for r in RUNS})))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    runs = [r for r in RUNS if not a.only or r[0] == a.only]
    if a.list or not runs:
        for r in RUNS:
            print(f"{r[0]:8s} {r[1]:22s} {r[2][0]:30s} {' '.join(r[3])}")
        sys.exit(0 if runs else 1)
    os.makedirs(os.path.join(RES, "dse5"), exist_ok=True)
    build(runs)
    print(f"{len(runs)} runs, {a.jobs} at a time")
    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(run_one, runs))
    summarize(results)


if __name__ == "__main__":
    main()
