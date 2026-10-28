#!/usr/bin/env python3
"""
summarize5.py - the numbers of step 5 (bench_dma*.elf result CSVs).

    python3 scripts/summarize5.py results/step5a_dbuf.csv [more.csv ...]
    python3 scripts/summarize5.py results/step5a_dbuf.csv --compare results/step3b_maeri_conv.csv

For every file and model it prints, per mode (DMA = one piece, TILE = tiled,
DBUF = tiled + double buffered):
  total      end-to-end cycles of all layers (what the application sees)
  vs DMA     speed-up over the one-piece job of the same run
  job        accelerator START -> DONE, summed
  load/compute/store   the three phases (in DBUF they overlap, so their sum
             is larger than the job time)
  CPU free   share of the waiting time the CPU could use: asleep (WFI) or
             doing background work (BG); 0 when it polls
  polls      STATUS reads while busy; resp = mean cycles from DONE until the
             CPU reacts (poll-loop exit or interrupt-handler entry)
  bus        cycles the CPU / DMA waited for the shared bus and the longest
             single CPU wait (0 on the ideal bus)
--compare checks that the DMA rows equal the DMA rows of a step-3b CSV.
"""
import csv
import os
import signal
import sys
from collections import OrderedDict, defaultdict

INT = ("M", "K", "N", "macs", "prep", "in", "run", "total", "tiles", "acc_load", "acc_compute", "acc_store",
       "acc_total", "bytes_in", "polls", "sleep", "bg_units", "resp", "cpu_txn", "cpu_wait", "cpu_maxw", "dma_wait")


def info_of(path):
    log = path[:-4] + ".log"
    if os.path.exists(log):
        for line in open(log):
            if line.startswith("INFO,"):
                f = line.strip().split(",")[1:]
                return {f[i]: f[i + 1] for i in range(0, len(f) - 1, 2)}
    return {}


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in INT:
            r[k] = int(r[k])
    return rows


def aggregate(rows, bg_unit):
    """{model: {mode: totals}}"""
    out = OrderedDict()
    for r in rows:
        m = out.setdefault(r["model"], OrderedDict()).setdefault(r["mode"], defaultdict(int))
        for k in ("total", "run", "acc_total", "acc_load", "acc_compute", "acc_store", "bytes_in", "polls",
                  "sleep", "bg_units", "resp", "cpu_txn", "cpu_wait", "dma_wait", "macs"):
            m[k] += r[k]
        m["cpu_maxw"] = max(m["cpu_maxw"], r["cpu_maxw"])
        m["layers"] += 1
        m["acc_layers"] += r["place"] == "ACC"
        m["free"] += r["sleep"] + r["bg_units"] * bg_unit
        m["fail"] += r["check"] == "FAIL"
    return out


def summarize(path):
    info = info_of(path)
    rows = load(path)
    bg_unit = int(info.get("bg_unit_cycles", "0"))
    fails = sum(r["check"] == "FAIL" for r in rows)
    desc = " ".join(f"{k}={v}" for k, v in info.items())
    print(f"\n=== {path}   {desc}")
    print(f"    {len(rows)} rows, {fails} FAIL")
    for model, by in aggregate(rows, bg_unit).items():
        base = by.get("DMA", {}).get("total")
        print(f"\n  {model}")
        print(f"  {'mode':5s} {'total':>11s} {'vs DMA':>7s} {'job':>11s} {'load':>10s} {'compute':>9s} "
              f"{'store':>10s} {'CPU free':>8s} {'polls':>7s} {'resp':>5s} {'CPU wait':>9s} {'max':>6s} {'DMA wait':>9s}")
        for mode, m in by.items():
            sp = f"{base / m['total']:6.2f}x" if base else "     - "
            free = f"{100.0 * m['free'] / m['run']:7.1f}%" if m["run"] else "      -"
            resp = m["resp"] // max(1, m["acc_layers"])
            print(f"  {mode:5s} {m['total']:11,d} {sp} {m['acc_total']:11,d} {m['acc_load']:10,d} "
                  f"{m['acc_compute']:9,d} {m['acc_store']:10,d} {free} {m['polls']:7,d} {resp:5d} "
                  f"{m['cpu_wait']:9,d} {m['cpu_maxw']:6,d} {m['dma_wait']:9,d}")
    return rows


def compare(rows, ref_path):
    ref = {(r["model"], r["layer"]): r for r in csv.DictReader(open(ref_path)) if r["mode"] == "DMA"}
    dma = [r for r in rows if r["mode"] == "DMA"]
    same = 0
    for r in dma:
        f = ref.get((r["model"], r["layer"]))
        if f and all(int(f[k]) == r[k] for k in ("total", "acc_load", "acc_compute", "acc_store")):
            same += 1
        else:
            print(f"  differs: {r['model']} {r['layer']}: {r['total']} vs {f['total'] if f else 'missing'}")
    ok = same == len(dma) and dma
    print(f"\nDMA rows equal to {ref_path}: {same}/{len(dma)} -> {'IDENTICAL' if ok else 'DIFFERENT'}")
    return ok


def main():
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    args = sys.argv[1:]
    ref = None
    if "--compare" in args:
        i = args.index("--compare")
        ref = args[i + 1]
        del args[i:i + 2]
    if not args:
        sys.exit(__doc__)
    ok = True
    for p in args:
        rows = summarize(p)
        if ref:
            ok = compare(rows, ref) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
