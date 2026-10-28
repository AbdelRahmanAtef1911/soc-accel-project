#!/usr/bin/env python3
"""
summarize.py - turn result CSVs into the numbers for the report.

    python3 scripts/summarize.py results/step2_tpu.csv [results/other.csv ...] [--layers]

For every file and every model it prints, per mode (SW / CPU-copy / DMA):
  total cycles for all layers, time at 100 MHz, speedup over SW, and where the
  time goes: prep (im2col on the CPU), in (moving A,B), run (START->DONE),
  out (moving C back) and cpu (depthwise layers that stay on the CPU).
With --layers it also prints one line per layer.
"""
import csv
import os
import signal
import sys
from collections import OrderedDict, defaultdict

CLK_MHZ = 100.0


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in ("M", "K", "N", "macs", "prep", "in", "run", "out", "total",
                  "acc_load", "acc_compute", "acc_store"):
            r[k] = int(r[k])
    info = ""
    log = path[:-4] + ".log"
    if os.path.exists(log):
        for line in open(log):
            if line.startswith("INFO,"):
                f = line.strip().split(",")[1:]
                info = " ".join(f"{f[i]}={f[i+1]}" for i in range(0, len(f) - 1, 2))
                break
    return rows, info


def summarize(path, per_layer):
    rows, info = load(path)
    print(f"\n=== {path}   {info}")
    fails = [r for r in rows if r["check"] != "PASS"]
    print(f"    {len(rows)} rows, {len(fails)} FAIL")

    models = OrderedDict()
    for r in rows:
        models.setdefault(r["model"], []).append(r)

    for model, mrows in models.items():
        by_mode = OrderedDict()
        for r in mrows:
            by_mode.setdefault(r["mode"], []).append(r)
        sw_total = sum(r["total"] for r in by_mode.get("SW", [])) or None
        macs = sum(r["macs"] for r in by_mode.get("SW", by_mode[next(iter(by_mode))]))
        print(f"\n  {model}: {len(by_mode[next(iter(by_mode))])} layers, {macs/1e6:.2f} M MACs")
        print(f"  {'mode':5s} {'total cycles':>13s} {'ms@100MHz':>10s} {'speedup':>8s}   "
              f"{'prep':>6s} {'in':>6s} {'run':>6s} {'out':>6s} {'cpu':>6s}   {'MACs/cyc':>8s}")
        for mode, mr in by_mode.items():
            tot = sum(r["total"] for r in mr)
            parts = defaultdict(int)
            for r in mr:
                if r["place"] == "CPU":
                    parts["cpu"] += r["total"]
                else:
                    for k in ("prep", "in", "run", "out"):
                        parts[k] += r[k]
            sp = f"{sw_total / tot:7.2f}x" if sw_total else "     -  "
            pct = lambda k: f"{100.0 * parts[k] / tot:5.1f}%"
            print(f"  {mode:5s} {tot:13,d} {tot / CLK_MHZ / 1e3:10.2f} {sp}   "
                  f"{pct('prep')} {pct('in')} {pct('run')} {pct('out')} {pct('cpu')}   {macs / tot:8.2f}")

        if per_layer:
            print(f"\n  {'layer':16s} {'type':4s} {'M':>5s} {'K':>4s} {'N':>4s} "
                  + " ".join(f"{m:>11s}" for m in by_mode) + "   compute  util")
            names = [r["layer"] for r in by_mode[next(iter(by_mode))]]
            for i, name in enumerate(names):
                rr = {m: by_mode[m][i] for m in by_mode}
                r0 = next(iter(rr.values()))
                acc = [r for r in rr.values() if r["place"] == "ACC"]
                comp = acc[0]["acc_compute"] if acc else 0
                pe = int(info.split("pe=")[1].split()[0]) if "pe=" in info else 0
                util = f"{100.0 * r0['macs'] / (pe * comp):5.1f}%" if comp and pe else "   - "
                print(f"  {name:16s} {r0['type']:4s} {r0['M']:5d} {r0['K']:4d} {r0['N']:4d} "
                      + " ".join(f"{rr[m]['total']:11,d}" for m in by_mode)
                      + f"   {comp:7d} {util}")


def main():
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # quiet when piped into head/less
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    for p in args:
        summarize(p, "--layers" in sys.argv)


if __name__ == "__main__":
    main()
