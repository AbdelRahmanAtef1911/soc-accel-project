#!/usr/bin/env python3
"""
gap_analyze.py - reads results/gap_summary.csv (scripts/gap_sweep.py) and
prints the gap experiment's findings; with --plot also draws
reference_results/figures/fig5_gap.png.

For each model and job mode it compares two choices of DMA setting
(bus mode x burst size) on the shared bus:
  inference-only pick  the setting with the shortest inference time,
                       deadlines ignored (what an inference-only study picks)
  deadline-aware pick  the shortest inference among settings where the
                       16 kHz CPU task never misses a deadline

    python3 scripts/gap_analyze.py [--summary results/gap_summary.csv] [--plot]
"""
import argparse
import csv
import os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ("resnet8", "dscnn", "mbnet")
NAMES = {"resnet8": "ResNet-8", "dscnn": "DS-CNN", "mbnet": "MobileNetV1-0.25"}


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r.setdefault("cpu", "wfi")
        for k in ("taps", "burst", "inference_cycles", "rt_releases", "rt_late", "rt_lost",
                  "rt_max_resp", "rt_iso_max", "layers_with_miss", "layers", "cpu_max_bus_wait"):
            r[k] = int(r[k])
        r["rt_missed_pct"] = float(r["rt_missed_pct"])
    return rows


def label(r):
    return f"{r['bus']} {r['burst']} B"


SCEN = ((16, "wfi", "16-tap task, CPU sleeps between samples"),
        (64, "wfi", "64-tap task, CPU sleeps between samples"),
        (16, "bg", "16-tap task, CPU busy with background work"))


def picks(rows, model, mode, taps=16, cpu="wfi"):
    shared = [r for r in rows if r["model"] == model and r["mode"] == mode and r["taps"] == taps
              and r["cpu"] == cpu and r["bus"] != "ideal"]
    if not shared:
        return None, None, []
    fast = min(shared, key=lambda r: r["inference_cycles"])
    ok = [r for r in shared if r["rt_late"] + r["rt_lost"] == 0]
    safe = min(ok, key=lambda r: r["inference_cycles"]) if ok else None
    return fast, safe, shared


def report(rows):
    print("== ideal bus (what an accelerator-only or ideal-bus model sees) ==")
    for m in MODELS:
        ideal = [r for r in rows if r["model"] == m and r["bus"] == "ideal"]
        miss = sum(r["rt_late"] + r["rt_lost"] for r in ideal)
        print(f"  {NAMES[m]:18s} {len(ideal)} settings, deadline misses: {miss}, "
              f"max response {max(r['rt_max_resp'] for r in ideal)} cycles (alone: {ideal[0]['rt_iso_max']})")
    for taps, cpu, title in SCEN:
        print(f"\n== shared bus, {title}: inference-only pick vs deadline-aware pick ==")
        for mode in ("DMA", "DBUF"):
            for m in MODELS:
                fast, safe, shared = picks(rows, m, mode, taps, cpu)
                if not fast:
                    continue
                s = f"  {mode:4s} {NAMES[m]:18s} fastest = {label(fast):16s} {fast['inference_cycles']:>9d} cyc, " \
                    f"{fast['rt_missed_pct']:5.1f}% samples late/lost"
                if safe:
                    s += f" | best on-time = {label(safe):16s} {safe['inference_cycles']:>9d} cyc " \
                         f"(+{100.0 * (safe['inference_cycles'] / fast['inference_cycles'] - 1):.1f}%)"
                else:
                    s += " | no setting meets every deadline"
                print(s)
    for taps, cpu, title in SCEN:
      print(f"\n== every setting, DMA mode (one piece), {title} ==")
      print(f"  {'setting':16s} " + " ".join(f"{NAMES[m]:>26s}" for m in MODELS))
      keys = sorted({(r['bus'], r['burst']) for r in rows if r['taps'] == taps and r['cpu'] == cpu})
      for b, bu in keys:
        cells = []
        for m in MODELS:
            r = [x for x in rows if x["model"] == m and x["mode"] == "DMA" and x["taps"] == taps
                 and x["cpu"] == cpu and x["bus"] == b and x["burst"] == bu]
            cells.append(f"{r[0]['inference_cycles'] / 1e5:7.3f} ms {r[0]['rt_missed_pct']:5.1f}% "
                         f"R={r[0]['rt_max_resp']:>6d}" if r else "")
        print(f"  {b + ' ' + str(bu):16s} " + " ".join(f"{c:>26s}" for c in cells))


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"ideal": "#9aa0a6", "block": "#d93025", "steal": "#1a73e8", "transparent": "#188038"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, m in zip(axes, MODELS):
        for b in ("ideal", "block", "steal", "transparent"):
            pts = sorted([r for r in rows if r["model"] == m and r["mode"] == "DMA" and r["taps"] == 16
                          and r["cpu"] == "bg" and r["bus"] == b], key=lambda r: r["burst"])
            if not pts:
                continue
            xs = [r["inference_cycles"] / 1e5 for r in pts]
            ys = [r["rt_missed_pct"] for r in pts]
            ax.plot(xs, ys, "o-" if b == "steal" else "o", color=colors[b], label=b, ms=5, lw=1)
            for r, x, y in zip(pts, xs, ys):
                if b in ("steal", "block"):
                    ax.annotate(str(r["burst"]), (x, y), textcoords="offset points", xytext=(4, 4),
                                fontsize=7, color=colors[b])
        ax.axhline(0, color="#5f6368", lw=0.8)
        ax.set_title(NAMES[m], fontsize=10)
        ax.set_xlabel("inference time (ms, 100 MHz)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("16 kHz samples late or lost (%)")
    axes[0].legend(fontsize=8, title="DMA bus mode", title_fontsize=8)
    fig.suptitle("CPU busy + 16 kHz task: the fastest DMA settings miss deadlines (labels = burst size, B)",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=os.path.join(PROJ, "results", "gap_summary.csv"))
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default=os.path.join(PROJ, "results", "figures", "fig5_gap.png"))
    a = ap.parse_args()
    rows = load(a.summary)
    report(rows)
    if a.plot:
        plot(rows, a.out)


if __name__ == "__main__":
    main()
