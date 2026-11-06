#!/usr/bin/env python3
"""
make_figures.py - the four final figures of the term paper, from the stage-B
results (reference_results/stageB_summary.csv, stageB_baselines.csv).

  fig1_failure_map.png   for each DNN burst setting and sensor rate: can any
                         policy meet all limits (strict FIFO / relaxed FIFO /
                         neither, with the limit that fails)
  fig2_min_fifo.png      exact minimum FIFO depth vs sensor rate, four policies,
                         with the sufficient bounds (sensor-first, round-robin)
  fig3_delay.png         smallest DNN delay vs sensor rate under each FIFO limit,
                         with the bus-overhead estimate
  fig4_tradeoff.png      minimum FIFO vs DNN delay over the theta sweep

    python3 docs/term_paper/make_figures.py [--data reference_results]
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
P4 = ("rr", "sensor-first", "cpu-first", "buffer")
COL = {"rr": "#1a73e8", "sensor-first": "#d93025", "cpu-first": "#f9ab00", "buffer": "#188038",
       "budget25": "#9334e6", "budget50": "#12b5cb"}
RATES = (8, 80, 800, 8000, 25000)
S = 4
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})


def load(d):
    rows = list(csv.DictReader(open(os.path.join(d, "stageB_summary.csv"))))
    for r in rows:
        for k in ("B", "rate_ksps", "theta", "F_min_num", "ok_cpu", "ok_dnn"):
            r[k] = int(r[k])
        for k in ("dT_pct", "rate_MBps", "gap_fastest_pct"):
            r[k] = float(r[k])
        for k in ("F_suf_sf", "F_suf_rr"):
            r[k] = int(r[k]) if r[k] else None
    return rows


def mbps(k):
    return k * S / 1000.0


def fig1(rows, out):
    confs = [("B = 64 B", lambda r: r["B"] == 64 and r["policy"] in P4),
             ("B = 256 B", lambda r: r["B"] == 256 and r["policy"] in P4),
             ("B = 1024 B", lambda r: r["B"] == 1024 and r["policy"] in P4),
             ("budget, ρ = 0.5", lambda r: r["policy"] == "budget50"),
             ("budget, ρ = 0.25", lambda r: r["policy"] == "budget25")]
    c = {"strict": "#1e8e3e", "relaxed": "#a8dab5", "fail": "#f28b82"}
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    for i, (lab, sel) in enumerate(confs):
        for j, k in enumerate(RATES):
            rs = [r for r in rows if sel(r) and r["rate_ksps"] == k]
            ok_s = [r for r in rs if r["pass_strict_10"] == "1"]
            ok_r = [r for r in rs if r["pass_relaxed_10"] == "1"]
            if ok_s:
                b, st = min(ok_s, key=lambda r: (r["dT_pct"], r["F_min_num"])), "strict"
                txt = f"F={b['F_min']}  θ={b['theta']}\nΔT={b['dT_pct']:+.1f}%"
            elif ok_r:
                b, st = min(ok_r, key=lambda r: (r["dT_pct"], r["F_min_num"])), "relaxed"
                txt = f"F={b['F_min']}  θ={b['theta']}\nΔT={b['dT_pct']:+.1f}%"
            else:
                st = "fail"
                cpu_ok = [r for r in rs if r["ok_cpu"]]
                if not cpu_ok:
                    txt = "C: control task\nlate"
                else:
                    best = min(r["dT_pct"] for r in cpu_ok)
                    txt = f"D: ΔT ≥ {best:.0f}%"
            ax.add_patch(plt.Rectangle((j - 0.47, -i - 0.45), 0.94, 0.9, color=c[st]))
            ax.text(j, -i, txt, ha="center", va="center", fontsize=6.3)
    ax.set_xlim(-0.55, len(RATES) - 0.45)
    ax.set_ylim(-len(confs) + 0.45, 0.55)
    ax.set_xticks(range(len(RATES)))
    ax.set_xticklabels([f"{mbps(k):g} MB/s" for k in RATES])
    ax.set_yticks([-i for i in range(len(confs))])
    ax.set_yticklabels([l for l, _ in confs])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(handles=[Patch(color=c["strict"], label="all limits met, FIFO ≤ 8 samples"),
                       Patch(color=c["relaxed"], label="all limits met, FIFO ≤ 1280 samples only"),
                       Patch(color=c["fail"], label="no configuration meets all limits")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig2(rows, out):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    for ax, B in zip(axes, (64, 256, 1024)):
        for p in P4:
            pts = []
            for k in RATES:
                rs = [r for r in rows if r["policy"] == p and r["B"] == B and r["rate_ksps"] == k]
                r = min(rs, key=lambda x: x["theta"])
                pts.append((mbps(k), r["F_min_num"]))
            ax.plot(*zip(*pts), "o-", color=COL[p], ms=3, lw=1, label=p, alpha=0.85)
        for col, ls, lab in (("F_suf_sf", "--", "bound, sensor-first"), ("F_suf_rr", ":", "bound, round-robin")):
            pol = "sensor-first" if col == "F_suf_sf" else "rr"
            pts = []
            for k in RATES:
                rs = [r for r in rows if r["policy"] == pol and r["B"] == B and r["rate_ksps"] == k]
                r = min(rs, key=lambda x: x["theta"])
                if r[col]:
                    pts.append((mbps(k), r[col]))
            ax.plot(*zip(*pts), ls, color="k", lw=0.9, label=lab)
        ax.axhline(8, color="#5f6368", lw=0.6, ls="-.")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"DNN burst B = {B} B", fontsize=8)
        ax.set_xlabel("sensor rate (MB/s)")
        ax.grid(alpha=0.3, which="both", lw=0.4)
    axes[0].set_ylabel("min. FIFO depth (samples)")
    axes[0].text(0.04, 9.5, "strict limit", fontsize=6, color="#5f6368")
    axes[2].legend(fontsize=5.8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig3(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    for ax, (lim, fl) in zip(axes, (("strict", 8), ("relaxed", 1280))):
        for B, mk in ((64, "o"), (256, "s"), (1024, "^")):
            xs, ys = [], []
            for k in RATES:
                rs = [r for r in rows if r["policy"] in P4 and r["B"] == B and r["rate_ksps"] == k
                      and r["F_min_num"] <= fl]
                if rs:
                    xs.append(mbps(k))
                    ys.append(min(r["dT_pct"] for r in rs))
            ax.plot(xs, ys, mk + "-", ms=3.5, lw=1, label=f"B = {B} B")
        # bus-overhead estimate s/(1-s) at the largest theta the FIFO limit allows
        xs, ys = [], []
        for k in RATES:
            th = 1
            while th * 2 <= min(fl, 128):
                th *= 2
            s = (k * 1000.0 / th) * (10 + -(-th * S // 4) + 1) / 100e6
            xs.append(mbps(k))
            ys.append(100 * s / (1 - s))
        ax.plot(xs, ys, "k--", lw=0.9, label="estimate s/(1-s)")
        ax.axhline(10, color="#5f6368", lw=0.6, ls="-.")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("sensor rate (MB/s)")
        ax.set_title(f"{lim} FIFO limit (≤ {fl} samples)", fontsize=8)
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].set_ylabel("smallest ΔT_acc (%)")
    axes[0].text(0.035, 11.5, "10 % margin", fontsize=6, color="#5f6368")
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig4(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, k in zip(axes, (800, 8000)):
        for B, mk in ((64, "o"), (256, "s"), (1024, "^")):
            for p in P4:
                rs = sorted([r for r in rows if r["policy"] == p and r["B"] == B and r["rate_ksps"] == k
                             and r["F_min_num"] < 10 ** 6], key=lambda r: r["theta"])
                if not rs:
                    continue
                ax.plot([r["F_min_num"] for r in rs], [r["dT_pct"] for r in rs], "-", color=COL[p], lw=0.6, alpha=0.6)
                for r in rs:
                    ax.scatter(r["F_min_num"], r["dT_pct"], marker=mk if r["ok_cpu"] else "x", s=14,
                               color=COL[p], linewidths=0.8)
        ax.axvline(8, color="#5f6368", lw=0.6, ls="-.")
        ax.axhline(10, color="#5f6368", lw=0.6, ls="-.")
        ax.set_xscale("log")
        ax.set_ylim(-1, 50)
        ax.set_xlabel("minimum FIFO depth (samples)")
        ax.set_title(f"{mbps(k):g} MB/s: theta from C2 minimum to 128", fontsize=8)
        ax.grid(alpha=0.3, lw=0.4)
        bud = [r for r in rows if r["policy"].startswith("budget") and r["rate_ksps"] == k]
        if bud:
            ax.text(0.98, 0.95, f"budget variants: ΔT = {min(r['dT_pct'] for r in bud):.0f}–"
                                f"{max(r['dT_pct'] for r in bud):.0f}% (off scale)",
                    transform=ax.transAxes, ha="right", va="top", fontsize=6)
    axes[0].set_ylabel("ΔT_acc vs no-sensor run (%)")
    h = [plt.Line2D([], [], color=COL[p], marker="o", ls="-", ms=3, lw=0.8, label=p) for p in P4]
    h += [plt.Line2D([], [], color="k", marker=m, ls="", ms=3, label=l)
          for m, l in (("o", "B = 64"), ("s", "B = 256"), ("^", "B = 1024"), ("x", "CPU deadline missed"))]
    axes[1].legend(handles=h, fontsize=5.8, loc="upper right", bbox_to_anchor=(1.0, 0.88))
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(PROJ, "reference_results"))
    a = ap.parse_args()
    rows = load(a.data)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    f = lambda n: os.path.join(HERE, "figures", n)
    fig1(rows, f("fig1_failure_map.png"))
    fig2(rows, f("fig2_min_fifo.png"))
    fig3(rows, f("fig3_delay.png"))
    fig4(rows, f("fig4_tradeoff.png"))
    print("wrote figures/fig1..fig4")


if __name__ == "__main__":
    main()
