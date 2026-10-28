#!/usr/bin/env python3
"""
plot_results.py - figures for the report (PNG, in results/figures/).

    python3 scripts/plot_results.py

Uses whatever exists in results/:
  step1_sw.csv, step2_tpu.csv, step3a_maeri_gemm.csv, step3b_maeri_conv.csv
      -> fig1_end_to_end.png   total time per model and design (log scale)
      -> fig2_breakdown.png    where the time goes in each design (DMA mode)
  dse_summary.csv (from dse.py)
      -> fig3_dse.png          the four sweeps
"""
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "results")
FIG = os.path.join(RES, "figures")

# validated categorical palette (light mode), used in this fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODELS = ["resnet8", "dscnn", "mbnet"]
MODEL_NAME = {"resnet8": "ResNet-8", "dscnn": "DS-CNN", "mbnet": "MobileNetV1-0.25"}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150,
})

STEPS = [  # file, label, mode used
    ("step1_sw.csv", "Step 1: software", "SW"),
    ("step2_tpu.csv", "Step 2: TPU 16x16", "DMA"),
    ("step3a_maeri_gemm.csv", "Step 3a: MAERI, same partition", "DMA"),
    ("step3b_maeri_conv.csv", "Step 3b: MAERI + conv offload", "DMA"),
]


def load(name):
    p = os.path.join(RES, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else None


def totals(rows, mode):
    t = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["mode"] != mode:
            continue
        m = t[r["model"]]
        tot = int(r["total"])
        m["total"] += tot
        if r["place"] == "CPU":
            m["cpu"] += tot
        else:
            m["prep"] += int(r["prep"])
            m["load"] += int(r["acc_load"])
            m["compute"] += int(r["acc_compute"])
            m["store"] += int(r["acc_store"])
            m["other"] += tot - int(r["prep"]) - int(r["acc_load"]) - int(r["acc_compute"]) - int(r["acc_store"])
    return t


def fig_end_to_end():
    data = []
    for f, label, mode in STEPS:
        rows = load(f)
        if rows:
            data.append((label, totals(rows, mode)))
    if not data:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    n = len(data)
    w = 0.8 / n
    base = data[0][1]
    for i, (label, t) in enumerate(data):
        xs = [k + (i - (n - 1) / 2) * w for k in range(len(MODELS))]
        ms = [t[m]["total"] / 1e5 for m in MODELS]  # cycles at 100 MHz -> ms
        ax.bar(xs, ms, w * 0.92, color=SERIES[i], label=label, edgecolor="white", linewidth=1)
        if i > 0 and data[0][0].startswith("Step 1"):
            for x, m, v in zip(xs, MODELS, ms):
                ax.text(x, v * 1.15, f"{base[m]['total'] / t[m]['total']:.0f}x", ha="center",
                        va="bottom", fontsize=8, color=INK2)
    ax.set_yscale("log")
    ax.set_ylabel("time per inference (ms at 100 MHz, log scale)")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_NAME[m] for m in MODELS])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)
    ax.set_title("End-to-end time per model (labels: speedup over step 1)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_end_to_end.png"))
    plt.close(fig)


def fig_breakdown():
    parts = [("prep", "im2col on CPU"), ("cpu", "depthwise on CPU"), ("load", "DMA load"),
             ("compute", "accelerator compute"), ("store", "DMA store"), ("other", "driver/polling")]
    bars = []
    for f, label, mode in STEPS[1:]:
        rows = load(f)
        if not rows:
            continue
        t = totals(rows, mode)
        for m in MODELS:
            if m in t:
                bars.append((f"{MODEL_NAME[m]}\n{label.split(':')[0]}", t[m]))
    if not bars:
        return
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(bars) + 1.4))
    ys = list(range(len(bars)))[::-1]
    left = [0.0] * len(bars)
    for i, (key, name) in enumerate(parts):
        vals = [100.0 * b[1][key] / b[1]["total"] for b in bars]
        ax.barh(ys, vals, 0.7, left=left, color=SERIES[i], label=name, edgecolor="white", linewidth=1)
        left = [l + v for l, v in zip(left, vals)]
    ax.set_yticks(ys)
    ax.set_yticklabels([b[0] for b in bars], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of total time (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    ax.set_title("Where the time goes (DMA mode)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_breakdown.png"))
    plt.close(fig)


def fig_dse():
    rows = load("dse_summary.csv")
    if not rows:
        return
    by = defaultdict(list)
    for r in rows:
        by[r["sweep"]].append(r)

    def series(rs, model, key, xf, pick=lambda r: True):
        sr = sorted((r for r in rs if r["model"] == model and pick(r)), key=xf)
        return [xf(r) for r in sr], [int(r[key]) for r in sr]

    # each panel: list of (label, x-function, filter, column, line style)
    panels = [
        ("tpu_size", "a) TPU mesh size (gemm partition)", "multipliers", lambda r: int(r["pe"]),
         [("end-to-end", lambda r: True, "total_cycles", "-"),
          ("accelerator compute", lambda r: True, "compute_cycles", "--")]),
        ("maeri_bw", "b) MAERI distribution bandwidth (gemm partition)", "dn_bw (elements/cycle)",
         lambda r: int(r["dn_bw"]),
         [("end-to-end", lambda r: True, "total_cycles", "-"),
          ("accelerator compute", lambda r: True, "compute_cycles", "--")]),
        ("dma_width", "c) DMA width", "DMA bytes per cycle", lambda r: int(r["dma_width"]),
         [("MAERI + conv offload, end-to-end", lambda r: r["partition"] == "conv", "total_cycles", "-"),
          ("TPU 16x16, end-to-end", lambda r: r["partition"] == "gemm", "total_cycles", ":")]),
        ("sparsity", "d) pruned weights (gemm partition)", "weights set to zero (%)", lambda r: int(r["wzero"]),
         [("SIGMA compute", lambda r: r["arch"] == "SIGMA", "compute_cycles", "--"),
          ("TPU 16x16 compute", lambda r: r["arch"] == "TPU", "compute_cycles", ":"),
          ("SIGMA end-to-end", lambda r: r["arch"] == "SIGMA", "total_cycles", "-")]),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 7.6))
    for ax, (sweep, title, xl, xf, lines) in zip(axs.flat, panels):
        rs = by.get(sweep, [])
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.set_xlabel(xl)
        ax.set_ylabel("cycles (log)")
        ax.set_yscale("log")
        if not rs:
            ax.text(0.5, 0.5, "not run", ha="center", transform=ax.transAxes, color=INK2)
            continue
        for mi, m in enumerate(MODELS):
            for label, pick, key, ls in lines:
                xs, ys = series(rs, m, key, xf, pick)
                if xs:
                    ax.plot(xs, ys, color=SERIES[mi], ls=ls, lw=2, marker="o", ms=4)
        if sweep != "sparsity":
            ax.set_xscale("log", base=2)
            xs = sorted({xf(r) for r in rs})
            ax.set_xticks(xs)
            ax.set_xticklabels([str(x) for x in xs])
        else:
            ax.set_xticks(sorted({xf(r) for r in rs}))
        handles = [plt.Line2D([], [], color=SERIES[i], lw=2, label=MODEL_NAME[m]) for i, m in enumerate(MODELS)]
        handles += [plt.Line2D([], [], color=INK2, lw=2, ls=ls, label=label) for label, _, _, ls in lines]
        ax.legend(handles=handles, fontsize=7, loc="best")
    fig.suptitle("Design-space exploration: all MLPerf Tiny layers, DMA mode (colour = model, line = measure)",
                 x=0.02, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_dse.png"))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    fig_end_to_end()
    fig_breakdown()
    fig_dse()
    print("figures in", FIG, ":", ", ".join(sorted(os.listdir(FIG))))


if __name__ == "__main__":
    main()
