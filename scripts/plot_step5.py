#!/usr/bin/env python3
"""
plot_step5.py - figure 4 of the report: the step-5 sweeps (results/dse5_summary.csv).

    python3 scripts/plot_step5.py      -> results/figures/fig4_step5.png

  a) double buffering vs DMA width      b) number of tiles
  c) bus ownership x wait style: time   d) ... CPU time left for other work
  e) burst size: accelerator job time   f) burst size: longest CPU wait for the bus
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

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODELS = ["resnet8", "dscnn", "mbnet"]
MODEL_NAME = {"resnet8": "ResNet-8", "dscnn": "DS-CNN", "mbnet": "MobileNetV1-0.25"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150,
})


def load():
    p = os.path.join(RES, "dse5_summary.csv")
    if not os.path.exists(p):
        raise SystemExit("results/dse5_summary.csv not found: run scripts/dse5.py first")
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for k in r:
            if k.endswith("_cycles") or k in ("burst", "dma_width", "ntiles", "polls", "bg_units", "resp_avg",
                                               "cpu_bus_txn", "cpu_bus_wait", "cpu_bus_maxwait", "dma_bus_wait",
                                               "bytes_in"):
                r[k] = int(r[k])
        r["cpu_free_pct"] = float(r["cpu_free_pct"])
    return rows


def pick(rows, **kw):
    return [r for r in rows if all(str(r[k]) == str(v) for k, v in kw.items())]


def one(rows, **kw):
    rs = pick(rows, **kw)
    return rs[0] if rs else None


def model_legend(ax, extra=(), **kw):
    h = [plt.Line2D([], [], color=SERIES[i], lw=2, marker="o", ms=4, label=MODEL_NAME[m]) for i, m in enumerate(MODELS)]
    ax.legend(handles=h + list(extra), fontsize=7, **({"loc": "best"} | kw))


def panel_overlap(ax, rows):
    ax.set_title("a) double buffering vs DMA width (4 tiles, CPU polls)", loc="left", fontsize=9)
    for i, m in enumerate(MODELS):
        xs, ys = [], []
        for w in (4, 8, 16):
            a = one(rows, run=f"overlap_dma{w}", model=m, mode="DMA")
            b = one(rows, run=f"overlap_dma{w}", model=m, mode="DBUF")
            if a and b:
                xs.append(w)
                ys.append(a["total_cycles"] / b["total_cycles"])
        ax.plot(xs, ys, color=SERIES[i], lw=2, marker="o", ms=4)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}x", (x, y), textcoords="offset points", xytext=(0, 5), ha="center",
                        fontsize=6.5, color=SERIES[i])
    ax.axhline(1.0, color=INK2, lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16])
    ax.set_xticklabels(["4", "8", "16"])
    ax.set_xlabel("DMA bytes per cycle")
    ax.set_ylabel("speedup of DBUF over one piece")
    model_legend(ax)


def panel_tiles(ax, rows):
    ax.set_title("b) number of tiles (double buffering)", loc="left", fontsize=9)
    for w, ls in ((4, "--"), (16, "-")):
        for i, m in enumerate(MODELS):
            base = one(rows, run=f"overlap_dma{w}", model=m, mode="DMA")
            if not base:
                continue
            xs, ys = [1], [1.0]
            for t in (2, 4, 8, 16):
                r = one(rows, run=(f"overlap_dma{w}" if t == 4 else f"dbuf_t{t}_dma{w}"), model=m, mode="DBUF")
                if r:
                    xs.append(t)
                    ys.append(r["total_cycles"] / base["total_cycles"])
            ax.plot(xs, ys, color=SERIES[i], lw=2 if w == 16 else 1.3, ls=ls, marker="o", ms=3.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1\n(one piece)", "2", "4", "8", "16"])
    ax.set_xlabel("tiles per layer")
    ax.set_ylabel("time relative to one piece")
    extra = [plt.Line2D([], [], color=INK2, lw=2, ls="-", label="DMA 16 B/cycle"),
             plt.Line2D([], [], color=INK2, lw=1.3, ls="--", label="DMA 4 B/cycle")]
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.45 * (hi - lo))  # room for the legend above the lines
    model_legend(ax, extra, loc="upper center", ncol=2)


BUSES = [("ideal", "ideal bus"), ("block", "block"), ("steal", "cycle steal"), ("transparent", "transparent")]
WAITS = [("poll", "POLL", "poll"), ("wfi", "WFI", "IRQ + WFI"), ("bg", "BG", "IRQ + background work")]


def bus_row(rows, bus, wait, model, mode="DBUF"):
    if bus == "ideal":
        run = "overlap_dma4" if wait == "poll" else f"ideal_{wait}"
    else:
        run = f"bus_{bus}_{wait}"
    return one(rows, run=run, model=model, mode=mode)


def panel_bus(ax, rows, key, title, ylabel, scale=1.0, total=True, legend=None):
    ax.set_title(title, loc="left", fontsize=9)
    n = len(WAITS)
    w = 0.8 / n
    for j, (wait, _, label) in enumerate(WAITS):
        xs, ys = [], []
        for k, (bus, _) in enumerate(BUSES):
            rs = [bus_row(rows, bus, wait, m) for m in MODELS]
            if any(r is None for r in rs):
                continue
            if total:
                v = sum(r[key] for r in rs) * scale
            else:  # weighted share: sum of free cycles / sum of waiting cycles
                v = 100.0 * sum(r["cpu_free_cycles"] for r in rs) / max(1, sum(r["wait_cycles"] for r in rs))
            xs.append(k + (j - (n - 1) / 2) * w)
            ys.append(v)
        ax.bar(xs, ys, w * 0.92, color=SERIES[j], label=label, edgecolor="white", linewidth=1)
        for x, y in zip(xs, ys):
            ax.text(x, y, f"{y:.2f}" if total else f"{y:.0f}%", ha="center", va="bottom", fontsize=6, color=INK2)
    ax.set_xticks(range(len(BUSES)))
    ax.set_xticklabels([b[1] for b in BUSES])
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, **(legend or {"loc": "upper left"}))


def panel_burst(ax, rows, key, title, ylabel, logy=False):
    ax.set_title(title, loc="left", fontsize=9)
    for j, (mode, label, ls) in enumerate((("block", "block", "-"), ("steal", "cycle steal", "--"))):
        xs, ys = [], []
        for b in (4, 16, 64, 256, 1024):
            run = f"bus_{mode}_bg" if b == 64 else f"burst_{mode}_{b}"
            rs = [one(rows, run=run, model=m, mode="DBUF") for m in MODELS]
            if any(r is None for r in rs):
                continue
            xs.append(b)
            ys.append(sum(r[key] for r in rs) / 1e6 if key == "job_cycles" else max(r[key] for r in rs))
        ax.plot(xs, ys, color=SERIES[3 + j], lw=2, ls=ls, marker="o", ms=4, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 16, 64, 256, 1024])
    ax.set_xticklabels(["4", "16", "64", "256", "1024"])
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("DMA burst size (bytes)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, loc="best")


def main():
    rows = load()
    os.makedirs(FIG, exist_ok=True)
    fig, axs = plt.subplots(3, 2, figsize=(9.5, 10.5))
    panel_overlap(axs[0, 0], rows)
    panel_tiles(axs[0, 1], rows)
    panel_bus(axs[1, 0], rows, "total_cycles", "c) bus ownership x wait style: inference time (DBUF)",
              "all three models (M cycles)", scale=1e-6)
    panel_bus(axs[1, 1], rows, None, "d) CPU time free for other work while it waits",
              "free / waiting time (%)", total=False, legend={"loc": "center", "bbox_to_anchor": (0.42, 0.62)})
    panel_burst(axs[2, 0], rows, "job_cycles", "e) burst size: accelerator time (CPU busy, BG)",
                "accelerator jobs, all models (M cycles)")
    panel_burst(axs[2, 1], rows, "cpu_bus_maxwait", "f) burst size: longest CPU wait for the bus",
                "cycles (log)", logy=True)
    fig.suptitle("Step 5: DMA and bus design (MAERI 256, conv partition, all MLPerf Tiny layers)",
                 x=0.02, ha="left", fontsize=10.5)
    fig.tight_layout()
    out = os.path.join(FIG, "fig4_step5.png")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
