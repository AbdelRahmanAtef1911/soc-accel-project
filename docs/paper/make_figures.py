#!/usr/bin/env python3
"""Paper figures (vector PDF) from a results folder.
    python3 make_figures.py <results-folder> <out-folder>
Needs the project's scripts/ folder next to the results folder, or SCRIPTS=<path>."""
import csv, os, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES, OUT = sys.argv[1], sys.argv[2]
SCRIPTS = os.environ.get("SCRIPTS", os.path.join(os.path.dirname(os.path.abspath(RES)), "scripts"))
sys.path.insert(0, SCRIPTS)
import plot_step5 as P5  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODELS = ["resnet8", "dscnn", "mbnet"]
NAME = {"resnet8": "ResNet-8", "dscnn": "DS-CNN", "mbnet": "MobileNetV1"}
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.edgecolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "pdf.fonttype": 42})


def totals(path, mode):
    t = defaultdict(lambda: defaultdict(int))
    for r in csv.DictReader(open(os.path.join(RES, path))):
        if r["mode"] != mode:
            continue
        m = t[r["model"]]
        tot = int(r["total"])
        m["total"] += tot
        if r["place"] == "CPU":
            m["cpu"] += tot
        else:
            for k in ("prep", "acc_load", "acc_compute", "acc_store"):
                m[k] += int(r[k])
            m["other"] += tot - int(r["prep"]) - int(r["acc_load"]) - int(r["acc_compute"]) - int(r["acc_store"])
    return t


STEPS = [("step1_sw.csv", "S1 software", "SW"), ("step2_tpu.csv", "S2 TPU 16x16", "DMA"),
         ("step3a_maeri_gemm.csv", "S3a MAERI", "DMA"), ("step3b_maeri_conv.csv", "S3b MAERI + conv", "DMA")]


def fig_e2e():
    data = [(lab, totals(f, mode)) for f, lab, mode in STEPS]
    fig, ax = plt.subplots(figsize=(3.45, 2.1))
    n, w = len(data), 0.8 / len(data)
    base = data[0][1]
    for i, (lab, t) in enumerate(data):
        xs = [k + (i - (n - 1) / 2) * w for k in range(3)]
        ys = [t[m]["total"] / 1e5 for m in MODELS]
        ax.bar(xs, ys, w * 0.9, color=SERIES[i], label=lab)
        if i:
            for x, m, y in zip(xs, MODELS, ys):
                ax.text(x, y * 1.18, f"{base[m]['total'] / t[m]['total']:.0f}x", ha="center", fontsize=5.5, color=INK2)
    ax.set_yscale("log")
    ax.set_ylabel("ms per inference @100 MHz (log)")
    ax.set_xticks(range(3)); ax.set_xticklabels([NAME[m] for m in MODELS])
    ax.legend(fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_e2e.pdf"))


def fig_breakdown():
    parts = [("prep", "im2col (CPU)"), ("cpu", "depthwise (CPU)"), ("acc_load", "DMA load"),
             ("acc_compute", "accel. compute"), ("acc_store", "DMA store"), ("other", "driver")]
    bars = []
    for f, lab, mode in STEPS[1:]:
        t = totals(f, mode)
        for m in MODELS:
            bars.append((f"{NAME[m]} {lab.split()[0]}", t[m]))
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    ys = list(range(len(bars)))[::-1]
    left = [0.0] * len(bars)
    for i, (k, lab) in enumerate(parts):
        v = [100.0 * b[1][k] / b[1]["total"] for b in bars]
        ax.barh(ys, v, 0.72, left=left, color=SERIES[i], label=lab, edgecolor="white", linewidth=0.5)
        left = [a + b for a, b in zip(left, v)]
    ax.set_yticks(ys); ax.set_yticklabels([b[0] for b in bars], fontsize=6)
    ax.set_xlim(0, 100); ax.set_xlabel("share of end-to-end time (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=5.8, ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.2))
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_breakdown.pdf"))


def fig_dse():
    rows = list(csv.DictReader(open(os.path.join(RES, "dse_summary.csv"))))
    by = defaultdict(list)
    for r in rows:
        by[r["sweep"]].append(r)
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.25))

    def line(ax, rs, xf, key, ls, pick=lambda r: True):
        for i, m in enumerate(MODELS):
            sr = sorted((r for r in rs if r["model"] == m and pick(r)), key=xf)
            ax.plot([xf(r) for r in sr], [int(r[key]) for r in sr], color=SERIES[i], ls=ls, lw=1.3, marker="o", ms=2.5)

    specs = [("tpu_size", "(a) TPU mesh (gemm partition)", "multipliers", lambda r: int(r["pe"]),
              [("total_cycles", "-", None), ("compute_cycles", "--", None)]),
             ("maeri_bw", "(b) MAERI bandwidth (gemm)", "dn_bw (elem./cycle)", lambda r: int(r["dn_bw"]),
              [("total_cycles", "-", None), ("compute_cycles", "--", None)]),
             ("dma_width", "(c) DMA width", "bytes/cycle", lambda r: int(r["dma_width"]),
              [("total_cycles", "-", lambda r: r["partition"] == "conv"), ("total_cycles", ":", lambda r: r["partition"] == "gemm")]),
             ("sparsity", "(d) zero weights (gemm)", "zeros (%)", lambda r: int(r["wzero"]),
              [("compute_cycles", "--", lambda r: r["arch"] == "SIGMA"), ("compute_cycles", ":", lambda r: r["arch"] == "TPU")])]
    for ax, (sw, title, xl, xf, lines) in zip(axs, specs):
        for key, ls, pk in lines:
            line(ax, by[sw], xf, key, ls, pk or (lambda r: True))
        ax.set_yscale("log"); ax.set_title(title, fontsize=7, loc="left"); ax.set_xlabel(xl)
        xs = sorted({xf(r) for r in by[sw]})
        if sw != "sparsity":
            ax.set_xscale("log", base=2)
        ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
        ax.minorticks_off()
    axs[0].set_ylabel("cycles (log)")
    leg = [plt.Line2D([], [], color=SERIES[i], lw=1.3, label=NAME[m]) for i, m in enumerate(MODELS)]
    leg += [plt.Line2D([], [], color=INK2, ls="-", label="end-to-end"),
            plt.Line2D([], [], color=INK2, ls="--", label="compute: accelerator (a, b), SIGMA (d)"),
            plt.Line2D([], [], color=INK2, ls=":", label="(c) TPU gemm partition; (d) TPU compute")]
    fig.legend(handles=leg, fontsize=6, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(pad=0.3, rect=(0, 0.17, 1, 1))
    fig.savefig(os.path.join(OUT, "fig_dse.pdf"))


def fig_step5():
    P5.RES = RES
    rows = P5.load()
    one = P5.one
    fig, axs = plt.subplots(2, 3, figsize=(7.16, 3.9))
    # (a) double buffering vs DMA width
    ax = axs[0, 0]
    for i, m in enumerate(MODELS):
        ys = [one(rows, run=f"overlap_dma{w}", model=m, mode="DMA")["total_cycles"] /
              one(rows, run=f"overlap_dma{w}", model=m, mode="DBUF")["total_cycles"] for w in (4, 8, 16)]
        ax.plot([4, 8, 16], ys, color=SERIES[i], lw=1.3, marker="o", ms=2.5, label=NAME[m])
    ax.set_xscale("log", base=2); ax.set_xticks([4, 8, 16]); ax.set_xticklabels(["4", "8", "16"]); ax.minorticks_off()
    ax.set_xlabel("DMA bytes/cycle"); ax.set_ylabel("speedup of DBUF")
    ax.set_title("(a) double buffering vs DMA width", loc="left", fontsize=7); ax.legend(fontsize=6)
    # (b) tiles
    ax = axs[0, 1]
    for w, ls in ((4, "--"), (16, "-")):
        for i, m in enumerate(MODELS):
            base = one(rows, run=f"overlap_dma{w}", model=m, mode="DMA")["total_cycles"]
            ys = [1.0] + [one(rows, run=(f"overlap_dma{w}" if t == 4 else f"dbuf_t{t}_dma{w}"), model=m,
                              mode="DBUF")["total_cycles"] / base for t in (2, 4, 8, 16)]
            ax.plot([1, 2, 4, 8, 16], ys, color=SERIES[i], ls=ls, lw=1.2, marker="o", ms=2.2)
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels(["1", "2", "4", "8", "16"])
    ax.minorticks_off(); ax.set_xlabel("tiles per layer (1 = one piece)"); ax.set_ylabel("time / one-piece time")
    ax.set_title("(b) number of tiles", loc="left", fontsize=7)
    ax.legend(handles=[plt.Line2D([], [], color=INK2, ls="-", label="16 B/cycle"),
                       plt.Line2D([], [], color=INK2, ls="--", label="4 B/cycle")], fontsize=6, loc="upper center", ncol=2)
    ax.set_ylim(0.76, 1.1)
    # (c), (d) ownership x wait
    buses = [("ideal", "ideal"), ("block", "block"), ("steal", "steal"), ("transparent", "transp.")]
    waits = [("poll", "poll"), ("wfi", "IRQ+WFI"), ("bg", "IRQ+background")]
    for ax, what in ((axs[1, 0], "time"), (axs[1, 1], "free")):
        for j, (wait, lab) in enumerate(waits):
            xs, ys = [], []
            for k, (bus, _) in enumerate(buses):
                rs = [P5.bus_row(rows, bus, wait, m) for m in MODELS]
                if what == "time":
                    v = sum(r["total_cycles"] for r in rs) / 1e6
                else:
                    v = 100.0 * sum(r["cpu_free_cycles"] for r in rs) / max(1, sum(r["wait_cycles"] for r in rs))
                xs.append(k + (j - 1) * 0.27); ys.append(v)
            ax.bar(xs, ys, 0.25, color=SERIES[j], label=lab)
        ax.set_xticks(range(4)); ax.set_xticklabels([b[1] for b in buses]); ax.grid(axis="x", visible=False)
    axs[1, 0].set_ylabel("M cycles, 3 models"); axs[1, 0].set_ylim(0, 1.95)
    axs[1, 0].set_title("(c) bus ownership x wait: time (DBUF)", loc="left", fontsize=7)
    axs[1, 0].legend(fontsize=6, loc="upper left")
    axs[1, 1].set_ylabel("CPU free while waiting (%)"); axs[1, 1].set_ylim(0, 100)
    axs[1, 1].set_title("(d) CPU time left for other work", loc="left", fontsize=7)
    # (e), (f) burst size
    for ax, key in ((axs[0, 2], "job_cycles"), (axs[1, 2], "cpu_bus_maxwait")):
        for j, (mode, lab, ls) in enumerate((("block", "block", "-"), ("steal", "cycle steal", "--"))):
            xs, ys = [], []
            for b in (4, 16, 64, 256, 1024):
                run = f"bus_{mode}_bg" if b == 64 else f"burst_{mode}_{b}"
                rs = [one(rows, run=run, model=m, mode="DBUF") for m in MODELS]
                xs.append(b); ys.append(sum(r[key] for r in rs) / 1e6 if key == "job_cycles" else max(r[key] for r in rs))
            ax.plot(xs, ys, color=SERIES[3 + j], ls=ls, lw=1.3, marker="o", ms=2.5, label=lab)
        ax.set_xscale("log", base=2); ax.set_xticks([4, 16, 64, 256, 1024])
        ax.set_xticklabels(["4", "16", "64", "256", "1024"]); ax.minorticks_off(); ax.legend(fontsize=6)
        ax.set_xlabel("burst size (bytes)")
    axs[0, 2].set_ylabel("accel. time, 3 models (M cyc.)")
    axs[0, 2].set_title("(e) burst size: accelerator time", loc="left", fontsize=7)
    axs[1, 2].set_yscale("log"); axs[1, 2].set_ylabel("cycles (log)")
    axs[1, 2].set_title("(f) burst size: longest CPU stall", loc="left", fontsize=7)
    fig.tight_layout(pad=0.3, h_pad=0.8)
    fig.savefig(os.path.join(OUT, "fig_step5.pdf"))


os.makedirs(OUT, exist_ok=True)
fig_e2e(); fig_breakdown(); fig_dse(); fig_step5()
print("figures in", OUT)
