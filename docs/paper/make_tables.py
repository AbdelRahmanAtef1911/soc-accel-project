#!/usr/bin/env python3
"""LaTeX tables (booktabs) from a results folder:  python3 make_tables.py <results> <out>"""
import csv, os, sys
from collections import defaultdict

RES, OUT = sys.argv[1], sys.argv[2]
MODELS = ["resnet8", "dscnn", "mbnet"]
NAME = {"resnet8": "ResNet-8", "dscnn": "DS-CNN", "mbnet": "MobileNetV1-0.25"}
f = lambda n: f"{n:,}".replace(",", "{,}")
load = lambda p: list(csv.DictReader(open(os.path.join(RES, p))))


def write(name, s):
    open(os.path.join(OUT, name), "w").write(s)


def totals(path, mode):
    t = defaultdict(lambda: defaultdict(int))
    for r in load(path):
        if r["mode"] != mode:
            continue
        m = t[r["model"]]
        tot = int(r["total"]); m["total"] += tot; m["macs"] += int(r["macs"]); m["layers"] += 1
        m[r["type"]] += 1
        if r["place"] == "CPU":
            m["cpu"] += tot
        else:
            for k in ("prep", "acc_load", "acc_compute", "acc_store"):
                m[k] += int(r[k])
    return t


# workload
t = totals("step1_sw.csv", "SW")
rows = "".join(f"{NAME[m]} & {t[m]['layers']} & {t[m]['CONV']} & {t[m]['PW']} & {t[m]['DW']} & {t[m]['FC']} & "
               f"{t[m]['macs'] / 1e6:.2f} \\\\\n" for m in MODELS)
write("tab_workload.tex", r"""\begin{table}[t]\centering\footnotesize
\caption{Workload: the INT8 layers of the three MLPerf Tiny models.}\label{tab:workload}
\begin{tabular}{@{}lrrrrrr@{}}\toprule
model & layers & CONV & PW & DW & FC & MMACs\\\midrule
""" + rows + r"""\bottomrule\end{tabular}\end{table}
""")

# step 0
rows = ""
for r in load("step0_stonne.csv"):
    macs, tp, ma = int(r["macs"]), int(r["tpu16x16"]), int(r["maeri256_bw256"])
    rows += (f"{r['layer'].replace('_', chr(92) + '_')} & {r['M']}$\\times${r['K']}$\\times${r['N']} & {f(tp)} & {f(ma)} & "
             f"{f(int(r['sigma256_bw256_dense']))} & {f(int(r['sigma256_bw256_50pct_zero_weights']))} & "
             f"{100 * macs / 256 / tp:.0f} & {100 * macs / 256 / ma:.0f} \\\\\n")
write("tab_stonne.tex", r"""\begin{table}[t]\centering\scriptsize\setlength{\tabcolsep}{2pt}
\caption{S0: STONNE alone, 256 multipliers each. Compute cycles and multiplier utilization (\%).}\label{tab:stonne}
\begin{tabular}{@{}llrrrrrr@{}}\toprule
layer & $M{\times}K{\times}N$ & TPU & MAERI & SIGMA & SIGMA & \multicolumn{2}{c}{util.\ (\%)}\\
 & & 16$\times$16 & bw 256 & dense & 50\% z. & TPU & MAERI\\\midrule
""" + rows + r"""\bottomrule\end{tabular}\end{table}
""")

# end-to-end S1-S3
steps = [("step1_sw.csv", "S1 software", "SW"), ("step2_tpu.csv", "S2 TPU 16$\\times$16", "DMA"),
         ("step3a_maeri_gemm.csv", "S3a MAERI", "DMA"), ("step3b_maeri_conv.csv", "S3b MAERI + conv", "DMA")]
sw = totals("step1_sw.csv", "SW")
rows = ""
for m in MODELS:
    for i, (p, lab, mode) in enumerate(steps):
        x = totals(p, mode)[m]
        tot = x["total"]
        pc = lambda k: f"{100 * x[k] / tot:.1f}"
        dma = 100 * (x["acc_load"] + x["acc_store"]) / tot
        rows += (f"{NAME[m] if i == 0 else ''} & {lab} & {f(tot)} & {(f(round(sw[m]['total'] / tot)) if sw[m]['total'] / tot >= 100 else f"{sw[m]['total'] / tot:.1f}")}$\\times$ & "
                 f"{pc('prep')} & {pc('cpu') if mode != 'SW' else '100'} & {dma:.1f} & {pc('acc_compute')} \\\\\n")
    rows += "\\midrule\n" if m != MODELS[-1] else ""
write("tab_e2e.tex", r"""\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{S1--S3: end-to-end cycles (DMA mode), speedup over software, and where the time goes (\% of total).}\label{tab:e2e}
\begin{tabular}{@{}llrrrrrr@{}}\toprule
model & design & cycles & speedup & im2col & CPU-only & DMA & compute\\\midrule
""" + rows + r"""\bottomrule\end{tabular}\end{table*}
""")


# step 5a-c
def t5(path):
    info = {}
    for line in open(os.path.join(RES, path[:-4] + ".log")):
        if line.startswith("INFO,"):
            x = line.strip().split(",")[1:]
            info = {x[i]: x[i + 1] for i in range(0, len(x) - 1, 2)}
    bg = int(info.get("bg_unit_cycles", 0))
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for r in load(path):
        o = out[r["model"]][r["mode"]]
        for k in ("total", "run", "acc_total", "acc_load", "acc_compute", "acc_store", "sleep", "bg_units", "resp", "polls"):
            o[k] += int(r[k])
        o["free"] += int(r["sleep"]) + int(r["bg_units"]) * bg
        o["n"] += 1
    return out


a, b, c = t5("step5a_dbuf.csv"), t5("step5b_irq_wfi.csv"), t5("step5c_bus_steal_bg.csv")
rows = ""
for m in MODELS:
    for i, mode in enumerate(("DMA", "TILE", "DBUF")):
        A, B, C = a[m][mode], b[m][mode], c[m][mode]
        rows += (f"{NAME[m] if i == 0 else ''} & {mode} & {f(A['total'])} & {a[m]['DMA']['total'] / A['total']:.2f}$\\times$ & "
                 f"{(B['total'] / A['total'] - 1) * 100:+.1f} & {100 * B['free'] / B['run']:.0f} & "
                 f"{(C['total'] / A['total'] - 1) * 100:+.1f} & {100 * C['free'] / C['run']:.0f} \\\\\n")
    rows += "\\midrule\n" if m != MODELS[-1] else ""
write("tab_step5.tex", r"""\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{2.6pt}
\caption{S5a--c (checked runs). 5a: ideal bus, CPU polls. 5b: interrupt + WFI, ideal bus. 5c: shared bus, cycle stealing (64-B bursts), interrupt + background work. Time change relative to 5a; ``free'' = share of the waiting time the CPU can use.}\label{tab:step5}
\begin{tabular}{@{}llrrrrrr@{}}\toprule
 & & \multicolumn{2}{c}{5a (poll, ideal)} & \multicolumn{2}{c}{5b (WFI)} & \multicolumn{2}{c}{5c (steal + BG)}\\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
model & mode & cycles & vs DMA & $\Delta$\% & free\% & $\Delta$\% & free\%\\\midrule
""" + rows + r"""\bottomrule\end{tabular}\end{table*}
""")

# bus x wait matrix (DBUF, 3 models)
d5 = load("dse5_summary.csv")
for r in d5:
    for k in r:
        if k.endswith("_cycles") or k in ("cpu_bus_wait", "cpu_bus_maxwait", "dma_bus_wait"):
            r[k] = int(r[k])
get = lambda run: [r for r in d5 if r["run"] == run and r["mode"] == "DBUF"]
base = sum(r["total_cycles"] for r in get("overlap_dma4"))
rows = ""
for bus, bl in (("ideal", "ideal"), ("block", "block"), ("steal", "cycle steal"), ("transparent", "transparent")):
    for j, (w, wl) in enumerate((("poll", "poll"), ("wfi", "IRQ + WFI"), ("bg", "IRQ + BG"))):
        run = ("overlap_dma4" if w == "poll" else f"ideal_{w}") if bus == "ideal" else f"bus_{bus}_{w}"
        rs = get(run)
        tot = sum(r["total_cycles"] for r in rs)
        free = 100 * sum(r["cpu_free_cycles"] for r in rs) / sum(r["wait_cycles"] for r in rs)
        rows += (f"{bl if j == 0 else ''} & {wl} & {f(tot)} & {(tot / base - 1) * 100:+.1f} & {free:.0f} & "
                 f"{f(max(r['cpu_bus_maxwait'] for r in rs))} & {f(sum(r['dma_bus_wait'] for r in rs))} \\\\\n")
    rows += "\\midrule\n" if bus != "transparent" else ""
write("tab_bus.tex", r"""\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{Bus ownership $\times$ how the CPU waits: double-buffered jobs, 64-B bursts, all three models together. $\Delta$ relative to the ideal bus with polling.}\label{tab:bus}
\begin{tabular}{@{}llrrrrr@{}}\toprule
bus & CPU waits by & cycles & $\Delta$\% & free\% & max CPU stall & DMA bus wait\\\midrule
""" + rows + r"""\bottomrule\end{tabular}\end{table*}
""")
print("tables in", OUT)
