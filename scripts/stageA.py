#!/usr/bin/env python3
"""
stageA.py - course project, stage A: first coarse failure map (plan v3.3,
Sections 4.3-4.6, 5.1, 6.1).

One DS-CNN inference (10 layers, double buffered, 4 tiles) is released after
a warm-up; the 62 us control task and the streaming sensor run throughout.
Model: request/grant bus (--bus-arbiter), handover a = 1 on ownership change,
CPU instruction fetches from a private 1-cycle memory (--cpu-fetch-private),
CPU loads/stores, sensor DMA and DNN DMA on the shared bus.

Sweep (stage A): policies rr, sensor-first, cpu-first, buffer
  x DNN burst B in {64, 1024} x sensor rate in {8, 80, 800, 8000, 25000} kS/s
  (S = 4 B: 32 KB/s ... 100 MB/s) x FIFO depth in {theta, 8 theta, 64 theta}
  (capped at 512), theta fixed per (rate, B) by condition C2 (RR blocking).
Plus a matched no-sensor baseline per (policy, B).

Per point it reports and judges (limits of Section 6.1):
  sensor   zero overflows in the window [release, release + T_inf)
  cpu      zero late or lost control-task releases in the window
  dnn      T_acc <= T_inf (release -> last layer done)
  delay    dT_acc <= 10 % of the matched no-sensor baseline (same policy, B)
  fifo     depth <= 8 samples (strict) / <= 1280 samples (relaxed)
  U_min    overload indicator; a point is labelled overloaded only if the
           measured window bus occupancy is >= 99 % and a limit fails
  bound    sensor-first points that satisfy C1 and C2 must not overflow

    python3 scripts/stageA.py [--jobs 2] [--skip-existing] [--summary-only] [--plot]
"""
import argparse
import csv
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "results")
OUT = os.path.join(RES, "stageA")

F_CLK = 100e6
A = 1            # handover cycles
S = 4            # bytes per sample
T_INF_US = 5000
T_INF_CYC = T_INF_US * 100
POLICIES = ("rr", "sensor-first", "cpu-first", "buffer")
BURSTS = (64, 1024)
RATES_KSPS = (8, 80, 800, 8000, 25000)
DELAY_MARGIN = 0.10
FMAX_STRICT, FMAX_RELAXED = 8, 1280
BASE = ["--accel-arch", "maeri", "--accel-ms", "256", "--accel-dnbw", "256", "--cpu-fetch-private"]
ELF = "bench_kws.elf"


def bcyc(nbytes):
    return 10 + math.ceil(nbytes / 4)


def theta_for(ksps, B):
    """smallest power of two theta <= 128 with C2 under round-robin blocking"""
    R = ksps * 1000.0
    beta_rr = bcyc(B) + 1 + A
    for k in range(8):
        th = 1 << k
        L = beta_rr + A + bcyc(th * S)
        if math.ceil(R * L / F_CLK) <= th:
            return th, True
    return 128, False


def bound_ok(ksps, B, th, depth):
    """sufficient no-overflow conditions C1 and C2 for sensor-first (policy 2)"""
    R = ksps * 1000.0
    L = bcyc(B) + A + bcyc(th * S)
    n = math.ceil(R * L / F_CLK)
    return n <= th and depth >= th + n, th + n


def runs():
    out = []
    for p in POLICIES:
        for B in BURSTS:
            out.append(dict(name=f"base_{p}_{B}", policy=p, B=B, ksps=0, theta=0, depth=0))
            for r in RATES_KSPS:
                th, _ = theta_for(r, B)
                depths = sorted({min(d, 512) for d in (th, 8 * th, 64 * th)})
                for d in depths:
                    out.append(dict(name=f"{p}_{B}_{r}k_th{th}_d{d}", policy=p, B=B, ksps=r, theta=th, depth=d))
    return out


def run(r, skip):
    csvp = os.path.join(OUT, r["name"] + ".csv")
    if skip and os.path.exists(csvp):
        return
    opts = BASE + ["--accel-dma-burst", str(r["B"]), "--bus-arbiter", r["policy"]]
    if r["ksps"]:
        opts += ["--sensor-rate-ksps", str(r["ksps"]), "--sensor-theta", str(r["theta"]),
                 "--sensor-depth", str(r["depth"]), "--sensor-bytes", str(S)]
    p = subprocess.run(["bash", os.path.join(PROJ, "scripts", "run_vp.sh"), ELF, f"stageA/{r['name']}.csv"] + opts,
                       capture_output=True, text=True)
    print(p.stdout.strip() or p.stderr.strip(), flush=True)


def parse(r):
    base = os.path.join(OUT, r["name"])
    if not os.path.exists(base + ".csv"):
        return None
    rows = list(csv.DictReader(open(base + ".csv")))
    if not rows:
        return None
    d = {k: int(v) if v.isdigit() else v for k, v in rows[0].items()}
    for ln in open(base + ".log"):
        f = ln.strip().split(",")
        if f[0] == "WINDOW":
            kv = dict(zip(f[1::2], f[2::2]))
            d["occ"] = float(kv["occupancy"])
            for m in ("cpu", "sensor", "dnn"):
                d[f"win_{m}_busy"] = int(kv[f"{m}_busy"])
        elif f[0] == "RGBUS" and f[1] in ("cpu", "sensor", "dnn"):
            d[f"{f[1]}_grants"], d[f"{f[1]}_busy"], d[f"{f[1]}_maxwait"] = int(f[2]), int(f[3]), int(f[5])
    return d


def summarize():
    rs = runs()
    data = {r["name"]: (r, parse(r)) for r in rs}
    rows = []
    for name, (r, d) in data.items():
        if d is None or r["ksps"] == 0:
            continue
        b = data[f"base_{r['policy']}_{r['B']}"][1]
        if b is None:
            continue
        R = r["ksps"] * 1000.0
        c_sen = (R * T_INF_US * 1e-6 / r["theta"]) * bcyc(r["theta"] * S)
        u_min = (b["dnn_busy"] + c_sen + b["win_cpu_busy"]) / T_INF_CYC
        dacc = d["t_acc"] - b["t_acc"]
        ok_sen = d["sen_ovf_win"] == 0
        ok_cpu = d["rt_miss"] + d["rt_skip"] == 0
        ok_dnn = d["t_acc"] <= T_INF_CYC
        ok_delay = dacc <= DELAY_MARGIN * b["t_acc"]
        fails = "".join(c for c, ok in (("S", ok_sen), ("C", ok_cpu), ("D", ok_dnn and ok_delay)) if not ok)
        bnd_applies, f_suf = bound_ok(r["ksps"], r["B"], r["theta"], r["depth"])
        rows.append(dict(
            policy=r["policy"], B=r["B"], rate_ksps=r["ksps"], rate_MBps=r["ksps"] * S / 1000.0, theta=r["theta"],
            depth=r["depth"], u_min=round(u_min, 3), occupancy=round(d.get("occ", 0), 3),
            ovf_win=d["sen_ovf_win"], peak_win=d["sen_peak_win"], sen_maxwait=d["sen_maxwait"],
            rt_rel=d["rt_rel"], rt_miss=d["rt_miss"], rt_skip=d["rt_skip"], rt_max=d["rt_max"],
            t_acc=d["t_acc"], base_t_acc=b["t_acc"], dT_acc_pct=round(100.0 * dacc / b["t_acc"], 2),
            base_rt_miss=b["rt_miss"] + b["rt_skip"], base_rt_max=b["rt_max"],
            ok_sensor=int(ok_sen), ok_cpu=int(ok_cpu), ok_dnn=int(ok_dnn), ok_delay=int(ok_delay),
            fails=fails or "-", pass_all=int(not fails),
            overloaded=int(bool(fails) and d.get("occ", 0) >= 0.99),
            fifo_strict=int(r["depth"] <= FMAX_STRICT), fifo_relaxed=int(r["depth"] <= FMAX_RELAXED),
            bound_C1C2=int(bnd_applies) if r["policy"] == "sensor-first" else "",
            bound_F=f_suf if r["policy"] == "sensor-first" else "",
            bound_violated=int(r["policy"] == "sensor-first" and bnd_applies and not ok_sen)))
    bases = []
    for name, (r, d) in data.items():
        if r["ksps"] == 0 and d:
            bases.append(dict(policy=r["policy"], B=r["B"], t_acc=d["t_acc"], dnn_met=d["dnn_met"],
                              rt_rel=d["rt_rel"], rt_miss=d["rt_miss"], rt_skip=d["rt_skip"], rt_max=d["rt_max"],
                              cpu_maxwait=d.get("cpu_maxwait", ""), occupancy=d.get("occ", "")))
    with open(os.path.join(RES, "stageA_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RES, "stageA_baselines.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bases[0].keys()))
        w.writeheader()
        w.writerows(bases)
    print(f"wrote results/stageA_summary.csv ({len(rows)} points), results/stageA_baselines.csv ({len(bases)})")
    return rows, bases


def report(rows, bases):
    print("\n== no-sensor baselines (matched, per policy and B) ==")
    for b in sorted(bases, key=lambda x: (x["B"], x["policy"])):
        print(f"  {b['policy']:12s} B={b['B']:5d}  T_acc {b['t_acc']:7d} cyc  control task late/lost "
              f"{b['rt_miss'] + b['rt_skip']:2d}/{b['rt_rel'] + b['rt_skip']:3d}, worst response {b['rt_max']} cyc")
    print("\n== bound check (sensor-first; points meeting C1 and C2 must not overflow) ==")
    app = [r for r in rows if r["bound_C1C2"] == 1]
    bad = [r for r in rows if r["bound_violated"]]
    print(f"  {len(app)} points satisfy C1-C2, {len(bad)} overflowed  [{'PASS' if not bad else 'FAIL'}]")
    print("\n== failure map: fails S = sensor overflow, C = control-task deadline, D = DNN deadline or >10 % delay ==")
    for B in BURSTS:
        print(f"\n  B = {B} bytes")
        hdr = f"  {'rate':>9s} {'theta':>5s} {'depth':>5s} |" + "".join(f" {p:>13s}" for p in POLICIES)
        print(hdr)
        keys = sorted({(r["rate_ksps"], r["theta"], r["depth"]) for r in rows if r["B"] == B})
        for k, th, dep in keys:
            cells = []
            for p in POLICIES:
                x = [r for r in rows if r["B"] == B and r["policy"] == p and r["rate_ksps"] == k and r["depth"] == dep]
                if not x:
                    cells.append(f"{'':>13s}")
                    continue
                x = x[0]
                tag = "pass" if x["pass_all"] else x["fails"]
                if x["overloaded"]:
                    tag += "*"
                cells.append(f" {tag + ' ' + format(x['dT_acc_pct'], '+.0f') + '%':>13s}")
            print(f"  {k * S / 1000:6.3g}MB/s {th:5d} {dep:5d} |" + "".join(cells))
    print("\n  (* = overloaded: window bus occupancy >= 99 % and a limit fails; % = dT_acc vs matched baseline)")
    print("\n== can any policy meet all limits with the strict FIFO (depth <= 8)? ==")
    for k in RATES_KSPS:
        ok = [(r["policy"], r["B"], r["depth"]) for r in rows if r["rate_ksps"] == k and r["pass_all"] and r["fifo_strict"]]
        print(f"  {k * S / 1000:6.3g} MB/s: " + (", ".join(f"{p} B={B} F={d}" for p, B, d in ok) if ok else "none"))


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    col = {"pass": "#34a853", "S": "#ea4335", "C": "#fbbc04", "D": "#4285f4", "multi": "#7e57c2"}
    fig, axes = plt.subplots(len(BURSTS), len(POLICIES), figsize=(13, 5.6), sharey=True)
    for i, B in enumerate(BURSTS):
        for j, p in enumerate(POLICIES):
            ax = axes[i][j]
            pts = [r for r in rows if r["B"] == B and r["policy"] == p]
            for r in pts:
                xi = RATES_KSPS.index(r["rate_ksps"])
                lvl = 0 if r["depth"] == r["theta"] else (1 if r["depth"] == min(8 * r["theta"], 512) else 2)
                f = r["fails"]
                c = col["pass"] if r["pass_all"] else (col[f] if len(f) == 1 else col["multi"])
                ax.add_patch(plt.Rectangle((xi - 0.45, lvl - 0.45), 0.9, 0.9, color=c, alpha=0.9,
                                           hatch="//" if r["overloaded"] else None))
                ax.text(xi, lvl, f"F={r['depth']}\n{'ok' if r['pass_all'] else f}", ha="center", va="center", fontsize=6.5)
            ax.set_xlim(-0.6, len(RATES_KSPS) - 0.4)
            ax.set_ylim(-0.6, 2.6)
            ax.set_xticks(range(len(RATES_KSPS)))
            ax.set_xticklabels([f"{k * S / 1000:g}" for k in RATES_KSPS], fontsize=7)
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(["F = θ", "F = 8θ", "F = 64θ"], fontsize=7, fontfamily="DejaVu Sans")
            if i == len(BURSTS) - 1:
                ax.set_xlabel("sensor rate (MB/s)", fontsize=8)
            ax.set_title(f"{p}, B = {B} B", fontsize=9)
    fig.legend(handles=[Patch(color=col["pass"], label="all limits met"), Patch(color=col["S"], label="S: sensor overflow"),
                        Patch(color=col["C"], label="C: control-task deadline"), Patch(color=col["D"], label="D: DNN deadline / >10 % delay"),
                        Patch(color=col["multi"], label="several limits")], loc="lower center", ncol=5, fontsize=8)
    fig.suptitle("Stage A coarse failure map (DS-CNN, 62 µs control task, streaming sensor; FIFO depth in samples)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    rs = runs()
    if a.list:
        for r in rs:
            print(r["name"])
        print(len(rs), "runs")
        return
    if not a.summary_only:
        subprocess.run(["make", "-s", "-C", os.path.join(PROJ, "sw"), "kws"], check=True)
        os.makedirs(OUT, exist_ok=True)
        with ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(lambda r: run(r, a.skip_existing), rs))
    rows, bases = summarize()
    report(rows, bases)
    if a.plot:
        plot(rows, os.path.join(RES, "figures", "fig6_stageA.png"))


if __name__ == "__main__":
    sys.exit(main())
