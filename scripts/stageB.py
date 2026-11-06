#!/usr/bin/env python3
"""
stageB.py - course project, stage B (plan v3.3, Sections 4.3-4.6, 5, 6).

Stage A showed that at coarse resolution the four policies behave alike, that
the CPU deadline is set by the DNN burst length, and that the sensor threshold
theta trades FIFO size against bus overhead. Stage B therefore:

  1. sweeps theta as an axis: powers of two from the C2 minimum (round-robin
     blocking) up to 128 samples, for sensor rates >= 3.2 MB/s (the C2
     minimum only for the two low rates, where overhead is negligible);
  2. finds the EXACT minimum FIFO depth F_min for every (policy, B, rate,
     theta) by integer bisection on [theta, 512]. This is valid because theta
     is fixed and no rule reads the depth: a run without overflow is identical
     at every larger depth (checked by test 9), so overflow-freedom is
     monotonic in F, and the CPU and DNN results of the passing run hold for
     every F >= F_min;
  3. adds DNN bursts of 256 B and policy variant 5 (budget + fragmentation,
     64-B fragments, window 1000 cycles, rho = 0.25 and 0.5, with B = 1024);
  4. checks the sufficient bound (sensor-first and round-robin): F_min <= F_suf
     wherever C2 holds, and reports how tight it is;
  5. marks points where bandwidth alone makes the delay margin unreachable:
     the sensor's raw data share s_raw = R*S/(4 f_clk) >= margin (an estimate,
     checked against the simulations: no configuration there may pass D).
     The estimate includes the per-burst overhead at the largest theta the
     FIFO limit allows: s_min = (R/theta_max)(10 + ceil(theta_max*S/4) + a)/f.

Delay is judged against the UNTHROTTLED no-sensor run at the same B (round-
robin; policies 1-4 have identical no-sensor runs): a policy's own throttling
(variant 5) is a protection cost and is counted. dT against the policy's own
no-sensor run is also reported (dT_own_pct).

    python3 scripts/stageB.py [--jobs 2] [--plot]       (results/stageB/, results/stageB_*.csv)
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
OUT = os.path.join(RES, "stageB")
F_CLK = 100e6
A = 1
S = 4
T_INF_US = 5000
T_INF_CYC = T_INF_US * 100
RATES_KSPS = (8, 80, 800, 8000, 25000)
FMAX_STRICT, FMAX_RELAXED, DEPTH_CAP = 8, 1280, 512
MARGINS = (0.05, 0.10, 0.25)
BASE = ["--accel-arch", "maeri", "--accel-ms", "256", "--accel-dnbw", "256", "--cpu-fetch-private"]
ELF = "bench_kws.elf"

# policy configurations: name -> (vp options, bursts, effective blocking burst)
BUDGET = lambda x: ["--bus-arbiter", "budget", "--bus-frag-bytes", "64", "--bus-window-cycles", "1000",
                    "--bus-budget-bytes", str(x)]
POLICIES = {
    "rr": (["--bus-arbiter", "rr"], (64, 256, 1024), None),
    "sensor-first": (["--bus-arbiter", "sensor-first"], (64, 256, 1024), None),
    "cpu-first": (["--bus-arbiter", "cpu-first"], (64, 256, 1024), None),
    "buffer": (["--bus-arbiter", "buffer"], (64, 256, 1024), None),
    "budget25": (BUDGET(1000), (1024,), 64),
    "budget50": (BUDGET(2000), (1024,), 64),
}


def bcyc(n):
    return 10 + math.ceil(n / 4)


def need(ksps, L):
    return math.ceil(ksps * 1000.0 * L / F_CLK)


def theta_min(ksps, Beff):
    beta = bcyc(Beff) + 1 + A
    for k in range(8):
        th = 1 << k
        if need(ksps, beta + A + bcyc(th * S)) <= th:
            return th
    return 128


def bound(ksps, Beff, th, rr):
    """(C2 holds, sufficient F) for sensor-first (rr=False) or round-robin (rr=True)"""
    beta = bcyc(Beff) + ((1 + A) if rr else 0)
    n = need(ksps, beta + A + bcyc(th * S))
    return n <= th, th + n


def thetas(ksps, Beff):
    t0 = theta_min(ksps, Beff)
    if ksps < 800:
        return [t0]
    out, t = [], t0
    while t <= 128:
        out.append(t)
        t *= 2
    return out


def combos():
    out = []
    for p, (_, bursts, beff) in POLICIES.items():
        for B in bursts:
            Be = beff or B
            for k in RATES_KSPS:
                for th in thetas(k, Be):
                    out.append(dict(policy=p, B=B, Beff=Be, ksps=k, theta=th))
    return out


def name(p, B, k=0, th=0, F=0):
    return f"base_{p}_{B}" if not k else f"{p}_{B}_{k}k_th{th}_d{F}"


def parse(nm):
    base = os.path.join(OUT, nm)
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
            d["win_cpu_busy"] = int(kv["cpu_busy"])
        elif f[0] == "RGBUS" and f[1] in ("cpu", "sensor", "dnn"):
            d[f"{f[1]}_busy"] = int(f[3])
    return d


def vp(p, B, k=0, th=0, F=0):
    nm = name(p, B, k, th, F)
    d = parse(nm)
    if d is not None:
        return d
    opts = BASE + ["--accel-dma-burst", str(B)] + POLICIES[p][0]
    if k:
        opts += ["--sensor-rate-ksps", str(k), "--sensor-theta", str(th), "--sensor-depth", str(F), "--sensor-bytes", str(S)]
    r = subprocess.run(["bash", os.path.join(PROJ, "scripts", "run_vp.sh"), ELF, f"stageB/{nm}.csv"] + opts,
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stdout, r.stderr, flush=True)
    return parse(nm)


def search(c):
    """exact F_min by bisection; returns (F_min or None, run dict at F_min or at the cap)"""
    p, B, k, th = c["policy"], c["B"], c["ksps"], c["theta"]
    rr = p not in ("sensor-first",)
    _, guess = bound(k, c["Beff"], th, rr)
    guess = max(th, min(guess, DEPTH_CAP))
    ok = lambda F: vp(p, B, k, th, F)["sen_ovf_win"] == 0
    lo, hi = th - 1, None  # lo fails (by convention, F >= theta), hi passes
    if ok(guess):
        hi = guess
    else:
        lo = guess
        if guess < DEPTH_CAP and ok(DEPTH_CAP):
            hi = DEPTH_CAP
    if hi is None:
        return None, vp(p, B, k, th, DEPTH_CAP)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi, vp(p, B, k, th, hi)


def test9(c, fmin):
    """depth monotonicity: the run at F_min and at the cap are identical except the depth"""
    p, B, k, th = c["policy"], c["B"], c["ksps"], c["theta"]
    a, b = vp(p, B, k, th, fmin), vp(p, B, k, th, DEPTH_CAP)
    # window results only: whole-run bus totals also include the final printing,
    # which differs because the program prints the depth itself
    keys = [x for x in a if x not in ("sen_depth", "cpu_busy", "sensor_busy", "dnn_busy")]
    return all(a[x] == b[x] for x in keys if x in b)


def analyse():
    rows = []
    bases = {}
    for p, (_, bursts, _) in POLICIES.items():
        for B in bursts:
            bases[(p, B)] = vp(p, B)
    for c in combos():
        fmin, d = search(c)
        b = bases[("rr", c["B"])]           # unthrottled no-sensor run, same B
        b_own = bases[(c["policy"], c["B"])]
        k, th, Be = c["ksps"], c["theta"], c["Beff"]
        R = k * 1000.0
        s_est = (R / th) * (bcyc(th * S) + A) / F_CLK
        s_raw = R * S / 4 / F_CLK
        c2_sf, f_sf = bound(k, Be, th, False)
        c2_rr, f_rr = bound(k, Be, th, True)
        dT = (d["t_acc"] - b["t_acc"]) / b["t_acc"]
        ok_cpu = d["rt_miss"] + d["rt_skip"] == 0
        ok_dnn = d["t_acc"] <= T_INF_CYC
        row = dict(policy=c["policy"], B=c["B"], rate_ksps=k, rate_MBps=k * S / 1000.0, theta=th,
                   F_min=fmin if fmin else f">{DEPTH_CAP}", F_min_num=fmin or 10 ** 6,
                   F_suf_sf=f_sf if c2_sf else "", F_suf_rr=f_rr if c2_rr else "",
                   peak=d["sen_peak_win"], sen_maxwait=d["sen_maxwait"],
                   rt_miss=d["rt_miss"] + d["rt_skip"], rt_max=d["rt_max"], ok_cpu=int(ok_cpu),
                   t_acc=d["t_acc"], base_t_acc=b["t_acc"], dT_pct=round(100 * dT, 2),
                   dT_own_pct=round(100 * (d["t_acc"] - b_own["t_acc"]) / b_own["t_acc"], 2),
                   gap_fastest_pct=0.0, ok_dnn=int(ok_dnn), occupancy=round(d.get("occ", 0), 3),
                   s_est=round(s_est, 4), s_raw=round(s_raw, 4))
        for m in MARGINS:
            okd = ok_dnn and dT <= m
            row[f"pass_strict_{int(m * 100)}"] = int(bool(fmin) and fmin <= FMAX_STRICT and ok_cpu and okd)
            row[f"pass_relaxed_{int(m * 100)}"] = int(bool(fmin) and fmin <= FMAX_RELAXED and ok_cpu and okd)
        rows.append(row)
    fastest = min(b["t_acc"] for b in bases.values())
    for r in rows:
        r["gap_fastest_pct"] = round(100.0 * (r["t_acc"] - fastest) / fastest, 2)
    with open(os.path.join(RES, "stageB_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RES, "stageB_baselines.csv"), "w", newline="") as f:
        bl = [dict(policy=p, B=B, t_acc=b["t_acc"], rt_late_lost=b["rt_miss"] + b["rt_skip"], rt_max=b["rt_max"])
              for (p, B), b in bases.items()]
        w = csv.DictWriter(f, fieldnames=list(bl[0].keys()))
        w.writeheader()
        w.writerows(bl)
    return rows, bases


def s_min(ksps, flimit):
    """smallest sensor bus share reachable with theta <= min(F limit, 128) (estimate)"""
    th = 1
    while th * 2 <= min(flimit, 128):
        th *= 2
    return (ksps * 1000.0 / th) * (bcyc(th * S) + A) / F_CLK


def report(rows, bases):
    print(f"\nstage B: {len(rows)} (policy, B, rate, theta) points, exact F_min by bisection "
          f"({len([x for x in os.listdir(OUT) if x.endswith('.csv')])} runs)")
    print("\n== baselines (no sensor) ==")
    for (p, B), b in sorted(bases.items(), key=lambda x: (x[0][1], x[0][0])):
        print(f"  {p:12s} B={B:5d} T_acc {b['t_acc']:7d} cyc, control task late/lost {b['rt_miss'] + b['rt_skip']:2d}, "
              f"worst response {b['rt_max']} cyc")
    print("\n== bound check: F_min <= F_suf where C2 holds ==")
    for pol, col in (("sensor-first", "F_suf_sf"), ("rr", "F_suf_rr")):
        app = [r for r in rows if r["policy"] == pol and r[col] != ""]
        bad = [r for r in app if r["F_min_num"] > r[col]]
        tight = [r["F_min_num"] / r[col] for r in app if r["F_min_num"] <= r[col]]
        print(f"  {pol:12s}: {len(app)} points, {len(bad)} violations  [{'PASS' if not bad else 'FAIL'}]; "
              f"F_min/F_suf from {min(tight):.2f} to {max(tight):.2f}" if tight else "")
    print("\n== test 9: run at F_min identical to run at F = 512 (depth monotonicity) ==")
    sample = [c for c in combos() if c["ksps"] >= 8000][:: 7][:6]
    res = []
    for c in sample:
        r = [x for x in rows if x["policy"] == c["policy"] and x["B"] == c["B"] and x["rate_ksps"] == c["ksps"]
             and x["theta"] == c["theta"]][0]
        if isinstance(r["F_min"], int):
            res.append(test9(c, r["F_min"]))
    print(f"  {sum(res)}/{len(res)} identical  [{'PASS' if all(res) else 'FAIL'}]")

    print("\n== best configuration per rate and policy (all limits; delay margin 10 %) ==")
    print("   strict = F <= 8 samples, relaxed = F <= 1280; shown: B, theta, F_min, dT, gap to fastest setting")
    for k in RATES_KSPS:
        s_raw = k * 1000.0 * S / 4 / F_CLK
        tag = "  [bandwidth: raw sensor share %.0f %% >= 10 %% margin]" % (100 * s_raw) if s_raw >= 0.10 else ""
        print(f"\n  {k * S / 1000:g} MB/s{tag}")
        for p in POLICIES:
            for lim in ("strict", "relaxed"):
                ok = [r for r in rows if r["policy"] == p and r["rate_ksps"] == k and r[f"pass_{lim}_10"]]
                if ok:
                    best = min(ok, key=lambda r: (r["gap_fastest_pct"], r["F_min_num"]))
                    s = (f"B={best['B']} th={best['theta']} F={best['F_min']} dT={best['dT_pct']:+.1f}% "
                         f"gap={best['gap_fastest_pct']:+.0f}%")
                else:
                    near = [r for r in rows if r["policy"] == p and r["rate_ksps"] == k]
                    why = set()
                    for r in near:
                        if not r["ok_cpu"]: why.add("C")
                        if r["dT_pct"] > 10 or not r["ok_dnn"]: why.add("D")
                        if lim == "strict" and r["F_min_num"] > FMAX_STRICT: why.add("F")
                    s = "none (" + "".join(sorted(why)) + ")"
                print(f"    {p:12s} {lim:7s}: {s}")
    print("\n== research-gap test (Section 6): feasible points where NO policy meets all limits ==")
    for lim, fl in (("strict", FMAX_STRICT), ("relaxed", FMAX_RELAXED)):
        for m in MARGINS:
            bad, excl = [], []
            for k in RATES_KSPS:
                if s_min(k, fl) >= m:
                    excl.append(f"{k * S / 1000:g}")  # bus overhead alone exceeds the margin
                    continue
                if not any(r[f"pass_{lim}_{int(m * 100)}"] for r in rows if r["rate_ksps"] == k):
                    bad.append(f"{k * S / 1000:g} MB/s")
            print(f"  FIFO {lim:7s}, margin {int(m * 100):2d} %: gap candidates: " + (", ".join(bad) if bad else "none")
                  + (f"   (excluded by bus overhead: {', '.join(excl)} MB/s)" if excl else ""))
    print("\n== check of the overhead estimate: no configuration within the FIFO limit may pass D where s_min >= margin ==")
    for lim, fl in (("strict", FMAX_STRICT), ("relaxed", FMAX_RELAXED)):
        for m in MARGINS:
            viol = [r for r in rows if s_min(r["rate_ksps"], fl) >= m and r["F_min_num"] <= fl
                    and r["ok_dnn"] and r["dT_pct"] <= 100 * m]
            print(f"  {lim:7s} margin {int(m * 100):2d} %: {len(viol)} violations  [{'PASS' if not viol else 'CHECK'}]"
                  + "".join(f"\n     {v['policy']} B={v['B']} {v['rate_MBps']} MB/s th={v['theta']} dT={v['dT_pct']}" for v in viol))


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"rr": "#1a73e8", "sensor-first": "#d93025", "cpu-first": "#f9ab00", "buffer": "#188038",
              "budget25": "#9334e6", "budget50": "#12b5cb"}
    os.makedirs(os.path.join(RES, "figures"), exist_ok=True)
    # figure 7: minimum FIFO vs rate (theta = C2 minimum), per policy, with bounds
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, B in zip(axes, (64, 256, 1024)):
        for p in POLICIES:
            pts = []
            for k in RATES_KSPS:
                rr_ = [r for r in rows if r["policy"] == p and r["B"] == B and r["rate_ksps"] == k]
                if rr_:
                    r = min(rr_, key=lambda x: x["theta"])
                    pts.append((k * S / 1000, r["F_min_num"] if r["F_min_num"] < 10 ** 6 else None))
            if pts:
                ax.plot([x for x, y in pts if y], [y for x, y in pts if y], "o-", color=colors[p], label=p, ms=4, lw=1.2)
        sf = [(r["rate_MBps"], r["F_suf_sf"]) for r in rows if r["policy"] == "sensor-first" and r["B"] == B
              and r["F_suf_sf"] != "" and r["theta"] == min(x["theta"] for x in rows if x["policy"] == "sensor-first"
                                                              and x["B"] == B and x["rate_ksps"] == r["rate_ksps"])]
        if sf:
            ax.plot(*zip(*sorted(sf)), "k--", lw=1, label="bound (sensor-first)")
        ax.axhline(FMAX_STRICT, color="#5f6368", lw=0.8, ls=":")
        ax.text(0.035, FMAX_STRICT * 1.15, "strict limit (8)", fontsize=7, color="#5f6368")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"DNN burst B = {B} B (theta = C2 minimum)", fontsize=9)
        ax.set_xlabel("sensor rate (MB/s)"); ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("minimum FIFO depth (samples)")
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "figures", "fig7_stageB_minfifo.png"), dpi=150)
    # figure 8: trade-off, dT vs F_min for every configuration at 3.2 and 32 MB/s
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, k in zip(axes, (800, 8000)):
        for p in POLICIES:
            for r in rows:
                if r["policy"] != p or r["rate_ksps"] != k or r["F_min_num"] >= 10 ** 6:
                    continue
                ax.scatter(r["F_min_num"], r["dT_pct"], color=colors[p], s=18 + r["B"] / 30,
                           marker="o" if r["ok_cpu"] else "x", label=p)
        ax.axvline(FMAX_STRICT, color="#5f6368", ls=":", lw=0.8)
        ax.axhline(10, color="#5f6368", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("minimum FIFO depth F_min (samples)")
        ax.set_ylabel("dT_acc vs matched baseline (%)")
        ax.set_title(f"{k * S / 1000:g} MB/s: every (policy, B, theta); x = CPU deadline missed; size ~ B", fontsize=9)
        ax.grid(alpha=0.3)
        h, l = ax.get_legend_handles_labels()
        u = dict(zip(l, h))
        ax.legend(u.values(), u.keys(), fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "figures", "fig8_stageB_tradeoff.png"), dpi=150)
    print("wrote results/figures/fig7_stageB_minfifo.png, fig8_stageB_tradeoff.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    cs = combos()
    if a.list:
        print(len(cs), "combos")
        return
    subprocess.run(["make", "-s", "-C", os.path.join(PROJ, "sw"), "kws"], check=True)
    os.makedirs(OUT, exist_ok=True)
    jobs = [("base", p, B) for p, (_, bs, _) in POLICIES.items() for B in bs] + [("combo", c) for c in cs]

    def work(j):
        if j[0] == "base":
            vp(j[1], j[2])
        else:
            search(j[1])
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(work, jobs))
    rows, bases = analyse()
    report(rows, bases)
    if a.plot:
        plot(rows)


if __name__ == "__main__":
    sys.exit(main())
