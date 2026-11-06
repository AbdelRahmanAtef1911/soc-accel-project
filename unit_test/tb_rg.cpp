/*
 * tb_rg.cpp - course project, week 1: unit tests of the request/grant bus
 * (accel/rg_bus.h) with stub masters, on the course SystemC 2.3.3.
 * Test numbers follow the project plan, Section 5.2:
 *
 *   1 policy order        every variant grants three simultaneous requests
 *                         in its documented order
 *   2 sensor-first bound  sensor wait <= beta + a under sensor-first
 *                         (and <= beta_RR + a under round-robin)
 *   3 round-robin         equal grant counts (+-1) for equal-size requests;
 *                         with unequal tenures, busy share = len_i / sum(len_j + a)
 *   4 starvation          fixed priority starves lower masters under saturation
 *   5 budget              DNN bytes per aligned window <= X; saturated windows
 *                         use the budget (>= X - fragment)
 *   6 handover            a cycles per ownership change, none between grants
 *                         to the same master
 *   7 accounting          busy <= elapsed; tenures never overlap
 *
 * Every test runs on its own RgBus instance; all of them run in one
 * simulation of RUN_US microseconds.
 *
 * usage: ./tb_rg
 */
#include <systemc>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "rg_bus.h"

using namespace sc_core;

static const sc_time CLK(10, SC_NS);
static const uint64_t CYC = CLK.value();
static const unsigned A = 1;          /* handover cycles */
static const double RUN_US = 2000.0;  /* simulated time for the saturating tests */
static int failures = 0;

static void check(bool ok, const std::string &what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) failures++;
}

static uint64_t burst_cyc(uint32_t bytes) { return 10 + (bytes + 3) / 4; }

struct Bus {
    std::string name;
    std::unique_ptr<RgBus> bus;
    Bus(const std::string &n, RgConfig c) : name(n) {
        c.clk = CLK;
        c.handover = A;
        c.keep_trace = true;
        bus.reset(new RgBus(n.c_str(), c));
    }
};

static RgConfig policy(RgPolicy p) { RgConfig c; c.policy = p; return c; }

/* masters */
static void once(RgBus &b, RgMaster m, uint32_t bytes, uint64_t cyc, uint64_t at_cyc = 0) {
    sc_spawn([&b, m, bytes, cyc, at_cyc]() {
        if (at_cyc) wait(at_cyc * CLK);
        b.acquire(m, bytes, cyc * CYC);
        wait(cyc * CLK);
    });
}
static void saturate(RgBus &b, RgMaster m, uint32_t bytes, uint64_t cyc, uint64_t count = 0) {
    sc_spawn([&b, m, bytes, cyc, count]() {
        for (uint64_t i = 0; count == 0 || i < count; i++) {
            b.acquire(m, bytes, cyc * CYC);
            wait(cyc * CLK);
        }
    });
}
static void periodic(RgBus &b, RgMaster m, uint32_t bytes, uint64_t cyc, uint64_t period_cyc) {
    sc_spawn([&b, m, bytes, cyc, period_cyc]() {
        while (true) {
            sc_time t0 = sc_time_stamp();
            b.acquire(m, bytes, cyc * CYC);
            wait(cyc * CLK);
            sc_time next = t0 + period_cyc * CLK;
            if (sc_time_stamp() < next) wait(next - sc_time_stamp());
        }
    });
}
/* request at pseudo-random gaps (deterministic LCG), lo..hi cycles apart */
static void random_gaps(RgBus &b, RgMaster m, uint32_t bytes, uint64_t cyc, unsigned lo, unsigned hi, uint32_t seed) {
    sc_spawn([&b, m, bytes, cyc, lo, hi, seed]() {
        uint32_t x = seed;
        while (true) {
            x = x * 1664525u + 1013904223u;
            wait((lo + (x >> 8) % (hi - lo + 1)) * CLK);
            b.acquire(m, bytes, cyc * CYC);
            wait(cyc * CLK);
        }
    });
}

static std::string order_of(const RgBus &b) {
    std::string s;
    for (auto &g : b.trace) s += std::string(s.empty() ? "" : ",") + rg_master_name(g.master);
    return s;
}

int sc_main(int, char **) {
    std::vector<Bus *> all;
    auto make = [&](const std::string &n, RgConfig c) { all.push_back(new Bus(n, c)); return all.back(); };

    /* ---- test 1: policy order (three requests at t = 0) ---- */
    struct OrderCase { const char *name; RgConfig cfg; const char *expect; };
    unsigned level_hi = 100, level_lo = 0;
    RgConfig buf_hi = policy(RG_BUFFER);
    buf_hi.urgent_level = 50;
    buf_hi.sensor_level = [&level_hi]() { return level_hi; };
    RgConfig buf_lo = buf_hi;
    buf_lo.sensor_level = [&level_lo]() { return level_lo; };
    RgConfig bud = policy(RG_BUDGET);
    bud.budget_bytes = 1 << 20;
    std::vector<OrderCase> oc = {
        {"t1_rr", policy(RG_RR), "cpu,sensor,dnn"},
        {"t1_sensor_first", policy(RG_SENSOR_FIRST), "sensor,cpu,dnn"},
        {"t1_cpu_first", policy(RG_CPU_FIRST), "cpu,sensor,dnn"},
        {"t1_buffer_urgent", buf_hi, "sensor,dnn,cpu"},
        {"t1_buffer_normal", buf_lo, "cpu,sensor,dnn"},
        {"t1_budget_ample", bud, "cpu,sensor,dnn"},
    };
    std::vector<Bus *> t1;
    for (auto &c : oc) {
        Bus *b = make(c.name, c.cfg);
        t1.push_back(b);
        once(*b->bus, RG_DNN, 64, 26);
        once(*b->bus, RG_SENSOR, 64, 26);
        once(*b->bus, RG_CPU, 4, 1);
    }

    /* ---- test 2: sensor waiting bound ---- */
    const uint64_t dnn_len = burst_cyc(1024);  /* 266 */
    const uint64_t sen_len = burst_cyc(128);   /* theta = 32 samples x 4 B -> 42 */
    Bus *t2sf = make("t2_sensor_first", policy(RG_SENSOR_FIRST));
    Bus *t2rr = make("t2_rr", policy(RG_RR));
    for (Bus *b : {t2sf, t2rr}) {
        saturate(*b->bus, RG_DNN, 1024, dnn_len);
        periodic(*b->bus, RG_CPU, 4, 1, 7);
        random_gaps(*b->bus, RG_SENSOR, 128, sen_len, 50, 900, 12345);
    }

    /* ---- test 3: round-robin ---- */
    Bus *t3eq = make("t3_rr_equal", policy(RG_RR));
    for (RgMaster m : {RG_CPU, RG_SENSOR, RG_DNN}) saturate(*t3eq->bus, m, 64, 26);
    Bus *t3un = make("t3_rr_unequal", policy(RG_RR));
    saturate(*t3un->bus, RG_CPU, 4, 1);
    saturate(*t3un->bus, RG_SENSOR, 128, sen_len);
    saturate(*t3un->bus, RG_DNN, 1024, dnn_len);

    /* ---- test 4: starvation under fixed priority ---- */
    Bus *t4cf = make("t4_cpu_first", policy(RG_CPU_FIRST));
    saturate(*t4cf->bus, RG_CPU, 4, 1);
    saturate(*t4cf->bus, RG_SENSOR, 64, 26);
    saturate(*t4cf->bus, RG_DNN, 64, 26);
    Bus *t4sf = make("t4_sensor_first", policy(RG_SENSOR_FIRST));
    saturate(*t4sf->bus, RG_SENSOR, 64, 26);
    saturate(*t4sf->bus, RG_CPU, 4, 1);
    saturate(*t4sf->bus, RG_DNN, 64, 26);

    /* ---- test 5: budget per aligned window ---- */
    RgConfig b5 = policy(RG_BUDGET);
    b5.frag_bytes = 64;
    b5.window_cycles = 1000;
    b5.budget_bytes = 1000;  /* rho = 0.25 of 4 B x 1000 cycles */
    Bus *t5 = make("t5_budget", b5);
    saturate(*t5->bus, RG_DNN, 64, burst_cyc(64));
    periodic(*t5->bus, RG_CPU, 4, 1, 50);

    /* ---- test 6: handover ---- */
    Bus *t6one = make("t6_single_master", policy(RG_RR));
    const uint64_t n6 = 100;
    saturate(*t6one->bus, RG_DNN, 64, 26, n6);
    Bus *t6alt = make("t6_alternating", policy(RG_RR));
    saturate(*t6alt->bus, RG_CPU, 4, 1);
    saturate(*t6alt->bus, RG_DNN, 64, 26);

    sc_start(sc_time(RUN_US, SC_US));
    const uint64_t elapsed = sc_time_stamp().value();
    printf("request/grant bus unit tests (handover a = %u cycle, %.0f us simulated)\n", A, RUN_US);

    printf("\nTest 1: policy order, three requests at t = 0\n");
    for (size_t i = 0; i < oc.size(); i++) {
        std::string got = order_of(*t1[i]->bus);
        check(got == oc[i].expect, std::string(oc[i].name) + ": " + got + " (expected " + oc[i].expect + ")");
    }

    printf("\nTest 2: sensor waiting time, DNN bursts of 1024 B (%llu cycles), sensor bursts %llu cycles\n",
           (unsigned long long)dnn_len, (unsigned long long)sen_len);
    {
        const RgBus &sf = *t2sf->bus, &rr = *t2rr->bus;
        uint64_t sf_max = sf.cycles(sf.stats[RG_SENSOR].max_wait), rr_max = rr.cycles(rr.stats[RG_SENSOR].max_wait);
        uint64_t sf_bound = dnn_len + A;                 /* beta + a, beta = max lower tenure */
        uint64_t rr_bound = (dnn_len + A) + (1 + A) + A;  /* one tenure of each other master + handovers */
        printf("  sensor-first: %llu sensor grants, max wait %llu cycles (bound %llu)\n",
               (unsigned long long)sf.stats[RG_SENSOR].grants, (unsigned long long)sf_max, (unsigned long long)sf_bound);
        printf("  round-robin : %llu sensor grants, max wait %llu cycles (bound %llu)\n",
               (unsigned long long)rr.stats[RG_SENSOR].grants, (unsigned long long)rr_max, (unsigned long long)rr_bound);
        check(sf.stats[RG_SENSOR].grants > 200 && sf_max <= sf_bound, "sensor-first: wait <= beta + a");
        check(sf_max + 30 >= sf_bound, "sensor-first: the bound is reached (within 30 cycles), so the test is not trivial");
        check(rr_max <= rr_bound, "round-robin: wait <= beta_RR + a");
    }

    printf("\nTest 3: round-robin\n");
    {
        const RgBus &e = *t3eq->bus;
        uint64_t g[3] = {e.stats[0].grants, e.stats[1].grants, e.stats[2].grants};
        uint64_t mx = std::max({g[0], g[1], g[2]}), mn = std::min({g[0], g[1], g[2]});
        printf("  equal 64 B requests: grants cpu %llu, sensor %llu, dnn %llu\n", (unsigned long long)g[0],
               (unsigned long long)g[1], (unsigned long long)g[2]);
        check(mx - mn <= 1, "equal-size requests: grant counts differ by at most one");
        const RgBus &u = *t3un->bus;
        const uint64_t len[3] = {1, sen_len, dnn_len};
        const double round = (double)(len[0] + len[1] + len[2] + 3 * A);
        bool ok = true;
        for (int m = 0; m < 3; m++) {
            double share = (double)u.stats[m].busy / elapsed, expect = len[m] / round;
            printf("  unequal: %-6s grants %6llu, busy share %.4f, expected %.4f\n", rg_master_name(m),
                   (unsigned long long)u.stats[m].grants, share, expect);
            ok = ok && std::abs(share - expect) <= round * CYC / (double)elapsed + 1e-9;
        }
        uint64_t gu[3] = {u.stats[0].grants, u.stats[1].grants, u.stats[2].grants};
        check(std::max({gu[0], gu[1], gu[2]}) - std::min({gu[0], gu[1], gu[2]}) <= 1,
              "unequal tenures: still equal grant counts (RR equalises grants)");
        check(ok, "unequal tenures: busy share = len_i / sum(len_j + a), within one round");
    }

    printf("\nTest 4: starvation under fixed priority (all masters saturating)\n");
    {
        const RgBus &c = *t4cf->bus, &s = *t4sf->bus;
        printf("  cpu-first   : cpu %llu, sensor %llu, dnn %llu grants\n", (unsigned long long)c.stats[RG_CPU].grants,
               (unsigned long long)c.stats[RG_SENSOR].grants, (unsigned long long)c.stats[RG_DNN].grants);
        printf("  sensor-first: sensor %llu, cpu %llu, dnn %llu grants\n", (unsigned long long)s.stats[RG_SENSOR].grants,
               (unsigned long long)s.stats[RG_CPU].grants, (unsigned long long)s.stats[RG_DNN].grants);
        check(c.stats[RG_SENSOR].grants == 0 && c.stats[RG_DNN].grants == 0, "cpu-first: sensor and DNN starve");
        check(s.stats[RG_CPU].grants == 0 && s.stats[RG_DNN].grants == 0, "sensor-first: CPU and DNN starve");
    }

    printf("\nTest 5: budget, X = %llu B per aligned window of %llu cycles, fragments of %u B\n",
           (unsigned long long)b5.budget_bytes, (unsigned long long)b5.window_cycles, b5.frag_bytes);
    {
        const RgBus &b = *t5->bus;
        const uint64_t W = b5.window_cycles * CYC;
        std::vector<uint64_t> per_win(elapsed / W + 1, 0);
        for (auto &g : b.trace) if (g.master == RG_DNN) per_win[g.start / W] += g.bytes;
        uint64_t mx = 0, mn = ~0ull;
        for (size_t w = 0; w + 1 < per_win.size(); w++) { mx = std::max(mx, per_win[w]); mn = std::min(mn, per_win[w]); }
        /* sliding window, for information only */
        uint64_t slide = 0;
        for (size_t i = 0; i < b.trace.size(); i++) {
            if (b.trace[i].master != RG_DNN) continue;
            uint64_t sum = 0;
            for (size_t j = i; j < b.trace.size() && b.trace[j].start < b.trace[i].start + W; j++)
                if (b.trace[j].master == RG_DNN) sum += b.trace[j].bytes;
            slide = std::max(slide, sum);
        }
        printf("  DNN bytes per aligned window: min %llu, max %llu; max over any sliding window %llu (may reach 2X)\n",
               (unsigned long long)mn, (unsigned long long)mx, (unsigned long long)slide);
        check(mx <= b5.budget_bytes, "every aligned window: DNN bytes <= X");
        check(mn + b5.frag_bytes > b5.budget_bytes, "saturated: every full window uses the budget (> X - fragment)");
        check(b.stats[RG_CPU].grants > 0, "CPU is still served while the DNN is throttled");
    }

    printf("\nTest 6: handover only on ownership change\n");
    {
        const RgBus &o = *t6one->bus;
        uint64_t end = o.trace.back().start + o.trace.back().len;
        printf("  single master: %llu bursts end at %llu cycles (%llu x 26), %llu handovers\n",
               (unsigned long long)o.trace.size(), (unsigned long long)o.cycles(end), (unsigned long long)n6,
               (unsigned long long)o.handovers);
        check(o.trace.size() == n6 && end == n6 * 26 * CYC && o.handovers == 0,
              "same master back to back: no handover, total = sum of bursts");
        const RgBus &t = *t6alt->bus;
        uint64_t changes = 0;
        bool gaps_ok = true;
        for (size_t i = 1; i < t.trace.size(); i++) {
            bool change = t.trace[i].master != t.trace[i - 1].master;
            changes += change;
            uint64_t gap = t.trace[i].start - (t.trace[i - 1].start + t.trace[i - 1].len);
            gaps_ok = gaps_ok && gap == (change ? A * CYC : 0) && t.trace[i].handover == change;
        }
        printf("  alternating CPU/DNN: %llu tenures, %llu ownership changes, %llu handovers\n",
               (unsigned long long)t.trace.size(), (unsigned long long)changes, (unsigned long long)t.handovers);
        check(changes == t.handovers && changes > 100, "handovers == ownership changes");
        check(gaps_ok, "gap between tenures = a on an ownership change, 0 otherwise");
    }

    printf("\nTest 7: accounting on every bus\n");
    {
        /* busy time inside [0, elapsed]: a tenure granted near the end may
         * run past it, so clip each tenure (the stats count it at grant) */
        bool ok = true, overlap = false;
        for (Bus *b : all) {
            const RgBus &r = *b->bus;
            uint64_t busy[RG_MASTERS] = {0, 0, 0}, total = 0, from_trace = 0;
            for (size_t i = 0; i < r.trace.size(); i++) {
                const RgGrant &g = r.trace[i];
                uint64_t e = std::min(g.start + g.len, elapsed);
                if (e > g.start) { busy[g.master] += e - g.start; total += e - g.start; }
                from_trace += g.len;
                if (i && g.start < r.trace[i - 1].start + r.trace[i - 1].len) overlap = true;
            }
            ok = ok && total <= elapsed && from_trace == r.total_busy();
            for (uint64_t x : busy) ok = ok && x <= elapsed;
        }
        check(ok, "busy (clipped to the run) <= elapsed for every master and bus; stats == trace");
        check(!overlap, "tenures never overlap");
    }

    printf("\n%s (%d failure%s)\n", failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED", failures,
           failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
