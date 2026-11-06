/*
 * rg_bus.h - course-project bus model: one shared bus with a request/grant
 * arbiter and three masters (CPU, sensor DMA, DNN accelerator DMA).
 *
 * It replaces the step-5 occupancy timeline (shared_bus.h, kept unchanged so
 * the step-5 results stay reproducible) with an arbiter that only grants the
 * bus when it is free:
 *
 *   - every master has at most one outstanding request (Req below);
 *   - a request names its tenure length (bus time it will hold the bus);
 *   - when the bus becomes free, the policy picks one pending request;
 *   - the granted tenure starts `handover` cycles later if ownership changes
 *     (granted master != master of the previous tenure), otherwise at once;
 *   - a tenure cannot be interrupted (no split transactions).
 *
 * Requests made in the same simulated time step are collected for one delta
 * cycle before the arbiter decides, so simultaneous requests see the policy,
 * not the SystemC process order.
 *
 * Fast path (optional): a master that declares itself quiet (set_quiet)
 * promises not to request until it declares otherwise. When every other
 * master is quiet and nothing is pending, a request can only be granted to
 * its requester, at its own time (plus the handover if ownership changes), so
 * grant_if_alone() books it directly, without synchronising the requester.
 * The arbitration result is the same, but the CPU then synchronises less
 * often, which can move the instruction at which it takes an interrupt by a
 * few cycles. So the fast path is off by default and used only for runs
 * without interrupts (CPU-only checks), where results are identical.
 *
 * Policy variants (project plan, Section 4.4):
 *   1 RG_RR            round-robin among requesting masters
 *   2 RG_SENSOR_FIRST  fixed priority sensor > CPU > DNN
 *   3 RG_CPU_FIRST     fixed priority CPU > sensor > DNN
 *   4 RG_BUFFER        sensor gets top priority when its FIFO holds at least
 *                      urgent_level samples, otherwise round-robin
 *                      (simplified, inspired by Roh et al. 2009)
 *   5 RG_BUDGET        DNN transfers split into frag_bytes fragments; the DNN
 *                      may be granted at most budget_bytes per aligned window
 *                      of window_cycles; a fragment is granted only if it fits
 *                      in the remaining budget of the window its tenure starts
 *                      in; eligible requests are served round-robin
 *                      (simplified, inspired by AXI-REALM / Carfield)
 *
 * Times are SystemC time values (sc_time::value(), picoseconds by default).
 */
#ifndef RG_BUS_H
#define RG_BUS_H

#include <systemc>
#include <tlm>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/simple_target_socket.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

enum RgMaster { RG_CPU = 0, RG_SENSOR = 1, RG_DNN = 2, RG_MASTERS = 3 };
enum RgPolicy { RG_RR = 1, RG_SENSOR_FIRST = 2, RG_CPU_FIRST = 3, RG_BUFFER = 4, RG_BUDGET = 5 };

inline const char *rg_master_name(int m) {
    static const char *n[] = {"cpu", "sensor", "dnn"};
    return (m >= 0 && m < RG_MASTERS) ? n[m] : "?";
}
inline const char *rg_policy_name(RgPolicy p) {
    switch (p) {
        case RG_RR: return "rr";
        case RG_SENSOR_FIRST: return "sensor-first";
        case RG_CPU_FIRST: return "cpu-first";
        case RG_BUFFER: return "buffer";
        case RG_BUDGET: return "budget";
    }
    return "?";
}
/* parse "rr", "sensor-first", ... or "1".."5"; returns 0 if unknown */
inline int rg_policy_parse(const std::string &s) {
    for (int p = RG_RR; p <= RG_BUDGET; p++)
        if (s == rg_policy_name((RgPolicy)p) || s == std::to_string(p)) return p;
    return 0;
}

struct RgConfig {
    RgPolicy policy = RG_RR;
    sc_core::sc_time clk = sc_core::sc_time(10, sc_core::SC_NS);
    unsigned handover = 1;                 /* cycles, charged on ownership change only */
    /* variant 4 */
    std::function<unsigned()> sensor_level;  /* current FIFO occupancy (samples) */
    unsigned urgent_level = 0;               /* 0 = never urgent */
    /* variant 5 */
    unsigned frag_bytes = 64;
    uint64_t window_cycles = 1000;
    uint64_t budget_bytes = 2000;
    bool keep_trace = false;                 /* record every grant (tests) */
};

struct RgStats {
    uint64_t grants = 0;    /* tenures granted                    */
    uint64_t busy = 0;      /* time holding the bus               */
    uint64_t wait = 0;      /* request -> tenure start (incl. handover) */
    uint64_t max_wait = 0;  /* longest single wait (resettable)   */
    uint64_t bytes = 0;     /* bytes moved                        */
};

struct RgGrant {
    uint64_t t_req, start, len;
    int master;
    uint32_t bytes;
    bool handover;
};

class RgBus : public sc_core::sc_module {
   public:
    RgConfig cfg;
    RgStats stats[RG_MASTERS];
    uint64_t handovers = 0;
    std::vector<RgGrant> trace;

#if SC_VERSION_MAJOR < 3
    SC_HAS_PROCESS(RgBus);
#endif
    RgBus(sc_core::sc_module_name n, const RgConfig &c) : sc_module(n), cfg(c) {
        if (cfg.policy == RG_BUDGET && cfg.budget_bytes < cfg.frag_bytes) cfg.budget_bytes = cfg.frag_bytes;
        SC_THREAD(arbitrate);
    }

    /* Request the bus for a tenure of `len` (time value) moving `bytes`.
     * Blocks the calling SC_THREAD until the tenure starts and returns the
     * start time (== sc_time_stamp() on return). The bus is then busy until
     * start + len; the caller decides whether it waits for that itself. */
    uint64_t acquire(RgMaster m, uint32_t bytes, uint64_t len) {
        Req &r = req_[m];
        if (r.pending) SC_REPORT_FATAL("RgBus", "second outstanding request from one master");
        r.pending = true;
        r.t_req = now();
        r.len = std::max<uint64_t>(len, 1);
        r.bytes = bytes;
        req_ev_.notify(sc_core::SC_ZERO_TIME);
        sc_core::wait(r.granted);
        return r.start;
    }

    /* A master that will not request until further notice (e.g. the DNN DMA
     * between jobs, or a master that is not present) declares itself quiet. */
    void set_quiet(RgMaster m, bool q) { quiet_[m] = q; }

    /* Fast path: grant master m a tenure at its time t (>= now) if no other
     * master can compete for it; returns false if the slow path is needed. */
    bool grant_if_alone(RgMaster m, uint64_t t, uint32_t bytes, uint64_t len, uint64_t &start) {
        if (any_pending() || t < busy_until_) return false;
        for (int o = 0; o < RG_MASTERS; o++)
            if (o != m && !quiet_[o]) return false;
        if (!budget_ok(m, start_for(m, t), bytes)) return false;
        start = start_for(m, t);
        book(m, t, start, std::max<uint64_t>(len, 1), bytes);
        fast_grants++;
        return true;
    }
    uint64_t fast_grants = 0;

    /* Lengthen the tenure that master m was just granted (called by the owner
     * at the start of its tenure, before anyone else can be granted). */
    void extend(RgMaster m, uint64_t extra) {
        if (!extra) return;
        busy_until_ += extra;
        stats[m].busy += extra;
        if (cfg.keep_trace && !trace.empty() && trace.back().master == m) trace.back().len += extra;
    }

    /* DNN transfer piece: the burst size, or the fragment size under variant 5 */
    uint32_t dnn_piece(uint32_t burst) const {
        return cfg.policy == RG_BUDGET ? std::min<uint32_t>(burst, cfg.frag_bytes) : burst;
    }

    uint64_t busy_until() const { return busy_until_; }
    uint64_t cycles(uint64_t t) const { return t / cfg.clk.value(); }
    uint64_t total_busy() const {
        uint64_t b = 0;
        for (auto &s : stats) b += s.busy;
        return b;
    }

   private:
    struct Req {
        bool pending = false;
        uint64_t t_req = 0, len = 0, start = 0;
        uint32_t bytes = 0;
        sc_core::sc_event granted;
    };
    Req req_[RG_MASTERS];
    sc_core::sc_event req_ev_;
    uint64_t busy_until_ = 0;
    int last_owner_ = -1;          /* master of the previous tenure (handover rule) */
    int rr_last_ = RG_MASTERS - 1; /* round-robin pointer: last granted master      */
    uint64_t win_idx_ = std::numeric_limits<uint64_t>::max(), win_used_ = 0;
    bool quiet_[RG_MASTERS] = {false, false, false};

    static uint64_t now() { return sc_core::sc_time_stamp().value(); }
    bool any_pending() const {
        for (auto &r : req_) if (r.pending) return true;
        return false;
    }
    uint64_t start_for(int m, uint64_t t) const {
        bool ho = last_owner_ >= 0 && m != last_owner_ && cfg.handover;
        return t + (ho ? cfg.handover * cfg.clk.value() : 0);
    }
    uint64_t window_len() const { return cfg.window_cycles * cfg.clk.value(); }
    /* variant 5: may the DNN be granted a tenure starting at `start`? */
    bool budget_ok(int m, uint64_t start, uint32_t bytes) const {
        if (cfg.policy != RG_BUDGET || m != RG_DNN) return true;
        uint64_t w = start / window_len();
        uint64_t used = (w == win_idx_) ? win_used_ : 0;
        return used + bytes <= cfg.budget_bytes;
    }
    bool eligible(int m, uint64_t t) const {
        return req_[m].pending && budget_ok(m, start_for(m, t), req_[m].bytes);
    }

    /* bookkeeping of one grant (both paths) */
    void book(int m, uint64_t t_req, uint64_t start, uint64_t len, uint32_t bytes) {
        const bool ho = last_owner_ >= 0 && m != last_owner_ && cfg.handover;
        busy_until_ = start + len;
        if (ho) handovers++;
        last_owner_ = m;
        rr_last_ = m;
        if (cfg.policy == RG_BUDGET && m == RG_DNN) {
            uint64_t w = start / window_len();
            if (w != win_idx_) { win_idx_ = w; win_used_ = 0; }
            win_used_ += bytes;
        }
        RgStats &s = stats[m];
        s.grants++;
        s.busy += len;
        s.wait += start - t_req;
        s.max_wait = std::max(s.max_wait, start - t_req);
        s.bytes += bytes;
        if (cfg.keep_trace) trace.push_back({t_req, start, len, m, bytes, ho});
    }

    int pick_rr(uint64_t t) const {
        for (int i = 1; i <= RG_MASTERS; i++) {
            int m = (rr_last_ + i) % RG_MASTERS;
            if (eligible(m, t)) return m;
        }
        return -1;
    }
    int pick_order(const int (&order)[RG_MASTERS], uint64_t t) const {
        for (int m : order) if (eligible(m, t)) return m;
        return -1;
    }
    int pick(uint64_t t) const {
        static const int sensor_first[RG_MASTERS] = {RG_SENSOR, RG_CPU, RG_DNN};
        static const int cpu_first[RG_MASTERS] = {RG_CPU, RG_SENSOR, RG_DNN};
        switch (cfg.policy) {
            case RG_SENSOR_FIRST: return pick_order(sensor_first, t);
            case RG_CPU_FIRST: return pick_order(cpu_first, t);
            case RG_BUFFER:
                if (eligible(RG_SENSOR, t) && cfg.urgent_level && cfg.sensor_level &&
                    cfg.sensor_level() >= cfg.urgent_level)
                    return RG_SENSOR;
                return pick_rr(t);
            case RG_RR:
            case RG_BUDGET:
            default: return pick_rr(t);
        }
    }

    void arbitrate() {
        while (true) {
            if (!any_pending()) sc_core::wait(req_ev_);
            sc_core::wait(sc_core::SC_ZERO_TIME);  /* collect requests of this time step */
            const uint64_t t = now();
            if (t < busy_until_) {
                sc_core::wait(sc_core::sc_time::from_value(busy_until_ - t));
                continue;
            }
            int m = pick(t);
            if (m < 0) {
                /* only an over-budget DNN request is pending: wait for the
                 * next budget window or for a new request */
                uint64_t next = (t / window_len() + 1) * window_len();
                sc_core::wait(sc_core::sc_time::from_value(next - t), req_ev_);
                continue;
            }
            Req &r = req_[m];
            const uint64_t start = start_for(m, t);
            r.pending = false;
            r.start = start;
            book(m, r.t_req, start, r.len, r.bytes);
            r.granted.notify(sc_core::sc_time::from_value(start - t));
        }
    }
};

/* CPU side: put between the core's memory interface and the bus router.
 * The CPU synchronises its local time before every access (so requests reach
 * the arbiter in time order), waits for the grant, and only then performs
 * the access (so e.g. an accelerator START write takes effect inside the
 * CPU's tenure). It holds the bus for max(target latency, one cycle). The CPU
 * sees the target latency plus any wait for the bus, as with the step-5 port. */
struct RgCpuPort : public sc_core::sc_module {
    tlm_utils::simple_target_socket<RgCpuPort> tsock;
    tlm_utils::simple_initiator_socket<RgCpuPort> isock;
    RgBus &bus;
    sc_core::sc_time min_tenure;

#if SC_VERSION_MAJOR < 3
    SC_HAS_PROCESS(RgCpuPort);
#endif
    RgCpuPort(sc_core::sc_module_name n, RgBus &b, sc_core::sc_time cycle)
        : sc_module(n), tsock("tsock"), isock("isock"), bus(b), min_tenure(cycle) {
        tsock.register_b_transport(this, &RgCpuPort::b_transport);
        tsock.register_transport_dbg(this, &RgCpuPort::transport_dbg);
    }

    /* address range that always takes the slow path (e.g. the accelerator's
     * registers, whose writes start DMA jobs); empty by default */
    uint64_t sync_lo = 1, sync_hi = 0;

    void b_transport(tlm::tlm_generic_payload &t, sc_core::sc_time &delay) {
        const uint64_t a = t.get_address();
        const bool may_fast = !(a >= sync_lo && a <= sync_hi);
        uint64_t start;
        const uint64_t t_local = (sc_core::sc_time_stamp() + delay).value();
        if (may_fast && bus.grant_if_alone(RG_CPU, t_local, t.get_data_length(), min_tenure.value(), start)) {
            /* nobody else can request: granted at the CPU's own time */
            sc_core::sc_time d = delay + sc_core::sc_time::from_value(start - t_local);
            const sc_core::sc_time before = d;
            isock->b_transport(t, d);
            const sc_core::sc_time lat = d - before;
            if (lat > min_tenure) bus.extend(RG_CPU, (lat - min_tenure).value());
            delay = d;
            return;
        }
        if (delay > sc_core::SC_ZERO_TIME) {  /* synchronise: the request happens now */
            sc_core::wait(delay);
            delay = sc_core::SC_ZERO_TIME;
        }
        bus.acquire(RG_CPU, t.get_data_length(), min_tenure.value());  /* returns at tenure start */
        sc_core::sc_time lat = sc_core::SC_ZERO_TIME;
        isock->b_transport(t, lat);  /* functional access + target latency */
        if (lat > min_tenure) bus.extend(RG_CPU, (lat - min_tenure).value());
        delay = lat;
    }

    unsigned transport_dbg(tlm::tlm_generic_payload &t) { return isock->transport_dbg(t); }
};

#endif /* RG_BUS_H */
