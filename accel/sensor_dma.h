/*
 * sensor_dma.h - course project: a streaming sensor with a FIFO and its own
 * DMA, the third master on the request/grant bus (accel/rg_bus.h).
 *
 * Model rules (project plan, Section 4.1):
 *   - one S-byte sample arrives every 1/R (strictly periodic, first sample
 *     one period after t = 0);
 *   - FIFO of DEPTH samples; a sample that arrives when DEPTH samples are held
 *     is dropped and counted as an overflow;
 *   - request rule: the DMA requests the bus as soon as the FIFO holds THETA
 *     samples and it has no request outstanding;
 *   - drain rule: once granted, it moves exactly THETA samples in one burst
 *     of 10 + ceil(THETA*S/4) cycles;
 *   - removal rule: the THETA samples leave the FIFO when the burst completes;
 *   - tie rule: a sample arriving at the instant a burst completes is counted
 *     before the removal (the drain thread removes one delta cycle later).
 * No rule reads DEPTH except the overflow check, so request times do not
 * depend on DEPTH as long as nothing overflows.
 *
 * Measurement window: software writes WINDOW_US, then MARK; the window is
 * [mark, mark + WINDOW_US). Overflows, peak occupancy and the sensor's longest
 * wait are counted inside it, and the bus statistics of every master are
 * snapshotted at its start and end (bus occupancy per master in the window).
 */
#ifndef SENSOR_DMA_H
#define SENSOR_DMA_H

#include <systemc>
#include <tlm>
#include <tlm_utils/simple_target_socket.h>

#include <cstdint>
#include <cstring>
#include <ostream>

#include "rg_bus.h"
#include "sensor_regs.h"

struct SensorConfig {
    unsigned rate_ksps = 0;   /* 0 = no sensor */
    unsigned bytes = 4;       /* S, bytes per sample (<= 4: one bus word) */
    unsigned theta = 1;       /* drain threshold and burst, in samples */
    unsigned depth = 8;       /* FIFO capacity, in samples */
    sc_core::sc_time clk = sc_core::sc_time(10, sc_core::SC_NS);
    unsigned burst_latency = 10, bytes_per_cycle = 4;
};

struct SensorDma : public sc_core::sc_module {
    tlm_utils::simple_target_socket<SensorDma> tsock;
    SensorConfig cfg;
    RgBus *bus = nullptr;

    /* state */
    unsigned occ = 0;
    bool outstanding = false;

    /* statistics */
    uint64_t arrivals = 0, ovf_total = 0, ovf_win = 0, drains = 0;
    unsigned peak_total = 0, peak_win = 0;
    uint64_t maxwait_win = 0, maxwait_total = 0;

    /* window */
    uint32_t window_us = 0;
    bool win_open = false, win_done = false;
    uint64_t win_t0 = 0, win_t1 = 0;
    RgStats at_open[RG_MASTERS], at_close[RG_MASTERS];
    uint64_t ho_open = 0, ho_close = 0;

#if SC_VERSION_MAJOR < 3
    SC_HAS_PROCESS(SensorDma);
#endif
    SensorDma(sc_core::sc_module_name n, const SensorConfig &c, RgBus *b) : sc_module(n), tsock("tsock"), cfg(c), bus(b) {
        tsock.register_b_transport(this, &SensorDma::b_transport);
        if (cfg.rate_ksps) {
            SC_THREAD(produce);
            SC_THREAD(drain);
        }
        SC_THREAD(window);
    }

    uint64_t burst_cycles() const {
        return cfg.burst_latency + (cfg.theta * cfg.bytes + cfg.bytes_per_cycle - 1) / cfg.bytes_per_cycle;
    }
    sc_core::sc_time period() const { return sc_core::sc_time(1.0e6 / cfg.rate_ksps, sc_core::SC_NS); }
    bool in_window(uint64_t t) const { return win_open && t >= win_t0 && (win_done ? t < win_t1 : true); }

    void produce() {
        const sc_core::sc_time p = period();
        while (true) {
            sc_core::wait(p);
            const uint64_t t = sc_core::sc_time_stamp().value();
            arrivals++;
            if (occ >= cfg.depth) {
                ovf_total++;
                if (in_window(t)) ovf_win++;
            } else {
                occ++;
                if (occ > peak_total) peak_total = occ;
                if (in_window(t) && occ > peak_win) peak_win = occ;
            }
            if (occ >= cfg.theta && !outstanding) {
                outstanding = true;
                drain_ev.notify(sc_core::SC_ZERO_TIME);
            }
        }
    }

    void drain() {
        const uint64_t c = burst_cycles() * cfg.clk.value();
        while (true) {
            sc_core::wait(drain_ev);
            while (true) {
                const uint64_t t_req = sc_core::sc_time_stamp().value();
                bus->acquire(RG_SENSOR, cfg.theta * cfg.bytes, c);
                const uint64_t w = (sc_core::sc_time_stamp().value() - t_req) / cfg.clk.value();
                if (w > maxwait_total) maxwait_total = w;
                if (in_window(t_req) && w > maxwait_win) maxwait_win = w;
                sc_core::wait(sc_core::sc_time::from_value(c));
                sc_core::wait(sc_core::SC_ZERO_TIME);  /* tie rule: arrivals at this instant first */
                occ -= cfg.theta;
                drains++;
                if (occ < cfg.theta) break;  /* else request again at once */
            }
            outstanding = false;
        }
    }

    void window() {
        while (true) {
            sc_core::wait(mark_ev);
            win_t0 = sc_core::sc_time_stamp().value();
            win_open = true;
            win_done = false;
            ovf_win = 0;
            peak_win = occ;
            maxwait_win = 0;
            if (bus) { for (int m = 0; m < RG_MASTERS; m++) at_open[m] = bus->stats[m]; ho_open = bus->handovers; }
            if (window_us == 0) continue;
            sc_core::wait(sc_core::sc_time(window_us, sc_core::SC_US));
            win_t1 = sc_core::sc_time_stamp().value();
            win_done = true;
            if (bus) { for (int m = 0; m < RG_MASTERS; m++) at_close[m] = bus->stats[m]; ho_close = bus->handovers; }
        }
    }

    void b_transport(tlm::tlm_generic_payload &t, sc_core::sc_time &delay) {
        const uint32_t off = (uint32_t)t.get_address();
        uint32_t v = 0;
        if (t.get_data_length() != 4) { t.set_response_status(tlm::TLM_BURST_ERROR_RESPONSE); return; }
        if (t.is_write()) {
            std::memcpy(&v, t.get_data_ptr(), 4);
            if (off == SENSOR_WINDOW_US) window_us = v;
            else if (off == SENSOR_MARK) mark_ev.notify(delay);
        } else {
            switch (off) {
                case SENSOR_ID: v = cfg.rate_ksps ? SENSOR_ID_VALUE : 0; break;
                case SENSOR_WINDOW_US: v = window_us; break;
                case SENSOR_OVF_WINDOW: v = (uint32_t)ovf_win; break;
                case SENSOR_OVF_TOTAL: v = (uint32_t)ovf_total; break;
                case SENSOR_PEAK_WINDOW: v = peak_win; break;
                case SENSOR_THETA: v = cfg.theta; break;
                case SENSOR_DEPTH: v = cfg.depth; break;
                case SENSOR_RATE_KSPS: v = cfg.rate_ksps; break;
                case SENSOR_MAXWAIT: v = (uint32_t)maxwait_win; break;
                case SENSOR_WINDOW_DONE: v = win_done ? 1 : 0; break;
                default: v = 0;
            }
            std::memcpy(t.get_data_ptr(), &v, 4);
        }
        delay += cfg.clk;  /* one cycle per register access */
        t.set_response_status(tlm::TLM_OK_RESPONSE);
    }

    /* machine-readable summary, printed by the platform at the end */
    void report(std::ostream &os) const {
        os << "SENSOR,rate_ksps," << cfg.rate_ksps << ",bytes," << cfg.bytes << ",theta," << cfg.theta << ",depth,"
           << cfg.depth << ",arrivals," << arrivals << ",drains," << drains << ",ovf_total," << ovf_total
           << ",ovf_window," << ovf_win << ",peak_total," << peak_total << ",peak_window," << peak_win
           << ",maxwait_window," << maxwait_win << ",maxwait_total," << maxwait_total << std::endl;
        if (bus && win_done) {
            const uint64_t c = cfg.clk.value(), len = win_t1 - win_t0;
            os << "WINDOW,t0_cycles," << win_t0 / c << ",cycles," << len / c;
            uint64_t total = 0;
            for (int m = 0; m < RG_MASTERS; m++) {
                uint64_t b = at_close[m].busy - at_open[m].busy;
                total += b;
                os << "," << rg_master_name(m) << "_busy," << b / c;
            }
            os << ",handovers," << (ho_close - ho_open) << ",occupancy," << (double)total / (double)len << std::endl;
        }
    }

   private:
    sc_core::sc_event drain_ev, mark_ev;
};

#endif /* SENSOR_DMA_H */
