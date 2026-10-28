/*
 * shared_bus.h - step 5: one shared system bus that the CPU and the
 * accelerator's DMA have to take turns on.
 *
 * The RISC-V VP++ SimpleBus only routes transactions: two masters can use it
 * at the same moment and never slow each other down. This file adds the
 * missing arbiter as an occupancy model:
 *
 *   - every bus tenure is a reservation [start, end) on one timeline;
 *   - a master that asks for the bus at time `req` gets the first free slot
 *     at or after `req` that is long enough (first come, first served, no
 *     pre-emption of a tenure that has already been granted);
 *   - the waiting time (start - req) is added to that master's delay.
 *
 * The CPU side is a TLM pass-through module (BusArbiterPort) put between the
 * core and the SimpleBus: the target's latency (memory 1 cycle, accelerator
 * register 1 cycle, ...) is how long the CPU holds the bus. The DMA side is
 * called directly by the accelerator model (StonneAccel::dma_xfer), which
 * holds the bus according to its bus-ownership mode (accel_regs.h):
 *
 *   DMA_MODE_BLOCK        one tenure for the whole transfer (CPU locked out)
 *   DMA_MODE_STEAL        one tenure per burst, the CPU gets the bus between
 *                         bursts (burst = 4 bytes is classic cycle stealing)
 *   DMA_MODE_TRANSPARENT  the DMA only uses cycles the CPU left idle and never
 *                         delays the CPU (harvest_idle below)
 *
 * Times are SystemC time values (sc_time::value(), picoseconds by default).
 * The model is exact for any order of requests, which matters because the
 * VP's CPU runs a little ahead of global time (TLM quantum, 1 cycle here):
 * a reservation made "in the future" is simply part of the timeline.
 */
#ifndef SHARED_BUS_H
#define SHARED_BUS_H

#include <systemc>
#include <tlm>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/simple_target_socket.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>

enum BusMaster { BUS_CPU = 0, BUS_DMA = 1, BUS_MASTERS = 2 };

struct BusStats {
    uint64_t txn = 0;          /* number of tenures                          */
    uint64_t busy = 0;         /* time holding the bus                       */
    uint64_t wait = 0;         /* time waiting for the bus                   */
    uint64_t max_wait = 0;     /* longest single wait (reset per job)        */
};

class SharedBus {
   public:
    BusStats stats[BUS_MASTERS];

    /* Reserve the bus for `len` time units, not before `req`. Returns the
     * start of the granted slot (>= req). */
    uint64_t reserve(BusMaster m, uint64_t req, uint64_t len) {
        prune();
        if (len == 0) len = 1;
        uint64_t start = req;
        auto it = busy_.upper_bound(start);
        if (it != busy_.begin()) {
            auto p = std::prev(it);
            if (p->second > start) start = p->second;
        }
        while (it != busy_.end() && it->first < start + len) {
            start = std::max(start, it->second);
            ++it;
        }
        insert(start, start + len);
        BusStats &s = stats[m];
        s.txn++;
        s.busy += len;
        s.wait += start - req;
        s.max_wait = std::max(s.max_wait, start - req);
        return start;
    }

    /* Idle bus time inside [a, b), i.e. not covered by any reservation. */
    uint64_t idle_between(uint64_t a, uint64_t b) const {
        if (b <= a) return 0;
        uint64_t busy = 0;
        auto it = busy_.upper_bound(a);
        if (it != busy_.begin()) --it;
        for (; it != busy_.end() && it->first < b; ++it) {
            uint64_t s = std::max(it->first, a), e = std::min(it->second, b);
            if (e > s) busy += e - s;
        }
        return (b - a) - busy;
    }

    /* Transparent DMA looks back at the timeline: keep reservations that end
     * after `t` until keep_history_from(max) is called again. */
    void keep_history_from(uint64_t t) { keep_from_ = t; }
    void release_history() { keep_from_ = std::numeric_limits<uint64_t>::max(); }

    void account(BusMaster m, uint64_t busy, uint64_t wait) {  /* for transparent DMA */
        BusStats &s = stats[m];
        s.txn++;
        s.busy += busy;
        s.wait += wait;
        s.max_wait = std::max(s.max_wait, wait);
    }

    size_t size() const { return busy_.size(); }

   private:
    std::map<uint64_t, uint64_t> busy_;  /* start -> end, non-overlapping */
    uint64_t keep_from_ = std::numeric_limits<uint64_t>::max();

    /* No request can be older than the current SystemC time, so slots that
     * ended before it are history and can go (unless transparent DMA needs them). */
    void prune() {
        uint64_t cut = std::min<uint64_t>(sc_core::sc_time_stamp().value(), keep_from_);
        while (!busy_.empty() && busy_.begin()->second <= cut) busy_.erase(busy_.begin());
    }

    void insert(uint64_t s, uint64_t e) {
        auto next = busy_.lower_bound(s);
        if (next != busy_.begin()) {  /* merge with a slot that ends exactly at s */
            auto p = std::prev(next);
            if (p->second == s) { s = p->first; busy_.erase(p); }
        }
        if (next != busy_.end() && next->first == e) { e = next->second; busy_.erase(next); }
        busy_[s] = e;
    }
};

/* CPU side: put between the core's memory interface and the bus. */
struct BusArbiterPort : public sc_core::sc_module {
    tlm_utils::simple_target_socket<BusArbiterPort> tsock;
    tlm_utils::simple_initiator_socket<BusArbiterPort> isock;
    SharedBus &bus;
    BusMaster who;
    sc_core::sc_time min_tenure;

#if SC_VERSION_MAJOR < 3
    SC_HAS_PROCESS(BusArbiterPort);
#endif
    BusArbiterPort(sc_core::sc_module_name n, SharedBus &b, BusMaster m, sc_core::sc_time cycle)
        : sc_module(n), tsock("tsock"), isock("isock"), bus(b), who(m), min_tenure(cycle) {
        tsock.register_b_transport(this, &BusArbiterPort::b_transport);
        tsock.register_transport_dbg(this, &BusArbiterPort::transport_dbg);
    }

    void b_transport(tlm::tlm_generic_payload &t, sc_core::sc_time &delay) {
        const uint64_t req = (sc_core::sc_time_stamp() + delay).value();
        const sc_core::sc_time before = delay;
        isock->b_transport(t, delay);  /* functional access + target latency */
        sc_core::sc_time hold = delay - before;
        if (hold < min_tenure) hold = min_tenure;  /* every access is at least one bus cycle */
        const uint64_t start = bus.reserve(who, req, hold.value());
        delay += sc_core::sc_time::from_value(start - req);  /* time spent waiting for the bus */
    }

    unsigned transport_dbg(tlm::tlm_generic_payload &t) { return isock->transport_dbg(t); }
};

#endif /* SHARED_BUS_H */
