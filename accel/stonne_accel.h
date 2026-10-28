/*
 * stonne_accel.h - SystemC/TLM-2.0 model of a memory-mapped GEMM accelerator
 * whose compute timing comes from the STONNE cycle-level simulator.
 *
 * The same file compiles in two places:
 *   1. unit_test/  against the course SystemC 2.3.3 (C++20)  -> step 3, step 5 tests
 *   2. the RISC-V VP++ platform (vendored SystemC 3.0.x)     -> steps 4-6
 * so it only uses standard SystemC and TLM-2.0; STONNE sits behind
 * stonne_backend.h, compiled in its own file.
 *
 * What the model does for one job (see accel_regs.h for the register map):
 *   CPU writes the job shape (+ A/B/C addresses in DMA mode) and CTRL=START
 *   [DMA mode]  load the two inputs from memory -> cycles = DMA timing model
 *   compute with STONNE                          -> cycles = STONNE n_cycles
 *       OP_GEMM: C = A x B^T      (TPU, MAERI, SIGMA)
 *       OP_CONV: Y = conv(X, Wt)  (MAERI only: standard or depthwise)
 *   [DMA mode]  store the output to memory      -> cycles = DMA timing model
 *   STATUS.DONE = 1, per-phase cycle counters updated
 * In CPU-copy mode the CPU itself writes A/B into the scratchpads and reads C
 * back, one 32-bit bus access at a time; the cost of that shows up in the
 * CPU's cycle count, not in the accelerator counters.
 *
 * Step 5 adds (all off unless software asks for them, so steps 1-4 are
 * unchanged):
 *   CTRL_TILE   the job is cut into tiles of TILE_ROWS rows: weights are
 *               loaded once, then per tile: load inputs, compute, store.
 *   CTRL_DBUF   double buffering: two input and two output buffers (halves of
 *               scratchpads A and C). A separate PE thread computes tile i
 *               while the DMA engine loads tile i+1 and stores tile i-1.
 *   CTRL_IRQ    raise the completion interrupt (irq_line callback -> PLIC);
 *               software acknowledges it in IRQ_STATUS.
 *   shared bus  if the platform gives the model a SharedBus, every DMA
 *               transfer must win the bus first, in the ownership mode set in
 *               DMA_MODE (block, cycle stealing, transparent).
 */
#ifndef STONNE_ACCEL_H
#define STONNE_ACCEL_H

#include <systemc>
#include <tlm>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/simple_target_socket.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "accel_regs.h"
#include "shared_bus.h"
#include "stonne_backend.h"  /* the only link to STONNE (separate .cpp file) */

enum AccelArch { ARCH_TPU = 0, ARCH_MAERI = 1, ARCH_SIGMA = 2 };

/* Hardware parameters. They are fixed when the platform is built/started,
 * like the parameters of a real IP block, so software can only read them. */
struct AccelConfig {
    AccelArch arch = ARCH_TPU;
    unsigned tpu_rows = 16, tpu_cols = 16;  /* TPU: rows x cols PEs, dn_bw = rows+cols   */
    unsigned num_ms = 256;                  /* MAERI / SIGMA: number of multipliers      */
    unsigned dn_bw = 64, rn_bw = 64;        /* MAERI / SIGMA: network bandwidths         */
    bool sigma_wst = true;                  /* SIGMA: weights stationary (skips zero weights) */
    sc_core::sc_time clk = sc_core::sc_time(10, sc_core::SC_NS); /* 100 MHz             */
    unsigned reg_latency = 1;               /* cycles per MMIO access to the accelerator */
    unsigned dma_bytes_per_cycle = 4;       /* DMA data width (32-bit bus)              */
    unsigned dma_burst_bytes = 64;          /* bytes per DMA burst (reset value)         */
    unsigned dma_burst_latency = 10;        /* fixed cycles per burst (address+memory)   */
    unsigned dma_mode = DMA_MODE_BLOCK;     /* bus ownership (reset value), step 5       */
    bool stonne_verbose = false;            /* let STONNE print to stdout                */
    bool stonne_stats = false;              /* let STONNE write its JSON stats files     */

    unsigned num_pe() const { return arch == ARCH_TPU ? tpu_rows * tpu_cols : num_ms; }
    unsigned caps() const { return CAPS_GEMM | (arch == ARCH_MAERI ? CAPS_CONV : 0) | CAPS_TILE | CAPS_IRQ; }
    unsigned dist_bw() const { return arch == ARCH_TPU ? tpu_rows + tpu_cols : dn_bw; }
    std::string arch_name() const {
        return arch == ARCH_TPU ? "TPU" : (arch == ARCH_MAERI ? "MAERI" : "SIGMA");
    }
};

struct StonneAccel : public sc_core::sc_module {
    tlm_utils::simple_target_socket<StonneAccel> tsock;     /* CPU -> registers/SPM */
    tlm_utils::simple_initiator_socket<StonneAccel> isock;  /* DMA -> main memory   */

    AccelConfig cfg;

    /* optional connections (set by the platform / testbench before sc_start) */
    SharedBus *bus = nullptr;                 /* shared-bus model; nullptr = ideal bus */
    std::function<void(bool)> irq_line;       /* interrupt output (level)               */

    /* registers */
    uint32_t M = 0, K = 0, N = 0, a_addr = 0, b_addr = 0, c_addr = 0;
    uint32_t op = OP_GEMM;
    struct { uint32_t H, W, C, R, S, KOUT, G, stride, pt, pl, Ho, Wo; } cv = {};
    uint32_t status = 0, jobs = 0;
    uint32_t cyc_load = 0, cyc_compute = 0, cyc_store = 0, bytes_in = 0, bytes_out = 0;
    bool use_dma = false;
    /* step 5 registers */
    uint32_t tile_rows = 0, irq_pending = 0, cyc_total = 0, ntiles = 0, polls = 0;
    uint32_t dma_mode, dma_burst;
    uint64_t done_at = 0;
    uint32_t job_cpu_txn = 0, job_cpu_wait = 0, job_cpu_maxw = 0, job_dma_wait = 0;
    bool tiled = false, dbuf = false, irq_req = false;

    /* scratchpads */
    std::vector<int8_t> spm_a, spm_b;
    std::vector<int32_t> spm_c;

    sc_core::sc_event start_ev;

    /* tile pipeline state (shared by run() = DMA engine and pe_thread()) */
    struct Tile {
        uint32_t a_off, a_bytes, c_off, c_bytes;  /* in main memory, relative to A/C base */
        unsigned r0, nr;                          /* first row and rows (GEMM M / CONV Ho) */
        unsigned in_rows, pt;                     /* CONV: input rows fetched, top padding */
    };
    std::vector<Tile> tiles;
    int n_loaded = 0, n_computed = 0, n_stored = 0;
    sc_core::sc_event pipe_ev, pe_go;

#if SC_VERSION_MAJOR < 3
    SC_HAS_PROCESS(StonneAccel);
#endif

    StonneAccel(sc_core::sc_module_name name, const AccelConfig &c)
        : sc_module(name), tsock("tsock"), isock("isock"), cfg(c),
          dma_mode(c.dma_mode), dma_burst(c.dma_burst_bytes),
          spm_a(ACCEL_SPM_A_SIZE), spm_b(ACCEL_SPM_B_SIZE), spm_c(ACCEL_SPM_C_SIZE / 4) {
        tsock.register_b_transport(this, &StonneAccel::b_transport);
        tsock.register_transport_dbg(this, &StonneAccel::transport_dbg);
        SC_THREAD(run);
        SC_THREAD(pe_thread);
        std::cout << "[accel] " << cfg.arch_name() << ", " << cfg.num_pe() << " PEs, dn_bw "
                  << cfg.dist_bw() << ", DMA " << cfg.dma_bytes_per_cycle << " B/cycle, burst "
                  << cfg.dma_burst_bytes << " B + " << cfg.dma_burst_latency << " cycles" << std::endl;
    }

    uint64_t now_cycles(const sc_core::sc_time &delay) const {
        return (uint64_t)((sc_core::sc_time_stamp() + delay) / cfg.clk);
    }
    uint64_t cycles_of(uint64_t time_value) const {
        return time_value / cfg.clk.value();
    }

    /* ------------------------------------------------------------------ */
    /* CPU side: register and scratchpad accesses                          */
    /* ------------------------------------------------------------------ */
    void b_transport(tlm::tlm_generic_payload &t, sc_core::sc_time &delay) {
        access(t, &delay);
        delay += cfg.reg_latency * cfg.clk;
    }

    unsigned transport_dbg(tlm::tlm_generic_payload &t) {
        access(t, nullptr);
        return t.get_data_length();
    }

    uint8_t *spm_ptr(uint64_t off, unsigned len) {
        if (off >= ACCEL_SPM_A && off + len <= ACCEL_SPM_A + ACCEL_SPM_A_SIZE)
            return (uint8_t *)spm_a.data() + (off - ACCEL_SPM_A);
        if (off >= ACCEL_SPM_B && off + len <= ACCEL_SPM_B + ACCEL_SPM_B_SIZE)
            return (uint8_t *)spm_b.data() + (off - ACCEL_SPM_B);
        if (off >= ACCEL_SPM_C && off + len <= ACCEL_SPM_C + ACCEL_SPM_C_SIZE)
            return (uint8_t *)spm_c.data() + (off - ACCEL_SPM_C);
        return nullptr;
    }

    void set_irq(bool on) {
        if (on == (irq_pending != 0)) return;
        irq_pending = on ? 1 : 0;
        if (irq_line) irq_line(on);
    }

    void access(tlm::tlm_generic_payload &t, sc_core::sc_time *delay) {
        uint64_t off = t.get_address();
        unsigned len = t.get_data_length();
        uint8_t *p = t.get_data_ptr();
        t.set_response_status(tlm::TLM_OK_RESPONSE);

        if (off >= ACCEL_SPM_A) { /* scratchpad window: any size */
            uint8_t *s = spm_ptr(off, len);
            if (!s) { t.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE); return; }
            if (t.is_read()) std::memcpy(p, s, len); else std::memcpy(s, p, len);
            return;
        }
        if (len != 4 || (off & 3)) { t.set_response_status(tlm::TLM_BURST_ERROR_RESPONSE); return; }

        uint32_t v = 0;
        if (t.is_write()) {
            std::memcpy(&v, p, 4);
            switch (off) {
                case ACCEL_CTRL:
                    if (v & CTRL_CLEAR_DONE) status &= ~(STATUS_DONE | STATUS_ERROR);
                    if (v & CTRL_START) {
                        if (status & STATUS_BUSY) { status |= STATUS_ERROR; break; }
                        use_dma = (v & CTRL_USE_DMA) != 0;
                        dbuf = (v & CTRL_DBUF) != 0;
                        tiled = dbuf || (v & CTRL_TILE) != 0;
                        irq_req = (v & CTRL_IRQ) != 0;
                        status = STATUS_BUSY;
                        /* the job starts when the CPU's write actually happens */
                        start_ev.notify(delay ? *delay : sc_core::SC_ZERO_TIME);
                    }
                    break;
                case ACCEL_M: M = v; break;
                case ACCEL_K: K = v; break;
                case ACCEL_N: N = v; break;
                case ACCEL_A_ADDR: a_addr = v; break;
                case ACCEL_B_ADDR: b_addr = v; break;
                case ACCEL_C_ADDR: c_addr = v; break;
                case ACCEL_OP: op = v; break;
                case ACCEL_CONV_H: cv.H = v; break;
                case ACCEL_CONV_W: cv.W = v; break;
                case ACCEL_CONV_C: cv.C = v; break;
                case ACCEL_CONV_R: cv.R = v; break;
                case ACCEL_CONV_S: cv.S = v; break;
                case ACCEL_CONV_KOUT: cv.KOUT = v; break;
                case ACCEL_CONV_G: cv.G = v; break;
                case ACCEL_CONV_STRIDE: cv.stride = v; break;
                case ACCEL_CONV_PAD_T: cv.pt = v; break;
                case ACCEL_CONV_PAD_L: cv.pl = v; break;
                case ACCEL_CONV_HO: cv.Ho = v; break;
                case ACCEL_CONV_WO: cv.Wo = v; break;
                case ACCEL_TILE_ROWS: tile_rows = v; break;
                case ACCEL_IRQ_STATUS: if (v & 1) set_irq(false); break;
                case ACCEL_DMA_MODE: if (v <= DMA_MODE_TRANSPARENT) dma_mode = v; break;
                case ACCEL_DMA_BURST: if (v >= 1) dma_burst = v; break;
                default: break; /* writes to read-only registers are ignored */
            }
            return;
        }
        sc_core::sc_time d = delay ? *delay : sc_core::SC_ZERO_TIME;
        switch (off) {
            case ACCEL_ID: v = ACCEL_ID_VALUE; break;
            case ACCEL_STATUS:
                v = status;
                if ((status & STATUS_BUSY) && delay) polls++;
                break;
            case ACCEL_M: v = M; break;
            case ACCEL_K: v = K; break;
            case ACCEL_N: v = N; break;
            case ACCEL_A_ADDR: v = a_addr; break;
            case ACCEL_B_ADDR: v = b_addr; break;
            case ACCEL_C_ADDR: v = c_addr; break;
            case ACCEL_ARCH: v = cfg.arch; break;
            case ACCEL_NUM_PE: v = cfg.num_pe(); break;
            case ACCEL_DN_BW: v = cfg.dist_bw(); break;
            case ACCEL_CYC_LOAD: v = cyc_load; break;
            case ACCEL_CYC_COMPUTE: v = cyc_compute; break;
            case ACCEL_CYC_STORE: v = cyc_store; break;
            case ACCEL_BYTES_IN: v = bytes_in; break;
            case ACCEL_BYTES_OUT: v = bytes_out; break;
            case ACCEL_NOW_LO: v = (uint32_t)now_cycles(d); break;
            case ACCEL_NOW_HI: v = (uint32_t)(now_cycles(d) >> 32); break;
            case ACCEL_JOBS: v = jobs; break;
            case ACCEL_OP: v = op; break;
            case ACCEL_CAPS: v = cfg.caps() | (bus ? CAPS_BUS : 0); break;
            case ACCEL_TILE_ROWS: v = tile_rows; break;
            case ACCEL_IRQ_STATUS: v = irq_pending; break;
            case ACCEL_CYC_TOTAL: v = cyc_total; break;
            case ACCEL_TILES: v = ntiles; break;
            case ACCEL_POLLS: v = polls; break;
            case ACCEL_DONE_LO: v = (uint32_t)done_at; break;
            case ACCEL_DONE_HI: v = (uint32_t)(done_at >> 32); break;
            case ACCEL_DMA_MODE: v = dma_mode; break;
            case ACCEL_DMA_BURST: v = dma_burst; break;
            case ACCEL_BUS_CPU_TXN: v = job_cpu_txn; break;
            case ACCEL_BUS_CPU_WAIT: v = job_cpu_wait; break;
            case ACCEL_BUS_CPU_MAXW: v = job_cpu_maxw; break;
            case ACCEL_BUS_DMA_WAIT: v = job_dma_wait; break;
            default: v = 0; break;
        }
        std::memcpy(p, &v, 4);
    }

    /* ------------------------------------------------------------------ */
    /* DMA: functional copy through the bus + explicit timing model        */
    /* ------------------------------------------------------------------ */
    uint64_t burst_cycles(uint32_t n) const {
        return cfg.dma_burst_latency + (n + cfg.dma_bytes_per_cycle - 1) / cfg.dma_bytes_per_cycle;
    }

    /* Moves the data (TLM transactions, one per burst) and returns the
     * transfer time in cycles on an ideal, private bus. */
    uint64_t dma(tlm::tlm_command cmd, uint32_t addr, uint8_t *buf, uint32_t bytes) {
        uint64_t cycles = 0;
        for (uint32_t off = 0; off < bytes; off += dma_burst) {
            uint32_t n = std::min<uint32_t>(dma_burst, bytes - off);
            tlm::tlm_generic_payload t;
            sc_core::sc_time ignored = sc_core::SC_ZERO_TIME;
            t.set_command(cmd);
            t.set_address(addr + off);
            t.set_data_ptr(buf + off);
            t.set_data_length(n);
            t.set_streaming_width(n);
            t.set_byte_enable_ptr(nullptr);
            t.set_dmi_allowed(false);
            /* some VP++ targets (e.g. SimpleMemory) never set the response,
             * so start from OK and let a target report an error */
            t.set_response_status(tlm::TLM_OK_RESPONSE);
            isock->b_transport(t, ignored);
            if (t.get_response_status() != tlm::TLM_OK_RESPONSE) {
                std::cerr << "[accel] DMA error at 0x" << std::hex << addr + off << std::dec << std::endl;
                status |= STATUS_ERROR;
            }
            cycles += burst_cycles(n);
        }
        return cycles;
    }

    /* One DMA transfer that also takes the simulated time: on an ideal bus
     * it just waits; on the shared bus it has to win the bus first, in the
     * selected ownership mode. Returns the elapsed cycles. */
    uint64_t dma_xfer(tlm::tlm_command cmd, uint32_t addr, uint8_t *buf, uint32_t bytes) {
        const uint64_t ideal = dma(cmd, addr, buf, bytes);
        if (!bus) {
            sc_core::wait(ideal * cfg.clk);
            return ideal;
        }
        const uint64_t clk = cfg.clk.value();
        const uint64_t t0 = sc_core::sc_time_stamp().value();
        if (dma_mode == DMA_MODE_BLOCK) {
            /* hold the bus for all bursts of this transfer */
            uint64_t start = bus->reserve(BUS_DMA, t0, ideal * clk);
            sc_core::wait(sc_core::sc_time::from_value(start + ideal * clk - t0));
        } else if (dma_mode == DMA_MODE_STEAL) {
            /* give the bus back after every burst; the CPU can get in between */
            for (uint32_t off = 0; off < bytes; off += dma_burst) {
                uint64_t c = burst_cycles(std::min<uint32_t>(dma_burst, bytes - off)) * clk;
                uint64_t now = sc_core::sc_time_stamp().value();
                uint64_t start = bus->reserve(BUS_DMA, now, c);
                sc_core::wait(sc_core::sc_time::from_value(start + c - now));
            }
        } else {
            /* transparent: progress only in bus cycles the CPU did not use.
             * Wait for the remaining work, then look back at how much of that
             * time the bus was really idle; repeat until all work is done.
             * (Looking back is exact: no new request can be older than now.) */
            uint64_t remaining = ideal * clk;
            while (remaining > 0) {
                uint64_t from = sc_core::sc_time_stamp().value();
                bus->keep_history_from(from);
                sc_core::wait(sc_core::sc_time::from_value(remaining));
                uint64_t got = bus->idle_between(from, sc_core::sc_time_stamp().value());
                remaining -= std::min(got, remaining);
            }
            bus->release_history();
            uint64_t elapsed = sc_core::sc_time_stamp().value() - t0;
            bus->account(BUS_DMA, ideal * clk, elapsed - ideal * clk);
        }
        return cycles_of(sc_core::sc_time_stamp().value() - t0);
    }

    /* ------------------------------------------------------------------ */
    /* Compute: hand the job to STONNE and return its cycle count          */
    /* ------------------------------------------------------------------ */
    StonneParams stonne_params() const {
        StonneParams p;
        p.arch = cfg.arch;
        p.tpu_rows = cfg.tpu_rows;
        p.tpu_cols = cfg.tpu_cols;
        p.num_ms = cfg.num_ms;
        p.dn_bw = cfg.dn_bw;
        p.rn_bw = cfg.rn_bw;
        p.sigma_wst = cfg.sigma_wst;
        p.verbose = cfg.stonne_verbose;
        p.stats_files = cfg.stonne_stats;
        return p;
    }

    std::string job_name() const { return std::string(name()) + "_job" + std::to_string(jobs); }

    /* sizes of the job's two inputs and its output, in bytes */
    bool job_sizes(uint64_t &in_a, uint64_t &in_b, uint64_t &out, std::string &why) const {
        if (op == OP_GEMM) {
            in_a = (uint64_t)M * K; in_b = (uint64_t)N * K; out = (uint64_t)M * N * 4;
            if (!M || !K || !N) { why = "M, K and N must be non-zero"; return false; }
        } else if (op == OP_CONV) {
            if (!(cfg.caps() & CAPS_CONV)) { why = "OP_CONV needs the MAERI architecture"; return false; }
            bool dw = cv.G == cv.C && cv.KOUT == cv.C && cv.G > 1;
            if (!(cv.G == 1 || dw)) { why = "only G = 1 or depthwise (G = C = KOUT)"; return false; }
            if (!cv.H || !cv.W || !cv.C || !cv.R || !cv.S || !cv.KOUT || !cv.stride || !cv.Ho || !cv.Wo) {
                why = "a CONV register is zero"; return false;
            }
            in_a = (uint64_t)cv.H * cv.W * cv.C;
            in_b = (uint64_t)cv.R * cv.S * cv.C * (dw ? 1 : cv.KOUT);
            out = (uint64_t)cv.Ho * cv.Wo * cv.KOUT * 4;
        } else {
            why = "unknown OP"; return false;
        }
        if (in_b > ACCEL_SPM_B_SIZE) { why = "does not fit the scratchpads"; return false; }
        if (!tiled && (in_a > ACCEL_SPM_A_SIZE || out > ACCEL_SPM_C_SIZE)) {
            why = "does not fit the scratchpads"; return false;
        }
        return true;
    }

    uint64_t compute() {
        StonneParams p = stonne_params();
        std::string job = job_name();
        if (op == OP_GEMM)
            return stonne_gemm(p, job.c_str(), spm_a.data(), spm_b.data(), spm_c.data(), M, K, N);
        return stonne_conv(p, job.c_str(), spm_a.data(), spm_b.data(), spm_c.data(), cv.H, cv.W, cv.C,
                           cv.R, cv.S, cv.KOUT, cv.G, cv.stride, cv.pt, cv.pl, cv.Ho, cv.Wo);
    }

    /* ------------------------------------------------------------------ */
    /* Step 5: tiles                                                       */
    /* ------------------------------------------------------------------ */
    /* Cut the job into tiles of tile_rows rows. With double buffering every
     * tile must fit half of scratchpad A (inputs) and half of C (outputs). */
    bool make_tiles(std::string &why) {
        tiles.clear();
        const unsigned rows = (op == OP_GEMM) ? M : cv.Ho;
        if (tile_rows == 0) { why = "TILE_ROWS is zero"; return false; }
        const uint64_t cap_a = dbuf ? ACCEL_SPM_A_SIZE / 2 : ACCEL_SPM_A_SIZE;
        const uint64_t cap_c = dbuf ? ACCEL_SPM_C_SIZE / 2 : ACCEL_SPM_C_SIZE;
        for (unsigned r0 = 0; r0 < rows; r0 += tile_rows) {
            Tile t = {};
            t.r0 = r0;
            t.nr = std::min(tile_rows, rows - r0);
            if (op == OP_GEMM) {
                t.a_off = r0 * K; t.a_bytes = t.nr * K;
                t.c_off = r0 * N * 4; t.c_bytes = t.nr * N * 4;
            } else {
                /* input rows needed by output rows r0 .. r0+nr-1 (with halo) */
                int ys = (int)(r0 * cv.stride) - (int)cv.pt;
                int ye = (int)((r0 + t.nr - 1) * cv.stride) - (int)cv.pt + (int)cv.R;
                int a = std::max(0, ys), b = std::min((int)cv.H, ye);
                t.in_rows = b > a ? (unsigned)(b - a) : 0;
                t.pt = (unsigned)(a - ys);
                t.a_off = (uint32_t)a * cv.W * cv.C;
                t.a_bytes = t.in_rows * cv.W * cv.C;
                t.c_off = r0 * cv.Wo * cv.KOUT * 4;
                t.c_bytes = t.nr * cv.Wo * cv.KOUT * 4;
            }
            if (t.a_bytes > cap_a || t.c_bytes > cap_c) {
                why = dbuf ? "a tile does not fit half a scratchpad (lower TILE_ROWS)"
                           : "a tile does not fit the scratchpads (lower TILE_ROWS)";
                return false;
            }
            tiles.push_back(t);
        }
        return true;
    }

    uint8_t *a_buf(int b) { return (uint8_t *)spm_a.data() + b * (ACCEL_SPM_A_SIZE / 2); }
    int32_t *c_buf(int b) { return spm_c.data() + b * (ACCEL_SPM_C_SIZE / 8); }

    uint64_t load_tile(unsigned i, int b) {
        const Tile &t = tiles[i];
        bytes_in += t.a_bytes;
        return dma_xfer(tlm::TLM_READ_COMMAND, a_addr + t.a_off, a_buf(b), t.a_bytes);
    }
    uint64_t store_tile(unsigned i, int b) {
        const Tile &t = tiles[i];
        bytes_out += t.c_bytes;
        return dma_xfer(tlm::TLM_WRITE_COMMAND, c_addr + t.c_off, (uint8_t *)c_buf(b), t.c_bytes);
    }
    uint64_t compute_tile(unsigned i, int b) {
        const Tile &t = tiles[i];
        StonneParams p = stonne_params();
        std::string job = job_name() + "_t" + std::to_string(i);
        const int8_t *a = (const int8_t *)a_buf(b);
        if (op == OP_GEMM)
            return stonne_gemm(p, job.c_str(), a, spm_b.data(), c_buf(b), t.nr, K, N);
        return stonne_conv(p, job.c_str(), a, spm_b.data(), c_buf(b), t.in_rows, cv.W, cv.C,
                           cv.R, cv.S, cv.KOUT, cv.G, cv.stride, t.pt, cv.pl, t.nr, cv.Wo);
    }

    void wait_until(const std::function<bool()> &cond) {
        while (!cond()) sc_core::wait(pipe_ev);
    }
    void pipe_changed() { pipe_ev.notify(sc_core::SC_ZERO_TIME); }

    /* The PE array: computes the tiles in order, each one as soon as its
     * inputs are loaded and its output buffer has been stored. */
    void pe_thread() {
        while (true) {
            sc_core::wait(pe_go);
            for (unsigned i = 0; i < tiles.size(); i++) {
                wait_until([&] { return n_loaded >= (int)i + 1 && n_stored >= (int)i - 1; });
                uint64_t c = compute_tile(i, i % 2);
                if (c == 0) status |= STATUS_ERROR;
                cyc_compute += (uint32_t)c;
                sc_core::wait(c * cfg.clk);
                n_computed = i + 1;
                pipe_changed();
            }
        }
    }

    /* tiled, no overlap: weights once, then load -> compute -> store per tile */
    void run_tiled_serial(uint64_t in_b) {
        cyc_load += dma_xfer(tlm::TLM_READ_COMMAND, b_addr, (uint8_t *)spm_b.data(), in_b);
        for (unsigned i = 0; i < tiles.size(); i++) {
            cyc_load += load_tile(i, 0);
            uint64_t c = compute_tile(i, 0);
            if (c == 0) status |= STATUS_ERROR;
            cyc_compute += (uint32_t)c;
            sc_core::wait(c * cfg.clk);
            cyc_store += store_tile(i, 0);
        }
    }

    /* tiled with double buffering: this thread is the DMA engine; while the
     * PE thread computes tile i it loads tile i+1 and stores tile i-1. */
    void run_tiled_dbuf(uint64_t in_b) {
        const int T = (int)tiles.size();
        n_loaded = n_computed = n_stored = 0;
        pe_go.notify(sc_core::SC_ZERO_TIME);
        cyc_load += dma_xfer(tlm::TLM_READ_COMMAND, b_addr, (uint8_t *)spm_b.data(), in_b);
        cyc_load += load_tile(0, 0);
        n_loaded = 1;
        pipe_changed();
        for (int i = 0; i < T; i++) {
            if (i + 1 < T) { /* input buffer (i+1)%2 is free once tile i-1 is computed */
                wait_until([&] { return n_computed >= i; });
                cyc_load += load_tile(i + 1, (i + 1) % 2);
                n_loaded = i + 2;
                pipe_changed();
            }
            if (i >= 1) {    /* tile i-1 is computed: store it, freeing its output buffer */
                wait_until([&] { return n_computed >= i; });
                cyc_store += store_tile(i - 1, (i - 1) % 2);
                n_stored = i;
                pipe_changed();
            }
        }
        wait_until([&] { return n_computed >= T; });
        cyc_store += store_tile(T - 1, (T - 1) % 2);
        n_stored = T;
        pipe_changed();
    }

    /* ------------------------------------------------------------------ */
    /* The accelerator's control thread                                    */
    /* ------------------------------------------------------------------ */
    BusStats bus_at_start[BUS_MASTERS];

    void job_begin() {
        polls = 0;
        ntiles = 0;
        job_cpu_txn = job_cpu_wait = job_cpu_maxw = job_dma_wait = 0;
        if (bus) {
            bus->stats[BUS_CPU].max_wait = 0;
            for (int m = 0; m < BUS_MASTERS; m++) bus_at_start[m] = bus->stats[m];
        }
    }

    void job_end(uint64_t t_start) {
        const uint64_t now = sc_core::sc_time_stamp().value();
        cyc_total = (uint32_t)cycles_of(now - t_start);
        done_at = cycles_of(now);
        if (bus) {
            job_cpu_txn = (uint32_t)(bus->stats[BUS_CPU].txn - bus_at_start[BUS_CPU].txn);
            job_cpu_wait = (uint32_t)cycles_of(bus->stats[BUS_CPU].wait - bus_at_start[BUS_CPU].wait);
            job_cpu_maxw = (uint32_t)cycles_of(bus->stats[BUS_CPU].max_wait);
            job_dma_wait = (uint32_t)cycles_of(bus->stats[BUS_DMA].wait - bus_at_start[BUS_DMA].wait);
        }
        status = (status & STATUS_ERROR) | STATUS_DONE;
        if (irq_req) set_irq(true);
    }

    void run() {
        while (true) {
            sc_core::wait(start_ev);
            const uint64_t t_start = sc_core::sc_time_stamp().value();
            cyc_load = cyc_compute = cyc_store = bytes_in = bytes_out = 0;
            job_begin();

            uint64_t in_a = 0, in_b = 0, out = 0;
            std::string why;
            bool ok = job_sizes(in_a, in_b, out, why);
            if (ok && tiled && !use_dma) { why = "tiling needs DMA mode"; ok = false; }
            if (ok && tiled) ok = make_tiles(why);
            if (!ok) {
                std::cerr << "[accel] job rejected: " << why << std::endl;
                status = STATUS_DONE | STATUS_ERROR;
                done_at = cycles_of(sc_core::sc_time_stamp().value());
                if (irq_req) set_irq(true);
                continue;
            }

            if (!tiled) {
                /* steps 2-4: one piece, load -> compute -> store */
                bytes_in = (uint32_t)(in_a + in_b);
                bytes_out = (uint32_t)out;
                ntiles = 1;
                if (use_dma) {
                    cyc_load = dma_xfer(tlm::TLM_READ_COMMAND, a_addr, (uint8_t *)spm_a.data(), in_a);
                    cyc_load += dma_xfer(tlm::TLM_READ_COMMAND, b_addr, (uint8_t *)spm_b.data(), in_b);
                }

                cyc_compute = compute();
                if (cyc_compute == 0) status |= STATUS_ERROR;
                sc_core::wait(cyc_compute * cfg.clk);

                if (use_dma) cyc_store = dma_xfer(tlm::TLM_WRITE_COMMAND, c_addr, (uint8_t *)spm_c.data(), out);
            } else {
                bytes_in = (uint32_t)in_b;
                ntiles = (uint32_t)tiles.size();
                if (dbuf) run_tiled_dbuf(in_b);
                else run_tiled_serial(in_b);
            }
            jobs++;
            job_end(t_start);
        }
    }
};

#endif /* STONNE_ACCEL_H */
