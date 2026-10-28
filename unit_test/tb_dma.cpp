/*
 * tb_dma.cpp - step 5 unit test: the DMA/bus extension of the accelerator,
 * on the course SystemC 2.3.3 (C++20), before using it in the RISC-V platform.
 *
 *            +-----------+   shared bus (SharedBus + BusArbiterPort)
 *   Driver --| CPU port  |--+--> Router --> Memory
 *   ("CPU")  +-----------+  |          \-> StonneAccel registers/SPM
 *                           |
 *   StonneAccel DMA --------+  (reserves the same bus inside dma_xfer)
 *   StonneAccel irq_line ------> Driver (interrupt event)
 *
 * Part 1  tiling and double buffering (CPU asleep, waiting for the
 *         interrupt, so the DMA has the bus to itself): every job is run three ways
 *         (one piece / tiled / tiled + double buffered) and must give the same
 *         result as plain C++. Timing checks:
 *           tiled      total == load + compute + store       (no overlap)
 *           dbuf       total <  tiled total                   (overlap)
 *           dbuf       total >= max(load + store, compute)    (one DMA engine,
 *                                                              one PE array)
 * Part 2  completion interrupt: the driver sleeps on the interrupt instead of
 *         polling. Checks: IRQ_STATUS is set, the line goes low after the
 *         acknowledge, no STATUS polls happened, polling jobs do count polls.
 * Part 3  bus ownership: the same DMA-heavy job, while the "CPU" keeps using
 *         the bus (one memory read, then one cycle of work, until the
 *         interrupt). Checks per mode:
 *           CPU asleep    job time == ideal-bus time in every mode
 *           block         the CPU waits up to a whole transfer
 *           steal         the CPU waits at most one burst
 *           transparent   the CPU never waits, the DMA gets slower
 *
 * usage: ./tb_dma [maeri|tpu|sigma] [num_pe] [dn_bw]     (CONV jobs need maeri)
 */
#include <systemc>
#include <tlm>
#include <tlm_utils/simple_initiator_socket.h>
#include <tlm_utils/simple_target_socket.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "shared_bus.h"
#include "stonne_accel.h"

using namespace sc_core;

/* ---------------- main memory: 1-cycle access, like the VP's ---------------- */
struct Memory : sc_module {
    tlm_utils::simple_target_socket<Memory> tsock;
    std::vector<uint8_t> data;
    sc_time latency;
    Memory(sc_module_name n, size_t bytes, sc_time lat)
        : sc_module(n), tsock("tsock"), data(bytes, 0), latency(lat) {
        tsock.register_b_transport(this, &Memory::b_transport);
    }
    void b_transport(tlm::tlm_generic_payload &t, sc_time &d) {
        uint64_t a = t.get_address();
        unsigned n = t.get_data_length();
        if (a + n > data.size()) { t.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE); return; }
        if (t.is_read()) memcpy(t.get_data_ptr(), &data[a], n);
        else memcpy(&data[a], t.get_data_ptr(), n);
        t.set_response_status(tlm::TLM_OK_RESPONSE);
        d += latency;
    }
};

/* ---------------- address decoder (two masters, two slaves) ---------------- */
struct Router : sc_module {
    tlm_utils::simple_target_socket<Router> from_cpu, from_dma;
    tlm_utils::simple_initiator_socket<Router> to_mem, to_acc;
#if SC_VERSION_MAJOR < 3
    SC_HAS_PROCESS(Router);
#endif
    Router(sc_module_name n) : sc_module(n), from_cpu("from_cpu"), from_dma("from_dma"), to_mem("to_mem"), to_acc("to_acc") {
        from_cpu.register_b_transport(this, &Router::b_transport);
        from_dma.register_b_transport(this, &Router::b_transport);
    }
    void b_transport(tlm::tlm_generic_payload &t, sc_time &d) {
        uint64_t a = t.get_address();
        if (a >= ACCEL_BASE && a < ACCEL_BASE + ACCEL_SIZE) {
            t.set_address(a - ACCEL_BASE);
            to_acc->b_transport(t, d);
        } else {
            to_mem->b_transport(t, d);
        }
    }
};

/* ---------------- the "CPU" ---------------- */
struct GemmJob { const char *name; unsigned M, K, N; };
struct ConvJob { const char *name; unsigned H, W, C, R, S, KOUT, G, stride; };

/* one prepared job: inputs in memory, registers to write, reference output */
struct Prepared {
    std::string name, shape;
    std::vector<std::pair<uint32_t, uint32_t>> regs;
    std::vector<int32_t> ref;
    unsigned rows = 0;  /* rows that TILE_ROWS refers to */
    uint64_t macs = 0;
};

struct Result { uint64_t total, load, compute, store, cyc_total, tiles, polls, bytes_in; bool ok; };

struct Driver : sc_module {
    tlm_utils::simple_initiator_socket<Driver> isock;
    Memory *mem = nullptr;
    StonneAccel *acc = nullptr;
    sc_time clk = sc_time(10, SC_NS);
    int failures = 0;
    bool conv_supported = false;
    std::vector<GemmJob> gemm_jobs;
    std::vector<ConvJob> conv_jobs;

    sc_event irq_ev;
    bool irq_level = false;
    unsigned irq_rising = 0;

    static constexpr uint32_t A_ADDR = 0x00000, B_ADDR = 0x40000, C_ADDR = 0x80000;
    static constexpr uint32_t SCRATCH = 0xF0000;  /* what the CPU reads in its busy loop */

#if SC_VERSION_MAJOR < 3
    SC_HAS_PROCESS(Driver);
#endif
    Driver(sc_module_name n) : sc_module(n), isock("isock") { SC_THREAD(test); }

    void irq(bool on) {  /* the accelerator's interrupt output */
        if (on && !irq_level) irq_rising++;
        irq_level = on;
        if (on) irq_ev.notify(SC_ZERO_TIME);
    }

    sc_time xfer(tlm::tlm_command cmd, uint32_t addr, void *p, unsigned len) {
        tlm::tlm_generic_payload t;
        sc_time d = SC_ZERO_TIME;
        t.set_command(cmd);
        t.set_address(addr);
        t.set_data_ptr((unsigned char *)p);
        t.set_data_length(len);
        t.set_streaming_width(len);
        t.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
        isock->b_transport(t, d);
        if (t.get_response_status() != tlm::TLM_OK_RESPONSE) {
            printf("  bus error at 0x%08x\n", addr);
            failures++;
        }
        wait(d);  /* latency of the target + any wait for the bus */
        return d;
    }
    void wr(uint32_t off, uint32_t v) { xfer(tlm::TLM_WRITE_COMMAND, ACCEL_BASE + off, &v, 4); }
    uint32_t rd(uint32_t off) { uint32_t v = 0; xfer(tlm::TLM_READ_COMMAND, ACCEL_BASE + off, &v, 4); return v; }
    uint64_t now() { return (uint64_t)(sc_time_stamp() / clk); }
    void check(bool cond, const char *what) {
        if (!cond) { printf("  CHECK FAILED: %s\n", what); failures++; }
    }

    /* TFLite SAME padding */
    static void same(unsigned in, unsigned k, unsigned s, unsigned &out, unsigned &pad) {
        out = (in + s - 1) / s;
        int total = (int)((out - 1) * s + k) - (int)in;
        pad = total > 0 ? total / 2 : 0;
    }

    Prepared prepare_gemm(const GemmJob &j, unsigned seed) {
        Prepared p;
        std::vector<int8_t> A(j.M * j.K), B(j.N * j.K);
        srand(seed);
        for (auto &x : A) x = (int8_t)(rand() % 256 - 128);
        for (auto &x : B) x = (int8_t)(rand() % 256 - 128);
        p.ref.assign(j.M * j.N, 0);
        for (unsigned m = 0; m < j.M; m++)
            for (unsigned n = 0; n < j.N; n++) {
                int32_t s = 0;
                for (unsigned k = 0; k < j.K; k++) s += A[m * j.K + k] * B[n * j.K + k];
                p.ref[m * j.N + n] = s;
            }
        memcpy(&mem->data[A_ADDR], A.data(), A.size());
        memcpy(&mem->data[B_ADDR], B.data(), B.size());
        p.name = j.name;
        char s[48]; snprintf(s, sizeof s, "GEMM %ux%ux%u", j.M, j.K, j.N); p.shape = s;
        p.regs = {{ACCEL_OP, OP_GEMM}, {ACCEL_M, j.M}, {ACCEL_K, j.K}, {ACCEL_N, j.N}};
        p.rows = j.M;
        p.macs = (uint64_t)j.M * j.K * j.N;
        return p;
    }

    Prepared prepare_conv(const ConvJob &j, unsigned seed) {
        Prepared p;
        unsigned Ho, Wo, pt, pl;
        same(j.H, j.R, j.stride, Ho, pt);
        same(j.W, j.S, j.stride, Wo, pl);
        const bool dw = j.G > 1;
        std::vector<int8_t> X(j.H * j.W * j.C), Wt(dw ? j.R * j.S * j.C : j.KOUT * j.R * j.S * j.C);
        srand(seed);
        for (auto &x : X) x = (int8_t)(rand() % 256 - 128);
        for (auto &x : Wt) x = (int8_t)(rand() % 256 - 128);
        p.ref.assign(Ho * Wo * j.KOUT, 0);
        for (unsigned oy = 0; oy < Ho; oy++)
            for (unsigned ox = 0; ox < Wo; ox++)
                for (unsigned oc = 0; oc < j.KOUT; oc++) {
                    int32_t acc = 0;
                    for (unsigned r = 0; r < j.R; r++)
                        for (unsigned q = 0; q < j.S; q++) {
                            int iy = (int)(oy * j.stride + r) - (int)pt, ix = (int)(ox * j.stride + q) - (int)pl;
                            if (iy < 0 || ix < 0 || iy >= (int)j.H || ix >= (int)j.W) continue;
                            if (dw) acc += X[(iy * j.W + ix) * j.C + oc] * Wt[(r * j.S + q) * j.C + oc];
                            else for (unsigned c = 0; c < j.C; c++)
                                acc += X[(iy * j.W + ix) * j.C + c] * Wt[((oc * j.R + r) * j.S + q) * j.C + c];
                        }
                    p.ref[(oy * Wo + ox) * j.KOUT + oc] = acc;
                }
        memcpy(&mem->data[A_ADDR], X.data(), X.size());
        memcpy(&mem->data[B_ADDR], Wt.data(), Wt.size());
        p.name = j.name;
        char s[48]; snprintf(s, sizeof s, "%s %ux%ux%u k%ux%u s%u", dw ? "DW" : "CONV", j.H, j.W, j.C, j.R, j.S, j.stride);
        p.shape = s;
        p.regs = {{ACCEL_OP, OP_CONV}, {ACCEL_CONV_H, j.H}, {ACCEL_CONV_W, j.W}, {ACCEL_CONV_C, j.C},
                  {ACCEL_CONV_R, j.R}, {ACCEL_CONV_S, j.S}, {ACCEL_CONV_KOUT, j.KOUT}, {ACCEL_CONV_G, j.G},
                  {ACCEL_CONV_STRIDE, j.stride}, {ACCEL_CONV_PAD_T, pt}, {ACCEL_CONV_PAD_L, pl},
                  {ACCEL_CONV_HO, Ho}, {ACCEL_CONV_WO, Wo}};
        p.rows = Ho;
        p.macs = (uint64_t)Ho * Wo * j.KOUT * j.R * j.S * (dw ? 1 : j.C);
        return p;
    }

    /* wait styles of the CPU while the job runs */
    enum Wait { POLL, SLEEP, BUSY };

    /* busy loop of part 3: read memory, one cycle of work, repeat */
    uint64_t busy_reads = 0;

    Result run(const Prepared &p, uint32_t ctrl_extra, unsigned tile_rows, Wait how) {
        for (size_t i = 0; i < p.ref.size(); i++) memcpy(&mem->data[C_ADDR + 4 * i], "\x55\x55\x55\x55", 4);
        wr(ACCEL_CTRL, CTRL_CLEAR_DONE);
        for (auto &r : p.regs) wr(r.first, r.second);
        wr(ACCEL_A_ADDR, A_ADDR); wr(ACCEL_B_ADDR, B_ADDR); wr(ACCEL_C_ADDR, C_ADDR);
        wr(ACCEL_TILE_ROWS, tile_rows);
        uint64_t t0 = now();
        uint32_t ctrl = CTRL_START | CTRL_USE_DMA | ctrl_extra | (how != POLL ? CTRL_IRQ : 0);
        wr(ACCEL_CTRL, ctrl);
        busy_reads = 0;
        if (how == POLL) {
            while (!(rd(ACCEL_STATUS) & STATUS_DONE)) wait(clk);
        } else {
            while (!irq_level) {
                if (how == SLEEP) wait(irq_ev);
                else { uint32_t v; xfer(tlm::TLM_READ_COMMAND, SCRATCH, &v, 4); busy_reads++; wait(clk); }
            }
            check(rd(ACCEL_IRQ_STATUS) == 1, "IRQ_STATUS set after the interrupt");
            wr(ACCEL_IRQ_STATUS, 1);  /* acknowledge */
            check(!irq_level && rd(ACCEL_IRQ_STATUS) == 0, "interrupt line low after acknowledge");
            check(rd(ACCEL_STATUS) & STATUS_DONE, "STATUS.DONE set when the interrupt came");
        }
        Result r;
        r.total = now() - t0;
        r.load = rd(ACCEL_CYC_LOAD); r.compute = rd(ACCEL_CYC_COMPUTE); r.store = rd(ACCEL_CYC_STORE);
        r.cyc_total = rd(ACCEL_CYC_TOTAL); r.tiles = rd(ACCEL_TILES); r.polls = rd(ACCEL_POLLS);
        r.bytes_in = rd(ACCEL_BYTES_IN);
        r.ok = !(rd(ACCEL_STATUS) & STATUS_ERROR);
        for (size_t i = 0; i < p.ref.size() && r.ok; i++) {
            int32_t y; memcpy(&y, &mem->data[C_ADDR + 4 * i], 4);
            r.ok = (y == p.ref[i]);
        }
        if (!r.ok) failures++;
        return r;
    }

    /* ---------------- part 1 ---------------- */
    void part1_job(const Prepared &p, unsigned ntiles) {
        unsigned tr = (p.rows + ntiles - 1) / ntiles;
        Result one = run(p, 0, 0, SLEEP);  /* CPU sleeps: the DMA has the bus to itself */
        Result til = run(p, CTRL_TILE, tr, SLEEP);
        Result dbf = run(p, CTRL_DBUF, tr, SLEEP);
        printf("%-13s %-21s| one %6llu | tiled x%-2llu %6llu (=%6llu+%5llu+%6llu) | dbuf %6llu -%4.1f%% | %s\n",
               p.name.c_str(), p.shape.c_str(), (unsigned long long)one.cyc_total, (unsigned long long)til.tiles,
               (unsigned long long)til.cyc_total, (unsigned long long)til.load, (unsigned long long)til.compute,
               (unsigned long long)til.store, (unsigned long long)dbf.cyc_total,
               100.0 * (1.0 - (double)dbf.cyc_total / til.cyc_total),
               one.ok && til.ok && dbf.ok ? "PASS" : "FAIL");
        check(one.tiles == 1 && til.tiles == dbf.tiles && til.tiles >= 2, "tile count");
        check(til.cyc_total == til.load + til.compute + til.store, "tiled: total == load + compute + store");
        check(dbf.load == til.load && dbf.store == til.store && dbf.compute == til.compute,
              "dbuf moves and computes the same as tiled");
        if (til.tiles >= 2) check(dbf.cyc_total < til.cyc_total, "dbuf faster than tiled");
        check(dbf.cyc_total >= std::max(dbf.load + dbf.store, dbf.compute), "dbuf >= max(DMA, compute)");
    }

    /* ---------------- part 3 ---------------- */
    void part3(const Prepared &p) {
        printf("\nPart 3: bus ownership. Job %s %s, one piece; the CPU either sleeps (WFI) or keeps\n"
               "reading memory (1 read + 1 cycle of work per loop) until the interrupt.\n", p.name.c_str(), p.shape.c_str());
        printf("%-22s %-6s | %9s %9s | %9s %9s %9s %9s | %s\n", "DMA mode", "CPU", "job cyc", "DMA wait",
               "CPU reads", "CPU txn", "CPU wait", "max wait", "result");
        struct M { const char *name; uint32_t mode, burst; };
        const M modes[] = {{"block (burst 64 B)", DMA_MODE_BLOCK, 64},
                           {"steal (burst 64 B)", DMA_MODE_STEAL, 64},
                           {"steal (burst 4 B)", DMA_MODE_STEAL, 4},
                           {"transparent", DMA_MODE_TRANSPARENT, 64}};
        uint64_t ideal = 0;
        for (const M &m : modes) {
            wr(ACCEL_DMA_MODE, m.mode);
            wr(ACCEL_DMA_BURST, m.burst);
            uint64_t burst_cyc = acc->cfg.dma_burst_latency + (m.burst + acc->cfg.dma_bytes_per_cycle - 1) / acc->cfg.dma_bytes_per_cycle;
            for (Wait how : {SLEEP, BUSY}) {
                Result r = run(p, 0, 0, how);
                uint32_t txn = rd(ACCEL_BUS_CPU_TXN), cw = rd(ACCEL_BUS_CPU_WAIT), mw = rd(ACCEL_BUS_CPU_MAXW),
                         dw = rd(ACCEL_BUS_DMA_WAIT);
                printf("%-22s %-6s | %9llu %9u | %9llu %9u %9u %9u | %s\n", m.name, how == SLEEP ? "sleep" : "busy",
                       (unsigned long long)r.cyc_total, dw, (unsigned long long)busy_reads, txn, cw, mw, r.ok ? "PASS" : "FAIL");
                if (how == SLEEP) {
                    if (m.burst == 64 && ideal == 0) ideal = r.cyc_total;
                    if (m.burst == 64) check(r.cyc_total == ideal, "CPU asleep: job time equals the ideal-bus time");
                    /* the only conflict: the DMA's first burst waits for the CPU's START write */
                    check(dw <= 1 && txn == 0, "CPU asleep: no bus conflicts");
                } else {
                    check(r.polls == 0, "interrupt-driven: no STATUS polls");
                    if (m.mode == DMA_MODE_BLOCK) check(mw > 4 * burst_cyc, "block: the CPU waits for a whole transfer");
                    if (m.mode == DMA_MODE_STEAL) check(mw <= burst_cyc + 1, "steal: the CPU waits at most one burst");
                    if (m.mode == DMA_MODE_TRANSPARENT) {
                        check(cw == 0, "transparent: the CPU never waits");
                        check(r.cyc_total > ideal, "transparent: the DMA is slower when the CPU is busy");
                    }
                }
            }
        }
        wr(ACCEL_DMA_MODE, DMA_MODE_BLOCK);
        wr(ACCEL_DMA_BURST, acc->cfg.dma_burst_bytes);
    }

    void test() {
        printf("ID register: 0x%08x, CAPS 0x%02x (tile %s, irq %s, shared bus %s)\n", rd(ACCEL_ID), rd(ACCEL_CAPS),
               (rd(ACCEL_CAPS) & CAPS_TILE) ? "yes" : "no", (rd(ACCEL_CAPS) & CAPS_IRQ) ? "yes" : "no",
               (rd(ACCEL_CAPS) & CAPS_BUS) ? "yes" : "no");
        conv_supported = (rd(ACCEL_CAPS) & CAPS_CONV) != 0;

        printf("\nPart 1: tiling and double buffering (4 tiles), cycles START->DONE;\n"
               "        tiled = load + compute + store; dbuf -x%% = that much faster than tiled\n");
        unsigned seed = 1;
        for (const GemmJob &j : gemm_jobs) part1_job(prepare_gemm(j, seed++), 4);
        if (conv_supported)
            for (const ConvJob &j : conv_jobs) part1_job(prepare_conv(j, seed++), 4);
        else
            printf("(CONV jobs skipped: this architecture has no OP_CONV)\n");
        {   /* tile sweep on one job */
            Prepared p = conv_supported ? prepare_conv(conv_jobs[2], 99) : prepare_gemm(gemm_jobs[1], 99);
            printf("tile sweep, %s %s:", p.name.c_str(), p.shape.c_str());
            for (unsigned t : {2u, 4u, 8u, 16u}) {
                Result r = run(p, CTRL_DBUF, (p.rows + t - 1) / t, SLEEP);
                printf("  %u tiles %llu%s", t, (unsigned long long)r.cyc_total, r.ok ? "" : " FAIL");
            }
            printf("\n");
        }
        {   /* a tile too big for half a scratchpad must be refused */
            Prepared p = prepare_gemm({"big", 2048, 64, 32}, 7);  /* C = 256 KB: needs >= 2 tiles with dbuf */
            Result r = run(p, CTRL_DBUF, 2048, POLL);
            printf("dbuf with one 256 KB output tile -> ERROR flag %s\n", r.ok ? "NOT set" : "set (expected)");
            if (!r.ok) failures--; else failures++;  /* the error is the expected outcome here */
            Result r2 = run(p, CTRL_DBUF, 512, POLL);
            printf("same job with 4 tiles -> %s\n", r2.ok ? "PASS" : "FAIL");
        }

        printf("\nPart 2: completion interrupt\n");
        Prepared p2 = prepare_gemm(gemm_jobs[1], 5);
        unsigned before = irq_rising;
        Result poll = run(p2, 0, 0, POLL);
        Result slp = run(p2, CTRL_DBUF, (p2.rows + 3) / 4, SLEEP);
        printf("polling job: %llu STATUS polls, CPU saw DONE %llu cycles after START   %s\n",
               (unsigned long long)poll.polls, (unsigned long long)poll.total, poll.ok ? "PASS" : "FAIL");
        printf("interrupt job (dbuf): %llu polls, %u interrupt(s), woke up %llu cycles after START   %s\n",
               (unsigned long long)slp.polls, irq_rising - before, (unsigned long long)slp.total, slp.ok ? "PASS" : "FAIL");
        check(poll.polls > 10, "polling job counted its STATUS reads");
        check(slp.polls == 0, "interrupt job: no polls");
        check(irq_rising - before == 1, "exactly one interrupt per job");

        if (acc->bus) part3(prepare_gemm({"dma_heavy", 1024, 64, 16}, 11));

        printf("\n%s (%d failure%s), simulated time %s\n", failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED",
               failures, failures == 1 ? "" : "s", sc_time_stamp().to_string().c_str());
        sc_stop();
    }
};

int sc_main(int argc, char **argv) {
    AccelConfig cfg;
    std::string arch = argc > 1 ? argv[1] : "maeri";
    unsigned pe = argc > 2 ? atoi(argv[2]) : 256;
    if (arch == "tpu") {
        cfg.arch = ARCH_TPU;
        unsigned r = (unsigned)std::lround(std::sqrt((double)pe));
        cfg.tpu_rows = cfg.tpu_cols = r;
    } else {
        cfg.arch = (arch == "maeri") ? ARCH_MAERI : ARCH_SIGMA;
        cfg.num_ms = pe;
        cfg.dn_bw = cfg.rn_bw = argc > 3 ? atoi(argv[3]) : 256;
    }
    const sc_time clk(10, SC_NS);

    SharedBus bus;
    Driver drv("cpu");
    BusArbiterPort cpu_port("cpu_port", bus, BUS_CPU, clk);
    Router router("router");
    StonneAccel acc("accel", cfg);
    Memory mem("mem", 1 << 20, clk);
    acc.bus = &bus;
    acc.irq_line = [&drv](bool on) { drv.irq(on); };
    drv.mem = &mem;
    drv.acc = &acc;

    drv.isock.bind(cpu_port.tsock);
    cpu_port.isock.bind(router.from_cpu);
    acc.isock.bind(router.from_dma);
    router.to_mem.bind(mem.tsock);
    router.to_acc.bind(acc.tsock);

    drv.gemm_jobs = {
        {"resnet8_conv0", 1024, 27, 16},
        {"dscnn_pw2", 125, 64, 64},
        {"mbnet_pw13", 9, 256, 256},
    };
    drv.conv_jobs = {
        /* name            H   W   C  R  S KOUT G  stride */
        {"conv3x3",         8,  8,  4, 3, 3, 8,   1, 1},
        {"dscnn_conv0",    49, 10,  1, 10, 4, 64, 1, 2},
        {"resnet8_conv3",  32, 32, 16, 3, 3, 32,  1, 2},
        {"mbnet_dw1",      48, 48,  8, 3, 3, 8,   8, 1},
        {"dscnn_dw1",      25,  5, 64, 3, 3, 64, 64, 1},
    };
    sc_start();
    return drv.failures ? 1 : 0;
}
