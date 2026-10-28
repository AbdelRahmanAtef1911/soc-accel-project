/*
 * accel32-vp: RISC-V VP++ "tiny" platform + the STONNE GEMM accelerator.
 *
 * Copy this folder to <riscv-vp-plusplus>/vp/src/platform/accel (the setup
 * script does it). It is the tiny32-vp platform with one extra bus master
 * (the accelerator's DMA), one extra bus target (its registers and
 * scratchpads) and, for step 5, an interrupt controller (PLIC) that carries
 * the accelerator's completion interrupt to the core.
 *
 *   address map
 *   0x0000_0000 - 0x01FF_FFFF  main memory (32 MB)
 *   0x0200_0000 - 0x0200_FFFF  CLINT
 *   0x0201_0000 - 0x0201_03FF  syscall device
 *   0x0C00_0000 - 0x0CFF_FFFF  PLIC (FE310 layout), accelerator = source 1
 *   0x1000_0000 - 0x103F_FFFF  accelerator (accel/accel_regs.h)
 *
 *   bus initiators: 0 = CPU, 1 = debug interface, 2 = accelerator DMA
 *
 * Step 5 option --accel-bus-model puts an arbiter in front of the CPU's bus
 * port and gives the accelerator the same SharedBus (accel/shared_bus.h):
 * then CPU accesses and DMA bursts really compete for one bus, in the DMA
 * ownership mode chosen with --accel-dma-mode. Without the option the bus is
 * ideal (two masters never slow each other down), as in steps 1-4.
 *
 * Derived from vp/src/platform/tiny/main.cpp of RISC-V VP++ (MIT License,
 * University of Bremen and Johannes Kepler University Linz).
 */
#ifndef TARGET_RV32
#define TARGET_RV32
#endif

#include <boost/program_options.hpp>
#include <cstdlib>
#include <ctime>
#include <iostream>

#include "core/common/debug_memory.h"
#include "core/common/gdb-mc/gdb_runner.h"
#include "core/common/gdb-mc/gdb_server.h"
#include "core/common/real_clint.h"
#include "core/rv32/elf_loader.h"
#include "core/rv32/iss.h"
#include "core/rv32/mem.h"
#include "core/rv32/mmu.h"
#include "core/rv32/syscall.h"
#include "platform/common/fe310_plic.h"
#include "platform/common/memory.h"
#include "platform/common/options.h"
#include "util/propertytree.h"

#include "stonne_accel.h"  /* from the project's accel/ folder */

using namespace rv32;
namespace po = boost::program_options;

struct AccelOptions : public Options {
   public:
	typedef uint64_t addr_t;

	addr_t mem_size = 1024 * 1024 * 32;
	addr_t mem_start_addr = 0x00000000;
	addr_t mem_end_addr = mem_start_addr + mem_size - 1;
	addr_t clint_start_addr = 0x02000000;
	addr_t clint_end_addr = 0x0200ffff;
	addr_t sys_start_addr = 0x02010000;
	addr_t sys_end_addr = 0x020103ff;
	addr_t plic_start_addr = PLIC_BASE;
	addr_t plic_end_addr = PLIC_BASE + 0x00ffffff;
	addr_t accel_start_addr = ACCEL_BASE;
	addr_t accel_end_addr = ACCEL_BASE + ACCEL_SIZE - 1;

	bool quiet = false;

	/* accelerator (hardware) parameters */
	std::string arch = "tpu";
	unsigned rows = 16, cols = 16, num_ms = 256, dn_bw = 64, rn_bw = 0;
	unsigned dma_width = 4, dma_burst = 64, dma_latency = 10, reg_latency = 1;
	bool accel_verbose = false, accel_stats = false;
	bool bus_model = false;
	std::string dma_mode = "block";

	AccelOptions(void) {
		// clang-format off
		add_options()
			("quiet", po::bool_switch(&quiet), "do not output register values on exit")
			("memory-size", po::value<uint64_t>(&mem_size), "set memory size")
			("accel-arch", po::value<std::string>(&arch), "accelerator: tpu | maeri | sigma (default tpu)")
			("accel-rows", po::value<unsigned>(&rows), "TPU mesh rows (default 16)")
			("accel-cols", po::value<unsigned>(&cols), "TPU mesh columns (default 16)")
			("accel-ms", po::value<unsigned>(&num_ms), "MAERI/SIGMA multipliers, power of 2 (default 256)")
			("accel-dnbw", po::value<unsigned>(&dn_bw), "MAERI/SIGMA distribution bandwidth, power of 2 (default 64)")
			("accel-rnbw", po::value<unsigned>(&rn_bw), "MAERI/SIGMA reduction bandwidth (default = dnbw)")
			("accel-dma-width", po::value<unsigned>(&dma_width), "DMA bytes per cycle (default 4)")
			("accel-dma-burst", po::value<unsigned>(&dma_burst), "DMA burst size in bytes (default 64)")
			("accel-dma-latency", po::value<unsigned>(&dma_latency), "fixed cycles per DMA burst (default 10)")
			("accel-reg-latency", po::value<unsigned>(&reg_latency), "cycles per CPU access to the accelerator (default 1)")
			("accel-bus-model", po::bool_switch(&bus_model), "step 5: CPU and DMA share one arbitrated bus (default: ideal bus)")
			("accel-dma-mode", po::value<std::string>(&dma_mode), "step 5: DMA bus ownership with --accel-bus-model: block | steal | transparent (default block)")
			("accel-verbose", po::bool_switch(&accel_verbose), "show STONNE's console output")
			("accel-stats", po::bool_switch(&accel_stats), "let STONNE write its stats files for every job");
		// clang-format on
	}

	void parse(int argc, char **argv) override {
		Options::parse(argc, argv);
		mem_end_addr = mem_start_addr + mem_size - 1;
		if (rn_bw == 0) rn_bw = dn_bw;
	}

	AccelConfig accel_config() const {
		AccelConfig c;
		if (arch == "tpu") c.arch = ARCH_TPU;
		else if (arch == "maeri") c.arch = ARCH_MAERI;
		else if (arch == "sigma") c.arch = ARCH_SIGMA;
		else {
			std::cerr << "unknown --accel-arch " << arch << " (use tpu, maeri or sigma)" << std::endl;
			exit(1);
		}
		c.tpu_rows = rows;
		c.tpu_cols = cols;
		c.num_ms = num_ms;
		c.dn_bw = dn_bw;
		c.rn_bw = rn_bw;
		c.dma_bytes_per_cycle = dma_width;
		c.dma_burst_bytes = dma_burst;
		c.dma_burst_latency = dma_latency;
		if (dma_mode == "block") c.dma_mode = DMA_MODE_BLOCK;
		else if (dma_mode == "steal") c.dma_mode = DMA_MODE_STEAL;
		else if (dma_mode == "transparent") c.dma_mode = DMA_MODE_TRANSPARENT;
		else {
			std::cerr << "unknown --accel-dma-mode " << dma_mode << " (use block, steal or transparent)" << std::endl;
			exit(1);
		}
		c.reg_latency = reg_latency;
		c.stonne_verbose = accel_verbose;
		c.stonne_stats = accel_stats;
		return c;
	}
};

int sc_main(int argc, char **argv) {
	AccelOptions opt;
	opt.parse(argc, argv);

	std::srand(std::time(nullptr));

	if (!opt.property_tree_is_loaded) {
		/* 100 MHz for the CPU, memory and bus models */
		VPPP_PROPERTY_SET("", "clock_cycle_period", sc_core::sc_time, sc_core::sc_time(10, sc_core::SC_NS));
	}
	tlm::tlm_global_quantum::instance().set(sc_core::sc_time(opt.tlm_global_quantum, sc_core::SC_NS));

	RV_ISA_Config isa_config(opt.use_E_base_isa, opt.en_ext_Zfh, opt.en_ext_Zbc);
	ISS core(&isa_config, 0);
	MMU mmu(core);
	CombinedMemoryInterface core_mem_if("MemoryInterface0", core, &mmu);
	SimpleMemory mem("SimpleMemory", opt.mem_size);

	ELFLoader loader(opt.input_program.c_str());
	SimpleBus<3, 5> bus("SimpleBus", nullptr, false);
	SyscallHandler sys("SyscallHandler");
	DebugMemoryInterface dbg_if("DebugMemoryInterface");

	std::vector<clint_interrupt_target *> clint_targets{&core};
	RealCLINT clint("CLINT", clint_targets);

	StonneAccel accel("Accelerator", opt.accel_config());
	FE310_PLIC<1, 64, 96, 32> plic("PLIC");

	/* step 5: the accelerator's completion interrupt goes to PLIC source 1;
	 * the PLIC raises the core's machine external interrupt (MEIP) */
	plic.target_harts[0] = &core;
	accel.irq_line = [&plic](bool on) {
		if (on) plic.gateway_trigger_interrupt(ACCEL_IRQ_ID);
	};

	/* step 5: optional shared-bus model (CPU port arbiter + DMA reservations) */
	SharedBus shared_bus;
	BusArbiterPort *cpu_arbiter = nullptr;
	if (opt.bus_model) {
		cpu_arbiter = new BusArbiterPort("CPU-Bus-Arbiter", shared_bus, BUS_CPU, sc_core::sc_time(10, sc_core::SC_NS));
		accel.bus = &shared_bus;
		std::cout << "[accel32-vp] shared bus model on, DMA mode " << opt.dma_mode << std::endl;
	}

	std::shared_ptr<BusLock> bus_lock = std::make_shared<BusLock>();
	core_mem_if.bus_lock = bus_lock;
	mmu.mem = &core_mem_if;

	instr_memory_if *instr_mem_if = &core_mem_if;
	data_memory_if *data_mem_if = &core_mem_if;

	MemoryDMI dmi = MemoryDMI::create_start_size_mapping(mem.data, opt.mem_start_addr, mem.get_size());
	InstrMemoryProxy instr_mem(dmi, core);
	if (opt.use_instr_dmi) instr_mem_if = &instr_mem;
	core_mem_if.dmi_add(dmi);
	core_mem_if.dmi_enable(opt.use_data_dmi);

	loader.load_executable_image(mem, mem.get_size(), opt.mem_start_addr);
	core.init(instr_mem_if, opt.use_dbbcache, data_mem_if, opt.use_lscache, &clint, loader.get_entrypoint(),
	          opt.mem_end_addr);

	sys.init(mem.data, opt.mem_start_addr, loader.get_heap_addr(mem.get_size(), opt.mem_start_addr));
	sys.register_core(&core);
	if (opt.intercept_syscalls) core.sys = &sys;
	core.error_on_zero_traphandler = opt.error_on_zero_traphandler;

	/* address map */
	bus.ports[0] = new PortMapping(opt.mem_start_addr, opt.mem_end_addr, mem);
	bus.ports[1] = new PortMapping(opt.clint_start_addr, opt.clint_end_addr, clint);
	bus.ports[2] = new PortMapping(opt.sys_start_addr, opt.sys_end_addr, sys);
	bus.ports[3] = new PortMapping(opt.accel_start_addr, opt.accel_end_addr, accel);
	bus.ports[4] = new PortMapping(opt.plic_start_addr, opt.plic_end_addr, plic);
	bus.mapping_complete();

	/* initiators */
	if (cpu_arbiter) {
		core_mem_if.isock.bind(cpu_arbiter->tsock);
		cpu_arbiter->isock.bind(bus.tsocks[0]);
	} else {
		core_mem_if.isock.bind(bus.tsocks[0]);
	}
	dbg_if.isock.bind(bus.tsocks[1]);
	PeripheralWriteConnector accel_connector("Accelerator-DMA-Connector");  /* respects LR/SC bus lock */
	accel_connector.bus_lock = bus_lock;
	accel.isock.bind(accel_connector.tsock);
	accel_connector.isock.bind(bus.tsocks[2]);

	/* targets */
	bus.isocks[0].bind(mem.tsock);
	bus.isocks[1].bind(clint.tsock);
	bus.isocks[2].bind(sys.tsock);
	bus.isocks[3].bind(accel.tsock);
	bus.isocks[4].bind(plic.tsock);

	core.enable_trace(opt.trace_mode);

	std::vector<debug_target_if *> threads;
	threads.push_back(&core);
	if (opt.use_debug_runner) {
		auto server = new GDBServer("GDBServer", threads, &dbg_if, opt.debug_port, opt.debug_cont_sim_on_wait);
		new GDBServerRunner("GDBRunner", server, &core);
	} else {
		new DirectCoreRunner(core);
	}

	opt.handle_property_export_and_exit();
	if (opt.quiet) sc_core::sc_report_handler::set_verbosity_level(sc_core::SC_NONE);

	sc_core::sc_start();
	if (!opt.quiet) core.show();
	std::cout << "[accel32-vp] simulated time " << sc_core::sc_time_stamp() << std::endl;
	if (opt.bus_model) {
		const double c = 10000.0; /* ps per cycle */
		std::cout << "[accel32-vp] bus: CPU " << shared_bus.stats[BUS_CPU].txn << " transactions, waited "
		          << (uint64_t)(shared_bus.stats[BUS_CPU].wait / c) << " cycles; DMA "
		          << shared_bus.stats[BUS_DMA].txn << " tenures, busy " << (uint64_t)(shared_bus.stats[BUS_DMA].busy / c)
		          << " cycles, waited " << (uint64_t)(shared_bus.stats[BUS_DMA].wait / c) << " cycles" << std::endl;
	}
	return 0;
}
