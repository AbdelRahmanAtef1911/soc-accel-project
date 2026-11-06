# Project idea and insights

**Project:** Where Should the Work Go? HW/SW Partitioning and DMA Design for TinyML Inference on a RISC-V SystemC Platform
**Repository:** https://github.com/AbdelRahmanAtef1911/soc-accel-project

## The idea

A small chip with a CPU and an AI accelerator has to decide what runs in hardware, what stays in software, and how data moves between them. The whole chip was built in SystemC: a RISC-V CPU, a bus, memory, and an accelerator whose timing comes from STONNE. All 48 layers of the MLPerf Tiny models were run on it in five steps, each moving more work into hardware: software only, a simple accelerator, a better fabric, native convolution in hardware, and smarter data movement. At every step the time was split into its parts to see where it goes.

## The insights

1. **Speeding up what isn't slow does nothing.** A faster accelerator changed the total by 0.1%, because the CPU work was 96–99% of the time.
2. **Where the work runs matters more than how fast the engine is.** Moving convolution into hardware gave 1,389× over software, 42× more than the better fabric alone.
3. **Every fix moves the bottleneck.** First it was CPU work, then data movement (76–92% of the time), then sharing the bus.
4. **Once compute is cheap, moving data is the cost.** A wider DMA gave 1.5–1.7×; a faster compute fabric would have given almost nothing.
5. **Overlapping transfers with compute only hides the compute.** Double buffering gave up to 1.28×, and only 4–13% where the time is almost all transfers.
6. **An interrupt frees the CPU; it doesn't make the job faster.** It is 2% slower than polling on a private bus. It wins when the bus is shared, because a polling CPU steals bus time from the DMA (+9.7% vs +2.2% with cycle stealing).
7. **How the DMA shares the bus decides who waits.** Block mode keeps the accelerator fast but freezes the CPU (up to 26,624 cycles). Transparent mode keeps the CPU free but slows inference by up to 80%. Cycle stealing with large bursts gave the best balance: 1024-byte bursts cut accelerator time by 35% while keeping each CPU stall to one burst.
