# HW/SW partitioning of MLPerf Tiny on a RISC-V SystemC platform

COE 407/661 System on Chip term project. Start with **docs/Guide.pdf**: it walks through
setup, every step, what each step does and how to verify it. The term paper is in
**docs/paper/** (`main.pdf`) and the proposal in **docs/proposal/**.

The project runs the convolution, depthwise and fully-connected layers of the three
MLPerf Tiny models on a RISC-V instruction-set simulator inside SystemC, first in
software, then on a simple accelerator, then on a better one, and explains where the
time goes. The accelerator is a SystemC/TLM-2.0 module whose compute timing comes from
the STONNE cycle-level simulator (anchor paper: Muñoz-Martínez et al., IEEE Computer
Architecture Letters, 2021).

## Folders

| Folder | What is in it |
|---|---|
| `docs/` | Guide.pdf (setup and every step); `paper/` the term paper (LaTeX, IEEE format, figures and tables generated from the results); `proposal/` the two-page proposal (LaTeX, .docx, .pdf) |
| `setup/` | scripts 01-05: packages, sources, STONNE build, VP build, shell setup |
| `accel/` | the accelerator: SystemC model, STONNE backend, register map |
| `unit_test/` | step A (`tb_accel`) and step 5 (`tb_dma`): the accelerator alone on the course SystemC 2.3.3 |
| `vp_platform/accel/` | the RISC-V VP++ platform with the accelerator (accel32-vp) |
| `sw/` | bare-metal C: kernels, driver, benchmark, layer table |
| `workloads/` | MLPerf Tiny layer extraction and the extracted layer table |
| `scripts/` | step scripts 0-5, run helper, summaries, DSE sweeps (`dse.py`, `dse5.py`), plots |
| `patches/` | one-line STONNE patch (exposes the cycle count) |
| `reference_results/` | the results these scripts produced when the package was tested |
| `results/` | your results go here |

## Quick start (inside the course container, after reading the guide)

```bash
cd /workspace/soc-accel-project
bash setup/01_install_packages.sh
bash setup/02_get_sources.sh
bash setup/03_build_stonne.sh
bash setup/04_build_vp.sh
source setup/env.sh
(cd unit_test && make test)               # step A
bash scripts/step0_reproduce_stonne.sh    # step 0
bash scripts/step1_software.sh            # step 1
bash scripts/step2_simple_accel.sh        # step 2
bash scripts/step3_better_accel.sh        # step 3
python3 scripts/dse.py                    # step 4
python3 scripts/plot_results.py
(cd unit_test && make test5)              # step 5: DMA/bus extension, model alone
bash scripts/step5_dma_extension.sh       # step 5: on the platform + sweeps + figure 4
```

## Step 5: the DMA and bus extension

Once the layers run on the accelerator (step 3b), more than half of the time is data
movement. Step 5 asks how the DMA should work, with three SoC-level changes to the
same accelerator. None of them is active unless software asks for it, so steps 1-4 give
exactly the same numbers as before.

* **Tiling and double buffering** (`CTRL_TILE`, `CTRL_DBUF`, `TILE_ROWS`): the job is cut
  into tiles; a separate PE thread computes tile *i* while the DMA engine loads tile
  *i+1* and stores tile *i-1* (ping-pong halves of scratchpads A and C).
* **Completion interrupt** (`CTRL_IRQ`, `IRQ_STATUS`): the accelerator raises PLIC
  source 1, the CPU sleeps in `WFI` or does other work instead of polling `STATUS`
  (`sw/trap.S`, `sw/irq.c`).
* **Shared bus with real contention** (`--accel-bus-model`, `accel/shared_bus.h`): the CPU
  and the DMA take turns on one arbitrated bus, with the DMA bus-ownership mode of the
  course (`--accel-dma-mode block | steal | transparent`) and a programmable burst size.

Updating an existing setup (the container already built in steps 0-4): copy the new
files over the project folder, then

```bash
source setup/env.sh
bash setup/04_build_vp.sh                 # rebuild accel32-vp (about 1 minute)
(cd unit_test && make && make test && make test5)
bash scripts/step5_dma_extension.sh
```

Tested commits: STONNE 22634b8, RISC-V VP++ 9156184, MLPerf Tiny 4addd0f.

## Rebuilding the paper's figures and tables from your results

```bash
python3 docs/paper/make_figures.py results docs/paper/figures
python3 docs/paper/make_tables.py  results docs/paper/tables
cd docs/paper && latexmk -pdf main.tex      # or upload docs/paper to Overleaf
```

## Licences

Code written for this project: free to use for the course and the paper. `vp_platform/accel/main.cpp` is derived from RISC-V VP++ (MIT). STONNE (MIT), RISC-V VP++ (MIT) and MLPerf Tiny (Apache-2.0) are downloaded by the setup scripts under their own licences.
