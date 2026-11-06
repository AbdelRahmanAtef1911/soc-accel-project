# Step 3: build the accelerator unit test against the course SystemC 2.3.3.
#   make               build tb_accel and tb_dma (first time also compiles STONNE, ~1-3 min)
#   make test          step 3: run tb_accel for TPU, MAERI and SIGMA
#   make test5         step 5: run tb_dma (tiling, double buffering, interrupt, shared bus)
#   make test_rg       course project: request/grant bus tests 1-8 (tb_rg, tb_dma_rg)
# Paths can be overridden: make STONNE_DIR=/somewhere/stonne/stonne

SYSTEMC_HOME ?= /opt/systemc
STONNE_DIR   ?= $(or $(EXT),/opt/soc-ext)/stonne/stonne
ACCEL        := ../accel
BUILD        := build

STONNE_SRC := $(filter-out %/src/main.cpp %/src/testbench.cpp, \
              $(wildcard $(STONNE_DIR)/src/*.cpp) \
              $(wildcard $(STONNE_DIR)/src/TileGenerator/*.cpp) \
              $(wildcard $(STONNE_DIR)/src/TileGenerator/mRNA/*.cpp) \
              $(wildcard $(STONNE_DIR)/src/TileGenerator/StonneMapper/*.cpp) \
              $(wildcard $(STONNE_DIR)/src/TileGenerator/Utils/*.cpp))
STONNE_OBJ := $(patsubst $(STONNE_DIR)/src/%.cpp,$(BUILD)/stonne/%.o,$(STONNE_SRC))

# STONNE is plain C++17; the SystemC side must be C++20 to match /opt/systemc
CXX_STONNE := g++ -std=c++17 -O3 -w -I$(STONNE_DIR)/include -I$(STONNE_DIR)/external -I$(ACCEL)
CXX_SC     := g++ -std=c++20 -O2 -Wall -I$(SYSTEMC_HOME)/include -I$(ACCEL)

all: tb_accel tb_dma tb_dma_rg tb_rg

$(BUILD)/stonne/%.o: $(STONNE_DIR)/src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX_STONNE) -c $< -o $@

$(BUILD)/stonne_backend.o: $(ACCEL)/stonne_backend.cpp $(ACCEL)/stonne_backend.h
	@mkdir -p $(BUILD)
	$(CXX_STONNE) -c $< -o $@

$(BUILD)/tb_accel.o: tb_accel.cpp $(ACCEL)/stonne_accel.h $(ACCEL)/accel_regs.h $(ACCEL)/stonne_backend.h $(ACCEL)/shared_bus.h
	@mkdir -p $(BUILD)
	$(CXX_SC) -c $< -o $@

$(BUILD)/tb_dma.o: tb_dma.cpp $(ACCEL)/stonne_accel.h $(ACCEL)/accel_regs.h $(ACCEL)/stonne_backend.h $(ACCEL)/shared_bus.h
	@mkdir -p $(BUILD)
	$(CXX_SC) -c $< -o $@

$(BUILD)/tb_dma_rg.o: tb_dma.cpp $(ACCEL)/stonne_accel.h $(ACCEL)/accel_regs.h $(ACCEL)/stonne_backend.h $(ACCEL)/rg_bus.h
	@mkdir -p $(BUILD)
	$(CXX_SC) -DRG_BUS -c $< -o $@

tb_dma_rg: $(BUILD)/tb_dma_rg.o $(BUILD)/stonne_backend.o $(STONNE_OBJ)
	$(CXX_SC) $^ -L$(SYSTEMC_HOME)/lib -lsystemc -o $@
	chmod +x $@

# course project: request/grant bus unit tests (no STONNE needed)
tb_rg: tb_rg.cpp $(ACCEL)/rg_bus.h
	$(CXX_SC) $< -L$(SYSTEMC_HOME)/lib -lsystemc -o $@
	chmod +x $@

tb_accel: $(BUILD)/tb_accel.o $(BUILD)/stonne_backend.o $(STONNE_OBJ)
	$(CXX_SC) $^ -L$(SYSTEMC_HOME)/lib -lsystemc -o $@
	chmod +x $@   # NTFS mounts drop the execute bit

tb_dma: $(BUILD)/tb_dma.o $(BUILD)/stonne_backend.o $(STONNE_OBJ)
	$(CXX_SC) $^ -L$(SYSTEMC_HOME)/lib -lsystemc -o $@
	chmod +x $@

test: tb_accel
	[ -d parameters ] || cp -r $(STONNE_DIR)/parameters .   # STONNE's tile mapper reads parameters/ from the run folder
	LD_LIBRARY_PATH=$(SYSTEMC_HOME)/lib ./tb_accel tpu 256
	LD_LIBRARY_PATH=$(SYSTEMC_HOME)/lib ./tb_accel maeri 256 256
	LD_LIBRARY_PATH=$(SYSTEMC_HOME)/lib ./tb_accel sigma 256 64

test5: tb_dma
	[ -d parameters ] || cp -r $(STONNE_DIR)/parameters .
	LD_LIBRARY_PATH=$(SYSTEMC_HOME)/lib ./tb_dma maeri 256 256
	LD_LIBRARY_PATH=$(SYSTEMC_HOME)/lib ./tb_dma tpu 256

# course project, week 1: arbiter tests 1-7, then test 8 (single master on the
# request/grant bus gives the step-5 part-1 cycle counts: identical with handover
# a = 0, and exactly +a per job with a = 1 - the one ownership change after the
# CPU's START write)
test_rg: tb_rg tb_dma_rg
	LD_LIBRARY_PATH=$(SYSTEMC_HOME)/lib ./tb_rg
	[ -d parameters ] || cp -r $(STONNE_DIR)/parameters .
	LD_LIBRARY_PATH=$(SYSTEMC_HOME)/lib ./tb_dma_rg maeri 256 256 0 > tb_dma_rg_a0_output.txt 2>&1
	LD_LIBRARY_PATH=$(SYSTEMC_HOME)/lib ./tb_dma_rg maeri 256 256 1 > tb_dma_rg_a1_output.txt 2>&1
	LD_LIBRARY_PATH=$(SYSTEMC_HOME)/lib ./tb_dma_rg maeri 256 256 1 fast > tb_dma_rg_a1_fast_output.txt 2>&1
	python3 check_rg_regression.py ../reference_results/unit_test5_output.txt tb_dma_rg_a0_output.txt tb_dma_rg_a1_output.txt tb_dma_rg_a1_fast_output.txt

clean:
	rm -rf $(BUILD) tb_accel tb_dma tb_dma_rg tb_rg tb_dma_rg_a0_output.txt tb_dma_rg_a1_output.txt tb_dma_rg_a1_fast_output.txt parameters output_stats_* StonneMapper_output_* mRNA_output_*

.PHONY: all test test5 test_rg clean
