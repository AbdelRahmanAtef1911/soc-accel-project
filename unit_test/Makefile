# Step 3: build the accelerator unit test against the course SystemC 2.3.3.
#   make               build tb_accel and tb_dma (first time also compiles STONNE, ~1-3 min)
#   make test          step 3: run tb_accel for TPU, MAERI and SIGMA
#   make test5         step 5: run tb_dma (tiling, double buffering, interrupt, shared bus)
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

all: tb_accel tb_dma

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

clean:
	rm -rf $(BUILD) tb_accel tb_dma parameters output_stats_* StonneMapper_output_* mRNA_output_*

.PHONY: all test test5 clean
