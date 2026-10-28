# accel32-vp: tiny RV32 platform + STONNE GEMM accelerator.
# Configure VP++ with:
#   -DACCEL_SRC_DIR=<project>/accel   (stonne_accel.h, stonne_backend.*, accel_regs.h)
#   -DSTONNE_DIR=<stonne repo>/stonne (its include/ and src/ folders)
set(ACCEL_SRC_DIR "" CACHE PATH "project accel/ folder")
set(STONNE_DIR "" CACHE PATH "STONNE stonne/ folder")

if(NOT ACCEL_SRC_DIR OR NOT STONNE_DIR)
	message(STATUS "accel32-vp skipped: configure with -DACCEL_SRC_DIR=... -DSTONNE_DIR=...")
	return()
endif()

# STONNE as a static library (its command-line main and testbench left out)
file(GLOB STONNE_SRC
	${STONNE_DIR}/src/*.cpp
	${STONNE_DIR}/src/TileGenerator/*.cpp
	${STONNE_DIR}/src/TileGenerator/mRNA/*.cpp
	${STONNE_DIR}/src/TileGenerator/StonneMapper/*.cpp
	${STONNE_DIR}/src/TileGenerator/Utils/*.cpp)
list(FILTER STONNE_SRC EXCLUDE REGEX ".*/src/(main|testbench)\\.cpp$")

add_library(stonne-accel STATIC ${STONNE_SRC} ${ACCEL_SRC_DIR}/stonne_backend.cpp)
target_include_directories(stonne-accel PRIVATE ${STONNE_DIR}/include ${STONNE_DIR}/external ${ACCEL_SRC_DIR})
target_compile_options(stonne-accel PRIVATE -w)

add_executable(accel32-vp main.cpp)
target_compile_definitions(accel32-vp PUBLIC TARGET_RV32)
target_include_directories(accel32-vp PRIVATE ${ACCEL_SRC_DIR})
target_link_libraries(accel32-vp rv32 platform-common gdb-mc stonne-accel ${Boost_LIBRARIES} systemc pthread)

install(TARGETS accel32-vp RUNTIME DESTINATION bin)
