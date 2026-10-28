#!/usr/bin/env bash
# Step 0.2 - download STONNE, RISC-V VP++ and the MLPerf Tiny models at the
# exact commits this package was tested with, and patch STONNE.
#   bash setup/02_get_sources.sh
set -e
source "$(dirname "$0")/env.sh"

STONNE_COMMIT=22634b8b7668c2691bc2df8f612232d90839df20   # stonne-simulator/stonne, 16 Jun 2025
VPPP_COMMIT=91561841fa74ee4e0fd7313a3f7034f9fe63fad4     # ics-jku/riscv-vp-plusplus, 19 Oct 2026
TINY_COMMIT=4addd0fa08d216e20637637874e084895f289da4     # mlcommons/tiny, 13 Jul 2026

mkdir -p "$EXT"

fetch() { # fetch <url> <dir> <commit>
    if [ -d "$2/.git" ]; then echo "$2 already exists, skipping"; return; fi
    git init -q "$2"
    git -C "$2" remote add origin "$1"
    git -C "$2" fetch -q --depth 1 origin "$3"
    git -C "$2" checkout -q FETCH_HEAD
    echo "$2 at $(git -C "$2" rev-parse --short HEAD)"
}

# 1. STONNE + the one-line patch that exposes the cycle count
fetch https://github.com/stonne-simulator/stonne.git "$EXT/stonne" "$STONNE_COMMIT"
if git -C "$EXT/stonne" apply --check "$PROJ/patches/stonne-accel.patch" 2>/dev/null; then
    git -C "$EXT/stonne" apply "$PROJ/patches/stonne-accel.patch"
    echo "STONNE patched (getNCycles)"
else
    grep -q getNCycles "$STONNE_DIR/include/STONNEModel.h" && echo "STONNE already patched"
fi

# 2. RISC-V VP++ and the two submodules the build needs (vendored SystemC, mpc)
fetch https://github.com/ics-jku/riscv-vp-plusplus.git "$VPPP_DIR" "$VPPP_COMMIT"
git -C "$VPPP_DIR" submodule update --init --depth 1 \
    vp/src/vendor/systemc vp/src/core/common/gdb-mc/libgdb/mpc

# 3. MLPerf Tiny model files (only the trained models, not the whole repo)
if [ ! -d "$TINY_DIR/.git" ]; then
    git clone -q --filter=blob:none --sparse https://github.com/mlcommons/tiny.git "$TINY_DIR"
    git -C "$TINY_DIR" sparse-checkout set \
        benchmark/training/image_classification/trained_models \
        benchmark/training/keyword_spotting/trained_models \
        benchmark/training/visual_wake_words/trained_models
    git -C "$TINY_DIR" checkout -q "$TINY_COMMIT" || true
fi
ls "$TINY_DIR"/benchmark/training/*/trained_models/*.tflite | head -3

echo "02_get_sources: OK"
