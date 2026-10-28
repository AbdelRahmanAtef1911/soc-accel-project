#!/usr/bin/env bash
# Step 0.4 - add the accelerator platform to RISC-V VP++ and build accel32-vp.
# First build: about 5-10 minutes (it also builds VP++'s own SystemC).
#   bash setup/04_build_vp.sh
set -e
source "$(dirname "$0")/env.sh"

# 1. put the platform folder into VP++ (VP++ builds every folder in platform/)
rm -rf "$VPPP_DIR/vp/src/platform/accel"
cp -r "$PROJ/vp_platform/accel" "$VPPP_DIR/vp/src/platform/accel"

# 2. configure: tell the platform where the accelerator model and STONNE are
mkdir -p "$VPPP_DIR/vp/build"
cd "$VPPP_DIR/vp/build"
cmake -DCMAKE_BUILD_TYPE=Release \
      -DACCEL_SRC_DIR="$PROJ/accel" \
      -DSTONNE_DIR="$STONNE_DIR" .. > cmake.log

# 3. build only what we need
if ! make -j"$(nproc)" accel32-vp > build.log 2>&1; then
    grep -m 20 -E "error" build.log; echo "build failed, full log: $VPPP_DIR/vp/build/build.log"; exit 1
fi

echo "built $(pwd)/bin/accel32-vp"
bin/accel32-vp --help | grep accel-arch
echo "04_build_vp: OK"
