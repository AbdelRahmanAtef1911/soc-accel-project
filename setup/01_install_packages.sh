#!/usr/bin/env bash
# Step 0.1 - install the build tools inside the course container (systemc-env).
# Run as root inside the container:   bash setup/01_install_packages.sh
set -e
export DEBIAN_FRONTEND=noninteractive

apt update
apt install -y git cmake build-essential python3 python3-pip python3-numpy python3-matplotlib \
    libboost-iostreams-dev libboost-program-options-dev libboost-log-dev \
    nlohmann-json3-dev libgoogle-perftools-dev libvncserver-dev \
    gcc-riscv64-unknown-elf

# optional: only needed to regenerate sw/layers.h from the .tflite models
pip install tflite --break-system-packages || echo "tflite not installed (optional)"

echo
echo "check:"
g++ --version | head -1
cmake --version | head -1
riscv64-unknown-elf-gcc --version | head -1
riscv64-unknown-elf-gcc -march=rv32im_zicsr -mabi=ilp32 -print-libgcc-file-name
ls /opt/systemc/lib/libsystemc.so && echo "SystemC 2.3.3 found in /opt/systemc"
echo "01_install_packages: OK"
