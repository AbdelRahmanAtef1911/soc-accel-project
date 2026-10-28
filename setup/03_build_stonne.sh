#!/usr/bin/env bash
# Step 0.3 - build STONNE's own command-line simulator and run one GEMM on
# each of the three accelerators. Each run checks STONNE's output against
# a CPU GEMM and must print "Test passed correctly".
#   bash setup/03_build_stonne.sh
set -e
source "$(dirname "$0")/env.sh"

cd "$STONNE_DIR"
make -j"$(nproc)" all > /dev/null
echo "built $STONNE_DIR/stonne"

mkdir -p /tmp/stonne_check && cd /tmp/stonne_check
run() { # label, args...
    local label=$1; shift
    local out; out=$("$STONNE_DIR/stonne" "$@" 2>&1)
    local cyc; cyc=$(echo "$out" | grep -o "Number of cycles running: [0-9]*" | grep -o "[0-9]*$")
    if echo "$out" | grep -q "Test passed correctly"; then
        echo "  $label: $cyc cycles, result PASS"
    else
        echo "  $label: FAILED"; echo "$out" | tail -5; exit 1
    fi
}
echo "ResNet-8 first conv as a GEMM (M=1024, K=27, N=16), 256 multipliers:"
run "TPU 16x16  " -DenseGEMM -M=1024 -N=16 -K=27 -ms_rows=16 -ms_cols=16 -dn_bw=32 -rn_bw=256 \
    -T_N=16 -T_M=16 -T_K=1 -accumulation_buffer=1 -rn_type=TEMPORALRN -mn_type=OS_MESH -mem_ctrl=TPU_OS_DENSE
run "MAERI bw256" -FC -M=1024 -N=16 -K=27 -num_ms=256 -dn_bw=256 -rn_bw=256 \
    -generate_tile=performance -accumulation_buffer=1
run "SIGMA bw256" -SparseGEMM -M=1024 -N=16 -K=27 -num_ms=256 -dn_bw=256 -rn_bw=256 \
    -MK_sparsity=0 -KN_sparsity=0 -dataflow=MK_STR_KN_STA
rm -rf /tmp/stonne_check
echo "03_build_stonne: OK"
