#!/usr/bin/env bash
# STEP 0 - reproduce the anchor paper's kind of result with STONNE alone
# (no SystemC, no CPU): the same layers on a rigid TPU-like mesh, the flexible
# MAERI fabric and the sparse SIGMA fabric, all with 256 multipliers.
# The paper's claim to check: flexible fabrics keep their multipliers busier
# on irregular layer shapes, and SIGMA gains from zeros while the others do not.
#
#   bash scripts/step0_reproduce_stonne.sh          8 representative layers (~3 min)
#   ALL=1 bash scripts/step0_reproduce_stonne.sh    every GEMM-shaped MLPerf Tiny layer
#
# Output: results/step0_stonne.csv and a table on screen.
set -e
cd "$(dirname "$0")/.."
source setup/env.sh
out=results/step0_stonne.csv
mkdir -p results
work=$(mktemp -d); cp -r "$STONNE_DIR/parameters" "$work/"; cd "$work"

cyc() { "$STONNE_DIR/stonne" "$@" 2>&1 | grep -o "Number of cycles running: [0-9]*" | grep -o "[0-9]*$" || echo NA; }

if [ -n "$ALL" ]; then
    shapes=$(awk -F, 'NR>1 && $3!="DW" {print $2","$16","$17","$18}' "$PROJ/workloads/layers_mlperf_tiny.csv")
else
    shapes="resnet8_conv0,1024,27,16
resnet8_conv1,1024,144,16
resnet8_conv7,64,576,64
dscnn_conv0,125,40,64
dscnn_pw2,125,64,64
mbnet_pw2,2304,8,16
mbnet_pw14,36,128,128
mbnet_pw26,9,256,256"
fi

echo "layer,M,K,N,macs,tpu16x16,maeri256_bw256,sigma256_bw256_dense,sigma256_bw256_50pct_zero_weights" > "$PROJ/$out"
printf "%-16s %5s %4s %4s | %9s %9s %9s %9s | util TPU / MAERI\n" layer M K N TPU MAERI SIGMA SIGMA50
echo "$shapes" | while IFS=, read name M K N; do
    t=$(cyc -DenseGEMM -M=$M -N=$N -K=$K -ms_rows=16 -ms_cols=16 -dn_bw=32 -rn_bw=256 -T_N=16 -T_M=16 -T_K=1 \
            -accumulation_buffer=1 -rn_type=TEMPORALRN -mn_type=OS_MESH -mem_ctrl=TPU_OS_DENSE)
    m=$(cyc -FC -M=$M -N=$N -K=$K -num_ms=256 -dn_bw=256 -rn_bw=256 -generate_tile=performance -accumulation_buffer=1)
    s=$(cyc -SparseGEMM -M=$M -N=$N -K=$K -num_ms=256 -dn_bw=256 -rn_bw=256 -MK_sparsity=0 -KN_sparsity=0 -dataflow=MK_STR_KN_STA)
    z=$(cyc -SparseGEMM -M=$M -N=$N -K=$K -num_ms=256 -dn_bw=256 -rn_bw=256 -MK_sparsity=0 -KN_sparsity=50 -dataflow=MK_STR_KN_STA)
    macs=$((M * K * N))
    echo "$name,$M,$K,$N,$macs,$t,$m,$s,$z" >> "$PROJ/$out"
    printf "%-16s %5d %4d %4d | %9s %9s %9s %9s | %5.1f%% / %5.1f%%\n" $name $M $K $N $t $m $s $z \
        $(python3 -c "print(100*$macs/256/$t, 100*$macs/256/$m)")
done
cd "$PROJ"; rm -rf "$work"
echo "wrote $out"
