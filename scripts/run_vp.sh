#!/usr/bin/env bash
# run_vp.sh <program.elf> <result.csv> [accel32-vp options...]
#
# Runs one bare-metal program on the accel32-vp platform and keeps the CSV
# lines it prints. Fails (exit 1) if any layer result is FAIL or the program
# did not reach DONE. The full console log is saved next to the CSV (.log).
#
# Platform timing used for every run (keep it fixed when comparing results):
#   --intercept-syscalls  printing/exit handled by the VP
#   (default VP++ timing: every fetch, load and store goes through the bus
#    and the 1-cycle memory model; no DMI shortcuts)

set -e
source "$(dirname "$0")/../setup/env.sh"

elf=$1; out=$2; shift 2 || { echo "usage: run_vp.sh <elf> <csv> [vp options]"; exit 2; }
case "$elf" in /*) ;; *) elf="$PROJ/sw/$elf" ;; esac
case "$out" in /*) ;; *) out="$PROJ/results/$out" ;; esac
mkdir -p "$(dirname "$out")"
log="${out%.csv}.log"

run_dir=$(mktemp -d)  # STONNE's tile mapper writes small text files here
cp -r "$STONNE_DIR/parameters" "$run_dir/"  # and reads its parameter file from there
start=$(date +%s)
( cd "$run_dir" && accel32-vp --intercept-syscalls --quiet "$@" "$elf" ) > "$log" 2>&1 || true
rm -rf "$run_dir"
secs=$(( $(date +%s) - start ))

grep '^CSV,' "$log" | sed 's/^CSV,//' > "$out"
rows=$(( $(wc -l < "$out") - 1 ))
fails=$(grep -c ',FAIL$' "$out" || true)
info=$(grep '^INFO,' "$log" | head -1)
echo "$(basename "$out"): $rows rows, $fails FAIL, ${secs}s host time  [$info]"
grep -q '^DONE' "$log" || { echo "  program did not finish, see $log"; tail -5 "$log"; exit 1; }
[ "$fails" = "0" ] || { echo "  some layers FAILED, see $out"; exit 1; }
