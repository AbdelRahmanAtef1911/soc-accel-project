#!/usr/bin/env python3
"""
check_rg_regression.py <step-5 reference> <tb_dma_rg a=0> <tb_dma_rg a=1> [<tb_dma_rg a=1 fast>]

Test 8 of the course-project plan (single master). In part 1 of tb_dma the CPU
sleeps while the accelerator's DMA runs, so the DMA is the only bus master
except for one moment: the DMA's first request comes right after the CPU's
START write, which is one ownership change (CPU -> DMA) per job.

  a = 0  every part-1 cycle count must equal the step-5 reference
  a = 1  every job total and load phase must be the reference + exactly 1
         cycle (that one handover); compute and store phases unchanged

  fast   (optional) the uncontested-CPU fast path gives output identical to
         the slow path (same a = 1 run), including part 2

Part 2 (a polling CPU = two masters) differs from step 5 by design and is only printed.
"""
import re
import sys


def part(path, first, stop):
    lines, on = [], False
    for ln in open(path):
        if ln.startswith(first):
            on = True
        elif on and (ln.startswith(stop) or "TESTS" in ln):
            break
        if on and ln.strip():
            lines.append(ln.rstrip("\n"))
    return lines


JOB = re.compile(r"\| one\s+(\d+) \| tiled x\d+\s+(\d+) \(=\s*(\d+)\+\s*(\d+)\+\s*(\d+)\) \| dbuf\s+(\d+)")


def jobs(lines):
    out = {}
    for ln in lines:
        m = JOB.search(ln)
        if m:
            out[ln.split()[0]] = [int(x) for x in m.groups()]  # one, tiled, load, compute, store, dbuf
        elif ln.startswith("tile sweep"):
            out["tile sweep"] = [int(x) for x in re.findall(r"tiles (\d+)", ln)]
    return out


ref, a0, a1 = (jobs(part(p, "Part 1:", "Part 2:")) for p in sys.argv[1:4])
ok = True
same0 = part(sys.argv[1], "Part 1:", "Part 2:") == part(sys.argv[2], "Part 1:", "Part 2:")
print(f"test 8, a = 0: part 1 identical to the step-5 reference ({len(ref)} rows)  [{'PASS' if same0 else 'FAIL'}]")
ok &= same0
bad = []
for name, r in ref.items():
    n = a1.get(name)
    if name == "tile sweep":
        exp = [x + 1 for x in r]
    else:  # one, tiled total, load, compute, store, dbuf
        exp = [r[0] + 1, r[1] + 1, r[2] + 1, r[3], r[4], r[5] + 1]
    if n != exp:
        bad.append(f"  {name}: got {n}, expected {exp}")
print(f"test 8, a = 1: every job and load phase = reference + 1 cycle, compute/store unchanged "
      f"({len(ref)} rows)  [{'PASS' if not bad else 'FAIL'}]")
for b in bad:
    print(b)
ok &= not bad
if len(sys.argv) > 4:
    strip = lambda p: [ln for ln in open(p) if not ln.startswith("bus:") and "SystemC" not in ln]
    same_fast = strip(sys.argv[3]) == strip(sys.argv[4])
    print(f"test 8, fast path: whole output identical to the slow path  [{'PASS' if same_fast else 'FAIL'}]")
    ok &= same_fast
for p in sys.argv[2:]:
    tail = [ln for ln in open(p) if "TESTS" in ln]
    print(f"{p}: {tail[-1].strip() if tail else 'no result line'}")
print("part 2 (polling CPU, two masters; differs by design):")
for ln in part(sys.argv[3], "Part 2:", "Part 3:")[1:3]:
    print("  a=1:", ln)
for ln in part(sys.argv[1], "Part 2:", "Part 3:")[1:3]:
    print("  ref:", ln)
sys.exit(0 if ok else 1)
