#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_ob13_make.py - derives the SUMMER BLOCK version from gg2_ob8_arms.py.

WHY IT IS DONE THIS WAY
  gg2_ob8_arms.py is 420 lines and produced the LOCKED numbers. Copying it by
  hand carries a risk of silent drift. This script reads the original, applies
  the FIVE CHANGES LISTED BELOW, and writes gg2_ob13_season.py.
  If any pattern does not match EXACTLY ONCE, the script STOPS and writes no
  file.

CHANGES (this is all of them)
  1. The --t0 / --t1 arguments are added.
  2. T0, T1 are read from the arguments instead of being fixed
     (default: the old behaviour).
  3. Output file ob8_{arm}.npz -> ob13_{T0}_{arm}.npz
     (the LOCKED ob8_*.npz files are NEVER touched.)
  4. If run outside the test block, a warning is printed that the violation
     decomposition is INVALID (REQ_BY_STEP is keyed by test-block steps).
  5. In that same case the "uncontrolled 15.11%" line is not printed (that
     number belongs to the test block; it was NOT MEASURED for another block).

  Controller, k*, battery, safe layer, w_econ, H, MARGIN, LDF: UNTOUCHED.

RUN (from the ROOT folder):
  python gg2/gg2_ob13_make.py

ASCII only.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path("gg2/gg2_ob8_arms.py")
DST = Path("gg2/gg2_ob13_season.py")

HEADER = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# AUTO-GENERATED: gg2_ob8_arms.py -> gg2_ob13_make.py -> this file.
# DO NOT EDIT BY HAND. If a change is needed, edit gg2_ob13_make.py and
# regenerate.
#
# SOLE PURPOSE: run the same controller on a DIFFERENT TIME BLOCK.
# With default arguments it must give the SAME result as gg2_ob8_arms.py
# (ANCHOR).
# ---------------------------------------------------------------------------
'''

EDITS = [
    (
        "argparse: --t0 / --t1",
        '    ap.add_argument("--resume", action="store_true")\n'
        '    ap.add_argument("--nwp", default=NPZ_NWP)\n'
        '    a = ap.parse_args()\n',
        '    ap.add_argument("--resume", action="store_true")\n'
        '    ap.add_argument("--nwp", default=NPZ_NWP)\n'
        '    ap.add_argument("--t0", type=int, default=None,\n'
        '                    help="start of the evaluation block '
        '(default: M.SPLIT)")\n'
        '    ap.add_argument("--t1", type=int, default=None,\n'
        '                    help="end of the evaluation block '
        '(default: M.N-1)")\n'
        '    a = ap.parse_args()\n',
    ),
    (
        "T0/T1 from the arguments",
        "    T0, T1 = int(M.SPLIT), int(M.N) - 1\n",
        "    T0 = int(M.SPLIT) if a.t0 is None else int(a.t0)\n"
        "    T1 = (int(M.N) - 1) if a.t1 is None else int(a.t1)\n"
        "    if not (0 <= T0 < T1 <= int(M.N)):\n"
        '        sys.exit("[ERROR] invalid slice: %d:%d (N=%d)"\n'
        "                 % (T0, T1, int(M.N)))\n",
    ),
    (
        "output file name",
        '        out = Path(os.path.join(OUT_DIR, "ob8_%s.npz" % arm))\n',
        '        out = Path(os.path.join(OUT_DIR, "ob13_%d_%s.npz" % (T0, arm)))\n',
    ),
    (
        "violation decomposition warning",
        '        print("  violation decomposition '
        '(not a tautology, vs TASK 3 measurement):")\n',
        "        if T0 != int(M.SPLIT):\n"
        '            print("  [WARNING] violation decomposition is INVALID '
        'in this block:")\n'
        '            print("            REQ_BY_STEP is keyed by the test-block '
        'steps,")\n'
        '            print("            so here everything looks SPATIAL. '
        'IGNORE IT.")\n'
        '        print("  violation decomposition '
        '(not a tautology, vs TASK 3 measurement):")\n',
    ),
    (
        "uncontrolled baseline",
        '    print("uncontrolled (no battery, TASK 2): 15.11%")\n',
        "    if T0 == int(M.SPLIT):\n"
        '        print("uncontrolled (no battery, TASK 2): 15.11%")\n'
        "    else:\n"
        '        print("uncontrolled baseline: NOT MEASURED FOR THIS BLOCK")\n',
    ),
]


def main() -> None:
    print("=" * 70)
    print("gg2_ob13_make.py - deriving the summer-block version")
    print("=" * 70)

    if not SRC.exists():
        sys.exit("[ERROR] no source: %s  (run from the ROOT folder)" % SRC)

    src = SRC.read_text(encoding="utf-8", errors="strict")
    print("\nsource: %s  (%d lines)" % (SRC, src.count("\n")))

    out = src
    for i, (name, old, new) in enumerate(EDITS, 1):
        c = out.count(old)
        print("\n[%d/%d] %s" % (i, len(EDITS), name))
        print("      match count: %d" % c)
        if c != 1:
            print("\n[STOP] Pattern did not match exactly once.")
            print("[STOP] Searched for:")
            for ln in old.rstrip("\n").split("\n"):
                print("        |" + ln)
            print("[STOP] No file was written.")
            sys.exit(1)
        out = out.replace(old, new)
        print("      applied.")

    out = HEADER + out

    bad = [b for b in out.encode("utf-8") if b > 127]
    print("\nnon-ASCII bytes: %d" % len(bad))
    if bad:
        print("[WARNING] non-ASCII characters present (Windows parser risk).")

    import ast
    try:
        ast.parse(out)
        print("syntax: OK")
    except SyntaxError as e:
        sys.exit("[STOP] syntax error in the generated file: %s" % e)

    DST.write_text(out, encoding="ascii", errors="strict")
    print("\nWRITTEN: %s  (%d lines)" % (DST, out.count("\n")))

    print("\n" + "=" * 70)
    print("NEXT STEP - ANCHOR (default slice, same result as ob8):")
    print("  python gg2/gg2_ob13_season.py --arm nwp 2>&1 "
          "| tee gg2/ob13_anchor.log")
    print("  EXPECTED: violations 6.34%  |  profit 45609.7 EUR")
    print("  If this does not hold, DO NOT proceed to the summer run.")
    print("=" * 70)


if __name__ == "__main__":
    main()
