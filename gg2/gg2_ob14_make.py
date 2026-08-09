#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_ob14_make.py - derives the PARAMETRIC bootstrap version from
gg2_ob9_boot.py.

WHY IT IS DONE THIS WAY
  gg2_ob9_boot.py produced the LOCKED headline numbers and contains the
  case33bw anchor (B1). Copying it by hand carries a risk of silent drift.
  This script reads the original, applies the CHANGES BELOW, and writes
  gg2_ob14_boot.py.

THE CORE PROBLEM AND THE SOLUTION
  In the original, PRICE is a single global: M.price[T0 : T0+NUSE], T0 = M.SPLIT.
  Both the B1 anchor and the oberrhein arms use the same load().
  The summer block needs T0=0 -- but the anchor must stay on the OLD slice,
  otherwise it computes with the wrong price and produces a FAKE failure.
  Solution: TWO price slices. PRICE (anchor, unchanged) and PRICE_OB (arms).

CHANGES
  Z1 --prefix / --t0 arguments (below the SEED line)
  Z2 the OB dict is built from the prefix (ob8_ -> whatever is requested)
  Z3 PRICE is split in two: PRICE (anchor) + PRICE_OB (arms)
  Z4 load(path) -> load(path, price=None)
  Z5 the profit computation uses the given price slice
  Z6 the oberrhein arms are loaded with PRICE_OB
  Z7 the placebo file comes from the prefix, with PRICE_OB
  Z8 (COSMETIC, not required) the "2429 steps" text is neutralised

  The B1 ANCHOR, SEED, NBOOT, NDAY, DAY, compare(): UNTOUCHED.
  With default arguments the file behaves the SAME as the original.

RUN (from the ROOT folder):
  python gg2/gg2_ob14_make.py

ASCII only.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path("gg2/gg2_ob9_boot.py")
DST = Path("gg2/gg2_ob14_boot.py")

HEADER = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# AUTO-GENERATED: gg2_ob9_boot.py -> gg2_ob14_make.py -> this file.
# DO NOT EDIT BY HAND.
#
# SOLE PURPOSE: apply the same bootstrap to the trajectories of a DIFFERENT
# TIME BLOCK. The B1 anchor (the case33bw headline) is UNCHANGED and runs ON
# EVERY RUN.
# With default arguments it must give the SAME output as gg2_ob9_boot.py.
#
# USAGE:
#   python gg2/gg2_ob14_boot.py                          (original behaviour)
#   python gg2/gg2_ob14_boot.py --prefix ob13_0_ --t0 0  (summer block)
# ---------------------------------------------------------------------------
'''

# (name, required, old, new)
EDITS = [
    (
        "Z1 arguments",
        True,
        "SEED = 12345\n",
        "SEED = 12345\n"
        '_PFX = "ob8_"\n'
        "_T0OB = None\n"
        "for _i in range(len(sys.argv) - 1):\n"
        '    if sys.argv[_i] == "--prefix":\n'
        "        _PFX = sys.argv[_i + 1]\n"
        '    if sys.argv[_i] == "--t0":\n'
        "        _T0OB = int(sys.argv[_i + 1])\n",
    ),
    (
        "Z2 OB dict from the prefix",
        True,
        'OB = {"naive": os.path.join(_HERE, "data", "ob8_naive.npz"),\n'
        '      "nwp": os.path.join(_HERE, "data", "ob8_nwp.npz"),\n'
        '      "oracle": os.path.join(_HERE, "data", "ob8_oracle.npz")}\n',
        'OB = {"naive": os.path.join(_HERE, "data", _PFX + "naive.npz"),\n'
        '      "nwp": os.path.join(_HERE, "data", _PFX + "nwp.npz"),\n'
        '      "oracle": os.path.join(_HERE, "data", _PFX + "oracle.npz")}\n',
    ),
    (
        "Z3 two price slices",
        True,
        "PRICE = np.asarray(M.price, dtype=float)[T0:T0 + NUSE]\n",
        "_PRICE_ALL = np.asarray(M.price, dtype=float)\n"
        "PRICE = _PRICE_ALL[T0:T0 + NUSE]\n"
        "_T0OB = T0 if _T0OB is None else _T0OB\n"
        "if _T0OB + NUSE > len(_PRICE_ALL):\n"
        '    sys.exit("[ERROR] t0=%d + %d steps exceed the series (N=%d)"\n'
        "             % (_T0OB, NUSE, len(_PRICE_ALL)))\n"
        "PRICE_OB = _PRICE_ALL[_T0OB:_T0OB + NUSE]\n"
        'print("anchor block T0=%d | arm block T0=%d | file prefix %s"\n'
        "      % (T0, _T0OB, _PFX))\n",
    ),
    (
        "Z4 load signature",
        True,
        "def load(path):\n",
        "def load(path, price=None):\n",
    ),
    (
        "Z5 profit computation",
        True,
        "    day_prof = (-(PRICE * p[:NUSE]) * dt)"
        ".reshape(NDAY, DAY).sum(axis=1)\n",
        "    pr = PRICE if price is None else price\n"
        "    day_prof = (-(pr * p[:NUSE]) * dt)"
        ".reshape(NDAY, DAY).sum(axis=1)\n",
    ),
    (
        "Z6 oberrhein arms with PRICE_OB",
        True,
        "    r = load(v)\n"
        "    if r is None:\n"
        '        print("[ERROR] %s is incomplete" % v)\n',
        "    r = load(v, PRICE_OB)\n"
        "    if r is None:\n"
        '        print("[ERROR] %s is incomplete" % v)\n',
    ),
    (
        "Z7a placebo file",
        True,
        'PB = os.path.join(_HERE, "data", "ob8_placebo.npz")\n',
        'PB = os.path.join(_HERE, "data", _PFX + "placebo.npz")\n',
    ),
    (
        "Z7b placebo with PRICE_OB",
        True,
        "    r = load(PB)\n",
        "    r = load(PB, PRICE_OB)\n",
    ),
    (
        "Z8 cosmetic text",
        False,
        '    print("  %-7s 50 days: violations %.2f%% | profit %.1f EUR   '
        '(2429 steps"\n',
        '    print("  %-7s 50 days: violations %.2f%% | profit %.1f EUR   '
        '(full block"\n',
    ),
]


def main() -> None:
    print("=" * 70)
    print("gg2_ob14_make.py - deriving the parametric bootstrap")
    print("=" * 70)

    if not SRC.exists():
        sys.exit("[ERROR] no source: %s  (run from the ROOT folder)" % SRC)

    out = SRC.read_text(encoding="utf-8")
    print("\nsource: %s  (%d lines)" % (SRC, out.count("\n")))

    for i, (name, req, old, new) in enumerate(EDITS, 1):
        c = out.count(old)
        tag = "" if req else "  (cosmetic)"
        print("\n[%d/%d] %s%s" % (i, len(EDITS), name, tag))
        print("      matches: %d" % c)
        if c != 1:
            if req:
                print("\n[STOP] Pattern did not match exactly once. Searched for:")
                for ln in old.rstrip("\n").split("\n"):
                    print("        |" + ln)
                print("[STOP] No file was written.")
                sys.exit(1)
            print("      SKIPPED (not required, does not affect the verdict).")
            continue
        out = out.replace(old, new)
        print("      applied.")

    out = HEADER + out

    nb = sum(1 for b in out.encode("utf-8") if b > 127)
    print("\nnon-ASCII bytes: %d" % nb)
    try:
        ast.parse(out)
        print("syntax: OK")
    except SyntaxError as e:
        sys.exit("[STOP] syntax error in the generated file: %s" % e)

    DST.write_text(out, encoding="utf-8")
    print("\nWRITTEN: %s  (%d lines)" % (DST, out.count("\n")))

    print("\n" + "=" * 70)
    print("1) ANCHOR -- original behaviour, B1 MUST PASS:")
    print("   python gg2/gg2_ob14_boot.py 2>&1 | tee gg2/ob14_anchor.log")
    print("")
    print("2) SUMMER BLOCK -- only after 1 has passed:")
    print("   python gg2/gg2_ob14_boot.py --prefix ob13_0_ --t0 0 2>&1 "
          "| tee gg2/ob14_season.log")
    print("=" * 70)


if __name__ == "__main__":
    main()
