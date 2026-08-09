#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_ob12_seasonfc.py - SUMMER BLOCK NWP FORECAST MATRIX (for mv_oberrhein)

PURPOSE
  To produce an NWP solar forecast matrix fitted while EXCLUDING the SUMMER
  BLOCK (0:2400), so that block can be run on mv_oberrhein.

WHY (leakage)
  The existing gg2/data/nwp_solar_fc_capnorm.npz was fitted with 0:13800.
  The summer block 0:2400 falls INSIDE that window -> measuring there would
  mean looking at our own training data. gg2_nwp_season.py already solved
  this: fit 2400:13800, evaluation 0:2400.

NO CODE IS REWRITTEN
  The forecaster is imported AS IS from gg2_nwp_season.build_nwp. No model,
  feature or parameter is redefined in this script.

ANCHOR (the second file is NOT WRITTEN unless this test passes)
  The output of build_nwp(fit0=0, fit1=13800) must match the existing
  nwp_solar_fc_capnorm.npz EXACTLY (max |diff| <= 1e-6). If it is exceeded
  the script STOPS and writes nothing.

OUTPUT
  gg2/data/nwp_solar_fc_season.npz
    solar   (16230, 48) float32   - fit 2400:13800
    fit0, fit1, ghi, K            - spec sheet

RUN (from the ROOT folder):
  python gg2/gg2_ob12_seasonfc.py 2>&1 | tee gg2/ob12_seasonfc.log

ASCII only.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

GHI = "gg2/data/nwp_ghi_roi.csv"
REF = Path("gg2/data/nwp_solar_fc_capnorm.npz")
OUT = Path("gg2/data/nwp_solar_fc_season.npz")

FIT0_REF, FIT1_REF = 0, 13800      # window of the existing file
FIT0_NEW, FIT1_NEW = 2400, 13800   # window that EXCLUDES the summer block
TOL = 1e-6


def die(msg: str) -> None:
    print("\n[STOP] " + msg)
    print("[STOP] No file was written.")
    sys.exit(1)


def main() -> None:
    print("=" * 70)
    print("gg2_ob12_seasonfc.py - summer block forecast matrix")
    print("=" * 70)

    if not Path(GHI).exists():
        die("no GHI file: " + GHI)
    if not REF.exists():
        die("no reference file: " + str(REF))

    print("\n[1/4] importing gg2_nwp_season (build_nwp is used as is)...")
    import gg2_nwp_season as S
    print("      OK. SP=%.1f CAP_WIN=%d LAG=%d" % (S.SP, S.CAP_WIN, S.LAG))

    ref = np.load(REF)
    ref_solar = ref["solar"]
    ref_split = int(ref["split"]) if "split" in ref.files else -1
    print("\n[2/4] Reference: %s" % REF)
    print("      solar %s %s   split=%s" %
          (ref_solar.shape, ref_solar.dtype, ref_split))
    if ref_split != FIT1_REF:
        print("      [WARNING] stored split (%s) is not the expected %d."
              % (ref_split, FIT1_REF))

    print("\n[3/4] ANCHOR: build_nwp(fit0=%d, fit1=%d) must reproduce the "
          "reference..." % (FIT0_REF, FIT1_REF))
    fc_ref, _ = S.build_nwp(GHI, FIT0_REF, FIT1_REF)
    fc_ref = np.asarray(fc_ref, dtype=np.float32)

    if fc_ref.shape != ref_solar.shape:
        die("Shape mismatch: produced %s, reference %s"
            % (fc_ref.shape, ref_solar.shape))

    d = np.abs(fc_ref.astype(np.float64) - ref_solar.astype(np.float64))
    dmax = float(d.max())
    print("      max |diff| = %.3e   (threshold %.0e)" % (dmax, TOL))
    if dmax > TOL:
        i, j = np.unravel_index(int(np.argmax(d)), d.shape)
        print("      largest deviation: row %d column %d" % (i, j))
        print("        produced %.8f  reference %.8f"
              % (fc_ref[i, j], ref_solar[i, j]))
        print("      mean |diff| = %.3e" % float(d.mean()))
        print("      fraction of cells greater than zero = %.4f"
              % float((d > 0).mean()))
        die("ANCHOR FAILED. The existing file may not have been produced by "
            "this function.")
    print("      ANCHOR PASSED.")

    print("\n[4/4] Summer block matrix: build_nwp(fit0=%d, fit1=%d)..."
          % (FIT0_NEW, FIT1_NEW))
    fc_new, _ = S.build_nwp(GHI, FIT0_NEW, FIT1_NEW)
    fc_new = np.asarray(fc_new, dtype=np.float32)
    print("      solar %s %s" % (fc_new.shape, fc_new.dtype))

    ev = slice(0, 2400)
    d_ev = np.abs(fc_new[ev].astype(np.float64) - ref_solar[ev].astype(np.float64))
    print("      deviation from the reference in the evaluation block (0:2400):")
    print("        max %.4f  mean %.4f" % (float(d_ev.max()), float(d_ev.mean())))
    print("      (if this comes out VERY close to zero, the leakage guard "
          "HAS NOT WORKED)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        solar=fc_new,
        fit0=np.int64(FIT0_NEW),
        fit1=np.int64(FIT1_NEW),
        K=np.int64(fc_new.shape[1]),
        ghi=np.array(GHI),
    )
    print("\n      WRITTEN: %s" % OUT)
    print("\n" + "=" * 70)
    print("DONE. Next step: run the ob8 arm on 0:2400 with this matrix.")
    print("=" * 70)


if __name__ == "__main__":
    main()
