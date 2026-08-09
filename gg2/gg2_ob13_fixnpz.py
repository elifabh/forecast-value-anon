#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_ob13_fixnpz.py - adds the missing LABEL fields to nwp_solar_fc_season.npz.

WHY
  The get_fc() function of gg2_ob8_arms.py reads the z["model"] and z["a_mean"]
  fields when printing the label of the nwp arm. gg2_ob12_seasonfc.py had not
  written these two fields (my omission). Only the TEXT PRINTED ON SCREEN is
  affected, no number at all.

WHAT IT DOES
  It DOES NOT TOUCH the 'solar' matrix. It only adds the model and a_mean
  fields. a_mean was not computed for this file -> NaN is written (no made-up
  number is inserted).

RUN (from the ROOT folder):
  python gg2/gg2_ob13_fixnpz.py

ASCII only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

P = Path("gg2/data/nwp_solar_fc_season.npz")


def main() -> None:
    if not P.exists():
        sys.exit("[ERROR] no such file: %s  (run from the ROOT folder)" % P)

    z = np.load(P, allow_pickle=True)
    have = list(z.files)
    print("existing fields: %s" % have)

    solar = np.array(z["solar"])
    fit0 = np.int64(z["fit0"]) if "fit0" in have else np.int64(2400)
    fit1 = np.int64(z["fit1"]) if "fit1" in have else np.int64(13800)
    K = np.int64(z["K"]) if "K" in have else np.int64(solar.shape[1])
    ghi = np.array(z["ghi"]) if "ghi" in have else np.array("gg2/data/nwp_ghi_roi.csv")
    z.close()

    chk = float(np.abs(solar).sum())
    print("solar %s %s   |total| = %.6f" % (solar.shape, solar.dtype, chk))

    np.savez_compressed(
        P,
        solar=solar,
        fit0=fit0,
        fit1=fit1,
        K=K,
        ghi=ghi,
        model=np.array("lgbm"),
        a_mean=np.float64(np.nan),
    )

    w = np.load(P, allow_pickle=True)
    chk2 = float(np.abs(np.array(w["solar"])).sum())
    print("new fields     : %s" % list(w.files))
    print("is solar unchanged: |diff| = %.3e" % abs(chk2 - chk))
    if abs(chk2 - chk) > 0:
        sys.exit("[STOP] the solar matrix changed - UNEXPECTED")
    print("\nDONE. Restart the run with the same command;")
    print("the naive arm is already written and will be skipped "
          "(nwp + oracle will run).")


if __name__ == "__main__":
    main()
