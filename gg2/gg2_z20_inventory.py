#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_z20_inventory.py - Z20 ONCESI ENVANTER. Hicbir hukum vermez.

NEDEN
  Z20 her kol icin TAHMIN MATRISINDEN hata istatistigi hesaplayacak.
  DEVIR'de yorunge dosya adlari var, tahmin matrisi adlari YOK.
  Tahmin edilmez, BAKILIR. Bu script sadece listeler.

NE YAPAR
  1. gg2/ ve kok klasordeki tum .npz / .npy dosyalarini bulur
  2. Her .npz icin: anahtarlar + sekiller + dtype
  3. Gercek solar serisinin nerede oldugunu arar (M modulunden)
  4. 17 kolun yorungelerinden hangileri mevcut, isaretler

CALISTIR (kok klasorden, ~10 sn):
  python gg2/gg2_z20_inventory.py 2>&1 | tee gg2/z20_inventory.log
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# DEVIR'de adi gecen 17 kolun yorungeleri
EXPECTED_TRAJ = [
    "nwp_fair_traj_naive.npz",
    "nwp_fair_traj_oracle.npz",
    "nwp_fair_traj_nwp.npz",
    "nwp_fair_s1_nwp.npz",
    "nwp_fair_s7_nwp.npz",
    "nwp_fair_ecmwf_ifs025_nwp.npz",
    "nwp_fair_gfs_seamless_nwp.npz",
    "nwp_fair_cal_nwp.npz",
    "nwp_fair_calcap_nwp.npz",
]


def describe(p: Path) -> None:
    try:
        if p.suffix == ".npy":
            a = np.load(p, allow_pickle=False)
            print("    %-52s  npy  shape=%s dtype=%s"
                  % (p.name, a.shape, a.dtype))
            return
        z = np.load(p, allow_pickle=False)
        keys = list(z.keys())
        print("    %-52s  npz  %d anahtar" % (p.name, len(keys)))
        for k in keys:
            a = z[k]
            sh = getattr(a, "shape", "()")
            dt = getattr(a, "dtype", "?")
            extra = ""
            if getattr(a, "ndim", 0) == 0:
                extra = "  deger=%s" % a
            elif getattr(a, "size", 0) > 0 and np.issubdtype(a.dtype, np.number):
                try:
                    extra = "  min=%.4g max=%.4g" % (float(np.nanmin(a)), float(np.nanmax(a)))
                except Exception:
                    extra = ""
            print("        %-22s shape=%-16s dtype=%-9s%s" % (k, str(sh), str(dt), extra))
    except Exception as e:
        print("    %-52s  [OKUNAMADI] %s" % (p.name, e))


def main() -> None:
    print("=" * 78)
    print("Z20 ENVANTER - hicbir hukum verilmez, sadece listelenir")
    print("=" * 78)
    print("  kok: %s" % ROOT)

    # ---------------------------------------------------------- 1. dosyalar
    for sub in (".", "gg2"):
        d = ROOT / sub
        if not d.exists():
            continue
        files = sorted([q for q in d.iterdir()
                        if q.is_file() and q.suffix in (".npz", ".npy")])
        print("\n" + "-" * 78)
        print("  KLASOR: %s   (%d dosya)" % (sub, len(files)))
        print("-" * 78)
        for q in files:
            describe(q)

    # ------------------------------------------------ 2. beklenen yorungeler
    print("\n" + "=" * 78)
    print("BEKLENEN YORUNGELER (DEVIR'den)")
    print("=" * 78)
    for name in EXPECTED_TRAJ:
        hit = None
        for cand in (ROOT / "gg2" / name, ROOT / name):
            if cand.exists():
                hit = cand
                break
        print("  %-40s %s" % (name, ("VAR  -> " + str(hit.relative_to(ROOT)))
                              if hit else "YOK"))
    print("\n  NOT: sekiz plasebo (p1..p21) yorunge adi DEVIR'de yok.")
    print("       Yukaridaki dosya listesinde 'plac' / 'shift' / 'p1' gecen")
    print("       adlar varsa Z20'de kullanilacak.")

    # ---------------------------------------------------- 3. gercek solar
    print("\n" + "=" * 78)
    print("GERCEK SOLAR SERISI (hata istatistigi bunun uzerinden hesaplanir)")
    print("=" * 78)
    try:
        import mpc_econ_voltage_pareto as M
        print("  mpc_econ_voltage_pareto YUKLENDI")
        print("  SPLIT=%s  dt=%s  LO=%.3f  HI=%.3f"
              % (getattr(M, "SPLIT", "?"), getattr(M, "dt", "?"),
                 getattr(M, "LO", float("nan")), getattr(M, "HI", float("nan"))))
        cands = [a for a in dir(M)
                 if not a.startswith("_")
                 and isinstance(getattr(M, a, None), np.ndarray)]
        print("  modul icindeki numpy dizileri:")
        for a in sorted(cands):
            arr = getattr(M, a)
            try:
                print("    %-18s shape=%-14s min=%.4g max=%.4g"
                      % (a, str(arr.shape), float(np.nanmin(arr)), float(np.nanmax(arr))))
            except Exception:
                print("    %-18s shape=%s" % (a, str(arr.shape)))
    except Exception as e:
        print("  [YUKLENEMEDI] %s" % e)

    # ------------------------------------------------- 4. tahmin matrisi kodu
    print("\n" + "=" * 78)
    print("TAHMIN MATRISI URETEN KOD - imzalar")
    print("=" * 78)
    for fn in ("forecast_matrix.py", "gg2_nwp_build_fc.py", "gg2_nwp_mpc_fair.py"):
        p = ROOT / fn
        if not p.exists():
            print("  %-26s YOK" % fn)
            continue
        n = 0
        print("  %-26s VAR" % fn)
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                s = line.strip()
                if s.startswith("def ") or s.startswith("class "):
                    print("      %4d  %s" % (i, s[:88]))
                    n += 1
                elif ".npz" in s or ".npy" in s:
                    if n < 200:
                        print("      %4d  [DOSYA] %s" % (i, s[:88]))
    print()


if __name__ == "__main__":
    main()