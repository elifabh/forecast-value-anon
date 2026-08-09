#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_z20_discover.py - TAM AGAC KESFI. Hicbir hukum vermez.

Envanter uc bosluk gosterdi:
  (1) sekiz plasebo yorungesi bulunamadi
  (2) tahmin matrisleri taranan klasorlerde degil (data/ olabilir)
  (3) gg2_nwp_build_fc.py / gg2_nwp_mpc_fair.py kokte yok

Bu script UCUNU BIRDEN cozer, tek kosumda. Tahmin yok, sadece bakar.

NE YAPAR
  A. Tum agaci gezer (.venv/.git/__pycache__ haric), her .npz/.npy'yi bulur
  B. (N,48) sekilli anahtar iceren npz'leri TAHMIN MATRISI ADAYI isaretler
  C. Tum .py dosyalarini listeler (kok + gg2 + alt klasorler)
  D. Plasebo/model/sifir-hava scriptlerinde HER savez satirini gosterir
     -> yorunge kaydediliyor mu, hangi ada?
  E. Ayni scriptlerde solar_fc / forecast matrisi gecen satirlari gosterir
     -> her kolun tahmin matrisi nasil uretiliyor, saklaniyor mu?

CALISTIR (kok klasorden, ~20 sn):
  python gg2/gg2_z20_discover.py 2>&1 | tee gg2/z20_discover.log
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SKIP = {".venv", ".git", "__pycache__", "node_modules", ".ipynb_checkpoints",
        "site-packages", ".mypy_cache", ".pytest_cache"}

# D/E adimlarinda taranacak scriptler
TARGETS = [
    "gg2_z16_placebo_multi.py",
    "gg2_z16_alpha.py",
    "gg2_z16_final.py",
    "gg2_z18_all_shifts.py",
    "gg2_z15_nwp_models.py",
    "gg2_z15_build_models.py",
    "gg2_z19_noweather.py",
    "gg2_nwp_mpc_fair.py",
    "gg2_nwp_build_fc.py",
    "gg2_nwp_fetch.py",
    "gg2_nwp_season.py",
    "forecast_matrix.py",
    "mpc_econ_voltage_pareto.py",
]

KEY_HINTS = ("savez", "np.save", "solar_fc", "forecast_matrix", ".npz",
             "shift", "placebo", "plac", "roll(")


def walk(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP and not d.startswith(".")]
        for fn in filenames:
            yield Path(dirpath) / fn


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def main() -> None:
    print("=" * 78)
    print("Z20 TAM AGAC KESFI - hicbir hukum verilmez")
    print("=" * 78)
    print("  kok: %s" % ROOT)

    allf = list(walk(ROOT))
    arrays = [p for p in allf if p.suffix in (".npz", ".npy")]
    pys = [p for p in allf if p.suffix == ".py"]

    # ------------------------------------------------------- A + B
    print("\n" + "=" * 78)
    print("A/B - TUM .npz / .npy (%d adet)" % len(arrays))
    print("=" * 78)
    cand = []
    for p in sorted(arrays, key=rel):
        try:
            if p.suffix == ".npy":
                a = np.load(p, allow_pickle=False, mmap_mode="r")
                print("  %-58s npy shape=%s" % (rel(p), a.shape))
                if a.ndim == 2 and a.shape[1] == 48:
                    cand.append((rel(p), "(dizi)", a.shape))
                continue
            z = np.load(p, allow_pickle=False)
            ks = list(z.keys())
            print("  %-58s npz %s" % (rel(p), ks))
            for k in ks:
                a = z[k]
                if getattr(a, "ndim", 0) == 2 and a.shape[1] == 48:
                    cand.append((rel(p), k, a.shape))
        except Exception as e:
            print("  %-58s [OKUNAMADI] %s" % (rel(p), e))

    print("\n  --- TAHMIN MATRISI ADAYLARI (sekil (N,48)) ---")
    if not cand:
        print("  YOK. Hicbir dosyada (N,48) sekilli dizi bulunamadi.")
    for f, k, sh in cand:
        print("  %-58s anahtar=%-12s shape=%s" % (f, k, sh))

    # ------------------------------------------------------------- C
    print("\n" + "=" * 78)
    print("C - TUM .py (%d adet)" % len(pys))
    print("=" * 78)
    for p in sorted(pys, key=rel):
        try:
            sz = p.stat().st_size
        except Exception:
            sz = -1
        print("  %-62s %8d B" % (rel(p), sz))

    # ------------------------------------------------------------- D/E
    print("\n" + "=" * 78)
    print("D/E - HEDEF SCRIPTLERDE KAYIT ve TAHMIN MATRISI SATIRLARI")
    print("=" * 78)
    byname = {}
    for p in pys:
        byname.setdefault(p.name, []).append(p)

    for name in TARGETS:
        hits = byname.get(name, [])
        if not hits:
            print("\n  --- %s : DOSYA YOK ---" % name)
            continue
        for p in hits:
            print("\n  --- %s ---" % rel(p))
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except Exception as e:
                print("      [OKUNAMADI] %s" % e)
                continue
            shown = 0
            for i, line in enumerate(lines, 1):
                s = line.rstrip()
                t = s.strip()
                if t.startswith("#"):
                    continue
                if any(h in s for h in KEY_HINTS):
                    print("      %4d  %s" % (i, s[:96]))
                    shown += 1
                    if shown >= 60:
                        print("      ... (kirpildi)")
                        break
            if shown == 0:
                print("      (eslesen satir yok)")
    print()


if __name__ == "__main__":
    main()