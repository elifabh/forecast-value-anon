#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_v3_percentile_ci.py - YUZDELIK GUVEN ARALIKLARI (denetim bulgusu F2).

SORUN
  Projedeki her hata cubugu bootstrap DAGILIMININ STANDART SAPMASI. Simetri
  varsayiyor. Denetim olctu: manset dagilimi SAGA CARPIK (+0.48). "1.54 +/- 0.72"
  simetrik bir aralik ima ediyor; gercek aralik simetrik DEGIL.

NE YAPAR
  Ayni yorungeler, ayni eslestirilmis gunluk-blok bootstrap (blok 48, 50 gun,
  10000 tekrar, seed 12345). Ek olarak her karsilastirma icin:
    ortalama | SD | %2.5 ve %97.5 yuzdelik | carpiklik | P(fark <= 0)
  Hicbir sayi degismez; sadece belirsizlik DURUSTCE raporlanir.

CIPA (gecmezse hukum yok)
  naive - bm : ortalama +1.54, SD 0.72

MAKALEDE NE YAZILACAK
  "+/- 0.72 SE" YERINE yuzdelik aralik yazilir. Aralik sifiri disliyorsa
  iddia AYAKTA; genisligi de durustce gosterilir.

CALISTIR (kok klasorden, ~1 dk):
  python gg2/gg2_v3_percentile_ci.py 2>&1 | tee gg2/v3_percentile_ci.log
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mpc_econ_voltage_pareto as M

BLOCK, NDAYS, NBOOT, SEED = 48, 50, 10000, 12345
ANCHOR_D, ANCHOR_S, TOL = 1.54, 0.72, 0.10

ARMS = {
    "naive":  "gg2/nwp_fair_traj_naive.npz",
    "oracle": "gg2/nwp_fair_traj_oracle.npz",
    "bm":     "gg2/nwp_fair_traj_nwp.npz",
    "s1":     "gg2/nwp_fair_s1_nwp.npz",
    "s7":     "gg2/nwp_fair_s7_nwp.npz",
    "ecmwf":  "gg2/nwp_fair_ecmwf_ifs025_nwp.npz",
    "gfs":    "gg2/nwp_fair_gfs_seamless_nwp.npz",
    "cal":    "gg2/nwp_fair_cal_nwp.npz",
    "calcap": "gg2/nwp_fair_calcap_nwp.npz",
}
for d in (1, 2, 3, 4, 5, 7, 14, 21):
    ARMS["p%d" % d] = "gg2/data/placebo_shift%d.npz" % d

PAIRS = [
    ("naive", "bm",     "MANSET"),
    ("naive", "ecmwf",  "hava modeli"),
    ("naive", "gfs",    "hava modeli"),
    ("bm",    "oracle", "bilgi boslugu"),
    ("cal",   "naive",  "sifir-hava vs naive"),
    ("cal",   "bm",     "sifir-hava vs NWP"),
    ("calcap", "cal",   "ozellik farki"),
]
for d in (1, 2, 3, 4, 5, 7, 14, 21):
    PAIRS.append(("p%d" % d, "bm", "plasebo vs NWP"))
    PAIRS.append(("p%d" % d, "naive", "plasebo vs naive"))


def die(m):
    sys.exit("\n[DUR] " + m + "\n")


def load(rel_):
    p = Path(rel_)
    if not p.exists():
        die("yorunge yok: %s" % rel_)
    z = np.load(p)
    n = int(z["done"])
    v = ((z["mn"][:n] < M.LO) | (z["mx"][:n] > M.HI)).astype(np.float64)
    return v, z["p"][:n], n


def stats(d):
    m = float(d.mean()); s = float(d.std(ddof=1))
    lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
    sk = float(((d - m) ** 3).mean() / (s ** 3)) if s > 0 else 0.0
    p0 = float((d <= 0).mean())
    return m, s, lo, hi, sk, p0


def main():
    print("=" * 88)
    print("V3 - YUZDELIK GUVEN ARALIKLARI (F2)")
    print("=" * 88)
    print("  blok %d | %d gun | %d tekrar | seed %d" % (BLOCK, NDAYS, NBOOT, SEED))

    V, P, D = {}, {}, {}
    for t, f in ARMS.items():
        V[t], P[t], D[t] = load(f)
    need = BLOCK * NDAYS
    if min(D.values()) < need:
        die("en kisa yorunge %d < %d" % (min(D.values()), need))
    print("  %d kol yuklendi, pencere ilk %d adim" % (len(ARMS), need))

    price = M.price[M.SPLIT:M.SPLIT + need].reshape(NDAYS, BLOCK)
    Vb = {t: V[t][:need].reshape(NDAYS, BLOCK) for t in V}
    Pb = {t: P[t][:need].reshape(NDAYS, BLOCK) for t in P}

    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, NDAYS, size=(NBOOT, NDAYS))
    BV, BK = {}, {}
    for t in V:
        BV[t] = Vb[t].mean(axis=1)[draws].mean(axis=1) * 100.0
        BK[t] = (-(price * Pb[t]).sum(axis=1) * M.dt)[draws].sum(axis=1)

    d = BV["naive"] - BV["bm"]
    m, s = float(d.mean()), float(d.std(ddof=1))
    print("\n  CIPA: beklenen %+.2f / SD %.2f | olculen %+.2f / SD %.2f -> %s"
          % (ANCHOR_D, ANCHOR_S, m, s,
             "GECTI" if abs(m - ANCHOR_D) <= TOL and abs(s - ANCHOR_S) <= TOL
             else "KALDI - HUKUM YOK"))
    if abs(m - ANCHOR_D) > TOL or abs(s - ANCHOR_S) > TOL:
        die("cipa tutmadi")

    print("\n" + "=" * 88)
    print("IHLAL FARKLARI (puan) - A eksi B")
    print("=" * 88)
    print("  %-16s %8s %7s %6s %18s %8s %9s  %s"
          % ("karsilastirma", "ort", "SD", "SD kat", "%95 yuzdelik CI",
             "carpik", "P(<=0)", "not"))
    print("  " + "-" * 100)
    for a, b, note in PAIRS:
        m, s, lo, hi, sk, p0 = stats(BV[a] - BV[b])
        k = abs(m) / s if s > 0 else float("inf")
        flag = "SIFIRI DISLIYOR" if (lo > 0 or hi < 0) else "sifiri iceriyor"
        print("  %-16s %+8.2f %7.2f %6.1f  [%+6.2f, %+6.2f] %8.2f %9.3f  %s | %s"
              % (a + "-" + b, m, s, k, lo, hi, sk, p0, flag, note))

    print("\n" + "=" * 88)
    print("KAR FARKLARI (EUR)")
    print("=" * 88)
    print("  %-16s %8s %7s %20s %9s  %s"
          % ("karsilastirma", "ort", "SD", "%95 yuzdelik CI", "P(<=0)", "not"))
    for a, b, note in PAIRS[:7]:
        m, s, lo, hi, sk, p0 = stats(BK[a] - BK[b])
        flag = "SIFIRI DISLIYOR" if (lo > 0 or hi < 0) else "sifiri iceriyor"
        print("  %-16s %+8.0f %7.0f  [%+8.0f, %+8.0f] %9.3f  %s | %s"
              % (a + "-" + b, m, s, lo, hi, p0, flag, note))

    print("\n" + "=" * 88)
    print("MAKALE ICIN MANSET CUMLESI")
    print("=" * 88)
    m, s, lo, hi, sk, p0 = stats(BV["naive"] - BV["bm"])
    km, ks, klo, khi, _, kp = stats(BK["naive"] - BK["bm"])
    print("  ihlal : %+.2f puan, %%95 CI [%+.2f, %+.2f], carpiklik %+.2f, P(<=0)=%.3f"
          % (m, lo, hi, sk, p0))
    print("  kar   : %+.0f EUR, %%95 CI [%+.0f, %+.0f], P(<=0)=%.3f"
          % (km, klo, khi, kp))
    print("\n  -> 'X +/- SE' YERINE bu araliklar yazilir.")
    print("  -> Ihlal araligi sifiri disliyorsa iyilesme AYAKTA; genisligi")
    print("     durustce gosterilir. Kar araligi sifiri iceriyorsa 'kar farki")
    print("     olculemedi' yazilir, 'kar artti' ASLA yazilmaz.\n")


if __name__ == "__main__":
    main()