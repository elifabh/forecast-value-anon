#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_z18_all_shifts.py - Z18: SEKIZ kaydirmanin hata cubuklariyla tam tablosu.

DURUM
  Kisa kaydirmalar (1,2,3,4,5 gun) bes koldan besinde 9.96-10.42 bandinda
  toplandi. Gercek NWP 9.80, naive 11.28. Uzun kaydirmalar (7,14,21) ise
  11.16 / 11.77 / 12.39 -> naive tarafinda.
  Bes bagimsiz kaydirmanin ayni banda dusmesi rastgele DEGILDIR.

  SORU: 1.48 puanlik kazancin ne kadari DOGRU HAVAYI BILMEKTEN geliyor?
  Eger yanlis havali bir tahmin de ayni sonucu veriyorsa, "gun-oncesi hava
  bilgisinin degeri" cercevesi zayiflar.

ON-KAYIT (koda girmeden, sonuclara bakilmadan)
  Olculecek : sekiz plasebonun gercek NWP'den ve naive'den farki,
              eslestirilmis gunluk-blok bootstrap ile.
  ANA SORU  : plac1 (en kisa kaydirma) gercek NWP'den 2 SE ile AYRISIYOR mu?

  HUKUM (1) : plac1 - NWP >= 2 SE  -> yanlis hava OLCULEBILIR sekilde kotu.
              Plasebo argumani ayakta, cerceve korunur; kisa kaydirmalarin
              naive'den iyi olmasi ayri bir SINIR olarak yazilir.
  HUKUM (2) : plac1 - NWP < 2 SE   -> yanlis havali tahmin gercek tahminden
              ayirt edilemiyor. Bu CIDDIDIR. "Gun-oncesi hava bilgisinin
              degeri" mansetti; olcum onu tasimiyorsa manset DEGISIR.
              Sayilar dogru, YORUM degisir. Gizlenmez.
  HUKUM (3) : plac1 - naive >= 2 SE (yani kisa kaydirma naive'den olculebilir
              sekilde IYI) -> bu kendi basina raporlanacak bir bulgudur:
              tahminin gunluk SEKLI, hava iceriginden bagimsiz olarak deger
              tasiyor.

  Ayrica: uzun kaydirmalar TEPE hatasi tasiyor (z17_peak: 3g -0.08, 7g -0.18,
  14g -0.30, 21g -0.41) -> mevsimsel seviye kaymasi. Plasebo merdiveni iki
  seyi karistiriyor: yanlis hava + yanlis seviye. Bu ayrim YAZILIR.

  CIPA: naive - NWP = +1.54 / SE 0.72.

CALISTIR (~1 dk, MPC KOSMAZ):
  python gg2/gg2_z18_all_shifts.py 2>&1 | tee gg2/z18_all_shifts.log
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mpc_econ_voltage_pareto as M

SP = 3.0
BLOCK, NDAYS, NBOOT, BSEED = 48, 50, 10000, 12345
ANCHOR_DIFF, ANCHOR_SE, TOL = 1.54, 0.72, 0.10
SHIFTS = (1, 2, 3, 4, 5, 7, 14, 21)


def _die(m: str) -> None:
    sys.exit("[HATA] " + m)


def main() -> None:
    ap = argparse.ArgumentParser(description="Z18: sekiz kaydirma tam tablo.")
    ap.add_argument("--joint", default="data/aligned_30min_joint.csv")
    ap.add_argument("--fcmat", default="data/forecast_matrix.npz")
    ap.add_argument("--nwp", default="gg2/data/nwp_solar_fc_capnorm.npz")
    ap.add_argument("--nboot", type=int, default=NBOOT)
    a = ap.parse_args()

    # ---------------------------------------------------------- alpha'lar
    df = pd.read_csv(a.joint, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    N = len(df)
    y = (df.solar_mw / df.iloc[:M.SPLIT].solar_mw.max() * SP).to_numpy()
    naive_m = np.load(a.fcmat)["solar"]
    K = int(naive_m.shape[1])
    nwp_m = np.load(a.nwp)["solar"]
    idx = np.minimum(np.arange(N)[:, None] + np.arange(1, K + 1)[None, :], N - 1)
    truth = y[idx]
    sl = slice(M.SPLIT, N - 1)
    mn_ref = float(np.abs(naive_m[sl] - truth[sl]).mean())

    alpha = {"NWP": float(np.abs(nwp_m[sl] - truth[sl]).mean()) / mn_ref, "naive": 1.0}
    peak_err = {}
    tmax = truth[sl].max(axis=1)
    for tag, mm in (("NWP", nwp_m), ("naive", naive_m)):
        peak_err[tag] = float((mm[sl].max(axis=1) - tmax).mean())
    for d in SHIFTS:
        ii = np.clip(np.arange(len(nwp_m)) - d * 48, 0, len(nwp_m) - 1)
        sm = nwp_m[ii]
        alpha["p%d" % d] = float(np.abs(sm[sl] - truth[sl]).mean()) / mn_ref
        peak_err["p%d" % d] = float((sm[sl].max(axis=1) - tmax).mean())

    # -------------------------------------------------------- yorungeler
    paths = {"NWP": "gg2/nwp_fair_traj_nwp.npz", "naive": "gg2/nwp_fair_traj_naive.npz"}
    for d in SHIFTS:
        paths["p%d" % d] = "gg2/data/placebo_shift%d.npz" % d

    V, P, dn = {}, {}, {}
    for t, p in paths.items():
        if not Path(p).exists():
            _die("yorunge yok: %s" % p)
        z = np.load(p)
        k = int(z["done"])
        V[t] = ((z["mn"] < M.LO) | (z["mx"] > M.HI)).astype(np.float64)[:k]
        P[t] = z["p"][:k]
        dn[t] = k

    need = BLOCK * NDAYS
    if min(dn.values()) < need:
        _die("yorunge cok kisa")

    T0 = M.SPLIT
    Vb = {t: V[t][:need].reshape(NDAYS, BLOCK) for t in V}
    rng = np.random.default_rng(BSEED)
    draws = rng.integers(0, NDAYS, size=(a.nboot, NDAYS))
    BV = {t: (Vb[t].mean(axis=1))[draws].mean(axis=1) * 100.0 for t in Vb}

    def cmp_(A, B_):
        d = BV[A] - BV[B_]
        m_, s_ = float(d.mean()), float(d.std(ddof=1))
        return m_, s_, (abs(m_) / s_ if s_ > 0 else float("inf"))

    order = ["NWP"] + ["p%d" % d for d in SHIFTS] + ["naive"]

    print("=" * 84)
    print("Z18 - SEKIZ KAYDIRMA, TAM TABLO")
    print("=" * 84)
    m_, s_, _ = cmp_("naive", "NWP")
    ok = abs(m_ - ANCHOR_DIFF) <= TOL and abs(s_ - ANCHOR_SE) <= TOL
    print("  CIPA naive-NWP = %+.2f +/- %.2f  -> %s" % (m_, s_, "GECTI" if ok else "KALDI"))
    if not ok:
        sys.exit(2)

    print("\n  %-7s %7s %10s %9s %11s" % ("kol", "alpha", "tepe hata", "ihlal %", "kar EUR"))
    print("  " + "-" * 48)
    for t in order:
        kk = float(-(M.price[T0:T0 + dn[t]] * P[t]).sum() * M.dt)
        print("  %-7s %7.3f %10.4f %9.2f %11.1f"
              % (t, alpha[t], peak_err[t], 100.0 * V[t].mean(), kk))

    print("\n" + "=" * 84)
    print("HER PLASEBO vs GERCEK NWP   /   vs NAIVE")
    print("=" * 84)
    print("  %-7s %9s %6s %6s %-14s %9s %6s %6s %s"
          % ("kol", "-NWP", "SE", "kat", "hukum", "-naive", "SE", "kat", "hukum"))
    print("  " + "-" * 78)
    res = {}
    for d in SHIFTS:
        t = "p%d" % d
        a1, s1, k1 = cmp_(t, "NWP")
        a2, s2, k2 = cmp_(t, "naive")
        res[d] = (a1, s1, k1, a2, s2, k2)
        print("  %-7s %+9.2f %6.2f %6.1f %-14s %+9.2f %6.2f %6.1f %s"
              % (t, a1, s1, k1, "NWP'den kotu" if k1 >= 2 else "cozulmuyor",
                 a2, s2, k2, "naive'den iyi" if (k2 >= 2 and a2 < 0)
                 else ("naive'den kotu" if k2 >= 2 else "cozulmuyor")))

    # ------------------------------------------------------------- hukum
    print("\n" + "=" * 84)
    print("ON-KAYITLI HUKUM")
    print("=" * 84)
    a1, s1, k1, a2, s2, k2 = res[1]
    print("  ANA SORU: plac1 (1 gun kaydirma) gercek NWP'den ayrisiyor mu?")
    print("    plac1 - NWP = %+.2f +/- %.2f  (%.1f SE)" % (a1, s1, k1))
    if k1 >= 2.0:
        print("    -> HUKUM (1): yanlis hava OLCULEBILIR sekilde kotu.")
        print("       Plasebo argumani ayakta, cerceve korunur.")
    else:
        print("    -> HUKUM (2): yanlis havali tahmin gercek tahminden AYIRT")
        print("       EDILEMIYOR. CIDDI. Sayilar dogru ama 'gun-oncesi hava")
        print("       bilgisinin degeri' manseti bu olcumu TASIMIYOR.")
        print("       Manset DEGISIR. Gizlenmez.")

    print("\n  EK: plac1 naive'den olculebilir sekilde iyi mi?")
    print("    plac1 - naive = %+.2f +/- %.2f  (%.1f SE)" % (a2, s2, k2))
    if k2 >= 2.0 and a2 < 0:
        print("    -> HUKUM (3): EVET. Tahminin gunluk SEKLI, hava iceriginden")
        print("       bagimsiz olarak deger tasiyor. Bu kendi basina bir bulgudur.")
    else:
        print("    -> cozulmuyor. Yorumlanmaz.")

    kisa = [d for d in (1, 2, 3, 4, 5)]
    uzun = [d for d in (7, 14, 21)]
    print("\n  KISA (1-5 gun) ihlal araligi : %.2f - %.2f"
          % (min(100.0 * V["p%d" % d].mean() for d in kisa),
             max(100.0 * V["p%d" % d].mean() for d in kisa)))
    print("  UZUN (7-21 gun) ihlal araligi: %.2f - %.2f"
          % (min(100.0 * V["p%d" % d].mean() for d in uzun),
             max(100.0 * V["p%d" % d].mean() for d in uzun)))
    print("  UZUN kaydirmalarin TEPE hatasi buyuyor (%s)"
          % ", ".join("%dg %.2f" % (d, peak_err["p%d" % d]) for d in uzun))
    print("  -> Uzun kaydirmalar sadece 'yanlis hava' degil, 'yanlis mevsimsel")
    print("     seviye' de tasiyor. Plasebo merdiveni IKI seyi karistiriyor.")
    print("     Bu ayrim makalede YAZILIR.\n")


if __name__ == "__main__":
    main()