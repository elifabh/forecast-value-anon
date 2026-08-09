#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_z16_alpha.py - Z16 ADIM 2: kaydirilmis tahminlerin BECERISI olculur.

NEDEN BU SCRIPT VAR
  Z16 Adim 1'de on-kayitli hukum (2) atesledi: 3-gun kaydirilmis plasebo
  esigin altinda kaldi (%10.21 < %10.54). Ihlal kaydirma uzunluguyla duzenli
  artiyor: 3g %10.21 | 7g %11.16 | 14g %11.77 | 21g %12.39.

  Akla gelen aciklama: "3 gun onceki hava bugunkuyle gercekten iliskilidir,
  o yuzden 3-gun kaydirilmis tahmin hala gercek bilgi tasir."
  BU ACIKLAMA KULLANILMAZ - cunku olculmedi. Bu script onu OLCER.
  Hikaye degil sayi: kaydirilmis matrisin naive'e gore MAE orani (alpha).

ON-KAYIT (koda girmeden, sonuclara bakilmadan yazildi)
  Olculecek : dort kaydirilmis matrisin TEST blogu alpha'si.
  HUKUM (A) : alpha_3 < 1 -> 3-gun kaydirma SAF PLASEBO DEGILDIR; olculebilir
              tahmin becerisi tasir. Makalede plasebo seti "7 gun ve uzeri"
              olarak tanimlanir, 3 gun AYRI raporlanir (gizlenmez).
  HUKUM (B) : alpha_3 >= 1 -> kaydirilmis tahminin becerisi YOK ama ihlali
              dusuruyor. CIDDI UYARI: kazancin bir kismi tahmin iceriginden
              degil. Aynen yazilir, aciklama uydurulmaz.
  EK SORU   : yedi kolun ihlal orani alpha'ya gore MONOTON mu?
              Monoton degilse YORUMLANMAZ - ne monoton ne degil denir.
              Bitisik ciftler 2 SE ile ayrismiyorsa siralama okunmaz.

  CIPA : gercek NWP alpha'si 0.799 cikmali (DEVIR 4.1). Cikmazsa hukum yok.

CALISTIR (GG-1 kok klasorunden, ~2 dk, MPC KOSMAZ):
  python gg2/gg2_z16_alpha.py 2>&1 | tee gg2/z16_alpha.log
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
SOLAR_H = (10, 16)
BLOCK = 48
NDAYS = 50
NBOOT = 10000
BOOT_SEED = 12345

REF_ALPHA = 0.799
TOL_ALPHA = 0.002
SHIFTS = (3, 7, 14, 21)

# tag -> (yorunge dosyasi, alpha kaynagi)
#   alpha kaynagi: "nwp" = matristen hesaplanir, "fixed" = bilinen sabit
TRAJ = [
    ("oracle",  "gg2/nwp_fair_traj_oracle.npz", 0.000),
    ("NWP",     "gg2/nwp_fair_traj_nwp.npz",    None),
    ("plac3",   "gg2/data/placebo_shift3.npz",  None),
    ("plac7",   "gg2/data/placebo_shift7.npz",  None),
    ("plac14",  "gg2/data/placebo_shift14.npz", None),
    ("plac21",  "gg2/data/placebo_shift21.npz", None),
    ("naive",   "gg2/nwp_fair_traj_naive.npz",  1.000),
]


def _die(m: str) -> None:
    sys.exit("[HATA] " + m)


def mae_alpha(fc, naive, truth, hr, sl):
    def _m(err, h):
        a = np.abs(err).ravel()
        mm = (h.ravel() >= SOLAR_H[0]) & (h.ravel() <= SOLAR_H[1])
        return float(a.mean()), float(np.quantile(a[mm], 0.95))
    mn, qn = _m(naive[sl] - truth[sl], hr[sl])
    mw, qw = _m(fc[sl] - truth[sl], hr[sl])
    return mw, mw / mn, qw / qn


def main() -> None:
    ap = argparse.ArgumentParser(description="Z16 Adim 2: kaydirilmis tahmin becerisi.")
    ap.add_argument("--joint", default="data/aligned_30min_joint.csv")
    ap.add_argument("--fcmat", default="data/forecast_matrix.npz")
    ap.add_argument("--nwp", default="gg2/data/nwp_solar_fc_capnorm.npz")
    ap.add_argument("--nboot", type=int, default=NBOOT)
    a = ap.parse_args()

    for p in (a.joint, a.fcmat, a.nwp):
        if not Path(p).exists():
            _die("bulunamadi: %s" % p)

    df = pd.read_csv(a.joint, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    N = len(df)
    y = (df.solar_mw / df.iloc[:M.SPLIT].solar_mw.max() * SP).to_numpy()
    hour = df.index.hour.to_numpy()
    naive = np.load(a.fcmat)["solar"]
    K = int(naive.shape[1])
    nwp = np.load(a.nwp)["solar"]

    idx = np.minimum(np.arange(N)[:, None] + np.arange(1, K + 1)[None, :], N - 1)
    truth = y[idx]
    hr = hour[idx]
    sl = slice(M.SPLIT, N - 1)

    print("=" * 76)
    print("Z16 ADIM 2 - KAYDIRILMIS TAHMINLERIN BECERISI")
    print("=" * 76)
    print("  test blogu %d:%d | K=%d" % (M.SPLIT, N - 1, K))

    # -------------------------------------------------- alpha hesaplari
    mw0, al0, tl0 = mae_alpha(nwp, naive, truth, hr, sl)
    print("\n  CIPA: gercek NWP alpha = %.3f  (beklenen %.3f)" % (al0, REF_ALPHA))
    if abs(al0 - REF_ALPHA) > TOL_ALPHA:
        print("  -> KALDI. Hicbir hukum verilmez.")
        sys.exit(2)
    print("  -> GECTI")

    alphas = {"NWP": al0}
    tails = {"NWP": tl0}
    maes = {"NWP": mw0}
    print("\n" + "=" * 76)
    print("KAYDIRILMIS MATRISLERIN BECERISI (test blogu)")
    print("=" * 76)
    print("  %-10s %9s %8s %9s   %s" % ("kol", "MAE", "alpha", "kuyruk", "beceri"))
    print("  " + "-" * 62)
    print("  %-10s %9.4f %8.3f %9.3f   %s" % ("NWP", mw0, al0, tl0, "naive'i yeniyor"))
    for d in SHIFTS:
        sh = d * 48
        ii = np.clip(np.arange(len(nwp)) - sh, 0, len(nwp) - 1)
        mw, al, tl = mae_alpha(nwp[ii], naive, truth, hr, sl)
        tag = "plac%d" % d
        alphas[tag], tails[tag], maes[tag] = al, tl, mw
        skill = "naive'i yeniyor" if al < 1.0 else "naive'den KOTU"
        print("  %-10s %9.4f %8.3f %9.3f   %s" % (tag, mw, al, tl, skill))
    alphas["naive"] = 1.000
    alphas["oracle"] = 0.000

    # ---------------------------------------------- yorungeler + bootstrap
    V, P, dones = {}, {}, {}
    for tag, path, _ in TRAJ:
        if not Path(path).exists():
            _die("yorunge yok: %s" % path)
        z = np.load(path)
        d = int(z["done"])
        V[tag] = ((z["mn"] < M.LO) | (z["mx"] > M.HI)).astype(np.float64)[:d]
        P[tag] = z["p"][:d]
        dones[tag] = d

    need = BLOCK * NDAYS
    if min(dones.values()) < need:
        _die("yorunge cok kisa")

    T0 = M.SPLIT
    prb = M.price[T0:T0 + need].reshape(NDAYS, BLOCK)
    Vb = {t: V[t][:need].reshape(NDAYS, BLOCK) for t in V}
    Pb = {t: P[t][:need].reshape(NDAYS, BLOCK) for t in P}
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, NDAYS, size=(a.nboot, NDAYS))
    BV = {t: (Vb[t].mean(axis=1))[draws].mean(axis=1) * 100.0 for t in Vb}

    print("\n" + "=" * 76)
    print("SEVIYELER - alpha'ya gore siralanmis")
    print("=" * 76)
    order = sorted(V.keys(), key=lambda t: alphas[t])
    print("  %-10s %8s %10s %11s" % ("kol", "alpha", "ihlal %", "kar EUR"))
    print("  " + "-" * 44)
    for t in order:
        d = dones[t]
        kk = float(-(M.price[T0:T0 + d] * P[t]).sum() * M.dt)
        print("  %-10s %8.3f %10.2f %11.1f" % (t, alphas[t], 100.0 * V[t].mean(), kk))

    print("\n" + "=" * 76)
    print("BITISIK CIFTLER (alpha sirasinda), eslestirilmis bootstrap")
    print("=" * 76)
    print("  %-22s %9s %7s %6s  %s" % ("cift", "fark pp", "SE", "kat", "hukum"))
    print("  " + "-" * 62)
    mono = True
    for i in range(len(order) - 1):
        A, B_ = order[i + 1], order[i]      # alpha buyuk - alpha kucuk
        dv = BV[A] - BV[B_]
        m_, s_ = float(dv.mean()), float(dv.std(ddof=1))
        k = abs(m_) / s_ if s_ > 0 else float("inf")
        if m_ < 0:
            mono = False
        print("  %-22s %+9.2f %7.2f %6.1f  %s"
              % (A + " - " + B_, m_, s_, k, "GERCEK" if k >= 2.0 else "yorumlanmaz"))

    # ------------------------------------------------------------- hukum
    print("\n" + "=" * 76)
    print("ON-KAYITLI HUKUM")
    print("=" * 76)
    a3 = alphas["plac3"]
    if a3 < 1.0:
        print("  HUKUM (A): alpha_3 = %.3f < 1" % a3)
        print("    -> 3-gun kaydirma SAF PLASEBO DEGIL; olculebilir beceri tasiyor.")
        print("       Makalede plasebo seti '7 gun ve uzeri' olarak tanimlanir;")
        print("       3 gun AYRI raporlanir, gizlenmez.")
    else:
        print("  HUKUM (B): alpha_3 = %.3f >= 1" % a3)
        print("    -> Kaydirilmis tahminin BECERISI YOK ama ihlali dusuruyor.")
        print("       CIDDI UYARI. Kazancin bir kismi tahmin iceriginden degil.")
        print("       Aynen yazilir, ACIKLAMA UYDURULMAZ.")

    print("\n  MONOTONLUK (alpha artarken ihlal artiyor mu):")
    print("    seviyeler bazinda: %s" % ("MONOTON" if mono else "MONOTON DEGIL"))
    print("    NOT: yukaridaki ciftlerden 2 SE'yi asmayanlar YORUMLANMAZ.")
    print("    Tum bitisik ciftler yorumlanamiyorsa siralama okunmaz.\n")


if __name__ == "__main__":
    main()