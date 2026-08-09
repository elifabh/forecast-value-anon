#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_z17_peak.py - Z17b: MPC'yi baglayan sey ORTALAMA degil, UFUKTAKI TEPE.

Z17a NE YAPTI, NEDEN YETMEDI
  Tahminin solar-saat ORTALAMA isaretli hatasi olculdu. plac3 naive'den DAHA AZ
  tahmin ediyordu (-0.050 vs -0.027) ama yine de bataryayi daha bos tutup daha
  az ihlal veriyordu. Zincir koptu, hipotez reddedildi.

  ATLANAN SEY: MPC'nin kisiti ortalama tahmini GORMEZ. `fc_windows_ldf`
  ufuktaki HER adim icin bir voltaj penceresi kurar ve cozucu bunlarin
  TEPESINE gore plan yapar. Yani baglayici nicelik:
      max_t  ( VM_BASE[i] + ldf_v(solar_fc[i,t]) - ldf_v(solar[i]) )
  Bir tahmin ortalamada dusuk kalip TEPEDE yuksek olabilir. Olculmedi.

BU SCRIPT NE OLCUYOR (iki katman, MPC kosmadan)
  [1] ufuktaki TEPE solar tahmini vs gercek tepe (isaretli hata)
  [2] ufuktaki TEPE ONGORULEN VOLTAJ - MPC'nin kisitinin gordugu asil sayi
      (fc_windows_ldf ile BIREBIR ayni formul, cozucu cagrilmadan)

ON-KAYIT (sonuclara bakilmadan)
  Hipotez: plac3 ufuk TEPESINDE naive'den YUKSEK voltaj ongoruyor -> MPC daha
  cok tehlike goruyor -> bataryayi daha bos tutuyor -> daha az ihlal.
  HUKUM (A): tepe ongorulen voltaj plac3 > naive VE alti kolda tepe-voltaj ile
             ihlal TERS yonde -> mekanizma OLCULDU, makaleye girer.
  HUKUM (B): plac3 <= naive -> mekanizma REDDEDILDI. "Aciklanamadi" yazilir.
             ACIKLAMA UYDURULMAZ. Anomali arastirmasi BURADA BITER.

  CIPA: naive kolunun tepe-voltaj ortalamasi, gercek NWP kolununkinden
        FARKLI olmali (ayni cikarsa kod solar_fc'yi kullanmiyordur -> bug).

CALISTIR (~2 dk, MPC/cozucu KOSMAZ):
  python gg2/gg2_z17_peak.py 2>&1 | tee gg2/z17_peak.log
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
import diag_mpc_forecastvolt as D

SP = 3.0
SHIFTS = (3, 7, 14, 21)
REF_ALPHA = 0.799
TOL_ALPHA = 0.002

TRAJ = {
    "NWP":    "gg2/nwp_fair_traj_nwp.npz",
    "plac3":  "gg2/data/placebo_shift3.npz",
    "plac7":  "gg2/data/placebo_shift7.npz",
    "plac14": "gg2/data/placebo_shift14.npz",
    "plac21": "gg2/data/placebo_shift21.npz",
    "naive":  "gg2/nwp_fair_traj_naive.npz",
}
ORDER = ["NWP", "plac3", "plac7", "plac14", "plac21", "naive"]


def _die(m: str) -> None:
    sys.exit("[HATA] " + m)


def spearman(x, y):
    def rank(v):
        return np.argsort(np.argsort(np.asarray(v, dtype=float))).astype(float)
    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Z17b: ufuk tepesi olcumu.")
    ap.add_argument("--joint", default="data/aligned_30min_joint.csv")
    ap.add_argument("--fcmat", default="data/forecast_matrix.npz")
    ap.add_argument("--nwp", default="gg2/data/nwp_solar_fc_capnorm.npz")
    a = ap.parse_args()

    for p in (a.joint, a.fcmat, a.nwp):
        if not Path(p).exists():
            _die("bulunamadi: %s" % p)

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

    al = float(np.abs(nwp_m[sl] - truth[sl]).mean()
               / np.abs(naive_m[sl] - truth[sl]).mean())
    if abs(al - REF_ALPHA) > TOL_ALPHA:
        _die("CIPA: NWP alpha %.3f, beklenen %.3f" % (al, REF_ALPHA))

    mats = {"NWP": nwp_m, "naive": naive_m}
    for d in SHIFTS:
        ii = np.clip(np.arange(len(nwp_m)) - d * 48, 0, len(nwp_m) - 1)
        mats["plac%d" % d] = nwp_m[ii]

    print("=" * 78)
    print("Z17b - UFUKTAKI TEPE (MPC kisitinin GORDUGU sey)")
    print("=" * 78)
    print("  test blogu %d:%d | K=%d | CIPA NWP alpha %.3f OK" % (M.SPLIT, N - 1, K, al))

    # ---------------------------------------------- [1] tepe solar tahmini
    print("\n" + "=" * 78)
    print("[1] UFUKTAKI TEPE SOLAR TAHMINI (max_k fc[i,k]) vs gercek tepe")
    print("=" * 78)
    tmax = truth[sl].max(axis=1)
    print("  %-8s %12s %12s %12s" % ("kol", "tepe ort", "tepe hata", "fazla tah.%"))
    print("  " + "-" * 48)
    L1 = {}
    for t in ORDER:
        pm = mats[t][sl].max(axis=1)
        err = pm - tmax
        L1[t] = float(err.mean())
        print("  %-8s %12.4f %12.4f %12.1f"
              % (t, float(pm.mean()), float(err.mean()), 100.0 * float((err > 0).mean())))

    # ------------------------------------- [2] tepe ongorulen voltaj (asil)
    print("\n" + "=" * 78)
    print("[2] UFUKTAKI TEPE ONGORULEN VOLTAJ - fc_windows_ldf ile ayni formul")
    print("=" * 78)
    print("  hesaplaniyor (~1-2 dk)...", flush=True)

    T0, T1 = M.SPLIT, M.N - 1
    n = T1 - T0
    peak = {t: np.zeros(n) for t in ORDER}
    for j in range(n):
        i = T0 + j
        Hh = min(M.H, M.N - i)
        v_now = D.ldf_v(M.solar[i], M.wind[i], M.dmult[i])
        for t in ORDER:
            fc = mats[t]
            best = M.VM_BASE[i].max()
            for s in range(1, Hh):
                kk = min(s - 1, fc.shape[1] - 1)
                v_fut = D.ldf_v(fc[i, kk], M.wind_fc[i, kk], M.dmult_fc[i, kk])
                vv = (M.VM_BASE[i] + (v_fut - v_now)).max()
                if vv > best:
                    best = vv
            peak[t][j] = best
        if (j + 1) % 800 == 0:
            print("    %d/%d" % (j + 1, n), flush=True)

    V, SOC = {}, {}
    for t in ORDER:
        z = np.load(TRAJ[t])
        d = int(z["done"])
        V[t] = 100.0 * float(((z["mn"] < M.LO) | (z["mx"] > M.HI))[:d].mean())
        SOC[t] = float(z["soc"][:d].mean())

    print("\n  %-8s %14s %14s %11s %10s"
          % ("kol", "tepe voltaj", "HI asma %", "SOC ort", "ihlal %"))
    print("  " + "-" * 60)
    L2 = {}
    for t in ORDER:
        pk = peak[t]
        L2[t] = float(pk.mean())
        print("  %-8s %14.5f %14.1f %11.3f %10.2f"
              % (t, pk.mean(), 100.0 * float((pk > M.HI).mean()), SOC[t], V[t]))

    # ------------------------------------------------------------- hukum
    pv = [L2[t] for t in ORDER]
    vi = [V[t] for t in ORDER]
    r = spearman(pv, vi)
    print("\n" + "=" * 78)
    print("ON-KAYITLI HUKUM")
    print("=" * 78)
    print("  tepe voltaj vs ihlal  rho = %+.3f  (hipotez: NEGATIF)" % r)
    d3 = L2["plac3"] - L2["naive"]
    print("  tepe voltaj farki (plac3 - naive) = %+.6f  %s"
          % (d3, "plac3 DAHA TEHLIKELI goruyor" if d3 > 0 else "plac3 DAHA GUVENLI goruyor"))
    print()
    if d3 > 0 and r < 0:
        print("  HUKUM (A): MEKANIZMA OLCULDU.")
        print("    plac3 ufuk tepesinde daha yuksek voltaj ongoruyor -> MPC daha")
        print("    cok tehlike goruyor -> bataryayi daha bos tutuyor -> az ihlal.")
        print("    Ortalama tahmin hatasi bunu GIZLIYOR; baglayici nicelik tepedir.")
        print("    -> Makaleye girer ve KATKI 3'u guclendirir.")
    else:
        print("  HUKUM (B): MEKANIZMA REDDEDILDI.")
        print("    Makaleye 'aciklanamadi' yazilir. ACIKLAMA UYDURULMAZ.")
        print("    Anomali arastirmasi BURADA BITER. Yazim fazina gecilir.")
    print()


if __name__ == "__main__":
    main()