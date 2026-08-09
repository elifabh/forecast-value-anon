#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_z19_noweather.py - Z19: HIC HAVA BILGISI OLMAYAN tahminci.

NEDEN
  Z18 olctu: 1 gun kaydirilmis (YANLIS gunun havasi) NWP tahmini, gercek
  NWP'den ayirt edilemiyor (0.21 +/- 0.36, 0.6 SE) ama naive'den olculebilir
  sekilde iyi (-1.33 +/- 0.48, 2.8 SE).
  Yani 1.48 puanlik kazanc DOGRU HAVAYI BILMEKTEN gelmiyor gorunuyor.

  Kalan aday: LightGBM tahmincisi iki sey uretiyor -
    (a) o gunun havasi (GHI girdisinden)
    (b) egitimden ogrenilen GUNLUK + MEVSIMSEL SEKIL (takvim ozelliklerinden)
  Kaydirma (a)'yi bozar, (b)'yi bozmaz. Bu script (a)'yi TAMAMEN kaldirir.

BU SCRIPT NE YAPAR
  Ayni LightGBM, ayni cap-norm, ayni fit dilimi (0:13800), ayni seed 42.
  TEK FARK: GHI kolonlari YOK. Iki degisken uretilir:
    cal     : sadece takvim (sin_h, cos_h, sin_d, cos_d)          -> SIFIR hava
    calcap  : takvim + cap_est (30-gun yuvarlanan tepe, 48 gecikmeli)
              cap_est hava degil, KURULU KAPASITE vekilidir; ayri tutulur ki
              "sekil" ile "seviye" karismasin.

ON-KAYIT (koda girmeden, sonuclara bakilmadan)
  Olculecek : cal ve calcap kollarinin ihlal orani (MPC, ana boru hatti).
  HUKUM (A) : cal ihlali gercek NWP'den 2 SE ile AYRISMIYOR
              -> Kazancin cogu OGRENILMIS SEKILDEN geliyor, havadan degil.
                 MANSET DEGISIR. Yeni cerceve: "gun-oncesi hava tahminine
                 atfedilen voltaj iyilesmesinin cogu, hic hava bilgisi
                 icermeyen ogrenilmis bir gunluk profille elde ediliyor."
                 Bu KATKI 3 ile ayni ailedendir ve onu tamamlar.
  HUKUM (B) : cal ihlali naive'den (11.28) 2 SE ile AYRISMIYOR
              -> Hava ONEMLI. Kisa kaydirma anomalisi ACIKLANAMAMIS bir
                 SINIR olarak kalir; manset korunur ama sinir yazilir.
  HUKUM (C) : ikisinin arasinda, her ikisinden de ayrisiyor
              -> Kazanc BOLUNMUS. Sekil payi ve hava payi ayri raporlanir,
                 oran verilir, sebep UYDURULMAZ.

  CIPA : bu script'in urettigi tahminciler naive'den KOTU MAE vermeli
         (alpha > 1). Hava yoksa nokta tahmini kotudur; iyi cikarsa
         sizinti vardir ve DURULUR.

CALISTIR (GG-1 kok klasorunden, ~2 dk):
  python gg2/gg2_z19_noweather.py 2>&1 | tee gg2/z19_noweather.log
Sonra MPC (iki pencere, ~15 dk):
  python gg2/gg2_nwp_mpc_fair.py --nwp gg2/data/nwp_solar_fc_cal.npz \
      --out gg2/nwp_fair_cal.npz 2>&1 | tee gg2/nwp_fair_cal.log
  python gg2/gg2_nwp_mpc_fair.py --nwp gg2/data/nwp_solar_fc_calcap.npz \
      --out gg2/nwp_fair_calcap.npz 2>&1 | tee gg2/nwp_fair_calcap.log
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SP = 3.0
SPLIT = 13800
CAP_WIN = 30 * 48
LAG = 48
SEED = 42
SOLAR_H = (10, 16)
REF_ALPHA = 0.799
TOL_ALPHA = 0.002


def _die(m: str) -> None:
    sys.exit("[HATA] " + m)


def fit_lgbm(X, y, tr, seed):
    """gg2_nwp_build_fc.fit_pred'in lgbm dali ile BIREBIR ayni parametreler."""
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                          min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                          random_state=seed, verbose=-1)
    m.fit(X[tr], y[tr])
    return m.predict(X)


def main() -> None:
    ap = argparse.ArgumentParser(description="Z19: hava bilgisi olmayan tahminci.")
    ap.add_argument("--joint", default="data/aligned_30min_joint.csv")
    ap.add_argument("--fcmat", default="data/forecast_matrix.npz")
    ap.add_argument("--ref", default="gg2/data/nwp_solar_fc_capnorm.npz")
    ap.add_argument("--outdir", default="gg2/data")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    for p in (a.joint, a.fcmat, a.ref):
        if not Path(p).exists():
            _die("bulunamadi: %s  (kok klasorden mi calistiriyorsun?)" % p)

    df = pd.read_csv(a.joint, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    N = len(df)
    y = (df.solar_mw / df.iloc[:SPLIT].solar_mw.max() * SP).to_numpy()
    hour = df.index.hour.to_numpy()
    naive = np.load(a.fcmat)["solar"]
    K = int(naive.shape[1])

    ref = np.load(a.ref, allow_pickle=True)
    ref_fc = ref["solar"]

    print("=" * 76)
    print("Z19 - HIC HAVA BILGISI OLMAYAN TAHMINCI")
    print("=" * 76)
    print("  joint %d adim | K=%d | fit 0:%d | seed %d" % (N, K, SPLIT, SEED))
    print("  referans NWP alpha (diskten): %.3f" % float(ref["a_mean"]))

    s = pd.Series(y, index=df.index)
    cap = (s.rolling(CAP_WIN, min_periods=48).max().shift(LAG)
             .bfill().clip(lower=0.05).to_numpy())
    yn = y / cap

    hod = df.index.hour + df.index.minute / 60.0
    doy = df.index.dayofyear.to_numpy().astype(float)
    cal = {
        "sin_h": np.sin(2 * np.pi * hod / 24.0),
        "cos_h": np.cos(2 * np.pi * hod / 24.0),
        "sin_d": np.sin(2 * np.pi * doy / 365.25),
        "cos_d": np.cos(2 * np.pi * doy / 365.25),
    }
    X_cal = np.column_stack([cal[k] for k in cal])
    X_calcap = np.column_stack([cal[k] for k in cal] + [cap])

    idx = np.minimum(np.arange(N)[:, None] + np.arange(1, K + 1)[None, :], N - 1)
    truth = y[idx]
    hr = hour[idx]
    sl = slice(SPLIT, N - 1)

    def alpha_of(mat):
        mn = float(np.abs(naive[sl] - truth[sl]).mean())
        mw = float(np.abs(mat[sl] - truth[sl]).mean())
        h = hr[sl].ravel()
        mm = (h >= SOLAR_H[0]) & (h <= SOLAR_H[1])
        qn = float(np.quantile(np.abs(naive[sl] - truth[sl]).ravel()[mm], 0.95))
        qw = float(np.quantile(np.abs(mat[sl] - truth[sl]).ravel()[mm], 0.95))
        return mw, mw / mn, qw / qn

    # ------------------------------------------------------------- CIPA
    _, a_ref, _ = alpha_of(ref_fc)
    print("\n  CIPA: diskteki NWP matrisinin alpha'si %.3f (beklenen %.3f)"
          % (a_ref, REF_ALPHA))
    if abs(a_ref - REF_ALPHA) > TOL_ALPHA:
        print("  -> KALDI. Hicbir dosya yazilmaz.")
        sys.exit(2)
    print("  -> GECTI")

    # ------------------------------------------------------------ egitim
    print("\n" + "=" * 76)
    print("EGITIM (GHI kolonu YOK)")
    print("=" * 76)
    print("  %-10s %-34s %9s %8s %8s" % ("kol", "ozellikler", "MAE", "alpha", "kuyruk"))
    print("  " + "-" * 66)
    out = {}
    specs = [
        ("cal",    X_cal,    "sin_h cos_h sin_d cos_d"),
        ("calcap", X_calcap, "sin_h cos_h sin_d cos_d cap_est"),
    ]
    for tag, X, desc in specs:
        pred = np.clip(fit_lgbm(X, yn, slice(0, SPLIT), SEED), 0.0, None) * cap
        fc = pred[idx].astype(np.float32)
        mw, al, tl = alpha_of(fc.astype(np.float64))
        out[tag] = (fc, al, tl)
        print("  %-10s %-34s %9.4f %8.3f %8.3f" % (tag, desc, mw, al, tl))
    print("  %-10s %-34s %9s %8.3f %8s"
          % ("naive", "dun tekrarlanir", "-", 1.000, "1.000"))
    print("  %-10s %-34s %9s %8.3f %8s"
          % ("NWP", "GHI 6 nokta + takvim + cap_est", "-", a_ref, "0.926"))

    # ------------------------------------------------------- cipa (sizinti)
    print("\n  SIZINTI KONTROLU: hava yoksa alpha > 1 olmali")
    bad = [t for t in out if out[t][1] <= 1.0]
    for tag in ("cal", "calcap"):
        al = out[tag][1]
        print("    %-8s alpha %.3f  -> %s" % (tag, al, "OK" if al > 1.0 else "SUPHELI"))
    if bad:
        print("\n  DUR: %s alpha <= 1. Hava bilgisi olmadan naive yenilemez;" % ", ".join(bad))
        print("  bu bir SIZINTI belirtisidir. Dosya YAZILMADI, sebep aranir.")
        sys.exit(2)

    # ------------------------------------------------------------- yazim
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 76)
    print("YAZIM")
    print("=" * 76)
    for tag in ("cal", "calcap"):
        fc, al, tl = out[tag]
        op = outdir / ("nwp_solar_fc_%s.npz" % tag)
        if op.exists() and not a.force:
            print("  %-8s ZATEN VAR, ezilmedi: %s  (--force ile ez)" % (tag, op))
            continue
        np.savez(op, solar=fc, model="lgbm", a_mean=al, a_tail=tl,
                 K=K, split=SPLIT, seed=SEED, weather="NONE")
        print("  %-8s yazildi: %s" % (tag, op))

    print("\n" + "=" * 76)
    print("SONRAKI ADIM - MPC (iki pencere, ~15 dk)")
    print("=" * 76)
    for tag in ("cal", "calcap"):
        print("  python gg2/gg2_nwp_mpc_fair.py \\")
        print("      --nwp gg2/data/nwp_solar_fc_%s.npz \\" % tag)
        print("      --out gg2/nwp_fair_%s.npz 2>&1 | tee gg2/nwp_fair_%s.log" % (tag, tag))
        print()
    print("  Kiyas noktalari: gercek NWP %9.80 | naive %11.28 | p1 %9.96")
    print("  Hukum (A/B/C) bootstrap ile verilir, seviyeye bakilarak DEGIL.\n")


if __name__ == "__main__":
    main()