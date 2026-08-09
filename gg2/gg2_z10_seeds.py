#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_z10_seeds.py - Z10: tahminci sonucu SEED'e duyarli mi?

SORUN
  Butun tahminci egitimleri tek seed ile yapildi (random_state=42). Makalenin
  ana sayisi (alpha = 0.799, MAE azaltmasi %20) baska bir seed'de baska cikarsa,
  "MAE %20 iyilesti -> ihlal 1.48 puan dustu" zincirinin ilk halkasi oynak demektir.

BU SCRIPT NE YAPAR
  Cap-norm NWP solar tahmincisini UC seed ile yeniden egitir (42, 1, 7) ve iki
  yerde alpha'yi olcer:
    [A] VALIDATION (fit 2400:13800, olcum 96:2352)  -> Z9 karari seed'e duyarli mi
    [B] TEST       (fit 0:13800,    olcum 13800:N-1) -> ana sayi seed'e duyarli mi

  Ridge'de random_state yok; sadece LightGBM seed'e duyarli, o yuzden lgbm.

  DIKKAT: [B] bir POST-HOC ROBUSTLUK KONTROLUDUR, model SECIMI DEGILDIR.
  Secim Z9'da validation'da yapildi ve degistirilmiyor. Burada test blogunda
  hicbir karar alinmaz, sadece mevcut kararin oynakligi raporlanir.

ON-KAYIT (kod yazilmadan once, duz Turkce)
  Olculecek : alpha'nin uc seed uzerindeki YAYILIMI (max - min).
  Esik      : 0.02 alpha birimi -- Z9'da kullanilan EPS ile ayni.
              Gerekce: alpha bloklar arasi 0.577-0.799 oynuyor; bundan kucuk
              fark o gurultunun icinde kaybolur.

  HUKUM:
   (1) yayilim < 0.02 -> "Sonuc seed'e duyarsiz. Tek seed yeterli; makalede
       ortalama ve yayilim birlikte raporlanir."
   (2) yayilim >= 0.02 -> "Sonuc seed'e duyarli. Makalede TEK bir alpha degil,
       uc seed'in ortalamasi +/- yayilimi raporlanir ve bu bir SINIR olarak
       yazilir." Ana tablonun MPC kosumu YENIDEN KOSULMAZ (pahali); bunun
       yerine tahminci oynakligi acikca belirtilir.
   (3) Ara durum yok. Sebep UYDURULMAZ.

  AYRICA: [A]'da cap-norm - ham farki uc seed'de de -0.02'nin ALTINDA kaliyor mu?
  Kalmiyorsa Z9'un gerekcesi seed'e bagimli demektir; bu da yazilir.

CALISTIR (kok klasorden, ~3 dk):
  python gg2/gg2_z10_seeds.py 2>&1 | tee gg2/z10_seeds.log
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*valid feature names.*")

SP = 3.0
SPLIT = 13800
VAL_END = 2400
CAP_WIN = 30 * 48
LAG = 48
WARM = 96
EPS = 0.02
SEEDS = (42, 1, 7)


def _die(m: str) -> None:
    sys.exit("[HATA] " + m)


def ghi30(joint: pd.DataFrame, ghi: pd.DataFrame, tag: str = "p1") -> pd.DataFrame:
    cols = [c for c in ghi.columns if c.startswith("ghi_" + tag + "_")]
    if not cols:
        _die("ghi_" + tag + "_* kolonu yok")
    g = ghi.set_index("datetime_utc")[cols].sort_index()
    g30 = g.reindex(g.index.union(joint.index)).interpolate(method="time").reindex(joint.index)
    if g30.isna().any().any():
        _die("hizalamadan sonra bos GHI var")
    return g30


def base_feats(joint: pd.DataFrame, g30: pd.DataFrame) -> dict:
    """gg2_nwp_build_fc.make_X ile BIREBIR ayni ozellikler (cap_est haric)."""
    mean = g30.mean(axis=1)
    f = {c: g30[c].to_numpy() for c in g30.columns}
    f["ghi_mean"] = mean.to_numpy()
    f["ghi_lag1"] = mean.shift(1).bfill().to_numpy()
    f["ghi_lead1"] = mean.shift(-1).ffill().to_numpy()
    hod = joint.index.hour + joint.index.minute / 60.0
    doy = joint.index.dayofyear.to_numpy().astype(float)
    f["sin_h"] = np.sin(2.0 * np.pi * hod / 24.0)
    f["cos_h"] = np.cos(2.0 * np.pi * hod / 24.0)
    f["sin_d"] = np.sin(2.0 * np.pi * doy / 365.25)
    f["cos_d"] = np.cos(2.0 * np.pi * doy / 365.25)
    return f


def fit_lgbm(X: np.ndarray, y: np.ndarray, tr: slice, seed: int) -> np.ndarray:
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                          min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                          random_state=seed, verbose=-1)
    m.fit(X[tr], y[tr])
    return m.predict(X)


def _idx(n: int, K: int) -> np.ndarray:
    return np.minimum(np.arange(n)[:, None] + np.arange(1, K + 1)[None, :], n - 1)


def mae_series(pred: np.ndarray, y: np.ndarray, K: int, r0: int, r1: int) -> float:
    ix = _idx(len(y), K)
    return float(np.abs(pred[ix][r0:r1] - y[ix][r0:r1]).mean())


def mae_matrix(mat: np.ndarray, y: np.ndarray, K: int, r0: int, r1: int) -> float:
    ix = _idx(len(y), K)
    return float(np.abs(mat[r0:r1] - y[ix][r0:r1]).mean())


def report(name: str, vals: list, unit: str = "") -> float:
    lo, hi = min(vals), max(vals)
    spread = hi - lo
    print("  %-22s %s | ort %.3f | yayilim %.3f%s"
          % (name, " ".join("%.3f" % v for v in vals), float(np.mean(vals)),
             spread, unit))
    return spread


def main() -> None:
    ap = argparse.ArgumentParser(description="Z10: tahminci seed duyarliligi.")
    ap.add_argument("--joint", default="data/aligned_30min_joint.csv")
    ap.add_argument("--fcmat", default="data/forecast_matrix.npz")
    ap.add_argument("--ghi", default="gg2/data/nwp_ghi_roi.csv")
    a = ap.parse_args()

    for p in (a.joint, a.fcmat, a.ghi):
        if not Path(p).exists():
            _die("bulunamadi: " + p)

    df = pd.read_csv(a.joint, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    n = len(df)
    y = (df.solar_mw / df.iloc[:SPLIT].solar_mw.max() * SP).to_numpy()
    naive = np.load(a.fcmat)["solar"]
    K = int(naive.shape[1])

    print("=" * 74)
    print("Z10 - TAHMINCI SEED DUYARLILIGI  (seed: %s)" % (", ".join(map(str, SEEDS))))
    print("=" * 74)
    print("  joint %d adim | K=%d | EPS=%.2f" % (n, K, EPS))

    g1 = ghi30(df, pd.read_csv(a.ghi, parse_dates=["datetime_utc"]), "p1")
    s = pd.Series(y, index=df.index)
    cap = s.rolling(CAP_WIN, min_periods=48).max().shift(LAG).bfill().clip(lower=0.05).to_numpy()

    f_raw = base_feats(df, g1)
    X_raw = np.column_stack([f_raw[k] for k in f_raw])
    f_cap = dict(f_raw)
    f_cap["cap_est"] = cap
    X_cap = np.column_stack([f_cap[k] for k in f_cap])

    # ---------------------------------------------------- [A] VALIDATION (Z9)
    v0, v1 = WARM, VAL_END - K
    tr_val = slice(VAL_END, SPLIT)
    m_nai_v = mae_matrix(naive, y, K, v0, v1)

    print("\n" + "=" * 74)
    print("[A] VALIDATION  (fit %d:%d, olcum %d:%d) - Z9 karari" % (VAL_END, SPLIT, v0, v1))
    print("=" * 74)
    a_raw_v, a_cap_v, d_v = [], [], []
    for sd in SEEDS:
        p_raw = np.clip(fit_lgbm(X_raw, y, tr_val, sd), 0.0, None)
        p_cap = np.clip(fit_lgbm(X_cap, y / cap, tr_val, sd), 0.0, None) * cap
        ar = mae_series(p_raw, y, K, v0, v1) / m_nai_v
        ac = mae_series(p_cap, y, K, v0, v1) / m_nai_v
        a_raw_v.append(ar)
        a_cap_v.append(ac)
        d_v.append(ac - ar)
        print("    seed %-3d  ham %.3f | cap-norm %.3f | fark %+.3f" % (sd, ar, ac, ac - ar))
    print()
    report("alpha ham (val)", a_raw_v)
    report("alpha cap-norm (val)", a_cap_v)
    sp_d = report("cap-norm - ham (val)", d_v)
    z9_ok = all(x < -EPS for x in d_v)
    print("\n  Z9 kontrolu: uc seed'de de fark < -EPS(%.2f) mi -> %s"
          % (EPS, "EVET" if z9_ok else "HAYIR"))

    # ---------------------------------------------------------- [B] TEST
    t0, t1 = SPLIT, n - 1
    tr_test = slice(0, SPLIT)
    m_nai_t = mae_matrix(naive, y, K, t0, t1)

    print("\n" + "=" * 74)
    print("[B] TEST  (fit %d:%d, olcum %d:%d) - POST-HOC ROBUSTLUK, SECIM DEGIL"
          % (0, SPLIT, t0, t1))
    print("=" * 74)
    a_cap_t = []
    for sd in SEEDS:
        p_cap = np.clip(fit_lgbm(X_cap, y / cap, tr_test, sd), 0.0, None) * cap
        ac = mae_series(p_cap, y, K, t0, t1) / m_nai_t
        a_cap_t.append(ac)
        print("    seed %-3d  alpha %.3f  (MAE azaltmasi %%%.1f)" % (sd, ac, 100 * (1 - ac)))
    print()
    sp_t = report("alpha cap-norm (test)", a_cap_t)
    print("\n  Yayinlanan deger (seed 42): 0.799")

    # ------------------------------------------------------------------ hukum
    print("\n" + "=" * 74)
    print("ON-KAYITLI HUKUM")
    print("=" * 74)
    print("  TEST alpha yayilimi = %.3f  (esik %.2f)" % (sp_t, EPS))
    if sp_t < EPS:
        print("  -> SEED'E DUYARSIZ. Tek seed yeterli.")
        print("     Makalede: alpha = %.3f +/- %.3f (3 seed)."
              % (float(np.mean(a_cap_t)), sp_t / 2))
    else:
        print("  -> SEED'E DUYARLI. Makalede TEK alpha degil, uc seed'in")
        print("     ortalamasi +/- yayilimi raporlanir ve SINIR olarak yazilir.")
        print("     Ana tablonun MPC kosumu yeniden KOSULMAZ (pahali); bunun")
        print("     yerine tahminci oynakligi acikca belirtilir.")
    print("\n  VALIDATION cap-norm-ham yayilimi = %.3f" % sp_d)
    if z9_ok:
        print("  -> Z9'un gerekcesi uc seed'de de ayakta.")
    else:
        print("  -> Z9'un gerekcesi SEED'E BAGIMLI. Makalede boyle yazilir;")
        print("     'validation kazandi' yerine 'seed'e gore degisiyor' denir.")
    print("\n  NOT: Bu script test blogunda HICBIR KARAR ALMAZ. Model secimi Z9'da")
    print("  validation'da yapildi ve degistirilmedi.\n")


if __name__ == "__main__":
    main()