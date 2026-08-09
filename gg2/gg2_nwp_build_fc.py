#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_nwp_build_fc.py - ADIM 4a: kapasite-normalize NWP solar tahmin matrisi.

NEDEN v2: Adim 3'te ham NWP tahmincisi naive'e KAYBETTI (alpha=1.096).
Adim 3b tanisi sebebi buldu: ROI kurulu solar kapasitesi pencere icinde
x1.51 buyumus (oran 0.00286 -> 0.00433). daily-naive bundan MUAF (dun zaten
bugunun kapasitesiyle uretti); eski veriyle egitilmis radyasyon->MW modeli DEGIL.
Cozum: hedefi kapasiteye normalize et, tahmin et, geri carp.
Tani sonucu: alpha 1.096 -> 0.799 (naive yenildi).

NEDENSELLIK (kritik - C'de dustugumuz hata sinifi):
  cap_est[T] = son 30 gunun tepesi, 48 adim (24h) GECIKMELI  -> T-48'e kadarki veri
  NWP        = shortwave_radiation_previous_day1 = T'den 24h once tahmin
  (i,k) hedefi i+k+1;  k<=47 icin her iki girdi de i aninda BILINIR.  Script kontrol eder.

PROTOKOL: model secimi SADECE validation (0:2400). fit 0:13800. Test'e dokunulmaz.

CIKTI: gg2/data/nwp_solar_fc_capnorm.npz  ->  anahtar 'solar', shape (N,K)
CALISTIR (GG-1 kok klasorunden):  python gg2/gg2_nwp_build_fc.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*valid feature names.*")

SP = 3.0            # gg2_solar_alpha_sweep.py ile birebir
SPLIT = 13800
VAL_END = 2400
SOLAR_H = (10, 16)
CAP_WIN = 30 * 48   # 30 gun
LAG = 48            # 24h gecikme -> nedensellik


def _die(m: str) -> None:
    sys.exit(f"[HATA] {m}")


def ghi30(joint, ghi, tag="p1"):
    cols = [c for c in ghi.columns if c.startswith(f"ghi_{tag}_")]
    if not cols:
        _die(f"ghi_{tag}_* kolonu yok")
    g = ghi.set_index("datetime_utc")[cols].sort_index()
    g30 = g.reindex(g.index.union(joint.index)).interpolate(method="time").reindex(joint.index)
    if g30.isna().any().any():
        _die("hizalamadan sonra bos GHI var")
    return g30


def make_X(joint, g30, cap):
    mean = g30.mean(axis=1)
    f = {c: g30[c].to_numpy() for c in g30.columns}
    f["ghi_mean"] = mean.to_numpy()
    f["ghi_lag1"] = mean.shift(1).bfill().to_numpy()
    f["ghi_lead1"] = mean.shift(-1).ffill().to_numpy()
    hod = joint.index.hour + joint.index.minute / 60.0
    doy = joint.index.dayofyear.to_numpy().astype(float)
    f["sin_h"] = np.sin(2 * np.pi * hod / 24.0)
    f["cos_h"] = np.cos(2 * np.pi * hod / 24.0)
    f["sin_d"] = np.sin(2 * np.pi * doy / 365.25)
    f["cos_d"] = np.cos(2 * np.pi * doy / 365.25)
    f["cap_est"] = cap
    return np.column_stack([f[k] for k in f])


def fit_pred(X, y, tr, name):
    if name == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    else:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                              min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                              random_state=42, verbose=-1)
    m.fit(X[tr], y[tr])
    return m.predict(X)


def main() -> None:
    ap = argparse.ArgumentParser(description="GG-2 Adim 4a: cap-norm NWP tahmin matrisi.")
    ap.add_argument("--joint", default="data/aligned_30min_joint.csv")
    ap.add_argument("--fcmat", default="data/forecast_matrix.npz")
    ap.add_argument("--ghi", default="gg2/data/nwp_ghi_roi.csv")
    ap.add_argument("--out", default="gg2/data/nwp_solar_fc_capnorm.npz")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and not a.force:
        _die(f"cikti zaten var, ezilmedi: {out}  (--force ile ez)")
    for p in (a.joint, a.fcmat, a.ghi):
        if not Path(p).exists():
            _die(f"bulunamadi: {p}")

    df = pd.read_csv(a.joint, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    N = len(df)
    y = (df.solar_mw / df.iloc[:SPLIT].solar_mw.max() * SP).to_numpy()
    hour = df.index.hour.to_numpy()
    naive = np.load(a.fcmat)["solar"]
    K = naive.shape[1]

    print("=" * 72)
    print("ADIM 4a - kapasite-normalize NWP solar tahmin matrisi")
    print(f"  joint {N} adim | K={K} | test {SPLIT}:{N-1}")
    if K - 1 > 47:
        _die(f"K-1={K-1} > 47 -> day1 NWP i aninda henuz yayinlanmamis olurdu (gizli oracle)")
    print(f"  nedensellik: k_max={K-1} <= 47  OK")

    g1 = ghi30(df, pd.read_csv(a.ghi, parse_dates=["datetime_utc"]), "p1")
    s = pd.Series(y, index=df.index)
    cap = s.rolling(CAP_WIN, min_periods=48).max().shift(LAG).bfill().clip(lower=0.05).to_numpy()
    yn = y / cap
    X = make_X(df, g1, cap)

    # --- model secimi: SADECE validation
    print("\n  MODEL SECIMI (validation 0:2400, val MAE - orijinal olcekte)")
    scores = {}
    for nm in ("ridge", "lgbm"):
        try:
            p = np.clip(fit_pred(X, yn, slice(VAL_END, SPLIT), nm), 0, None) * cap
        except ImportError as e:
            print(f"    {nm:6s} atlandi ({e})")
            continue
        scores[nm] = float(np.abs(p[:VAL_END] - y[:VAL_END]).mean())
        print(f"    {nm:6s} val MAE = {scores[nm]:.4f}")
    if not scores:
        _die("model egitilemedi")
    best = min(scores, key=scores.get)
    print(f"    -> secilen: {best}")

    pred = np.clip(fit_pred(X, yn, slice(0, SPLIT), best), 0, None) * cap

    idx = np.minimum(np.arange(N)[:, None] + np.arange(1, K + 1)[None, :], N - 1)
    fc, truth, hr = pred[idx], y[idx], hour[idx]
    sl = slice(SPLIT, N - 1)

    def m(e, h):
        a_ = np.abs(e).ravel(); mm = (h.ravel() >= SOLAR_H[0]) & (h.ravel() <= SOLAR_H[1])
        return a_.mean(), np.quantile(a_[mm], 0.95)

    mn, qn = m(naive[sl] - truth[sl], hr[sl])
    mw, qw = m(fc[sl] - truth[sl], hr[sl])
    print("\n" + "=" * 72)
    print(f"OLCUM - TEST ({SPLIT}:{N-1})")
    print("=" * 72)
    print(f"  {'':20s} {'MAE':>8s} {'alpha':>7s} {'q95':>8s} {'alpha_q95':>10s}")
    print(f"  {'daily-naive':20s} {mn:8.4f} {1.0:7.3f} {qn:8.4f} {1.0:10.3f}")
    print(f"  {'NWP cap-norm':20s} {mw:8.4f} {mw/mn:7.3f} {qw:8.4f} {qw/qn:10.3f}")
    print(f"\n  alpha_gercek = {mw/mn:.3f}  (MAE azaltmasi %{100*(1-mw/mn):.1f})")
    print(f"  alpha_kuyruk = {qw/qn:.3f}  (kuyruk azaltmasi %{100*(1-qw/qn):.1f})")
    if mw >= mn:
        _die("cap-norm tahminci naive'i yenemedi -> Adim 4b'ye GECILMEZ, hukum verilemez")
    print("\n  Naive yenildi -> Adim 4b (MPC) calistirilabilir.")
    print(f"  Supurmedeki en yakin nokta: alpha ~ "
          f"{min([0.25,0.50,0.75], key=lambda x: abs(x-mw/mn)):.2f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, solar=fc.astype(np.float32), model=best, a_mean=mw / mn,
             a_tail=qw / qn, K=K, split=SPLIT)
    print(f"\n  yazildi: {out}  (shape {fc.shape}, model={best})")


if __name__ == "__main__":
    main()