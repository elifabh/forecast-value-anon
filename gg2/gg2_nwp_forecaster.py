#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_nwp_forecaster.py - ADIM 3: NWP tabanli solar tahminci + olcum.

NE YAPAR
  1. gg2/data/nwp_ghi_roi.csv (saatlik GHI, day1 = 24h once tahmin) -> 30-dk izgara
  2. Ozellik matrisi kurar (6 nokta GHI + zaman ozellikleri)
  3. Ridge ve LightGBM'i egitir; SECIM SADECE VALIDATION'da (0:2400) - <author>'in kurali
  4. TEST bloğunda (13800:N-1) olcer: MAE + kuyruk (q95), daily-naive'e karsi
  5. (N,K) tahmin matrisini kaydeder -> Adim 4'te MPC'ye verilecek

BU ADIMDA HUKUM YOK. Sadece iki sayi uretir:
  alpha_gercek = MAE_nwp / MAE_naive        (supurmedeki alpha'nin gercek karsiligi)
  alpha_kuyruk = q95_nwp / q95_naive        (kuyruk ustunde ayni oran)
Iddia: alpha_kuyruk < alpha_gercek ise, supurmenin "tekduze hata olcekleme"
varsayimi NWP'yi yanlis modelliyor demektir (vekil sorunu kanitlanir).

PROTOKOL
  validation = 0:2400      -> model secimi SADECE burada
  train      = 2400:13800  -> fit
  test       = 13800:N-1   -> TEK bakis, olcum
  Test hicbir secim/ayar kararinda kullanilmaz.

CALISTIR (GG-1 kok klasorunden):  python gg2/gg2_nwp_forecaster.py
CIKTI: gg2/data/nwp_solar_fc.npz   (mevcut varsa --force gerekir)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- GG-1 ile birebir ayni sabitler (gg2_solar_alpha_sweep.py'dan)
SP = 3.0            # olceklenmis solar tepe (MW)
SPLIT = 13800       # test baslangici
VAL_END = 2400      # validation sonu (protokol)
SOLAR_H = (10, 16)  # kuyruk olcumu icin gunes saatleri (UTC), Figur 3 ile ayni


def _die(msg: str) -> None:
    sys.exit(f"[HATA] {msg}")


# ------------------------------------------------------------------ features
def build_features(joint: pd.DataFrame, ghi: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """30-dk izgarada ozellik matrisi. SADECE day1 (24h-lead) kullanilir."""
    p1_cols = [c for c in ghi.columns if c.startswith("ghi_p1_")]
    if not p1_cols:
        _die("nwp_ghi_roi.csv icinde ghi_p1_* kolonu yok")

    g = ghi.set_index("datetime_utc")[p1_cols].sort_index()
    # saatlik -> 30-dk: zaman-bazli interpolasyon, sonra joint izgarasina hizala
    g30 = g.reindex(g.index.union(joint.index)).interpolate(method="time").reindex(joint.index)
    if g30.isna().any().any():
        n = int(g30.isna().any(axis=1).sum())
        _die(f"hizalamadan sonra {n} adimda GHI bos - cekim araligi joint'i kapsamiyor olabilir")

    mean = g30[p1_cols].mean(axis=1)
    feats = {c: g30[c].to_numpy() for c in p1_cols}
    feats["ghi_mean"] = mean.to_numpy()
    feats["ghi_mean_lag1"] = mean.shift(1).bfill().to_numpy()    # -30 dk
    feats["ghi_mean_lead1"] = mean.shift(-1).ffill().to_numpy()  # +30 dk
    # zaman ozellikleri: gunluk ve yillik periyodiklik
    hod = joint.index.hour + joint.index.minute / 60.0
    doy = joint.index.dayofyear.to_numpy().astype(float)
    feats["sin_h"] = np.sin(2 * np.pi * hod / 24.0)
    feats["cos_h"] = np.cos(2 * np.pi * hod / 24.0)
    feats["sin_d"] = np.sin(2 * np.pi * doy / 365.25)
    feats["cos_d"] = np.cos(2 * np.pi * doy / 365.25)

    names = list(feats.keys())
    X = np.column_stack([feats[k] for k in names])
    return X, names


# ------------------------------------------------------------------ models
def fit_predict(X: np.ndarray, y: np.ndarray, tr: slice, name: str) -> np.ndarray:
    if name == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    elif name == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                              min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                              random_state=42, verbose=-1)
    else:
        _die(f"bilinmeyen model: {name}")
    m.fit(X[tr], y[tr])
    return np.clip(m.predict(X), 0.0, None)   # solar negatif olamaz


# ------------------------------------------------------------------ metrics
def metrics(err: np.ndarray, hour: np.ndarray) -> dict:
    """err ve hour ayni sekilde (T,K) duzlestirilmis olmali."""
    a = np.abs(err).ravel()
    m = (hour.ravel() >= SOLAR_H[0]) & (hour.ravel() <= SOLAR_H[1])
    return {
        "mae": float(a.mean()),
        "q95_solar": float(np.quantile(a[m], 0.95)) if m.any() else float("nan"),
        "max_solar": float(a[m].max()) if m.any() else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="GG-2 Adim 3: NWP solar tahminci.")
    ap.add_argument("--joint", default="data/aligned_30min_joint.csv")
    ap.add_argument("--fcmat", default="data/forecast_matrix.npz")
    ap.add_argument("--ghi", default="gg2/data/nwp_ghi_roi.csv")
    ap.add_argument("--out", default="gg2/data/nwp_solar_fc.npz")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        _die(f"cikti zaten var, ezilmedi: {out}  (--force ile ez)")

    for p in (args.joint, args.fcmat, args.ghi):
        if not Path(p).exists():
            _die(f"bulunamadi: {p}")

    # --- veri
    df = pd.read_csv(args.joint, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    N = len(df)
    train_slice = df.iloc[:SPLIT]
    y = (df.solar_mw / train_slice.solar_mw.max() * SP).to_numpy()   # sweep ile birebir

    ghi = pd.read_csv(args.ghi, parse_dates=["datetime_utc"])
    f = np.load(args.fcmat)
    naive = f["solar"]                       # (N,K) daily-naive, olceklenmis
    K = naive.shape[1]

    print("=" * 72)
    print("ADIM 3 - NWP solar tahminci")
    print(f"  joint: {N} adim | forecast_matrix K = {K} | test = {SPLIT}:{N-1}")

    # --- nedensellik capasi: day1 = 24h lead = 48 adim
    #     (i,k) hedefi i+k+1; o hedefin NWP kosumu i+k+1-48'de yayinlanmis.
    #     i aninda bilinir olmasi icin: k <= 47
    if K - 1 > 47:
        _die(f"K-1={K-1} > 47: day1 (24h lead) NWP {K}. adim icin i aninda HENUZ YAYINLANMAMIS "
             f"olurdu -> nedensellik ihlali. Daha uzun lead (day2) gerekir.")
    print(f"  nedensellik: k_max = {K-1} <= 47  -> day1 NWP i aninda bilinir  OK")

    X, names = build_features(df, ghi)
    print(f"  ozellik: {X.shape[1]} adet -> {', '.join(names[:4])}, ...")

    # --- model secimi: SADECE validation (0:2400)
    tr_for_val = slice(VAL_END, SPLIT)       # val'i gormeden fit
    val = slice(0, VAL_END)
    print("\n  MODEL SECIMI (validation 0:2400, val MAE - <author>'in kurali)")
    scores = {}
    for name in ("ridge", "lgbm"):
        try:
            pred = fit_predict(X, y, tr_for_val, name)
        except ImportError as e:
            print(f"    {name:6s} -> atlandi ({e})")
            continue
        scores[name] = float(np.abs(pred[val] - y[val]).mean())
        print(f"    {name:6s} val MAE = {scores[name]:.4f}")
    if not scores:
        _die("hicbir model egitilemedi (sklearn/lightgbm eksik?)")
    best = min(scores, key=scores.get)
    print(f"    -> secilen: {best}")

    # --- secilen modeli tum train-oncesi veriyle yeniden fit et (0:13800), TEST'e dokunma
    pred = fit_predict(X, y, slice(0, SPLIT), best)

    # --- (N,K) matris: row i, col k -> hedef zaman i+k+1  (naive ile ayni hizalama)
    idx = np.minimum(np.arange(N)[:, None] + np.arange(1, K + 1)[None, :], N - 1)
    nwp_fc = pred[idx]
    truth = y[idx]
    hour = df.index.hour.to_numpy()[idx]

    # --- TEST bloğunda olcum (sweep ile birebir ayni dilim)
    T0, T1 = SPLIT, N - 1
    sl = slice(T0, T1)
    m_naive = metrics(naive[sl] - truth[sl], hour[sl])
    m_nwp = metrics(nwp_fc[sl] - truth[sl], hour[sl])

    print("\n" + "=" * 72)
    print(f"OLCUM - TEST bloğu ({T0}:{T1}), olceklenmis MW")
    print("=" * 72)
    print(f"  {'':14s} {'MAE':>9s} {'q95(solar)':>12s} {'max(solar)':>12s}")
    print(f"  {'daily-naive':14s} {m_naive['mae']:9.4f} {m_naive['q95_solar']:12.4f} "
          f"{m_naive['max_solar']:12.4f}")
    print(f"  {'NWP day1':14s} {m_nwp['mae']:9.4f} {m_nwp['q95_solar']:12.4f} "
          f"{m_nwp['max_solar']:12.4f}")

    a_mean = m_nwp["mae"] / m_naive["mae"]
    a_tail = m_nwp["q95_solar"] / m_naive["q95_solar"]
    print(f"\n  alpha_gercek (MAE orani)   = {a_mean:.3f}   "
          f"-> MAE azaltmasi %{100*(1-a_mean):.1f}")
    print(f"  alpha_kuyruk (q95 orani)   = {a_tail:.3f}   "
          f"-> kuyruk azaltmasi %{100*(1-a_tail):.1f}")

    print("\n  VEKIL SORUNU TESTI (supurme tekduze varsayiyordu):")
    if a_tail < a_mean - 0.02:
        print(f"    alpha_kuyruk ({a_tail:.3f}) < alpha_gercek ({a_mean:.3f})")
        print("    -> NWP kuyrugu ortalamadan DAHA COK dusuruyor. Supurmenin tekduze")
        print("       varsayimi NWP'yi yanlis modelliyor. Vekil sorunu DOGRULANDI.")
    elif a_tail > a_mean + 0.02:
        print(f"    alpha_kuyruk ({a_tail:.3f}) > alpha_gercek ({a_mean:.3f})")
        print("    -> NWP kuyrugu ortalamadan DAHA AZ dusuruyor. Supurme NWP hakkinda")
        print("       IYIMSERMIS; C'nin hukmu bu yonde guclenir.")
    else:
        print(f"    alpha_kuyruk ~= alpha_gercek ({a_tail:.3f} vs {a_mean:.3f})")
        print("    -> tekduze varsayim bu veride kabaca tutuyor.")

    print(f"\n  Supurmedeki en yakin nokta: alpha ~ {min([0.25,0.50,0.75], key=lambda a: abs(a-a_mean)):.2f}")
    print("  (Kesin cevap Adim 4'te: bu tahmini MPC'ye verip ihlal olcecegiz.)")

    np.savez(out, solar=nwp_fc.astype(np.float32), model=best,
             a_mean=a_mean, a_tail=a_tail, K=K, split=SPLIT)
    print(f"\n  yazildi: {out}  (shape {nwp_fc.shape}, model={best})")


if __name__ == "__main__":
    main()