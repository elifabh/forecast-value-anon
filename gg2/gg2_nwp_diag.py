#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_nwp_diag.py - ADIM 3b: NWP tahminci neden naive'e kaybetti?

ADIM 3 SONUCU: NWP day1 tahmincisi daily-naive'den KOTU (alpha=1.096).
Bu, on-kayitta yazdigimiz "hukum verilemez" durumu. Suc NWP'de mi, bizim
radyasyon->MW donusumumuzde mi? Bu script onu ayirir.

UC KONTROL (hepsi sadece ekrana basar, DOSYA YAZMAZ):

  [1] KAPASITE KAYMASI
      Aylik (solar_mw tepesi / GHI tepesi) orani. Yukari trend varsa, ayni
      radyasyon zamanla daha cok MW uretiyor demektir = kurulu kapasite buyumus.
      daily-naive bundan MUAF (dun zaten bugunun kapasitesiyle uretti);
      eski veriyle egitilmis her radyasyon->MW modeli DEGIL.

  [2] p0 vs p1
      Ayni boru hatti, iki girdiyle:
        p1 = 24h once tahmin (gercek gun-oncesi bilgi)
        p0 = en taze model kosumu (~analiz; GERCEKCI DEGIL, sadece tani)
      p0 DA naive'i yenemiyorsa -> suc NWP'nin gecikmesinde degil, BIZIM
      donusum modelimizde/kapasite kaymasinda. Boru hatti bozuk.
      p0 yenip p1 yenemiyorsa -> suc 24h gecikmede. Bu gercek bulgu olur.

  [3] KAPASITE-NORMALIZE HEDEF
      Hedefi solar_mw yerine solar_mw / cap_est tahmin et; cap_est = son 30
      gunun tepesi (SADECE gecmis -> nedensel). Sonra geri carp.
      Bu, naive'in yapisal avantajini bize de verir. Duzelirse, Adim 3'teki
      kayip NWP'nin degil normalizasyonun sucuymus demektir.

PROTOKOL: fit 0:13800, olcum 13800:N-1. Test hicbir secim kararinda kullanilmaz.
Model sabit (lgbm) - Adim 3'te val secimi zaten lgbm demisti, yeniden secmiyoruz.

CALISTIR (GG-1 kok klasorunden):  python gg2/gg2_nwp_diag.py
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
SOLAR_H = (10, 16)
CAP_WIN = 30 * 48        # 30 gun, 30-dk adim
LAG = 48                 # nedensellik: cap_est en az 24h gecikmeli kullanilir


def _die(msg: str) -> None:
    sys.exit(f"[HATA] {msg}")


def ghi30(joint: pd.DataFrame, ghi: pd.DataFrame, tag: str) -> pd.DataFrame:
    cols = [c for c in ghi.columns if c.startswith(f"ghi_{tag}_")]
    if not cols:
        _die(f"ghi_{tag}_* kolonu yok")
    g = ghi.set_index("datetime_utc")[cols].sort_index()
    g30 = g.reindex(g.index.union(joint.index)).interpolate(method="time").reindex(joint.index)
    if g30.isna().any().any():
        _die(f"{tag}: hizalamadan sonra bos deger var")
    return g30


def make_X(joint: pd.DataFrame, g30: pd.DataFrame, extra: dict | None = None) -> np.ndarray:
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
    if extra:
        f.update(extra)
    return np.column_stack([f[k] for k in f])


def fit_lgbm(X: np.ndarray, y: np.ndarray, tr: slice) -> np.ndarray:
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                          min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                          random_state=42, verbose=-1)
    m.fit(X[tr], y[tr])
    return m.predict(X)


def score(pred_series: np.ndarray, y: np.ndarray, N: int, K: int,
          hour_all: np.ndarray) -> dict:
    idx = np.minimum(np.arange(N)[:, None] + np.arange(1, K + 1)[None, :], N - 1)
    fc, truth, hour = pred_series[idx], y[idx], hour_all[idx]
    sl = slice(SPLIT, N - 1)
    e = np.abs(fc[sl] - truth[sl]).ravel()
    m = (hour[sl].ravel() >= SOLAR_H[0]) & (hour[sl].ravel() <= SOLAR_H[1])
    return {"mae": float(e.mean()), "q95": float(np.quantile(e[m], 0.95))}


def main() -> None:
    ap = argparse.ArgumentParser(description="GG-2 Adim 3b: NWP tani.")
    ap.add_argument("--joint", default="data/aligned_30min_joint.csv")
    ap.add_argument("--fcmat", default="data/forecast_matrix.npz")
    ap.add_argument("--ghi", default="gg2/data/nwp_ghi_roi.csv")
    a = ap.parse_args()
    for p in (a.joint, a.fcmat, a.ghi):
        if not Path(p).exists():
            _die(f"bulunamadi: {p}")

    df = pd.read_csv(a.joint, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    N = len(df)
    y = (df.solar_mw / df.iloc[:SPLIT].solar_mw.max() * SP).to_numpy()
    hour_all = df.index.hour.to_numpy()
    naive = np.load(a.fcmat)["solar"]
    K = naive.shape[1]
    ghi = pd.read_csv(a.ghi, parse_dates=["datetime_utc"])
    g1 = ghi30(df, ghi, "p1")
    g0 = ghi30(df, ghi, "p0")

    # daily-naive referansi (Adim 3 ile birebir ayni dilim/hizalama)
    idx = np.minimum(np.arange(N)[:, None] + np.arange(1, K + 1)[None, :], N - 1)
    sl = slice(SPLIT, N - 1)
    en = np.abs(naive[sl] - y[idx][sl]).ravel()
    hn = hour_all[idx][sl].ravel()
    mn = (hn >= SOLAR_H[0]) & (hn <= SOLAR_H[1])
    NAIVE = {"mae": float(en.mean()), "q95": float(np.quantile(en[mn], 0.95))}

    # ---------------------------------------------------------- [1] kapasite
    print("=" * 72)
    print("[1] KAPASITE KAYMASI  (aylik: solar tepesi / GHI tepesi)")
    print("=" * 72)
    tmp = pd.DataFrame({"solar": y, "ghi": g1.mean(axis=1).to_numpy()}, index=df.index)
    mo = tmp.resample("MS").max()
    mo = mo[mo.ghi > 100]
    mo["oran"] = mo.solar / mo.ghi
    for t, r in mo.iterrows():
        bar = "#" * int(r.oran * 800)
        print(f"  {t:%Y-%m}  solar_tepe {r.solar:5.2f} | ghi_tepe {r.ghi:6.1f} "
              f"| oran {r.oran:.5f} {bar}")
    if len(mo) >= 4:
        ilk, son = mo.oran.iloc[:2].mean(), mo.oran.iloc[-2:].mean()
        print(f"\n  ilk 2 ay ort {ilk:.5f} -> son 2 ay ort {son:.5f}  "
              f"= x{son/ilk:.2f}")
        if son / ilk > 1.15:
            print("  -> KAPASITE BUYUMESI DOGRULANDI. Ayni radyasyon artik daha cok MW.")
            print("     daily-naive bundan muaf; eski veriyle egitilmis modelimiz degil.")
        else:
            print("  -> belirgin kapasite kaymasi yok; sebep baska.")

    # ---------------------------------------------------------- [2] p0 vs p1
    print("\n" + "=" * 72)
    print("[2] p0 vs p1  (fit 0:13800, olcum 13800:N-1)")
    print("=" * 72)
    r1 = score(np.clip(fit_lgbm(make_X(df, g1), y, slice(0, SPLIT)), 0, None),
               y, N, K, hour_all)
    r0 = score(np.clip(fit_lgbm(make_X(df, g0), y, slice(0, SPLIT)), 0, None),
               y, N, K, hour_all)
    print(f"  {'':22s} {'MAE':>8s} {'alpha':>7s} {'q95':>8s}")
    print(f"  {'daily-naive':22s} {NAIVE['mae']:8.4f} {1.0:7.3f} {NAIVE['q95']:8.4f}")
    print(f"  {'NWP p1 (24h once)':22s} {r1['mae']:8.4f} "
          f"{r1['mae']/NAIVE['mae']:7.3f} {r1['q95']:8.4f}")
    print(f"  {'NWP p0 (~analiz)':22s} {r0['mae']:8.4f} "
          f"{r0['mae']/NAIVE['mae']:7.3f} {r0['q95']:8.4f}")
    print("\n  OKUMA:")
    if r0["mae"] >= NAIVE["mae"]:
        print("    p0 DA naive'i yenemiyor -> suc NWP'nin gecikmesinde DEGIL.")
        print("    Bizim radyasyon->MW donusumumuz / kapasite kaymasi. Boru hatti bozuk.")
        print("    NWP hakkinda hicbir hukum verilemez.")
    else:
        print("    p0 naive'i yeniyor, p1 yenemiyor -> suc 24 SAATLIK GECIKMEDE.")
        print("    Bu gercek bir bulgu: gun-oncesi ufkunda NWP bu seride yeterli degil.")

    # ------------------------------------------------- [3] kapasite-normalize
    print("\n" + "=" * 72)
    print("[3] KAPASITE-NORMALIZE HEDEF  (cap_est = son 30 gun tepesi, 24h gecikmeli)")
    print("=" * 72)
    s = pd.Series(y, index=df.index)
    cap = s.rolling(CAP_WIN, min_periods=48).max().shift(LAG)   # nedensel
    cap = cap.bfill().clip(lower=0.05).to_numpy()
    yn = y / cap                                                # normalize hedef
    Xc = make_X(df, g1, extra={"cap_est": cap})
    pn = np.clip(fit_lgbm(Xc, yn, slice(0, SPLIT)), 0, None) * cap   # geri carp
    r3 = score(pn, y, N, K, hour_all)
    print(f"  {'daily-naive':22s} {NAIVE['mae']:8.4f} {1.0:7.3f} {NAIVE['q95']:8.4f}")
    print(f"  {'NWP p1 ham':22s} {r1['mae']:8.4f} "
          f"{r1['mae']/NAIVE['mae']:7.3f} {r1['q95']:8.4f}")
    print(f"  {'NWP p1 + cap-norm':22s} {r3['mae']:8.4f} "
          f"{r3['mae']/NAIVE['mae']:7.3f} {r3['q95']:8.4f}")
    print("\n  OKUMA:")
    if r3["mae"] < NAIVE["mae"]:
        print(f"    cap-norm ile naive YENILDI (alpha={r3['mae']/NAIVE['mae']:.3f}).")
        print("    Adim 3'teki kayip normalizasyon sucuymus. Bu tahminci Adim 4'e girer.")
    elif r3["mae"] < r1["mae"]:
        print("    cap-norm iyilestirdi ama naive'i hala yenemedi -> kismi sebep.")
    else:
        print("    cap-norm yardim etmedi -> sebep baska (uzamsal ornekleme? ozellikler?).")
    print()


if __name__ == "__main__":
    main()