#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_z9_valid.py - Z9: cap-norm karari TEST blogundan bagimsiz gerekcelendirilir mi?

SORUN
  gg2_nwp_diag.py kontrol [3], "cap-norm ile naive YENILDI -> bu tahminci
  Adim 4'e girer" hukmunu TEST blogunda olculen MAE ile verdi.

BU SCRIPT TEST BLOGUNA HIC DOKUNMAZ. Iki bagimsiz gerekce arar:
  [A] Kapasite kaymasi SADECE egitim verisinde (0:13800) gorunuyor mu?
  [B] Validation'da (0:2400) cap-norm, ham NWP'yi yeniyor mu? (fit 2400:13800)

ON-KAYIT (kod yazilmadan once):
  EPS = 0.02 alpha birimi. Gerekce: alpha bloklar arasi 0.577-0.799 oynuyor.
  [A] son2/ilk2 aylik oran > 1.15          -> kayma DOGRULANDI
  [B] alpha_cap < alpha_ham - EPS          -> cap-norm KAZANDI
      |fark| <= EPS                        -> AYIRT EDILEMIYOR
      alpha_cap > alpha_ham + EPS          -> KAYBETTI (ana tablo gozden gecirilir)

METRIK PENCERESI: satirlar [96 : 2400-K]
  96 : cap_est ilk 95 adimda tanimsiz, bfill ILERIDEN dolduruyor -> disla
  -K : (i,k) hedefi i+k+1; son K satir fit bolgesine uzanir -> disla
  Karsilastirma icin bfill'li tam pencere de basilir (etkinin buyuklugu gorunsun).

CALISTIR (kok klasorden):
  python gg2/gg2_z9_valid.py 2>&1 | tee gg2/z9_valid.log
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
DRIFT_GATE = 1.15


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


def fit_lgbm(X: np.ndarray, y: np.ndarray, tr: slice) -> np.ndarray:
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                          min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                          random_state=42, verbose=-1)
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Z9: cap-norm karari validation kontrolu.")
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

    r0, r1 = WARM, VAL_END - K
    if r1 <= r0:
        _die("validation penceresi bos (K cok buyuk)")

    print("=" * 74)
    print("Z9 - cap-norm karari TEST'ten bagimsiz gerekcelendirilir mi?")
    print("=" * 74)
    print("  joint %d adim | K=%d" % (n, K))
    print("  TEST BLOGU (%d:%d) BU SCRIPTTE KULLANILMIYOR" % (SPLIT, n - 1))
    print("  validation olcum satirlari: %d:%d  (%d adim)" % (r0, r1, r1 - r0))
    print("  tahminci fit              : %d:%d" % (VAL_END, SPLIT))

    g1 = ghi30(df, pd.read_csv(a.ghi, parse_dates=["datetime_utc"]), "p1")

    # -------------------------------------------------- [A] kapasite kaymasi
    print("\n" + "=" * 74)
    print("[A] KAPASITE KAYMASI - SADECE EGITIM VERISI (0:%d)" % SPLIT)
    print("=" * 74)
    tmp = pd.DataFrame({"solar": y[:SPLIT],
                        "ghi": g1.mean(axis=1).to_numpy()[:SPLIT]},
                       index=df.index[:SPLIT])
    mo = tmp.resample("MS").max()
    mo = mo[mo.ghi > 100]
    if len(mo) < 4:
        _die("egitim verisinde 4'ten az tam ay var; kayma olculemez")
    mo["oran"] = mo.solar / mo.ghi
    for t, r in mo.iterrows():
        print("  %s  solar_tepe %5.2f | ghi_tepe %6.1f | oran %.5f %s"
              % (t.strftime("%Y-%m"), r.solar, r.ghi, r.oran, "#" * int(r.oran * 800)))
    ilk = float(mo.oran.iloc[:2].mean())
    son = float(mo.oran.iloc[-2:].mean())
    drift = son / ilk if ilk > 0 else float("nan")
    print("\n  ilk 2 ay %.5f -> son 2 ay %.5f  = x%.2f   (esik %.2f)"
          % (ilk, son, drift, DRIFT_GATE))
    A_ok = bool(drift > DRIFT_GATE)
    print("  [A] " + ("KAYMA VAR -> cap-norm egitim verisinden motive edilir"
                      if A_ok else "belirgin kayma YOK"))

    # ------------------------------------------ [B] validation: ham vs cap-norm
    print("\n" + "=" * 74)
    print("[B] VALIDATION KARSILASTIRMASI (fit %d:%d, olcum %d:%d)"
          % (VAL_END, SPLIT, r0, r1))
    print("=" * 74)

    s = pd.Series(y, index=df.index)
    cap_raw = s.rolling(CAP_WIN, min_periods=48).max().shift(LAG)
    n_nan = int(cap_raw.isna().sum())
    cap = cap_raw.bfill().clip(lower=0.05).to_numpy()
    print("  cap_est ilk %d adimda tanimsizdi (bfill ileriden doldurdu)" % n_nan)

    f_raw = base_feats(df, g1)
    X_raw = np.column_stack([f_raw[k] for k in f_raw])
    f_cap = dict(f_raw)
    f_cap["cap_est"] = cap
    X_cap = np.column_stack([f_cap[k] for k in f_cap])

    tr = slice(VAL_END, SPLIT)
    p_raw = np.clip(fit_lgbm(X_raw, y, tr), 0.0, None)
    p_cap = np.clip(fit_lgbm(X_cap, y / cap, tr), 0.0, None) * cap

    m_nai = mae_matrix(naive, y, K, r0, r1)
    m_raw = mae_series(p_raw, y, K, r0, r1)
    m_cap = mae_series(p_cap, y, K, r0, r1)
    a_raw = m_raw / m_nai
    a_cap = m_cap / m_nai

    print("\n  %-22s %8s %8s" % ("", "MAE", "alpha"))
    print("  %-22s %8.4f %8.3f" % ("daily-naive", m_nai, 1.0))
    print("  %-22s %8.4f %8.3f" % ("NWP ham", m_raw, a_raw))
    print("  %-22s %8.4f %8.3f" % ("NWP cap-norm", m_cap, a_cap))

    f0, f1 = 0, VAL_END - K
    b_nai = mae_matrix(naive, y, K, f0, f1)
    print("\n  [bfill etkisi] tam pencere %d:%d -> ham %.3f | cap-norm %.3f"
          % (f0, f1,
             mae_series(p_raw, y, K, f0, f1) / b_nai,
             mae_series(p_cap, y, K, f0, f1) / b_nai))
    print("  (yukaridaki dar pencereyle farki buyukse, yaz blogundaki bfill")
    print("   sizintisi da o buyuklukte demektir -> limitation olarak yazilir)")

    d = a_cap - a_raw
    print("\n  ON-KAYITLI HUKUM (EPS = %.2f):" % EPS)
    print("    alpha_cap - alpha_ham = %+.3f" % d)
    if d < -EPS:
        B = "KAZANDI"
        print("    -> cap-norm validation'da ham NWP'yi YENIYOR.")
    elif d > EPS:
        B = "KAYBETTI"
        print("    -> cap-norm validation'da ham NWP'ye KAYBEDIYOR.")
    else:
        B = "AYIRT EDILEMIYOR"
        print("    -> iki varyant validation'da AYIRT EDILEMIYOR.")
    print("    cap-norm naive'i validation'da yeniyor mu: %s (alpha_cap=%.3f)"
          % ("EVET" if a_cap < 1.0 else "HAYIR", a_cap))

    print("\n" + "=" * 74)
    print("Z9 SONUC")
    print("=" * 74)
    if A_ok and B == "KAZANDI":
        print("  KAPANDI. Post-processing secimi (i) egitim verisindeki kapasite")
        print("  kaymasiyla motive edilir, (ii) validation MAE ile dogrulanir.")
        print("  Test blogu bu karara girmez.")
    elif B == "KAYBETTI":
        print("  ACIK ve CIDDI. Secim validation'da desteklenmiyor.")
        print("  Ana tablonun tahminci secimi yeniden ele alinmalidir.")
    elif A_ok:
        print("  KISMEN KAPANDI. Kapasite kaymasi egitim verisinde gorunuyor")
        print("  (a-priori gerekce var), ama validation iki varyanti ayirt")
        print("  edemiyor. Makalede boyle yazilir; 'validation kazandi' DENMEZ.")
    else:
        print("  KAPANMADI. Ne kapasite kaymasi ne validation ayrimi bulundu.")
    print()


if __name__ == "__main__":
    main()