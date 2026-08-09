#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_nwp_season.py - ADIM 6: MEVSIM KONTROLU (leave-block-out).

NEDEN: Test bloğunda (bahar 2026) sonuc guclu cikti:
  Fair MPC + naive  : %11.28 | 9681 EUR
  Fair MPC + NWP    : % 9.80 | 10248 EUR   (bosluğun %69'u, kar +567)
  Fair MPC + oracle : % 9.14 | 10574 EUR
Ama GG-1'de hibrit tam burada yanmisti: validation'da MPC'yi domine etti,
TESTE TASINMADI (mevsimsel dagilim kaymasi). Ayni tuzaga dusmemek icin
sonucu BASKA bir mevsimde tekrarliyoruz.

SIZINTI ONLEMI (kritik):
  Ana kosumdaki tahminci 0:13800'de fit edildi -> validation bloğu (0:2400) ICINDE.
  Orada olcersek kendi egitim verimize bakmis oluruz. Bu script tahminciyi
  DEGERLENDIRME BLOĞUNU DISLAYARAK yeniden fit eder.
  Varsayilan: fit 2400:13800  ->  degerlendirme 0:2400 (yaz 2025).

KONTROLCU KUNYESI: gg2_nwp_mpc_fair.py ile BIREBIR AYNI
  (diag_mpc_forecastvolt.fc_windows_ldf mantigi, w_econ=1.0, ayni safe layer).
  TEK degisken: solar tahmini (naive / oracle / NWP) ve degerlendirme bloğu.

ON-KAYIT (kosumdan ONCE yazildi):
  Ayni desen tekrarlaniyor mu?  Iki kosul birden aranir:
    (1) NWP, naive'e gore ihlali dusuruyor mu
    (2) NWP, naive'e gore kari ARTIRIYOR mu (yani takas yok)
  Ikisi de tutarsa   -> desen mevsimden bagimsiz, bulgu guclu.
  Biri tutmazsa      -> desen mevsime bagli; test sonucu tek-mevsim bulgusu
                        olarak, hibritteki gibi limitation'la raporlanir.
  Kapanma orani icin ayni formul: (naive - NWP) / (naive - oracle).

CALISTIR (GG-1 kok klasorunden):  python gg2/gg2_nwp_season.py
  Ucunu de sirayla kosar (~30 dk). Ara kayit yok - blok kisa.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", message=".*valid feature names.*")

import mpc_econ_voltage_pareto as M
import diag_mpc_forecastvolt as D

W_ECON = 1.0
SP = 3.0
CAP_WIN = 30 * 48
LAG = 48
SOLAR_H = (10, 16)


def _die(m: str) -> None:
    sys.exit(f"[HATA] {m}")


def build_nwp(ghi_path: str, fit0: int, fit1: int) -> tuple[np.ndarray, dict]:
    """Cap-norm NWP tahminci; SADECE [fit0:fit1] ile fit edilir."""
    import lightgbm as lgb

    df = pd.read_csv("data/aligned_30min_joint.csv", index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    N = len(df)
    y = (df.solar_mw / df.iloc[:M.SPLIT].solar_mw.max() * SP).to_numpy()

    g = pd.read_csv(ghi_path, parse_dates=["datetime_utc"])
    cols = [c for c in g.columns if c.startswith("ghi_p1_")]
    gg = g.set_index("datetime_utc")[cols].sort_index()
    g30 = gg.reindex(gg.index.union(df.index)).interpolate(method="time").reindex(df.index)
    if g30.isna().any().any():
        _die("GHI hizalamasinda bos deger")

    s = pd.Series(y, index=df.index)
    cap = s.rolling(CAP_WIN, min_periods=48).max().shift(LAG).bfill().clip(lower=0.05).to_numpy()

    mean = g30.mean(axis=1)
    f = {c: g30[c].to_numpy() for c in cols}
    f["ghi_mean"] = mean.to_numpy()
    f["ghi_lag1"] = mean.shift(1).bfill().to_numpy()
    f["ghi_lead1"] = mean.shift(-1).ffill().to_numpy()
    hod = df.index.hour + df.index.minute / 60.0
    doy = df.index.dayofyear.to_numpy().astype(float)
    f["sin_h"] = np.sin(2 * np.pi * hod / 24.0)
    f["cos_h"] = np.cos(2 * np.pi * hod / 24.0)
    f["sin_d"] = np.sin(2 * np.pi * doy / 365.25)
    f["cos_d"] = np.cos(2 * np.pi * doy / 365.25)
    f["cap_est"] = cap
    X = np.column_stack([f[k] for k in f])

    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                          min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                          random_state=42, verbose=-1)
    m.fit(X[fit0:fit1], (y / cap)[fit0:fit1])
    pred = np.clip(m.predict(X), 0, None) * cap

    K = M.solar_fc.shape[1]
    idx = np.minimum(np.arange(N)[:, None] + np.arange(1, K + 1)[None, :], N - 1)
    return pred[idx], {"y": y, "idx": idx, "hour": df.index.hour.to_numpy()[idx]}


def run_block(solar_fc: np.ndarray, e0: int, e1: int, label: str) -> tuple[float, float]:
    soc = M.CAP / 2
    n = e1 - e0
    mn = np.zeros(n); mx = np.zeros(n); P = np.zeros(n)
    for j in range(n):
        i = e0 + j
        Hh = min(M.H, M.N - i)
        v_now = D.ldf_v(M.solar[i], M.wind[i], M.dmult[i])
        vmb = np.empty((Hh, D.n)); vmb[0] = M.VM_BASE[i]
        for t in range(1, Hh):
            kk = min(t - 1, solar_fc.shape[1] - 1)
            v_fut = D.ldf_v(solar_fc[i, kk], M.wind_fc[i, kk], M.dmult_fc[i, kk])
            vmb[t] = M.VM_BASE[i] + (v_fut - v_now)
        pr = np.empty(Hh); pr[0] = M.price[i]
        for t in range(1, Hh):
            kk = min(t - 1, M.price_fc.shape[1] - 1)
            pr[t] = M.price_fc[i, kk]
        p_plan, _ = M.solve_window(vmb, pr, soc, W_ECON)
        p_plan = float(np.clip(p_plan, -M.P_MAX, M.P_MAX))
        p_lo = max(-M.P_MAX, -soc * M.ETA / M.dt)
        p_hi = min(M.P_MAX, (M.CAP - soc) / (M.ETA * M.dt))
        p_plan = float(np.clip(p_plan, p_lo, p_hi))
        M.set_grid(M.solar[i], M.wind[i], M.dmult[i], 0.0)
        p_safe, sinfo = M.sl.project(p_plan, p_lo, p_hi)
        mn[j] = sinfo["Vmin"]; mx[j] = sinfo["Vmax"]; P[j] = p_safe
        soc = min(max(soc + (M.ETA * p_safe * M.dt if p_safe >= 0 else p_safe * M.dt / M.ETA),
                      0), M.CAP)
        if (j + 1) % 500 == 0:
            print(f"      {label}: {j+1}/{n}", flush=True)
    V = ((mn < M.LO) | (mx > M.HI)).mean() * 100
    kar = float(-(M.price[e0:e1] * P).sum() * M.dt)
    return V, kar


def main() -> None:
    ap = argparse.ArgumentParser(description="GG-2 Adim 6: mevsim kontrolu.")
    ap.add_argument("--ghi", default="gg2/data/nwp_ghi_roi.csv")
    ap.add_argument("--fit-start", type=int, default=2400)
    ap.add_argument("--fit-end", type=int, default=13800)
    ap.add_argument("--eval-start", type=int, default=0)
    ap.add_argument("--eval-end", type=int, default=2400)
    a = ap.parse_args()

    if not Path(a.ghi).exists():
        _die(f"bulunamadi: {a.ghi}")
    if max(a.eval_start, a.eval_end) > a.fit_start and min(a.eval_start, a.eval_end) < a.fit_end:
        if not (a.eval_end <= a.fit_start or a.eval_start >= a.fit_end):
            _die(f"SIZINTI: degerlendirme [{a.eval_start}:{a.eval_end}] fit "
                 f"[{a.fit_start}:{a.fit_end}] ile kesisiyor")

    e0, e1 = a.eval_start, a.eval_end
    K = M.solar_fc.shape[1]

    print("=" * 74)
    print("ADIM 6 - MEVSIM KONTROLU (leave-block-out)")
    print("=" * 74)
    print(f"  tahminci fit : {a.fit_start}:{a.fit_end}   (degerlendirme bloğu DISLANDI)")
    print(f"  degerlendirme: {e0}:{e1}  ({e1-e0} adim)")
    print(f"  kontrolcu    : diag_mpc_forecastvolt mantigi | w_econ={W_ECON}")
    print(f"  ort solar (degerlendirme): {M.solar[e0:e1].mean():.3f} "
          f"| (test blogu): {M.solar[M.SPLIT:M.N-1].mean():.3f}")
    print("=" * 74)

    nwp_fc, aux = build_nwp(a.ghi, a.fit_start, a.fit_end)

    # tahminci kalitesi bu blokta (kiyas capasi)
    sl = slice(e0, e1)
    tr, hr = aux["y"][aux["idx"]], aux["hour"]
    en = np.abs(M.solar_fc[sl] - tr[sl]).ravel()
    ew = np.abs(nwp_fc[sl] - tr[sl]).ravel()
    print(f"\n  tahminci MAE bu blokta: naive {en.mean():.4f} | NWP {ew.mean():.4f} "
          f"| alpha {ew.mean()/en.mean():.3f}")

    oracle_fc = aux["y"][aux["idx"]]

    res = {}
    for label, fc in (("naive", M.solar_fc), ("oracle", oracle_fc), ("NWP", nwp_fc)):
        print(f"\n  --- {label} kosuluyor...")
        res[label] = run_block(fc, e0, e1, label)
        print(f"      {label}: ihlal {res[label][0]:.2f}% | kar {res[label][1]:.1f} EUR")

    vn, kn = res["naive"]; vo, ko = res["oracle"]; vw, kw = res["NWP"]
    print("\n" + "=" * 74)
    print(f"SONUC - degerlendirme bloğu {e0}:{e1}")
    print("=" * 74)
    print(f"  {'':22s} {'ihlal%':>8s} {'kar(EUR)':>10s}")
    print(f"  {'Fair MPC + naive':22s} {vn:8.2f} {kn:10.0f}")
    print(f"  {'Fair MPC + NWP':22s} {vw:8.2f} {kw:10.0f}")
    print(f"  {'Fair MPC + oracle':22s} {vo:8.2f} {ko:10.0f}   [referans]")
    kapanma = 100 * (vn - vw) / (vn - vo) if abs(vn - vo) > 1e-9 else float("nan")
    print(f"\n  ihlal degisimi : {vw-vn:+.2f} puan   | bosluk kapanmasi: {kapanma:.1f}%")
    print(f"  kar degisimi   : {kw-kn:+.0f} EUR")
    print(f"\n  TEST bloğu (bahar 2026) referansi: ihlal -1.48 puan | kar +567 EUR "
          f"| kapanma %69.2")

    print("\n  ON-KAYITLI HUKUM:")
    c1, c2 = vw < vn, kw > kn
    print(f"    (1) NWP ihlali dusuruyor mu : {'EVET' if c1 else 'HAYIR'}")
    print(f"    (2) NWP kari artiriyor mu   : {'EVET' if c2 else 'HAYIR'}")
    if c1 and c2:
        print("    -> DESEN TEKRARLANDI. Bulgu mevsime bagli degil; iki mevsimde de")
        print("       iyi tahmin hem guvenligi hem kari artiriyor (takas yok).")
    else:
        print("    -> DESEN TEKRARLANMADI. Test sonucu TEK-MEVSIM bulgusu olarak")
        print("       raporlanir; hibrittekiyle ayni limitation yazilir.")
    print()


if __name__ == "__main__":
    main()