#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_oracle_anomaly.py - Oracle neden NWP'den az kar ediyor? OLCUM.

GOZLEM (yaz blogu 0:2400, w_econ=1.0):
  oracle : ihlal 2.08% | kar 12414 EUR | SOC(solar-saat) 4.007
  NWP    : ihlal 2.38% | kar 12798 EUR | SOC(solar-saat) 3.627
Mukemmel bilgili referans, kusurlu tahminden AZ kar ediyor. Hakem bunu sorar.

ONCEKI TESTIM BOZUKTU (gg2_verify.py [4]): 'or' kullanmisim, throughput farki
%1 (627 vs 621) tek basina "dogrulandi" dedirtti. Ayrica SOC yorumum tersti:
4.007 > 3.627 "dusuk SOC" DEGIL. O hukum GECERSIZ.

VERININ SOYLEDIGI: oracle solar saatlerde DAHA DOLU ve DAHA AZ ihlal ediyor.
Yuksek SOC = hazirliksiz degil, AKTIF EMIYOR demek (gunes tepesinde sarj ediyor,
SOC o yuzden yukseliyor). Oglen sarj = oglen fiyatini odemek = kar kaybi.

HIPOTEZ (test edilecek): oracle ~384 EUR odeyip 0.30 puan ihlal satin aliyor.

IKI TEST:
  [A] KAR ANATOMISI (w_econ=1.0)
      Para farki nereden? Sarj/desarj enerjisi + o anlardaki ORTALAMA FIYAT.
      GECER : oracle daha cok VE/VEYA daha pahaliya sarj ediyor, ve fark
              yaklasik 384 EUR'yu acikliyor.
      KALIR : aciklamiyor -> mekanizma baska, hipotez curur.

  [B] w_econ SUPURMESI (KESIN OLAN)
      Oracle'i {1, 3, 10, 30} agirliklarinda kosar -> oracle'in Pareto onu.
      NWP noktasi (2.38%, 12798 EUR) bu onun ICINDE/UZERINDE mi, DISINDA mi?
      GECER : bir agirlikta oracle ihlal<=2.38 VE kar>=12798 -> anomali YOK,
              sadece agirlik konumu. Oracle satiri referans olarak kalabilir.
      KALIR : hicbir agirlikta yakalayamiyor -> NWP oracle'i GERCEKTEN domine
              ediyor. Bu aciklanamaz; oracle satiri tablodan CIKARILIR.

HUKUM MANTIGI: 'and' ile. Tek bir gostergenin %1 farki hicbir sey kanitlamaz.

CALISTIR:  python gg2/gg2_oracle_anomaly.py 2>&1 | tee gg2/oracle_anomaly.log
Sure: ~5 kosum x ~8 dk = ~40 dk.
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

SP = 3.0
CAP_WIN = 30 * 48
LAG = 48
E0, E1 = 0, 2400              # yaz blogu (anomalinin goruldugu yer)
W_LIST = [1.0, 3.0, 10.0, 30.0]
NWP_V_REF, NWP_K_REF = 2.38, 12798.0    # on-kayit: gozlenen NWP noktasi


def _die(m: str) -> None:
    sys.exit(f"[HATA] {m}")


def build_nwp(ghi_path: str):
    import lightgbm as lgb
    df = pd.read_csv("data/aligned_30min_joint.csv", index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    y = (df.solar_mw / df.iloc[:M.SPLIT].solar_mw.max() * SP).to_numpy()
    g = pd.read_csv(ghi_path, parse_dates=["datetime_utc"])
    cols = [c for c in g.columns if c.startswith("ghi_p1_")]
    gg = g.set_index("datetime_utc")[cols].sort_index()
    g30 = gg.reindex(gg.index.union(df.index)).interpolate(method="time").reindex(df.index)
    s = pd.Series(y, index=df.index)
    cap = s.rolling(CAP_WIN, min_periods=48).max().shift(LAG).bfill().clip(lower=0.05).to_numpy()
    mean = g30.mean(axis=1)
    f = {c: g30[c].to_numpy() for c in cols}
    f["ghi_mean"] = mean.to_numpy()
    f["ghi_lag1"] = mean.shift(1).bfill().to_numpy()
    f["ghi_lead1"] = mean.shift(-1).ffill().to_numpy()
    hod = df.index.hour + df.index.minute / 60.0
    doy = df.index.dayofyear.to_numpy().astype(float)
    f["sin_h"] = np.sin(2 * np.pi * hod / 24.0); f["cos_h"] = np.cos(2 * np.pi * hod / 24.0)
    f["sin_d"] = np.sin(2 * np.pi * doy / 365.25); f["cos_d"] = np.cos(2 * np.pi * doy / 365.25)
    f["cap_est"] = cap
    X = np.column_stack([f[k] for k in f])
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                          min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                          random_state=42, verbose=-1)
    m.fit(X[2400:M.SPLIT], (y / cap)[2400:M.SPLIT])   # yaz blogu DISLANDI (sizinti yok)
    pred = np.clip(m.predict(X), 0, None) * cap
    K = M.solar_fc.shape[1]
    idx = np.minimum(np.arange(len(df))[:, None] + np.arange(1, K + 1)[None, :], len(df) - 1)
    return pred[idx], y[idx], df.index.hour.to_numpy()


def run(solar_fc, w_econ, label):
    soc = M.CAP / 2
    n = E1 - E0
    mn = np.zeros(n); mx = np.zeros(n); P = np.zeros(n); SOC = np.zeros(n)
    for j in range(n):
        i = E0 + j
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
        p_plan, _ = M.solve_window(vmb, pr, soc, w_econ)
        p_plan = float(np.clip(p_plan, -M.P_MAX, M.P_MAX))
        p_lo = max(-M.P_MAX, -soc * M.ETA / M.dt)
        p_hi = min(M.P_MAX, (M.CAP - soc) / (M.ETA * M.dt))
        p_plan = float(np.clip(p_plan, p_lo, p_hi))
        M.set_grid(M.solar[i], M.wind[i], M.dmult[i], 0.0)
        p_safe, sinfo = M.sl.project(p_plan, p_lo, p_hi)
        mn[j] = sinfo["Vmin"]; mx[j] = sinfo["Vmax"]; P[j] = p_safe
        soc = min(max(soc + (M.ETA * p_safe * M.dt if p_safe >= 0 else p_safe * M.dt / M.ETA),
                      0), M.CAP)
        SOC[j] = soc
        if (j + 1) % 800 == 0:
            print(f"      {label}: {j+1}/{n}", flush=True)
    V = ((mn < M.LO) | (mx > M.HI)).mean() * 100
    kar = float(-(M.price[E0:E1] * P).sum() * M.dt)
    return dict(V=V, kar=kar, P=P, SOC=SOC)


def anatomy(r, price, hour):
    """Kar = -sum(price*P*dt). P>0 sarj (para GIDER), P<0 desarj (para GELIR)."""
    P, dt = r["P"], M.dt
    ch, di = P > 1e-6, P < -1e-6
    e_ch = float((P[ch] * dt).sum())
    e_di = float((-P[di] * dt).sum())
    cost_ch = float((price[ch] * P[ch] * dt).sum())          # odenen
    rev_di = float((-price[di] * P[di] * dt).sum())          # kazanilan
    pw_ch = cost_ch / e_ch if e_ch > 1e-9 else float("nan")  # sarj-agirlikli fiyat
    pw_di = rev_di / e_di if e_di > 1e-9 else float("nan")
    sm = (hour >= 10) & (hour <= 16)
    e_ch_solar = float((P[ch & sm] * dt).sum())
    return dict(e_ch=e_ch, e_di=e_di, cost_ch=cost_ch, rev_di=rev_di,
                pw_ch=pw_ch, pw_di=pw_di, e_ch_solar=e_ch_solar)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ghi", default="gg2/data/nwp_ghi_roi.csv")
    a = ap.parse_args()
    if not Path(a.ghi).exists():
        _die(f"bulunamadi: {a.ghi}")

    nwp_fc, oracle_fc, hour_all = build_nwp(a.ghi)
    price = M.price[E0:E1]
    hour = hour_all[E0:E1]

    print("=" * 78)
    print("ORACLE-KAR ANOMALISI - blok 0:2400 (yaz 2025)")
    print("=" * 78)

    # ---------------------------------------------------------------- [A]
    print("\n[A] KAR ANATOMISI (w_econ=1.0)\n")
    res = {}
    for lab, fc in (("oracle", oracle_fc), ("NWP", nwp_fc)):
        print(f"  --- {lab} kosuluyor...")
        res[lab] = run(fc, 1.0, lab)
        res[lab]["an"] = anatomy(res[lab], price, hour)
        print(f"      {lab}: ihlal {res[lab]['V']:.2f}% | kar {res[lab]['kar']:.0f} EUR")

    o, w = res["oracle"], res["NWP"]
    print(f"\n  {'':26s} {'oracle':>10s} {'NWP':>10s} {'fark':>10s}")
    rows = [("ihlal %", o["V"], w["V"]),
            ("kar (EUR)", o["kar"], w["kar"]),
            ("sarj enerjisi (MWh)", o["an"]["e_ch"], w["an"]["e_ch"]),
            ("  ...solar saatlerde", o["an"]["e_ch_solar"], w["an"]["e_ch_solar"]),
            ("desarj enerjisi (MWh)", o["an"]["e_di"], w["an"]["e_di"]),
            ("sarj-agirlikli fiyat", o["an"]["pw_ch"], w["an"]["pw_ch"]),
            ("desarj-agirlikli fiyat", o["an"]["pw_di"], w["an"]["pw_di"]),
            ("sarj maliyeti (EUR)", o["an"]["cost_ch"], w["an"]["cost_ch"]),
            ("desarj geliri (EUR)", o["an"]["rev_di"], w["an"]["rev_di"])]
    for nm, a_, b_ in rows:
        print(f"  {nm:26s} {a_:10.2f} {b_:10.2f} {a_-b_:+10.2f}")

    dk = w["kar"] - o["kar"]
    d_cost = o["an"]["cost_ch"] - w["an"]["cost_ch"]
    d_rev = w["an"]["rev_di"] - o["an"]["rev_di"]
    print(f"\n  kar farki (NWP - oracle) = {dk:+.0f} EUR")
    print(f"    fazla sarj maliyetinden : {d_cost:+.0f} EUR")
    print(f"    eksik desarj gelirinden : {d_rev:+.0f} EUR")
    print(f"    toplam aciklanan        : {d_cost + d_rev:+.0f} EUR "
          f"(kar farkinin %{100*abs(d_cost+d_rev)/max(abs(dk),1e-9):.0f}'i)")

    print("\n  [A] HUKMU:")
    c1 = o["an"]["e_ch_solar"] > w["an"]["e_ch_solar"] * 1.02   # oracle solar-saat daha cok sarj
    c2 = abs(d_cost + d_rev - dk) < 0.10 * abs(dk)              # anatomi kar farkini acikliyor
    print(f"    oracle solar saatlerde daha cok sarj ediyor mu (>%2)? "
          f"{'EVET' if c1 else 'HAYIR'}")
    print(f"    anatomi kar farkini acikliyor mu (+/-%10)?          "
          f"{'EVET' if c2 else 'HAYIR'}")
    if c1 and c2:
        print("    -> HIPOTEZ DOGRULANDI: oracle ihlali onlemek icin oglen fazladan")
        print("       sarj ediyor, bedelini kardan oduyor. Bug degil, bilincli takas.")
    else:
        print("    -> HIPOTEZ CURUDU. Mekanizma bu degil. [B] kesin cevabi verecek.")

    # ---------------------------------------------------------------- [B]
    print("\n" + "=" * 78)
    print("[B] w_econ SUPURMESI - oracle'in Pareto onu (KESIN TEST)")
    print("=" * 78)
    print(f"  Hedef: bir agirlikta oracle ihlal<={NWP_V_REF:.2f}% VE kar>={NWP_K_REF:.0f} EUR\n")
    front = [(1.0, o["V"], o["kar"])]
    for wv in W_LIST[1:]:
        print(f"  --- oracle w_econ={wv} kosuluyor...")
        r = run(oracle_fc, wv, f"oracle w={wv}")
        front.append((wv, r["V"], r["kar"]))
        print(f"      w={wv}: ihlal {r['V']:.2f}% | kar {r['kar']:.0f} EUR")

    print(f"\n  {'w_econ':>8s} {'ihlal%':>8s} {'kar(EUR)':>10s}   NWP'yi yakaladi mi?")
    hit = []
    for wv, V, k in front:
        ok = (V <= NWP_V_REF) and (k >= NWP_K_REF)
        if ok:
            hit.append(wv)
        print(f"  {wv:8.1f} {V:8.2f} {k:10.0f}   {'EVET' if ok else 'hayir'}")
    print(f"  {'--':>8s} {NWP_V_REF:8.2f} {NWP_K_REF:10.0f}   <- NWP (w_econ=1.0)")

    print("\n  [B] HUKMU:")
    if hit:
        print(f"    GECTI - oracle w_econ={hit} agirlik(lar)inda NWP'yi ikisinde birden")
        print("    yakaliyor. Anomali YOK: iki nokta ayni Pareto onunun farkli yerleri.")
        print("    Oracle satiri tabloda 'referans' olarak KALABILIR; dipnotta")
        print("    'w_econ=1.0'da raporlanmistir, agirlik degistiginde domine eder' denir.")
    else:
        print("    KALDI - hicbir agirlikta oracle NWP'yi ikisinde birden yakalayamiyor.")
        print("    NWP mukemmel-bilgili referansi GERCEKTEN domine ediyor. Bu aciklanamaz.")
        print("    -> Oracle satiri tablodan CIKARILIR. Sebep bulunana kadar kullanilmaz.")
    print()


if __name__ == "__main__":
    main()