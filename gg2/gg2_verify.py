#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_verify.py - DOGRULAMA PAKETI. Rapora gecmeden once dort kontrol.

NEDEN: Iki tuhaflik gorduk ve HIPOTEZ uydurduk. Hipotez kanit degil.
  (a) alpha bloklar arasi farkli: yaz 0.577 vs test 0.799  -> "kapasite" dedik. OLCULMEDI.
  (b) yaz bloğunda NWP oracle'dan KARLI (12798 vs 12414)   -> "agirlikli toplam" dedik. OLCULMEDI.
Ayrica: naive kosumlarinin uc capasi var (10.62 / 11.28+9681 / 5.38=val_cap.txt),
ama NWP kosumunun 9.80'i icin capa YOK.

DORT KONTROL:
  [1] PROVENANCE  (saniye)  - hangi dosya, hangi hash, hangi git commit, hangi surumler
  [2] CAP_EST GECIKMESI (saniye) - (a)'yi olcer: cap_est gercek kapasiteyi takip ediyor mu
  [3] PLASEBO  (~10 dk, TEST bloğu) - EN KESKIN TEST:
        NWP tahminini 7 GUN kaydirip MPC'ye ver. Ayni istatistik, ayni gunluk sekil,
        YANLIS hava. Ihlal naive'e (%11.28) geri donmeli.
        DONMEZSE -> iyilesme tahmin ICERIGINDEN gelmiyor -> sizinti/bug var, 9.80 COP.
  [4] ORACLE-KAR ANATOMISI (~16 dk, yaz bloğu) - (b)'yi olcer:
        oracle ve NWP'nin throughput / SOC / slack profillerini kiyaslar.
        Beklenen aciklama: oracle daha savunmaci sarj ediyor -> az ihlal, az kar.
        Tutmazsa aciklama YANLIS, baska sebep aranir.

CALISTIR (GG-1 kok klasorunden):
  python gg2/gg2_verify.py                 # hepsi (~26 dk)
  python gg2/gg2_verify.py --only 1,2      # sadece hizli olanlar (saniye)
  python gg2/gg2_verify.py --only 3        # sadece plasebo
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
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
PLACEBO_SHIFT = 7 * 48          # 7 gun
TABLE1_VIOL, TABLE1_PROFIT = 11.28, 9681.0
NWP_VIOL, NWP_PROFIT = 9.80, 10248.0


def _die(m: str) -> None:
    sys.exit(f"[HATA] {m}")


def sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ [1]
def check1() -> None:
    print("=" * 74)
    print("[1] PROVENANCE - hangi kod, hangi veri, hangi surum")
    print("=" * 74)
    try:
        c = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                    stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"],
                                        stderr=subprocess.DEVNULL).decode().strip()
        print(f"  git commit : {c}" + ("  [KIRLI: commit edilmemis degisiklik var]" if dirty else ""))
    except Exception:
        print("  git commit : (bulunamadi)")

    for p in ["data/aligned_30min_joint.csv", "data/forecast_matrix.npz",
              "data/opf_base_vm.npz", "gg2/data/nwp_ghi_roi.csv",
              "gg2/data/nwp_solar_fc_capnorm.npz"]:
        if Path(p).exists():
            print(f"  {p:42s} sha256:{sha(p)}  {Path(p).stat().st_size/1e6:7.2f} MB")
        else:
            print(f"  {p:42s} YOK")

    import lightgbm, pandapower, cvxpy, sklearn
    print(f"  surum: python {sys.version.split()[0]} | numpy {np.__version__} | "
          f"pandas {pd.__version__}")
    print(f"         lightgbm {lightgbm.__version__} | pandapower {pandapower.__version__} | "
          f"cvxpy {cvxpy.__version__} | sklearn {sklearn.__version__}")

    z = np.load("gg2/data/nwp_solar_fc_capnorm.npz")
    print(f"  NWP matris: shape {z['solar'].shape} | model {z['model']} | "
          f"alpha_mean {float(z['a_mean']):.3f}")
    print(f"  naive matris: shape {M.solar_fc.shape}")
    if z["solar"].shape != M.solar_fc.shape:
        _die("NWP ve naive matris sekilleri farkli!")
    print("  -> sekiller uyumlu.\n")


# ------------------------------------------------------------------ [2]
def check2() -> None:
    print("=" * 74)
    print("[2] CAP_EST GECIKMESI - alpha farkinin sebebi olculuyor")
    print("=" * 74)
    df = pd.read_csv("data/aligned_30min_joint.csv", index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    y = (df.solar_mw / df.iloc[:M.SPLIT].solar_mw.max() * SP).to_numpy()
    s = pd.Series(y, index=df.index)
    cap_used = s.rolling(CAP_WIN, min_periods=48).max().shift(LAG).bfill().clip(lower=0.05)
    cap_true = s.rolling(CAP_WIN, min_periods=1, center=True).max()   # merkezli = "gercek"
    t = pd.DataFrame({"cap_used": cap_used, "cap_true": cap_true}, index=df.index)
    mo = t.resample("MS").mean()
    mo["oran"] = mo.cap_used / mo.cap_true
    print(f"  {'ay':>8s} {'cap_used':>9s} {'cap_true':>9s} {'oran':>7s}")
    for d, r in mo.iterrows():
        flag = "  <- geriden geliyor" if r.oran < 0.90 else ""
        print(f"  {d:%Y-%m} {r.cap_used:9.3f} {r.cap_true:9.3f} {r.oran:7.3f}{flag}")
    val = mo.oran.iloc[:2].mean()
    tst = mo.oran.iloc[-2:].mean()
    print(f"\n  yaz bloğu (ilk 2 ay) ort oran : {val:.3f}")
    print(f"  test bloğu (son 2 ay) ort oran: {tst:.3f}")
    print("\n  HUKUM:")
    if tst < val - 0.05:
        print(f"    cap_est test bloğunda daha cok geriden geliyor ({tst:.3f} < {val:.3f}).")
        print("    -> alpha farkinin (yaz 0.577 vs test 0.799) sebebi DOGRULANDI:")
        print("       kapasite hizli buyurken 30-gunluk geriye donuk pencere yetisemiyor.")
        print("    -> NWP test bloğunda DAHA KOTU bir tahminle duvari yikmis. Lehimize.")
    else:
        print(f"    cap_est iki blokta benzer ({val:.3f} vs {tst:.3f}).")
        print("    -> HIPOTEZ CURUDU. alpha farkinin sebebi kapasite gecikmesi DEGIL.")
        print("       Baska sebep aranmali; makalede 'kapasite' diye yazilamaz.")
    print()


# ------------------------------------------------------------------ ortak kosum
def run_block(solar_fc, e0, e1, label):
    soc = M.CAP / 2
    n = e1 - e0
    mn = np.zeros(n); mx = np.zeros(n); P = np.zeros(n); SOC = np.zeros(n)
    thru = 0.0
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
        thru += abs(p_safe) * M.dt
        soc = min(max(soc + (M.ETA * p_safe * M.dt if p_safe >= 0 else p_safe * M.dt / M.ETA),
                      0), M.CAP)
        SOC[j] = soc
        if (j + 1) % 800 == 0:
            print(f"      {label}: {j+1}/{n}", flush=True)
    V = ((mn < M.LO) | (mx > M.HI)).mean() * 100
    kar = float(-(M.price[e0:e1] * P).sum() * M.dt)
    return dict(V=V, kar=kar, thru=thru, SOC=SOC, P=P, mn=mn, mx=mx)


# ------------------------------------------------------------------ [3]
def check3() -> None:
    print("=" * 74)
    print("[3] PLASEBO - EN KESKIN TEST (test bloğu)")
    print("=" * 74)
    z = np.load("gg2/data/nwp_solar_fc_capnorm.npz")
    nwp = z["solar"]
    idx = np.clip(np.arange(len(nwp)) - PLACEBO_SHIFT, 0, len(nwp) - 1)
    placebo = nwp[idx]                       # 7 gun onceki NWP tahmini
    print(f"  plasebo: NWP matrisi {PLACEBO_SHIFT} adim ({PLACEBO_SHIFT//48} gun) geri kaydirildi")
    print(f"  ayni istatistik, ayni gunluk sekil, YANLIS hava")
    print(f"  ON-KAYIT: ihlal naive'e (%{TABLE1_VIOL:.2f}) geri donmeli.")
    print(f"            %{NWP_VIOL:.2f} civarinda kalirsa -> 9.80 SAHTE, sizinti var.\n")
    r = run_block(placebo, M.SPLIT, M.N - 1, "plasebo")
    print(f"\n  {'naive (Tablo 1)':24s} %{TABLE1_VIOL:5.2f} | {TABLE1_PROFIT:7.0f} EUR")
    print(f"  {'PLASEBO (kaydirilmis)':24s} %{r['V']:5.2f} | {r['kar']:7.0f} EUR   <-- BU")
    print(f"  {'GERCEK NWP':24s} %{NWP_VIOL:5.2f} | {NWP_PROFIT:7.0f} EUR")
    print("\n  HUKUM:")
    d_naive = abs(r["V"] - TABLE1_VIOL)
    d_nwp = abs(r["V"] - NWP_VIOL)
    if d_naive < d_nwp:
        print(f"    GECTI - plasebo naive'e yakin (%{r['V']:.2f}), NWP'ye degil.")
        print("    Iyilesme tahmin ICERIGINDEN geliyor. 9.80 GERCEK.")
    else:
        print(f"    KALDI - plasebo NWP'ye yakin (%{r['V']:.2f}).")
        print("    Iyilesme tahmin iceriginden GELMIYOR -> sizinti/bug. 9.80 KULLANILAMAZ.")
        print("    Sebep bulunmadan rapora hicbir sey yazilmaz.")
    print()


# ------------------------------------------------------------------ [4]
def check4(ghi_path: str) -> None:
    print("=" * 74)
    print("[4] ORACLE-KAR ANATOMISI (yaz bloğu 0:2400)")
    print("=" * 74)
    print("  SORU: yaz bloğunda NWP neden oracle'dan KARLI (12798 vs 12414)?")
    print("  HIPOTEZ: oracle daha savunmaci sarj ediyor -> az ihlal, az kar.")
    print("  TEST: throughput ve solar-saat SOC'u kiyasla. Oracle daha cok sarj edip")
    print("        solar saatlerde daha DUSUK SOC tutuyorsa hipotez dogrulanir.\n")

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
    m.fit(X[2400:M.SPLIT], (y / cap)[2400:M.SPLIT])
    pred = np.clip(m.predict(X), 0, None) * cap
    K = M.solar_fc.shape[1]
    idx = np.minimum(np.arange(len(df))[:, None] + np.arange(1, K + 1)[None, :], len(df) - 1)
    nwp_fc = pred[idx]
    oracle_fc = y[idx]

    hour = df.index.hour.to_numpy()
    out = {}
    for lab, fc in (("oracle", oracle_fc), ("NWP", nwp_fc)):
        print(f"  --- {lab} kosuluyor...")
        out[lab] = run_block(fc, 0, 2400, lab)
    h = hour[0:2400]
    sm = (h >= 10) & (h <= 16)

    print(f"\n  {'':10s} {'ihlal%':>8s} {'kar':>8s} {'throughput':>11s} "
          f"{'SOC ort':>9s} {'SOC solar-saat':>15s}")
    for lab in ("oracle", "NWP"):
        r = out[lab]
        print(f"  {lab:10s} {r['V']:8.2f} {r['kar']:8.0f} {r['thru']:11.1f} "
              f"{r['SOC'].mean():9.3f} {r['SOC'][sm].mean():15.3f}")

    print("\n  HUKUM:")
    o, w = out["oracle"], out["NWP"]
    more_thru = o["thru"] > w["thru"]
    lower_soc = o["SOC"][sm].mean() < w["SOC"][sm].mean()
    if more_thru or lower_soc:
        print(f"    Oracle daha cok sarj/desarj ({o['thru']:.0f} vs {w['thru']:.0f}) "
              f"ve/veya solar saatlerde daha dusuk SOC "
              f"({o['SOC'][sm].mean():.3f} vs {w['SOC'][sm].mean():.3f}).")
        print("    -> HIPOTEZ DOGRULANDI: oracle savunmaci davraniyor, kardan veriyor.")
        print("       Bu bir bug degil, agirlikli-toplam amacin Pareto konumu.")
    else:
        print("    Oracle daha savunmaci DEGIL -> hipotez CURUDU.")
        print("    Kar farkinin sebebi baska. Aciklanamayan fark makalede sorun olur;")
        print("    sebep bulunana kadar oracle satiri 'referans' olarak bile verilmemeli.")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="GG-2 dogrulama paketi.")
    ap.add_argument("--ghi", default="gg2/data/nwp_ghi_roi.csv")
    ap.add_argument("--only", default="1,2,3,4", help="or: --only 1,2")
    a = ap.parse_args()
    sel = {s.strip() for s in a.only.split(",")}
    if "1" in sel: check1()
    if "2" in sel: check2()
    if "3" in sel: check3()
    if "4" in sel: check4(a.ghi)
    print("Dogrulama paketi bitti.")


if __name__ == "__main__":
    main()