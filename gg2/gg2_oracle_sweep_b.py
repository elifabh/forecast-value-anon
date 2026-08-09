#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_oracle_sweep_b.py - [B] TEKRAR: oracle w_econ supurmesi, DOGRU aralikta.

ONCEKI DENEME GECERSIZDI:
  w_econ {1,3,10,30} supurmustum. Ama W_SLACK=1e4. 30, 10000'in yaninda hicbir sey
  -> ekonomi agirligi HIC devreye girmedi. Kanit cikti icindeydi:
     ihlal 2.08 -> 2.17 (kipirdamadi) | kar 12414 -> 12528 (%0.9)
  Agirlik isirmiyorsa supurme YAPILMAMIS demektir. O hukum GECERSIZ.

DOGRU ARALIK NEREDEN: <author>'in kendi raporu "Economic-weight sweep (1e-2 to 1e3)"
diyor; mpc_econ_voltage_pareto.py'nin kendi w_list'i [100, 1000]. Ikisi de
elimdeydi, bakmamistim. Bu sefer bakildi.

SORU: Oracle'in Pareto onu, NWP noktasini (2.38%, 12798 EUR) iceriyor mu?
  GECER : bir agirlikta oracle ihlal<=2.38 VE kar>=12798
          -> anomali YOK. Iki nokta ayni onun farkli yerleri. Oracle satiri
             tabloda kalabilir; dipnotta agirlik belirtilir.
  KALIR : hicbir agirlikta yakalayamiyor
          -> NWP mukemmel-bilgili referansi GERCEKTEN domine ediyor.
             Aciklanamaz -> oracle satiri tablodan CIKARILIR.

YENI ZORUNLU KONTROL (bu hatanin tekrarlamamasi icin):
  Supurme yorumlanmadan ONCE "dugme donuyor mu" test edilir.
  Kar araligi < %5 ise -> agirlik isirmiyor -> HUKUM VERILMEZ, aralik genisletilir.

BILINEN NOKTALAR (on-kayit, onceki kosumlardan; yeniden kosulmaz):
  oracle w_econ=1.0 : ihlal 2.08% | kar 12414 EUR
  NWP    w_econ=1.0 : ihlal 2.38% | kar 12798 EUR

CALISTIR:
  python gg2/gg2_oracle_sweep_b.py 2>&1 | tee gg2/oracle_sweep_b.log
Sure: 4 kosum x ~8 dk = ~32 dk.
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
E0, E1 = 0, 2400
W_LIST = [100.0, 300.0, 1000.0, 3000.0]      # W_SLACK=1e4 -> bu aralik ISIRIR
NWP_V_REF, NWP_K_REF = 2.38, 12798.0         # on-kayit
ORA_W1_V, ORA_W1_K = 2.08, 12414.0           # on-kayit (w_econ=1.0)
KNOB_MIN_RANGE_PCT = 5.0                     # dugme donuyor mu esigi


def _die(m: str) -> None:
    sys.exit(f"[HATA] {m}")


def build_oracle():
    df = pd.read_csv("data/aligned_30min_joint.csv", index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    y = (df.solar_mw / df.iloc[:M.SPLIT].solar_mw.max() * SP).to_numpy()
    K = M.solar_fc.shape[1]
    idx = np.minimum(np.arange(len(df))[:, None] + np.arange(1, K + 1)[None, :], len(df) - 1)
    return y[idx]


def run(solar_fc, w_econ, label):
    soc = M.CAP / 2
    n = E1 - E0
    mn = np.zeros(n); mx = np.zeros(n); P = np.zeros(n)
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
        if (j + 1) % 800 == 0:
            print(f"      {label}: {j+1}/{n}", flush=True)
    V = ((mn < M.LO) | (mx > M.HI)).mean() * 100
    kar = float(-(M.price[E0:E1] * P).sum() * M.dt)
    return V, kar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", default=",".join(str(w) for w in W_LIST),
                    help="virgullu w_econ listesi")
    a = ap.parse_args()
    wlist = [float(x) for x in a.w.split(",")]

    print("=" * 78)
    print("[B] TEKRAR - oracle w_econ supurmesi, blok 0:2400 (yaz 2025)")
    print("=" * 78)
    print(f"  W_SLACK = 1e4 (sabit, solve_window icinde)")
    print(f"  supurulen w_econ: {wlist}")
    print(f"  on-kayit hedef  : oracle ihlal<={NWP_V_REF:.2f}% VE kar>={NWP_K_REF:.0f} EUR")
    print(f"  bilinen noktalar: oracle w=1.0 -> {ORA_W1_V:.2f}% / {ORA_W1_K:.0f} EUR")
    print(f"                    NWP    w=1.0 -> {NWP_V_REF:.2f}% / {NWP_K_REF:.0f} EUR")
    print("=" * 78)

    oracle_fc = build_oracle()
    front = [(1.0, ORA_W1_V, ORA_W1_K)]          # onceki kosumdan, yeniden kosulmaz
    for wv in wlist:
        print(f"\n  --- oracle w_econ={wv:g} kosuluyor...")
        V, k = run(oracle_fc, wv, f"w={wv:g}")
        front.append((wv, V, k))
        print(f"      w={wv:g}: ihlal {V:.2f}% | kar {k:.0f} EUR")

    print("\n" + "=" * 78)
    print("ORACLE PARETO ONU")
    print("=" * 78)
    print(f"  {'w_econ':>9s} {'ihlal%':>8s} {'kar(EUR)':>10s}   NWP'yi yakaladi mi?")
    hit = []
    for wv, V, k in front:
        ok = (V <= NWP_V_REF) and (k >= NWP_K_REF)
        if ok:
            hit.append(wv)
        tag = "  [onceki kosum]" if wv == 1.0 else ""
        print(f"  {wv:9.4g} {V:8.2f} {k:10.0f}   {'EVET' if ok else 'hayir'}{tag}")
    print(f"  {'--':>9s} {NWP_V_REF:8.2f} {NWP_K_REF:10.0f}   <- NWP")

    # --- ZORUNLU: dugme donuyor mu? (hukumden ONCE)
    ks = [k for _, _, k in front]
    vs = [V for _, V, _ in front]
    k_rng = 100.0 * (max(ks) - min(ks)) / max(abs(min(ks)), 1e-9)
    v_rng = max(vs) - min(vs)
    print("\n  DUGME KONTROLU (hukumden ONCE - onceki denemeyi bu yakalardi):")
    print(f"    kar araligi   : {min(ks):.0f} -> {max(ks):.0f} EUR  (%{k_rng:.1f})")
    print(f"    ihlal araligi : {min(vs):.2f} -> {max(vs):.2f}  ({v_rng:.2f} puan)")
    if k_rng < KNOB_MIN_RANGE_PCT:
        print(f"    -> AGIRLIK ISIRMIYOR (kar araligi %{k_rng:.1f} < %{KNOB_MIN_RANGE_PCT}).")
        print("       SUPURME GECERSIZ. HUKUM VERILMEZ. Aralik genisletilmeli:")
        print(f"       ornek: python gg2/gg2_oracle_sweep_b.py --w 1e4,3e4,1e5")
        print()
        return
    print(f"    -> agirlik isiriyor (kar araligi %{k_rng:.1f}). Hukum verilebilir.")

    print("\n  HUKUM:")
    if hit:
        print(f"    GECTI - oracle w_econ={hit} agirlik(lar)inda NWP'yi ikisinde birden")
        print("    yakaliyor. ANOMALI YOK: iki nokta ayni Pareto onunun farkli yerleri.")
        print("    -> Oracle satiri tabloda 'referans' olarak KALIR. Dipnot:")
        print("       'w_econ=1.0'da raporlanmistir; agirlik artirilinca NWP'yi domine eder.'")
    else:
        print("    KALDI - agirlik isiriyor AMA hicbir noktada oracle NWP'yi ikisinde")
        print("    birden yakalayamiyor. NWP, mukemmel-bilgili referansi GERCEKTEN")
        print("    domine ediyor. Bu aciklanamaz.")
        print("    -> Oracle satiri tablodan CIKARILIR. Ana bulgu (naive 11.28 -> NWP 9.80)")
        print("       bundan ETKILENMEZ; oracle sadece referans satiriydi.")
    print()


if __name__ == "__main__":
    main()