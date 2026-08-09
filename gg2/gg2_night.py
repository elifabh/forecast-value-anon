#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_night.py - GECE PAKETI: research raporunun Stage 1+2+3 kontrolleri.

Kaynak: adversarial literatur taramasi (14 Tem). Uc is:

[1] HATA CUBUKLARI (saniyeler - mevcut npz'lerden, yeniden kosum YOK)
    Soru: oracle-vs-NWP kar farki (384 EUR) ve ana bulgu (naive-vs-NWP) gurultu mu?
    Yontem: gunluk-blok ESLI bootstrap (ayni gunler, fark uzerinden), 10000 tekrar.
    ON-KAYIT: |fark| < 2*SE -> GURULTU, yorum yapilamaz, null olarak raporlanir.
              |fark| >= 2*SE -> gercek fark.

[2] ORACLE-FIYAT SUPURMESI (~35 dk - KESIN TEST, research Stage 2)
    Soru: w_econ ters cephesi tahmin hatasindan mi, kurulum artefaktindan mi?
    Yontem: ayni supurme ama MUKEMMEL fiyat + MUKEMMEL solar. Tahmin hatasi sifir.
    ON-KAYIT: ters cephe SURUYORSA -> tahmin hatasi DEGIL, artefakt (agirlik olcegi
              + projeksiyon). KAYBOLURSA -> tahmin-hatasi amplifikasyonu dogrulanir
              (optimizer's curse ailesi, Smith&Winkler 2006).

[3] PROJEKSIYON TANISI ([2]'nin icinde, bedava - research item f + b)
    Her kosumda loglanir: L1 = sum|p_plan - p_safe| (safe layer plani ne kadar
    eziyor), override sayisi, SOC-sinirda-kalma orani.
    ON-KAYIT: L1 w_econ ile buyuyorsa -> "ters cephe projeksiyon artefakti"
    aciklamasi desteklenir.

CALISTIR (gece, tek komut):
  python gg2/gg2_night.py 2>&1 | tee gg2/night_$(date +%Y%m%d).log
Sadece [1]: python gg2/gg2_night.py --only 1
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
E0, E1 = 0, 2400                    # yaz blogu (ters cephenin goruldugu yer)
W_LIST = [1.0, 100.0, 300.0, 1000.0]
NBOOT = 10000
BLOCK = 48                          # 1 gun


def _die(m: str) -> None:
    sys.exit(f"[HATA] {m}")


# ====================================================================== [1]
def paired_bootstrap(vals_a: np.ndarray, vals_b: np.ndarray, rng) -> tuple[float, float]:
    """Gunluk-blok esli bootstrap: fark ortalamasinin SE ve %95 CI yari-genisligi."""
    nb = len(vals_a) // BLOCK
    da = vals_a[:nb * BLOCK].reshape(nb, BLOCK).mean(axis=1)
    db = vals_b[:nb * BLOCK].reshape(nb, BLOCK).mean(axis=1)
    diff = da - db
    boots = np.array([diff[rng.integers(0, nb, nb)].mean() for _ in range(NBOOT)])
    return float(boots.std()), float(np.quantile(np.abs(boots - boots.mean()), 0.95))


def check1() -> None:
    print("=" * 74)
    print("[1] HATA CUBUKLARI - gunluk-blok esli bootstrap (test blogu)")
    print("=" * 74)
    T0 = M.SPLIT
    price = M.price[T0:M.N - 1]
    runs = {}
    for tag in ("naive", "oracle", "nwp"):
        p = Path(f"gg2/nwp_fair_traj_{tag}.npz")
        if not p.exists():
            _die(f"bulunamadi: {p}")
        c = np.load(p)
        viol = ((c["mn"] < M.LO) | (c["mx"] > M.HI)).astype(float)
        prof = -price * c["p"] * M.dt
        runs[tag] = (viol, prof)
        print(f"  {tag:7s}: ihlal {viol.mean()*100:6.2f}% | kar {prof.sum():8.0f} EUR")

    rng = np.random.default_rng(42)
    print(f"\n  esli bootstrap ({NBOOT} tekrar, {BLOCK}-adimlik gunluk bloklar):")
    for a, b, isim in (("oracle", "nwp", "YAN GOZLEM 1 (oracle vs NWP)"),
                       ("naive", "nwp", "ANA BULGU (naive vs NWP)")):
        va, pa = runs[a]; vb, pb = runs[b]
        n = min(len(va), len(vb))
        se_v, _ = paired_bootstrap(va[:n], vb[:n], rng)
        se_p, _ = paired_bootstrap(pa[:n], pb[:n], rng)
        nb = n // BLOCK
        dv = (va[:nb*BLOCK].mean() - vb[:nb*BLOCK].mean()) * 100
        dp = (pa[:nb*BLOCK].reshape(nb, BLOCK).mean(1).mean()
              - pb[:nb*BLOCK].reshape(nb, BLOCK).mean(1).mean()) * nb * BLOCK
        se_v_pp = se_v * 100
        se_p_tot = se_p * nb * BLOCK
        zv = abs(dv) / max(se_v_pp, 1e-12)
        zp = abs(dp) / max(se_p_tot, 1e-12)
        print(f"\n  {isim}")
        print(f"    ihlal farki : {dv:+7.2f} pp   SE {se_v_pp:5.2f}  -> {zv:4.1f} SE  "
              f"[{'GERCEK' if zv >= 2 else 'GURULTU OLABILIR'}]")
        print(f"    kar farki   : {dp:+7.0f} EUR  SE {se_p_tot:5.0f}  -> {zp:4.1f} SE  "
              f"[{'GERCEK' if zp >= 2 else 'GURULTU OLABILIR'}]")

    print("\n  ON-KAYITLI OKUMA:")
    print("    - 'GURULTU OLABILIR' cikan hicbir fark makalede yorumlanmaz;")
    print("      null sonuc olarak durustce raporlanir.")
    print("    - Ana bulgunun ihlal farki gurultuyse TUM hikaye degisir (beklenmiyor;")
    print("      plasebo testi zaten icerik etkisini gosterdi, ama olcelim).")
    print()


# ====================================================================== [2]+[3]
def build_oracle_solar():
    df = pd.read_csv("data/aligned_30min_joint.csv", index_col=0, parse_dates=True)
    y = (df.solar_mw / df.iloc[:M.SPLIT].solar_mw.max() * SP).to_numpy()
    K = M.solar_fc.shape[1]
    idx = np.minimum(np.arange(len(df))[:, None] + np.arange(1, K + 1)[None, :], len(df) - 1)
    return y[idx]


def run_oracle_prices(solar_fc, w_econ, label):
    """Supurme kosumu: MUKEMMEL fiyat penceresi + verilen solar. Tani loglu."""
    soc = M.CAP / 2
    n = E1 - E0
    mn = np.zeros(n); mx = np.zeros(n); P = np.zeros(n)
    l1 = 0.0; n_override = 0; n_soc_edge = 0
    for j in range(n):
        i = E0 + j
        Hh = min(M.H, M.N - i)
        v_now = D.ldf_v(M.solar[i], M.wind[i], M.dmult[i])
        vmb = np.empty((Hh, D.n)); vmb[0] = M.VM_BASE[i]
        for t in range(1, Hh):
            kk = min(t - 1, solar_fc.shape[1] - 1)
            v_fut = D.ldf_v(solar_fc[i, kk], M.wind_fc[i, kk], M.dmult_fc[i, kk])
            vmb[t] = M.VM_BASE[i] + (v_fut - v_now)
        # MUKEMMEL FIYAT: gercek gelecek fiyat penceresi (tek degisiklik)
        pr = np.array([M.price[min(i + t, M.N - 1)] for t in range(Hh)])
        p_plan, _ = M.solve_window(vmb, pr, soc, w_econ)
        p_plan = float(np.clip(p_plan, -M.P_MAX, M.P_MAX))
        p_lo = max(-M.P_MAX, -soc * M.ETA / M.dt)
        p_hi = min(M.P_MAX, (M.CAP - soc) / (M.ETA * M.dt))
        p_plan = float(np.clip(p_plan, p_lo, p_hi))
        M.set_grid(M.solar[i], M.wind[i], M.dmult[i], 0.0)
        p_safe, sinfo = M.sl.project(p_plan, p_lo, p_hi)
        l1 += abs(p_plan - p_safe)
        if abs(p_plan - p_safe) > 1e-6:
            n_override += 1
        mn[j] = sinfo["Vmin"]; mx[j] = sinfo["Vmax"]; P[j] = p_safe
        soc = min(max(soc + (M.ETA * p_safe * M.dt if p_safe >= 0 else p_safe * M.dt / M.ETA),
                      0), M.CAP)
        if soc < 0.05 * M.CAP or soc > 0.95 * M.CAP:
            n_soc_edge += 1
        if (j + 1) % 800 == 0:
            print(f"      {label}: {j+1}/{n}", flush=True)
    V = ((mn < M.LO) | (mx > M.HI)).mean() * 100
    kar = float(-(M.price[E0:E1] * P).sum() * M.dt)
    return dict(V=V, kar=kar, l1=l1, n_ov=n_override, n_soc=n_soc_edge)


def check23() -> None:
    print("=" * 74)
    print("[2] KESIN TEST - w_econ supurmesi, MUKEMMEL fiyat + MUKEMMEL solar")
    print("=" * 74)
    print("  Tahmin hatasi SIFIR. Ters cephe yine cikarsa sebep tahmin hatasi DEGIL.")
    print(f"  Referans (kusurlu fiyatla, onceki kosum):")
    print(f"    w=1: 2.08%/12414 | w=100: 3.25%/12510 | w=300: 5.21%/11433 "
          f"| w=1000: 5.96%/9592\n")
    oracle_solar = build_oracle_solar()
    rows = []
    for w in W_LIST:
        print(f"  --- w_econ={w:g} kosuluyor...")
        r = run_oracle_prices(oracle_solar, w, f"w={w:g}")
        rows.append((w, r))
        print(f"      w={w:g}: ihlal {r['V']:.2f}% | kar {r['kar']:.0f} EUR "
              f"| L1 {r['l1']:.1f} MW | override {r['n_ov']} | SOC-sinir {r['n_soc']}")

    print("\n" + "=" * 74)
    print("SONUC TABLOSU (mukemmel fiyat + mukemmel solar)")
    print("=" * 74)
    print(f"  {'w_econ':>8s} {'ihlal%':>8s} {'kar(EUR)':>10s} {'L1(MW)':>8s} "
          f"{'override':>9s} {'SOC-sinir':>10s}")
    for w, r in rows:
        print(f"  {w:8.4g} {r['V']:8.2f} {r['kar']:10.0f} {r['l1']:8.1f} "
              f"{r['n_ov']:9d} {r['n_soc']:10d}")

    kars = [r["kar"] for _, r in rows]
    k1, kmax, kend = kars[0], max(kars), kars[-1]
    inverted = kend < k1 * 0.98            # son nokta ilk noktadan belirgin dusuk mu
    l1s = [r["l1"] for _, r in rows]
    l1_rising = l1s[-1] > l1s[0] * 1.5

    print("\n  ON-KAYITLI HUKUM [2]:")
    if inverted:
        print("    Ters cephe MUKEMMEL fiyatla da SURUYOR.")
        print("    -> Sebep tahmin hatasi DEGIL. Kurulum artefakti (agirlik olcegi +")
        print("       projeksiyon). w_econ supurmesi 'bulgu' olarak RAPORLANMAZ;")
        print("       sadece metodolojik not olur. Benim 'fiyat tahmini' hipotezim OLDU.")
    else:
        print("    Ters cephe mukemmel fiyatla KAYBOLDU.")
        print("    -> Tahmin-hatasi amplifikasyonu dogrulandi (optimizer's curse ailesi,")
        print("       Smith&Winkler 2006). Atifla raporlanabilir.")

    print("\n  ON-KAYITLI OKUMA [3] (projeksiyon tanisi):")
    if l1_rising:
        print(f"    L1 diverjansi w ile buyuyor ({l1s[0]:.0f} -> {l1s[-1]:.0f} MW).")
        print("    -> Safe layer agresif planlari eziyor; 'planlanan kar gerceklesmiyor'")
        print("       mekanizmasi (research item f) DESTEKLENDI.")
    else:
        print(f"    L1 diverjansi buyumuyor ({l1s[0]:.0f} -> {l1s[-1]:.0f} MW).")
        print("    -> Projeksiyon aciklamasi desteklenmedi; agirlik-olcegi (item 0/c)")
        print("       tek basina kaliyor.")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="1,2")
    a = ap.parse_args()
    sel = {s.strip() for s in a.only.split(",")}
    if "1" in sel:
        check1()
    if "2" in sel:
        check23()
    print("Gece paketi bitti.")


if __name__ == "__main__":
    main()