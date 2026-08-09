#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_v2_season_boot.py - YAZ BLOGU HATA CUBUKLARI (denetim: DOGRULANMAMIS #3).

SORUN
  gg2_nwp_season.py seviyeleri uretiyor (5.38 / 2.38 / 2.08) ama `run_block`
  sadece iki float donduruyor; adim-bazli mn/mx/P DISKE YAZILMIYOR.
  Bu yuzden yaz blogunun HICBIR hata cubugu YOK. Yaz blogu, genelleme
  kanitinin iki ayagindan biri; hata cubuksuz kullanilamaz.

NE YAPAR
  1. `gg2_nwp_season.build_nwp` AYNEN import edilir (yeniden yazilmaz).
  2. `run_block` govdesi birebir kopyalanir, TEK FARK: mn/mx/P/SOC saklanir.
  3. CIPA: seviyeler gg2_nwp_season ciktisiyla BIREBIR eslesmeli.
     Eslesmezse kopya bozuktur, HICBIR hukum verilmez.
  4. Eslestirilmis gunluk-blok bootstrap + YUZDELIK guven araligi.

CIPA (on-kayit, gg2/nwp_season.log'dan)
  naive  5.38% / 11247.4 EUR
  oracle 2.08% / 12413.9 EUR
  NWP    2.38% / 12798.4 EUR
  tahminci: naive MAE 0.2404 | NWP 0.1388 | alpha 0.577

ON-KAYITLI HUKUM
  naive - NWP ihlal farkinin %95 yuzdelik araligi sifiri DISLIYORSA
    -> yaz replikasyonu hata cubuguyla AYAKTA, genelleme kaniti kullanilabilir.
  Sifiri ICERIYORSA
    -> yaz blogu "ayni yonde ama cozulmemis" olarak yazilir; test blogu
       TEK olculmus blok olarak kalir ve makalede boyle soylenir.
  KAR icin ayri hukum VERILMEZ. DEVIR 5.9'da isaret-testi zaten DUSURULDU;
  kar sadece aralikla raporlanir.

SURE: ~30-45 dk (3 kol x 2400 adim). Kol basina ara kayit var; script
tekrar calistirilirsa mevcut yorungeler ATLANIR.

CALISTIR (kok klasorden):
  python gg2/gg2_v2_season_boot.py 2>&1 | tee gg2/v2_season_boot.log
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore", message=".*valid feature names.*")

import mpc_econ_voltage_pareto as M
import diag_mpc_forecastvolt as D
import gg2_nwp_season as S            # build_nwp AYNEN kullanilir

W_ECON = 1.0
E0, E1 = 0, 2400
BLOCK, NDAYS, NBOOT, SEED = 48, 50, 10000, 12345

ANCHOR = {                             # gg2/nwp_season.log
    "naive":  (5.38, 11247.4),
    "oracle": (2.08, 12413.9),
    "NWP":    (2.38, 12798.4),
}
TOL_V, TOL_K = 0.02, 5.0
OUT = Path(ROOT) / "gg2" / "data"


def die(m):
    sys.exit("\n[DUR] " + m + "\n")


def run_block_save(solar_fc, e0, e1, label):
    """gg2_nwp_season.run_block ile BIREBIR AYNI govde.
    TEK FARK: mn/mx/P/SOC dondurulur ve diske yazilir."""
    soc = M.CAP / 2
    n = e1 - e0
    mn = np.zeros(n); mx = np.zeros(n); P = np.zeros(n); SOC = np.zeros(n)
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
        SOC[j] = soc
        if (j + 1) % 500 == 0:
            print("      %s: %d/%d" % (label, j + 1, n), flush=True)
    return mn, mx, P, SOC


def stats(d):
    m = float(d.mean()); s = float(d.std(ddof=1))
    lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
    sk = float(((d - m) ** 3).mean() / s ** 3) if s > 0 else 0.0
    return m, s, lo, hi, sk, float((d <= 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ghi", default="gg2/data/nwp_ghi_roi.csv")
    ap.add_argument("--fit-start", type=int, default=2400)
    ap.add_argument("--fit-end", type=int, default=13800)
    a = ap.parse_args()

    print("=" * 84)
    print("V2 - YAZ BLOGU YORUNGE + BOOTSTRAP")
    print("=" * 84)
    print("  degerlendirme %d:%d | tahminci fit %d:%d | w_econ=%.1f"
          % (E0, E1, a.fit_start, a.fit_end, W_ECON))
    if not (E1 <= a.fit_start or E0 >= a.fit_end):
        die("SIZINTI: degerlendirme ve fit pencereleri kesisiyor")
    OUT.mkdir(parents=True, exist_ok=True)

    print("\n  tahminci kuruluyor (gg2_nwp_season.build_nwp) ...")
    nwp_fc, aux = S.build_nwp(a.ghi, a.fit_start, a.fit_end)
    sl = slice(E0, E1)
    tr = aux["y"][aux["idx"]]
    en = float(np.abs(M.solar_fc[sl] - tr[sl]).mean())
    ew = float(np.abs(nwp_fc[sl] - tr[sl]).mean())
    print("  MAE naive %.4f | NWP %.4f | alpha %.3f  (cipa: 0.2404 / 0.1388 / 0.577)"
          % (en, ew, ew / en))
    if abs(ew / en - 0.577) > 0.005:
        die("tahminci cipasi tutmadi (alpha %.3f)" % (ew / en))
    print("  -> tahminci cipasi GECTI")

    oracle_fc = aux["y"][aux["idx"]]
    ARMS = (("naive", M.solar_fc), ("oracle", oracle_fc), ("NWP", nwp_fc))

    V, P = {}, {}
    print("\n" + "=" * 84)
    print("KOSUM + SEVIYE CIPASI")
    print("=" * 84)
    for label, fc in ARMS:
        f = OUT / ("season_%s.npz" % label)
        if f.exists():
            z = np.load(f)
            mn, mx, p = z["mn"], z["mx"], z["p"]
            print("  --- %s: mevcut yorunge kullaniliyor (%s)" % (label, f.name))
        else:
            print("  --- %s kosuluyor ..." % label)
            mn, mx, p, soc = run_block_save(fc, E0, E1, label)
            np.savez(f, mn=mn, mx=mx, p=p, soc=soc, start=E0, done=E1 - E0)
        v = ((mn < M.LO) | (mx > M.HI)).astype(np.float64)
        k = float(-(M.price[E0:E1] * p).sum() * M.dt)
        av, ak = ANCHOR[label]
        dv, dk = abs(100 * v.mean() - av), abs(k - ak)
        ok = dv <= TOL_V and dk <= TOL_K
        print("      ihlal %.2f%% (cipa %.2f, fark %.3f) | kar %.1f (cipa %.1f, fark %.1f) -> %s"
              % (100 * v.mean(), av, dv, k, ak, dk, "GECTI" if ok else "KALDI"))
        if not ok:
            die("%s seviye cipasi tutmadi. Kopyalanan run_block orijinalden\n"
                "      farkli davraniyor. HICBIR hukum verilmez." % label)
        V[label], P[label] = v, p

    # ------------------------------------------------------------ bootstrap
    need = BLOCK * NDAYS
    if (E1 - E0) != need:
        die("blok uzunlugu %d, %d bekleniyordu" % (E1 - E0, need))
    price = M.price[E0:E1].reshape(NDAYS, BLOCK)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, NDAYS, size=(NBOOT, NDAYS))
    BV, BK = {}, {}
    for t in V:
        BV[t] = V[t].reshape(NDAYS, BLOCK).mean(axis=1)[draws].mean(axis=1) * 100.0
        BK[t] = (-(price * P[t].reshape(NDAYS, BLOCK)).sum(axis=1) * M.dt)[draws].sum(axis=1)

    print("\n" + "=" * 84)
    print("YAZ BLOGU - ESLESTIRILMIS BOOTSTRAP (%d gun, %d tekrar)" % (NDAYS, NBOOT))
    print("=" * 84)
    print("  %-16s %8s %7s %6s %18s %8s %9s" %
          ("karsilastirma", "ort", "SD", "SD kat", "%95 yuzdelik CI", "carpik", "P(<=0)"))
    pairs = [("naive", "NWP"), ("naive", "oracle"), ("NWP", "oracle")]
    res = {}
    for A, B in pairs:
        m, s, lo, hi, sk, p0 = stats(BV[A] - BV[B])
        res[(A, B)] = (m, s, lo, hi, p0)
        print("  %-16s %+8.2f %7.2f %6.1f  [%+6.2f, %+6.2f] %8.2f %9.3f  %s"
              % (A + "-" + B, m, s, abs(m) / s if s else 0, lo, hi, sk, p0,
                 "SIFIRI DISLIYOR" if (lo > 0 or hi < 0) else "sifiri iceriyor"))
    print("\n  kar farklari (EUR) - HUKUM VERILMEZ, sadece aralik:")
    for A, B in pairs:
        m, s, lo, hi, sk, p0 = stats(BK[A] - BK[B])
        print("  %-16s %+8.0f %7.0f  [%+8.0f, %+8.0f]  P(<=0)=%.3f  %s"
              % (A + "-" + B, m, s, lo, hi, p0,
                 "SIFIRI DISLIYOR" if (lo > 0 or hi < 0) else "sifiri iceriyor"))

    print("\n" + "=" * 84)
    print("ON-KAYITLI HUKUM")
    print("=" * 84)
    m, s, lo, hi, p0 = res[("naive", "NWP")]
    print("  yaz  naive-NWP = %+.2f puan, %%95 CI [%+.2f, %+.2f], P(<=0)=%.3f" % (m, lo, hi, p0))
    print("  test naive-bm  = +1.54 puan, %95 CI [+0.29, +3.12], P(<=0)=0.005")
    if lo > 0:
        print("\n  -> YAZ REPLIKASYONU HATA CUBUGUYLA AYAKTA.")
        print("     Bulgu iki blokta da olculdu; genelleme kaniti KULLANILABILIR.")
    else:
        print("\n  -> YAZ BLOGU COZULMEDI (aralik sifiri iceriyor).")
        print("     'Ayni yonde ama cozulmemis' yazilir; test blogu TEK olculmus")
        print("     blok olarak kalir ve makalede acikca boyle soylenir.")
    print("\n  NOT: kar icin hukum VERILMEDI. DEVIR 5.9 isaret-testini dusurdu;")
    print("  'kar artti' ASLA yazilmaz. gg2_nwp_season.py'nin bastigi")
    print("  '(2) NWP kari artiriyor mu: EVET' satiri makaleye GIRMEZ.\n")


if __name__ == "__main__":
    main()