#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_v4_curtailment.py - KISINTI: voltaj metrigini IKLIM metrigine cevirir.

NEDEN
  "Ihlal %11.28 -> %9.80" operasyonel bir metrik. CCAI bir IKLIM workshop'u.
  "Gun-oncesi hava tahmini X MWh temiz enerji kurtariyor" cok daha guclu ve
  dogrudan iklim etkisine baglanabilir bir ifadedir.

NE OLCUYOR
  Batarya ihlali gideremediginde gercek bir sebekede yapilacak sey solar'i
  KISMAKTIR. Her ihlal adiminda sorulur:
    "Voltaji tam 1.05'e indirmek icin ne kadar solar kisilmaliydi?"
  Cevap gercek AC guc akisiyla ikili aramayla bulunur. Yaklasim DEGIL, TAM.

  Kisilan enerji = toplam(kisilan MW) * dt  [MWh]

  MPC YENIDEN KOSULMAZ. Mevcut yorungeler kullanilir; hicbir mevcut sayi
  degismez. Tamamen TURETILMIS yeni bir metrik eklenir.

CIPA (gecmezse hicbir sayi raporlanmaz)
  Kisinti = 0 ve batarya = yorungedeki p_safe iken yeniden hesaplanan Vmax,
  yorungedeki mx[j] degerini <= 1e-6 farkla vermeli. Vermezse sebeke durumu
  yanlis kuruluyor demektir.

ON-KAYIT (sonuclara bakilmadan)
  Raporlanacak tek sayi: naive ve NWP kollarinin kisilan enerjisi ve FARKI,
  gunluk-blok eslestirilmis bootstrap + %95 yuzdelik araligiyla.
  Aralik sifiri DISLIYORSA  -> "day-ahead forecast avoids X MWh of curtailment
                               (95% CI [a,b])" yazilabilir.
  Aralik sifiri ICERIYORSA -> "curtailment difference not resolved" yazilir.
                               Ihlal sonucu bundan ETKILENMEZ.
  Kisintinin toplam solar uretimine ORANI da raporlanir (baglam icin).

SURE: ~2-4 dk (sadece ihlal adimlari, adim basina ~10 runpp).

CALISTIR (kok klasorden):
  python gg2/gg2_v4_curtailment.py 2>&1 | tee gg2/v4_curtailment.log
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandapower as pp
import mpc_econ_voltage_pareto as M

BLOCK, NDAYS, NBOOT, SEED = 48, 50, 10000, 12345
BISECT = 40
TOL_MW = 1e-4
ANCHOR_TOL = 1e-6

ARMS = {
    "naive":  "gg2/nwp_fair_traj_naive.npz",
    "NWP":    "gg2/nwp_fair_traj_nwp.npz",
    "oracle": "gg2/nwp_fair_traj_oracle.npz",
}


def die(m):
    sys.exit("\n[DUR] " + m + "\n")


def vmax_at(i, cut, p_batt):
    """Gercek AC: adim i, solar'dan `cut` MW kisilmis, batarya p_batt."""
    M.set_grid(max(M.solar[i] - cut, 0.0), M.wind[i], M.dmult[i], p_batt)
    pp.runpp(M.net)
    return float(M.net.res_bus.vm_pu.max())


def needed_cut(i, p_batt, v0):
    """Vmax'i HI'ya indiren en kucuk kisinti (ikili arama, gercek AC)."""
    hi = float(M.solar[i])
    if hi <= 0:
        return 0.0
    if vmax_at(i, hi, p_batt) > M.HI:
        return hi                      # tum solar kisilsa bile yetmiyor
    lo = 0.0
    for _ in range(BISECT):
        if hi - lo < TOL_MW:
            break
        mid = 0.5 * (lo + hi)
        if vmax_at(i, mid, p_batt) > M.HI:
            lo = mid
        else:
            hi = mid
    return hi


def stats(d):
    m = float(d.mean()); s = float(d.std(ddof=1))
    lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
    return m, s, lo, hi, float((d <= 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=NBOOT)
    a = ap.parse_args()

    print("=" * 84)
    print("V4 - KISILAN SOLAR ENERJI (test blogu)")
    print("=" * 84)
    T0 = M.SPLIT
    need = BLOCK * NDAYS
    tot_solar = float(M.solar[T0:T0 + need].sum() * M.dt)
    print("  pencere ilk %d adim | toplam solar uretimi %.1f MWh | HI=%.3f"
          % (need, tot_solar, M.HI))

    CUT = {}
    for tag, f in ARMS.items():
        p = Path(ROOT) / f
        if not p.exists():
            die("yorunge yok: %s" % f)
        z = np.load(p)
        n = int(z["done"])
        mn, mx, ps = z["mn"][:n], z["mx"][:n], z["p"][:n]
        viol = np.where(mx[:need] > M.HI)[0]

        # ---- CIPA: sebeke durumu dogru kuruluyor mu?
        j = int(viol[0])
        v_re = vmax_at(T0 + j, 0.0, float(ps[j]))
        d0 = abs(v_re - float(mx[j]))
        print("\n  --- %s: %d ihlal adimi" % (tag, len(viol)))
        print("      cipa adim %d: yorunge mx=%.6f | yeniden %.6f | fark %.2e -> %s"
              % (j, mx[j], v_re, d0, "GECTI" if d0 <= ANCHOR_TOL else "KALDI"))
        if d0 > ANCHOR_TOL:
            die("%s: sebeke durumu yeniden kurulamadi (fark %.2e). Hicbir\n"
                "      kisinti sayisi raporlanmaz." % (tag, d0))

        cuts = np.zeros(need)
        t0 = time.time()
        for k, j in enumerate(viol):
            cuts[j] = needed_cut(T0 + int(j), float(ps[j]), float(mx[j]))
            if (k + 1) % 50 == 0:
                el = time.time() - t0
                print("      %d/%d  (%.0f sn, kalan ~%.0f sn)"
                      % (k + 1, len(viol), el, el / (k + 1) * (len(viol) - k - 1)),
                      flush=True)
        CUT[tag] = cuts
        e = float(cuts.sum() * M.dt)
        print("      kisilan enerji %.2f MWh  (toplam solar'in %%%.3f'u)"
              % (e, 100.0 * e / tot_solar))
        print("      tam kisintinin bile yetmedigi adim: %d"
              % int(((cuts > 0) & (cuts >= M.solar[T0:T0 + need] - TOL_MW)).sum()))

    # ------------------------------------------------------------ bootstrap
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, NDAYS, size=(a.nboot, NDAYS))
    B = {t: (CUT[t].reshape(NDAYS, BLOCK).sum(axis=1) * M.dt)[draws].sum(axis=1)
         for t in CUT}

    print("\n" + "=" * 84)
    print("KISILAN ENERJI FARKLARI (MWh, 50 gun)")
    print("=" * 84)
    print("  %-16s %9s %8s %20s %9s" % ("karsilastirma", "ort", "SD", "%95 CI", "P(<=0)"))
    for A, Bt in (("naive", "NWP"), ("naive", "oracle"), ("NWP", "oracle")):
        m, s, lo, hi, p0 = stats(B[A] - B[Bt])
        print("  %-16s %+9.2f %8.2f  [%+8.2f, %+8.2f] %9.3f  %s"
              % (A + "-" + Bt, m, s, lo, hi, p0,
                 "SIFIRI DISLIYOR" if (lo > 0 or hi < 0) else "sifiri iceriyor"))

    print("\n" + "=" * 84)
    print("ON-KAYITLI HUKUM")
    print("=" * 84)
    m, s, lo, hi, p0 = stats(B["naive"] - B["NWP"])
    en, ew = float(CUT["naive"].sum() * M.dt), float(CUT["NWP"].sum() * M.dt)
    print("  naive kisinti %.2f MWh | NWP kisinti %.2f MWh | fark %+.2f MWh" % (en, ew, m))
    print("  %%95 CI [%+.2f, %+.2f]  P(<=0)=%.3f" % (lo, hi, p0))
    if lo > 0:
        print("\n  -> YAZILABILIR: day-ahead forecast avoids %.2f MWh of curtailment" % m)
        print("     over %d days (95%% CI [%.2f, %.2f]), i.e. %.3f%% of solar output."
              % (NDAYS, lo, hi, 100.0 * m / tot_solar))
    else:
        print("\n  -> KISINTI FARKI COZULMEDI. 'not resolved' yazilir.")
        print("     Ihlal sonucu bundan ETKILENMEZ.")
    print("\n  SINIR: kisinti, ihlali gidermek icin GEREKEN miktardir; gercek bir")
    print("  DSO'nun kisinti politikasi modellenmedi. Bu bir ust-sinir tahminidir.\n")


if __name__ == "__main__":
    main()