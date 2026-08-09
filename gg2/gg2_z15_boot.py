#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_z15_boot.py - Z14+Z15 SONUC: tum kollar uzerinde eslestirilmis bootstrap.

NE OLCUYOR
  Elimizde ayni kontrolcude, TEK degisken solar tahmini olacak sekilde
  kosulmus yedi yorunge var:
    naive | oracle | best_match | best_match seed1 | best_match seed7
    | ecmwf_ifs025 | gfs_seamless
  Seviyeler zaten biliniyor. Bu script FARKLARIN hata cubugunu uretir.

  Yontem: gunluk blok (48 adim), eslestirilmis, 10.000 tekrar.
  Eslestirilmis = her tekrarda AYNI gunler tum kollara uygulanir.
  Gunlerin ortak gurultusu boylece farktan dusulur.

CIPA (gecmezse hicbir hukum verilmez)
  naive - best_match ihlal farki = +1.54 puan, SE = 0.72  (DEVIR 4.1)
  Bu sayi 2400 adim (50 tam gun) uzerinden hesaplanmisti; ayni pencere.

ON-KAYITLI SORULAR (sonuclara bakilmadan yazildi)
  S1 [Z15 hukum 1] : uc hava modelinin ihlal yayilimi 0.72 puani asiyor mu?
                     ASMIYORSA "etki hava modeli secimine duyarsiz."
  S2 [monotonluk]  : ECMWF (alpha .759) ile GFS (alpha .924) farki 2 SE'yi
                     asiyor mu? ASMIYORSA siralama YORUMLANMAZ - ne
                     "monoton" ne "monoton degil" denir; ayirt edilemiyor.
                     ASIYORSA ve yon alpha ile ters ise, bu bir BULGUDUR ve
                     ACIKLAMA UYDURULMADAN raporlanir.
  S3 [seed]        : uc seed'in yayilimi (zaten 0.08) 2 SE icinde mi?
  S4              : her NWP kolu naive'den 2 SE ile ayrisiyor mu?

  |fark| < 2 SE -> YORUMLANMAZ. Bu kural her satira uygulanir.

CALISTIR (GG-1 kok klasorunden, ~2 dk):
  python gg2/gg2_z15_boot.py 2>&1 | tee gg2/z15_boot.log
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ayni ihlal/kar tanimlari icin orijinal modul kullanilir (kopyalanmaz).
import mpc_econ_voltage_pareto as M

BLOCK = 48          # 1 gun
NDAYS = 50          # DEVIR 3.14: SE'ler 50 tam gun uzerinden
NBOOT = 10000
SEED = 12345

ANCHOR_DIFF = 1.54
ANCHOR_SE = 0.72
TOL_DIFF = 0.10
TOL_SE = 0.10

ARMS = [
    ("naive",       "gg2/nwp_fair_traj_naive.npz",        1.000),
    ("oracle",      "gg2/nwp_fair_traj_oracle.npz",       0.000),
    ("best_match",  "gg2/nwp_fair_traj_nwp.npz",          0.799),
    ("bm_seed1",    "gg2/nwp_fair_s1_nwp.npz",            0.781),
    ("bm_seed7",    "gg2/nwp_fair_s7_nwp.npz",            0.795),
    ("ecmwf",       "gg2/nwp_fair_ecmwf_ifs025_nwp.npz",  0.759),
    ("gfs",         "gg2/nwp_fair_gfs_seamless_nwp.npz",  0.924),
]

# (A, B) -> raporlanan fark A - B
PAIRS = [
    ("naive", "best_match", "CIPA: DEVIR 4.1 = +1.54 / SE 0.72"),
    ("naive", "ecmwf",      "S4"),
    ("naive", "gfs",        "S4"),
    ("ecmwf", "gfs",        "S2 monotonluk"),
    ("ecmwf", "best_match", "S1 hava modeli"),
    ("gfs",   "best_match", "S1 hava modeli"),
    ("best_match", "bm_seed1", "S3 seed"),
    ("best_match", "bm_seed7", "S3 seed"),
    ("best_match", "oracle", "bilgi boslugu"),
    ("ecmwf", "oracle",     "bilgi boslugu"),
]


def _die(m: str) -> None:
    sys.exit("[HATA] " + m)


def load_arm(path: str):
    p = Path(path)
    if not p.exists():
        _die("yorunge yok: %s" % path)
    z = np.load(p)
    for k in ("mn", "mx", "p", "done"):
        if k not in z:
            _die("%s icinde '%s' yok" % (path, k))
    done = int(z["done"])
    viol = ((z["mn"] < M.LO) | (z["mx"] > M.HI)).astype(np.float64)
    return viol[:done], z["p"][:done], done


def main() -> None:
    ap = argparse.ArgumentParser(description="Z15: tum kollar eslestirilmis bootstrap.")
    ap.add_argument("--nboot", type=int, default=NBOOT)
    a = ap.parse_args()

    print("=" * 78)
    print("Z14+Z15 - ESLESTIRILMIS GUNLUK-BLOK BOOTSTRAP")
    print("=" * 78)
    print("  blok %d adim | %d gun | %d tekrar | LO=%.3f HI=%.3f"
          % (BLOCK, NDAYS, a.nboot, M.LO, M.HI))

    V, P, dones = {}, {}, {}
    for tag, path, _al in ARMS:
        v, p, d = load_arm(path)
        V[tag], P[tag], dones[tag] = v, p, d

    nmin = min(dones.values())
    need = BLOCK * NDAYS
    if nmin < need:
        _die("en kisa yorunge %d adim, %d gerekiyor" % (nmin, need))
    print("  yorunge uzunluklari: %s" % ", ".join("%s=%d" % (t, dones[t]) for t in dones))
    print("  bootstrap penceresi: ilk %d adim (%d x %d)" % (need, NDAYS, BLOCK))

    T0 = M.SPLIT
    price = M.price[T0:T0 + nmin]

    # ------------------------------------------------------------ seviyeler
    print("\n" + "=" * 78)
    print("SEVIYELER (tam yorunge, %d adim)" % nmin)
    print("=" * 78)
    print("  %-12s %6s %9s %11s" % ("kol", "alpha", "ihlal %", "kar EUR"))
    for tag, _path, al in ARMS:
        d = dones[tag]
        vv = 100.0 * V[tag].mean()
        kk = float(-(M.price[T0:T0 + d] * P[tag]).sum() * M.dt)
        print("  %-12s %6.3f %9.2f %11.1f" % (tag, al, vv, kk))

    # ------------------------------------------------------------ bootstrap
    Vb = {t: V[t][:need].reshape(NDAYS, BLOCK) for t in V}
    Pb = {t: P[t][:need].reshape(NDAYS, BLOCK) for t in P}
    prb = price[:need].reshape(NDAYS, BLOCK)

    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, NDAYS, size=(a.nboot, NDAYS))

    def boot_stats(tag):
        vd = Vb[tag].mean(axis=1)                       # gun basina ihlal orani
        kd = -(prb * Pb[tag]).sum(axis=1) * M.dt        # gun basina kar
        return vd[draws].mean(axis=1) * 100.0, kd[draws].sum(axis=1)

    BV, BK = {}, {}
    for tag in V:
        BV[tag], BK[tag] = boot_stats(tag)

    print("\n" + "=" * 78)
    print("FARKLAR (A - B), eslestirilmis")
    print("=" * 78)
    print("  %-22s %9s %7s %6s   %s" % ("karsilastirma", "fark pp", "SE", "kat", "hukum"))
    print("  " + "-" * 74)
    anchor_row = None
    for A, Bt, note in PAIRS:
        dv = BV[A] - BV[Bt]
        m_, s_ = float(dv.mean()), float(dv.std(ddof=1))
        k = abs(m_) / s_ if s_ > 0 else float("inf")
        verdict = "GERCEK" if k >= 2.0 else "yorumlanmaz"
        print("  %-22s %+9.2f %7.2f %6.1f   %-12s %s"
              % (A + " - " + Bt, m_, s_, k, verdict, note))
        if A == "naive" and Bt == "best_match":
            anchor_row = (m_, s_)

    print("\n  kar farklari (EUR):")
    for A, Bt, note in PAIRS:
        dk = BK[A] - BK[Bt]
        m_, s_ = float(dk.mean()), float(dk.std(ddof=1))
        k = abs(m_) / s_ if s_ > 0 else float("inf")
        print("  %-22s %+9.0f %7.0f %6.1f   %s"
              % (A + " - " + Bt, m_, s_, k, "GERCEK" if k >= 2.0 else "yorumlanmaz"))

    # ---------------------------------------------------------------- CIPA
    print("\n" + "=" * 78)
    print("CIPA - DEVIR 4.1 yeniden uretildi mi?")
    print("=" * 78)
    if anchor_row is None:
        _die("cipa satiri bulunamadi")
    m_, s_ = anchor_row
    ok = (abs(m_ - ANCHOR_DIFF) <= TOL_DIFF) and (abs(s_ - ANCHOR_SE) <= TOL_SE)
    print("  beklenen: %+.2f pp / SE %.2f" % (ANCHOR_DIFF, ANCHOR_SE))
    print("  olculen : %+.2f pp / SE %.2f" % (m_, s_))
    print("  -> %s" % ("GECTI" if ok else "KALDI - hukum verilmez, sebep aranir"))

    # ------------------------------------------------------- on-kayitli S1
    print("\n" + "=" * 78)
    print("ON-KAYITLI SORULAR")
    print("=" * 78)
    lv = {t: 100.0 * V[t].mean() for t in V}
    hav = ["best_match", "ecmwf", "gfs"]
    spread = max(lv[t] for t in hav) - min(lv[t] for t in hav)
    print("  S1 hava modeli yayilimi = %.2f puan  (esik 0.72)" % spread)
    print("     -> %s" % ("DUYARSIZ: etki hava modeli secimine bagli degil."
                          if spread < 0.72 else
                          "DUYARLI: buyukluk hava modeline bagli, SINIRDIR."))
    dv = BV["ecmwf"] - BV["gfs"]
    k2 = abs(float(dv.mean())) / float(dv.std(ddof=1))
    print("  S2 ecmwf - gfs = %+.2f pp, %.1f SE" % (float(dv.mean()), k2))
    if k2 < 2.0:
        print("     -> AYIRT EDILEMIYOR. Siralama YORUMLANMAZ.")
        print("        'monoton' da 'monoton degil' de YAZILMAZ.")
    else:
        print("     -> AYRISIYOR. Yon alpha ile ters ise BULGUDUR;")
        print("        aciklama UYDURULMAZ, olculmeden yazilmaz.")
    sd = ["best_match", "bm_seed1", "bm_seed7"]
    print("  S3 seed yayilimi = %.2f puan"
          % (max(lv[t] for t in sd) - min(lv[t] for t in sd)))
    print()


if __name__ == "__main__":
    main()