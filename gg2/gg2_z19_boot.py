#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_z19_boot.py - Z19 (sifir-hava kontrolu) kollarinin eslestirilmis bootstrap'i.

Bu script gg2_z15_boot.py'nin MAKINESIDIR. Bootstrap mantigi, blok uzunlugu,
gun sayisi, tekrar sayisi, RNG seed'i, ihlal tanimi ve kar tanimi BIREBIR
kopyalandi. Degisen: hangi kollar yuklendi ve hangi sorular soruldu.

NE OLCUYOR
  Dort yorunge:
    naive       - persistence tahmini            (DEVIR 3.1)
    best_match  - gercek NWP, on-kayitli manset  (DEVIR 3.1)
    cal         - hava YOK, sadece takvim ozellikleri        (DEVIR 3.6)
    calcap      - hava YOK, takvim + cap_est                 (DEVIR 3.6)
  Seviyeler biliniyor: 11.28 / 9.80 / 12.06 / 12.76.
  Bu script FARKLARIN hata cubugunu uretir.

CIPA (gecmezse HICBIR hukum verilmez)
  naive - best_match ihlal farki = +1.54 puan, SE = 0.72  (DEVIR 3.1)
  Ayni pencere: 2400 adim = 50 tam gun.

ON-KAYITLI SORULAR (sonuclara BAKILMADAN yazildi, esikler 2 SE cinsinden)
  S1 [DEVIR 5.5'in dayanagi]
     cal - naive >= 2 SE mi?
     EVET -> 5.5 su haliyle yazilir: hava girdisi kaldirilinca sonuc
             persistence'in OLCULEBILIR sekilde altina duser.
     HAYIR -> 5.5 YUMUSATILIR: "above the persistence baseline" cumlesi
             cikarilir, yerine "not resolved against persistence" yazilir.
             Manset (KATKI 1) bundan ETKILENMEZ; sadece 5.5 zayiflar.

  S2 [DEVIR 3.6'nin ic celiskisi]
     calcap - cal >= 2 SE mi?  (ham fark +0.70 puan)
     EVET -> iki sifir-hava kolu OLCULEBILIR sekilde farkli. DEVIR 3.6'daki
             "cal ve calcap birebir ayni" cumlesi YANLIS, silinir. Fark
             SARI LISTEYE eklenir: tek bir ozellik eklemek manset etkinin
             yarisi kadar oynatiyor.
     HAYIR -> "birebir ayni" cumlesi yine SILINIR (tabloda 0.70 fark var,
             "ayni" degil), yerine "ayrismiyor (X SE)" yazilir.
     HER IKI DURUMDA DA "birebir ayni" ifadesi makaleye GIRMEZ.

  S3 [dogrudan olcum, bootstrap degil]
     cal ve calcap yorungeleri eleman-eleman AYNI mi?
     AYNI ISE  -> tablodaki 12.06 / 12.76 farki bir AKTARIM HATASIDIR;
                  Z19 durur, sayilar yeniden uretilir, 5.5 yazilmaz.
     FARKLI ISE-> DEVIR 3.6'daki "cap_est bolmede sadelesiyor" gerekcesi
                  YANLIS'tir ve silinir. cap_est bir GIRDI OZELLIGIDIR.

  |fark| < 2 SE -> YORUMLANMAZ. Her satira uygulanir.

CALISTIR (GG-1 kok klasorunden, ~1 dk):
  python gg2/gg2_z19_boot.py 2>&1 | tee gg2/z19_boot.log
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

# (tag, aday dosya yollari, bilinen alpha)
ARMS = [
    ("naive",      ["gg2/nwp_fair_traj_naive.npz", "nwp_fair_traj_naive.npz"], 1.000),
    ("best_match", ["gg2/nwp_fair_traj_nwp.npz",   "nwp_fair_traj_nwp.npz"],   0.799),
    ("cal",        ["gg2/nwp_fair_cal_nwp.npz",    "nwp_fair_cal_nwp.npz"],    1.594),
    ("calcap",     ["gg2/nwp_fair_calcap_nwp.npz", "nwp_fair_calcap_nwp.npz"], 1.594),
]

# (A, B, not) -> raporlanan fark A - B
PAIRS = [
    ("naive",  "best_match", "CIPA: DEVIR 3.1 = +1.54 / SE 0.72"),
    ("cal",    "naive",      "S1 - DEVIR 5.5'in dayanagi"),
    ("calcap", "cal",        "S2 - DEVIR 3.6 ic celiskisi (ham +0.70)"),
    ("cal",    "best_match", "destek: sifir-hava vs gercek NWP"),
    ("calcap", "best_match", "destek: sifir-hava vs gercek NWP"),
]


def _die(m):
    sys.exit("[HATA] " + m)


def _resolve(cands):
    for c in cands:
        if Path(c).exists():
            return c
    return None


def load_arm(cands):
    path = _resolve(cands)
    if path is None:
        _die("yorunge yok, denenen yollar: %s" % ", ".join(cands))
    z = np.load(path)
    for k in ("mn", "mx", "p", "done"):
        if k not in z:
            _die("%s icinde '%s' yok" % (path, k))
    done = int(z["done"])
    viol = ((z["mn"] < M.LO) | (z["mx"] > M.HI)).astype(np.float64)
    raw = {"mn": z["mn"][:done], "mx": z["mx"][:done], "p": z["p"][:done]}
    return viol[:done], z["p"][:done], done, path, raw


def main():
    ap = argparse.ArgumentParser(description="Z19: sifir-hava kollari eslestirilmis bootstrap.")
    ap.add_argument("--nboot", type=int, default=NBOOT)
    a = ap.parse_args()

    print("=" * 78)
    print("Z19 - SIFIR-HAVA KOLLARI, ESLESTIRILMIS GUNLUK-BLOK BOOTSTRAP")
    print("=" * 78)
    print("  blok %d adim | %d gun | %d tekrar | LO=%.3f HI=%.3f"
          % (BLOCK, NDAYS, a.nboot, M.LO, M.HI))

    V, P, dones, paths, RAW = {}, {}, {}, {}, {}
    for tag, cands, _al in ARMS:
        v, p, d, pth, raw = load_arm(cands)
        V[tag], P[tag], dones[tag], paths[tag], RAW[tag] = v, p, d, pth, raw
        print("  yuklendi: %-11s <- %s  (done=%d)" % (tag, pth, d))

    nmin = min(dones.values())
    need = BLOCK * NDAYS
    if nmin < need:
        _die("en kisa yorunge %d adim, %d gerekiyor" % (nmin, need))
    print("  bootstrap penceresi: ilk %d adim (%d x %d)" % (need, NDAYS, BLOCK))

    T0 = M.SPLIT
    price = M.price[T0:T0 + nmin]

    # ------------------------------------------------------------ seviyeler
    print("\n" + "=" * 78)
    print("SEVIYELER (tam yorunge)")
    print("=" * 78)
    print("  %-12s %6s %7s %9s %11s" % ("kol", "alpha", "adim", "ihlal %", "kar EUR"))
    for tag, _c, al in ARMS:
        d = dones[tag]
        vv = 100.0 * V[tag].mean()
        kk = float(-(M.price[T0:T0 + d] * P[tag]).sum() * M.dt)
        print("  %-12s %6.3f %7d %9.2f %11.1f" % (tag, al, d, vv, kk))

    # ------------------------------------ S3: dogrudan yorunge kimlik testi
    print("\n" + "=" * 78)
    print("S3 - cal / calcap YORUNGE KIMLIK TESTI (bootstrap degil, dogrudan)")
    print("=" * 78)
    nc = min(dones["cal"], dones["calcap"])
    identical = True
    for k in ("mn", "mx", "p"):
        A = np.asarray(RAW["cal"][k][:nc], dtype=np.float64)
        B = np.asarray(RAW["calcap"][k][:nc], dtype=np.float64)
        dmax = float(np.max(np.abs(A - B))) if nc > 0 else 0.0
        ndiff = int(np.count_nonzero(A != B))
        print("  %-3s  max|fark| = %.6e   farkli adim = %d / %d" % (k, dmax, ndiff, nc))
        if ndiff != 0:
            identical = False
    vA = V["cal"][:nc]
    vB = V["calcap"][:nc]
    print("  ihlal bayragi farkli adim = %d / %d" % (int(np.count_nonzero(vA != vB)), nc))
    if identical:
        print("  -> YORUNGELER BIREBIR AYNI.")
        print("     O ZAMAN ihlal oranlari da ayni olmali. DEVIR 3.6 tablosundaki")
        print("     12.06 / 12.76 farki AKTARIM HATASIDIR. Z19 DURUR, sayilar")
        print("     yeniden uretilir, DEVIR 5.5 YAZILMAZ.")
    else:
        print("  -> YORUNGELER FARKLI.")
        print("     DEVIR 3.6'daki 'cal ve calcap birebir ayni' cumlesi YANLIS.")
        print("     'cap_est bolmede sadelesiyor' gerekcesi de YANLIS: cap_est")
        print("     bir GIRDI OZELLIGIDIR, bolen degil. Her iki cumle SILINIR.")

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
    print("  %-24s %9s %7s %6s   %s" % ("karsilastirma", "fark pp", "SE", "kat", "hukum"))
    print("  " + "-" * 76)
    anchor_row = None
    res = {}
    for A, Bt, note in PAIRS:
        dv = BV[A] - BV[Bt]
        m_, s_ = float(dv.mean()), float(dv.std(ddof=1))
        k = abs(m_) / s_ if s_ > 0 else float("inf")
        res[(A, Bt)] = (m_, s_, k)
        verdict = "GERCEK" if k >= 2.0 else "yorumlanmaz"
        print("  %-24s %+9.2f %7.2f %6.1f   %-12s %s"
              % (A + " - " + Bt, m_, s_, k, verdict, note))
        if A == "naive" and Bt == "best_match":
            anchor_row = (m_, s_)

    print("\n  kar farklari (EUR):")
    for A, Bt, note in PAIRS:
        dk = BK[A] - BK[Bt]
        m_, s_ = float(dk.mean()), float(dk.std(ddof=1))
        k = abs(m_) / s_ if s_ > 0 else float("inf")
        print("  %-24s %+9.0f %7.0f %6.1f   %s"
              % (A + " - " + Bt, m_, s_, k, "GERCEK" if k >= 2.0 else "yorumlanmaz"))

    # ---------------------------------------------------------------- CIPA
    print("\n" + "=" * 78)
    print("CIPA - DEVIR 3.1 yeniden uretildi mi?")
    print("=" * 78)
    if anchor_row is None:
        _die("cipa satiri bulunamadi")
    m_, s_ = anchor_row
    ok = (abs(m_ - ANCHOR_DIFF) <= TOL_DIFF) and (abs(s_ - ANCHOR_SE) <= TOL_SE)
    print("  beklenen: %+.2f pp / SE %.2f" % (ANCHOR_DIFF, ANCHOR_SE))
    print("  olculen : %+.2f pp / SE %.2f" % (m_, s_))
    print("  -> %s" % ("GECTI" if ok else "KALDI - HUKUM VERILMEZ, sebep aranir"))

    # ------------------------------------------------- on-kayitli hukumler
    print("\n" + "=" * 78)
    print("ON-KAYITLI HUKUMLER")
    print("=" * 78)
    if not ok:
        print("  CIPA KALDI -> S1 ve S2 hukumleri VERILMEZ.")
        print()
        return

    m1, s1, k1 = res[("cal", "naive")]
    print("  S1  cal - naive = %+.2f pp, SE %.2f, %.1f SE" % (m1, s1, k1))
    if k1 >= 2.0:
        print("      -> DEVIR 5.5 SU HALIYLE YAZILIR.")
        print("         Hava girdisi kaldirilinca sonuc persistence'in OLCULEBILIR")
        print("         sekilde altina duser. NWP degeri gercek GHI alanindan gelir.")
    else:
        print("      -> DEVIR 5.5 YUMUSATILIR.")
        print("         'above the persistence baseline' CIKARILIR; yerine")
        print("         'not resolved against persistence (%.1f SE)' yazilir." % k1)
        print("         Manset (KATKI 1) etkilenmez.")

    m2, s2, k2 = res[("calcap", "cal")]
    print("  S2  calcap - cal = %+.2f pp, SE %.2f, %.1f SE" % (m2, s2, k2))
    if k2 >= 2.0:
        print("      -> IKI SIFIR-HAVA KOLU OLCULEBILIR SEKILDE FARKLI.")
        print("         'birebir ayni' SILINIR. Fark SARI LISTEYE eklenir:")
        print("         tek bir ozellik eklemek sonucu %.2f puan oynatiyor" % abs(m2))
        print("         (manset etki 1.48, SE 0.72).")
    else:
        print("      -> AYRISMIYOR (%.1f SE)." % k2)
        print("         'birebir ayni' YINE SILINIR (tabloda 0.70 fark var);")
        print("         yerine 'not resolved (%.1f SE)' yazilir." % k2)
    print()


if __name__ == "__main__":
    main()