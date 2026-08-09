#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_z16_final.py - Z16 SONUC: plasebonun DOGRU testi.

NEDEN BU SCRIPT VAR (hata itirafi)
  Z16 Adim 1'in on-kaydi HATALI yazildi: plasebo esigi "naive ile NWP arasi
  orta nokta" olarak, HATA CUBUGU OLMADAN tanimlandi. Bu, DEVIR 3.8'de
  belgelenmis yaz-kar hatasinin ayni sinifidir (isaret testi, hata cubuksuz).
  Sonucu: "plasebo 3 esigin altinda" uyarisi atesledi, ama eslestirilmis
  bootstrap plac3 - naive = -1.08 +/- 0.67 (1.6 SE) veriyor -> COZULMUYOR.
  Yani uyari gurultuyu okumus.

  Ayrica Z16 Adim 2'de test edilen "kaydirilmis tahmin beceri tasir" hipotezi
  CURUDU: dort kaydirmanin alpha'si 1.285 / 1.461 / 1.291 / 1.636, hepsi 1'in
  USTUNDE ve kaydirma uzunluguyla monoton DEGIL. Bu iyi bir sonuctur:
  kaydirilmis tahminler gercekten becerisiz, yani gercek plasebo.

PLASEBONUN DOGRU SORUSU
  "Plasebo naive'e mi yakin" DEGIL.
  "Plasebo, GERCEK NWP'den anlamli olcude kotu mu" -> iddia buysa test bu olmali.

ON-KAYIT (bu script yazilmadan once, sonuclara bakilmadan)
  Olculecek : her plasebonun (a) gercek NWP'den, (b) naive'den farki,
              eslestirilmis gunluk-blok bootstrap ile.
  HUKUM (1) : dort plasebonun HEPSI gercek NWP'den 2 SE ile KOTU
              -> "Kazanc tahmin iceriginden geliyor; dort kaydirmada dogrulandi."
  HUKUM (2) : bazilari 2 SE'yi asmiyor
              -> Asanlar plasebo kanitidir, asmayanlar COZULMEDI diye
                 raporlanir. Hicbiri gizlenmez, hicbiri secilmez.
  HUKUM (3) : herhangi bir plasebo gercek NWP'den 2 SE ile IYI ise
              -> iddia CURUR, 9.80 kullanilamaz. Sebep aranir.

  CIPA : naive - NWP = +1.54 / SE 0.72 (DEVIR 4.1). Cikmazsa hukum yok.

CALISTIR (~1 dk, MPC KOSMAZ):
  python gg2/gg2_z16_final.py 2>&1 | tee gg2/z16_final.log
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mpc_econ_voltage_pareto as M

BLOCK = 48
NDAYS = 50
NBOOT = 10000
BOOT_SEED = 12345
ANCHOR_DIFF, ANCHOR_SE = 1.54, 0.72
TOL = 0.10

ARMS = {
    "naive":  ("gg2/nwp_fair_traj_naive.npz",  1.000),
    "NWP":    ("gg2/nwp_fair_traj_nwp.npz",    0.799),
    "plac3":  ("gg2/data/placebo_shift3.npz",  1.285),
    "plac7":  ("gg2/data/placebo_shift7.npz",  1.461),
    "plac14": ("gg2/data/placebo_shift14.npz", 1.291),
    "plac21": ("gg2/data/placebo_shift21.npz", 1.636),
}
PLACEBOS = ["plac3", "plac7", "plac14", "plac21"]


def _die(m: str) -> None:
    sys.exit("[HATA] " + m)


def main() -> None:
    ap = argparse.ArgumentParser(description="Z16 sonuc: plasebonun dogru testi.")
    ap.add_argument("--nboot", type=int, default=NBOOT)
    a = ap.parse_args()

    V, P, dones = {}, {}, {}
    for tag, (path, _al) in ARMS.items():
        if not Path(path).exists():
            _die("yorunge yok: %s" % path)
        z = np.load(path)
        d = int(z["done"])
        V[tag] = ((z["mn"] < M.LO) | (z["mx"] > M.HI)).astype(np.float64)[:d]
        P[tag] = z["p"][:d]
        dones[tag] = d

    need = BLOCK * NDAYS
    if min(dones.values()) < need:
        _die("yorunge cok kisa")

    T0 = M.SPLIT
    prb = M.price[T0:T0 + need].reshape(NDAYS, BLOCK)
    Vb = {t: V[t][:need].reshape(NDAYS, BLOCK) for t in V}
    Pb = {t: P[t][:need].reshape(NDAYS, BLOCK) for t in P}
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, NDAYS, size=(a.nboot, NDAYS))
    BV = {t: (Vb[t].mean(axis=1))[draws].mean(axis=1) * 100.0 for t in Vb}
    BK = {t: (-(prb * Pb[t]).sum(axis=1) * M.dt)[draws].sum(axis=1) for t in Pb}

    print("=" * 76)
    print("Z16 SONUC - PLASEBONUN DOGRU TESTI")
    print("=" * 76)
    print("  %d gun x %d adim | %d tekrar | eslestirilmis" % (NDAYS, BLOCK, a.nboot))

    print("\n  SEVIYELER")
    print("  %-10s %8s %10s %11s" % ("kol", "alpha", "ihlal %", "kar EUR"))
    print("  " + "-" * 44)
    for t in ["NWP", "plac3", "plac7", "plac14", "plac21", "naive"]:
        d = dones[t]
        kk = float(-(M.price[T0:T0 + d] * P[t]).sum() * M.dt)
        print("  %-10s %8.3f %10.2f %11.1f" % (t, ARMS[t][1], 100.0 * V[t].mean(), kk))

    def cmp_(A, B_):
        dv = BV[A] - BV[B_]
        m_, s_ = float(dv.mean()), float(dv.std(ddof=1))
        return m_, s_, (abs(m_) / s_ if s_ > 0 else float("inf"))

    print("\n" + "=" * 76)
    print("CIPA")
    print("=" * 76)
    m_, s_, k = cmp_("naive", "NWP")
    ok = abs(m_ - ANCHOR_DIFF) <= TOL and abs(s_ - ANCHOR_SE) <= TOL
    print("  naive - NWP = %+.2f +/- %.2f  (beklenen %+.2f +/- %.2f)  -> %s"
          % (m_, s_, ANCHOR_DIFF, ANCHOR_SE, "GECTI" if ok else "KALDI"))
    if not ok:
        print("  KALDI -> hicbir hukum verilmez.")
        sys.exit(2)

    print("\n" + "=" * 76)
    print("ASIL TEST: her plasebo GERCEK NWP'den kotu mu?")
    print("=" * 76)
    print("  %-18s %9s %7s %6s  %s" % ("cift", "fark pp", "SE", "kat", "hukum"))
    print("  " + "-" * 58)
    worse, unresolved, better = [], [], []
    for t in PLACEBOS:
        m_, s_, k = cmp_(t, "NWP")
        if k >= 2.0 and m_ > 0:
            v, lst = "PLASEBO KOTU (2 SE)", worse
        elif k >= 2.0 and m_ < 0:
            v, lst = "PLASEBO IYI -> IDDIA CURUR", better
        else:
            v, lst = "cozulmuyor", unresolved
        lst.append(t)
        print("  %-18s %+9.2f %7.2f %6.1f  %s" % (t + " - NWP", m_, s_, k, v))

    print("\n  referans: her plasebo naive'e gore")
    print("  %-18s %9s %7s %6s  %s" % ("cift", "fark pp", "SE", "kat", "hukum"))
    print("  " + "-" * 58)
    for t in PLACEBOS:
        m_, s_, k = cmp_(t, "naive")
        print("  %-18s %+9.2f %7.2f %6.1f  %s"
              % (t + " - naive", m_, s_, k, "GERCEK" if k >= 2.0 else "cozulmuyor"))

    print("\n  kar (plasebo - NWP, EUR)")
    for t in PLACEBOS:
        dk = BK[t] - BK["NWP"]
        m_, s_ = float(dk.mean()), float(dk.std(ddof=1))
        k = abs(m_) / s_ if s_ > 0 else float("inf")
        print("  %-18s %+9.0f %7.0f %6.1f  %s"
              % (t + " - NWP", m_, s_, k, "GERCEK" if k >= 2.0 else "cozulmuyor"))

    print("\n" + "=" * 76)
    print("ON-KAYITLI HUKUM")
    print("=" * 76)
    if better:
        print("  HUKUM (3): %s gercek NWP'den 2 SE ile IYI." % ", ".join(better))
        print("    -> IDDIA CURUDU. 9.80 kullanilamaz. Sebep aranir.")
    elif not unresolved:
        print("  HUKUM (1): DORT plasebonun HEPSI gercek NWP'den 2 SE ile kotu.")
        print("    -> Kazanc tahmin ICERIGINDEN geliyor; dort kaydirmada dogrulandi.")
    else:
        print("  HUKUM (2): 2 SE ile kotu olanlar: %s"
              % (", ".join(worse) if worse else "yok"))
        print("             cozulmeyenler          : %s" % ", ".join(unresolved))
        print("    -> Asanlar plasebo kanitidir. Asmayanlar COZULMEDI diye")
        print("       raporlanir. Hicbiri gizlenmez, hicbiri secilmez.")
    print("\n  EK OLCUM (Adim 2): kaydirilmis tahminlerin alpha'si")
    print("    plac3 1.285 | plac7 1.461 | plac14 1.291 | plac21 1.636")
    print("    Hepsi 1'in USTUNDE -> kaydirilmis tahminlerin becerisi YOK.")
    print("    Kaydirma uzunluguyla monoton DEGIL; sebep BILINMIYOR, uydurulmaz.\n")


if __name__ == "__main__":
    main()