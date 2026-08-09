#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_mpc2_floor.py - IKI-BATARYA TABAN OLCUMU (perfect mod). SON KOSUM.

NEDEN: Toplam-3.0 kosumu ara bolgede kaldi (naive 3.09, NWP 2.76). Karar icin
eksik tek sayi: iki-bataryanin TABANI - mukemmel bilgiyle MPC nereye inebiliyor?

MANTIK:
  perfect sonucu     ~ fiziksel taban (kapasite siniri; bilgiyle inilemeyen yer)
  NWP - perfect      = bilgi boslugu (daha iyi bilgi/kararla kapatilabilir kisim)
  Buyuk fark yalnizca "ogrenme MUMKUN OLABILIR"i gosterir, "ogrenme kazanir"i
  DEGIL - o ancak DRL pilotuyla (Kapi 2) kanitlanir.

KUNYE (toplam-3.0 kosumuyla BIREBIR; tek fark mod=perfect):
  toplam solar 3.0 (site basina 1.5) | 2 x 6 MWh | dilim 13800:16229 | koordineli

ON-KAYITLI HUKUM (<author> onayladi, 17 Tem):
  NWP(2.76) - perfect < 1.0 puan  -> kapi KAPALI: iyi tahmin boslugu zaten
                                     kapatiyor; ogrenmeye kalan yer yok
  NWP - perfect >= 1.0 puan       -> kapi ACIK ADAY: tahminin kapatamadigi
                                     bosluk var; GG-2 adayi, DRL pilotu tartisilir
                                     (bu "ogrenme kazanir" DEMEK DEGILDIR)

CALISTIR:
  python gg2/gg2_mpc2_floor.py 2>&1 | tee gg2/mpc2_floor.log
Sure: ~50 dk (tek kosum).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 74)
print("IKI-BATARYA TABAN OLCUMU (perfect) - toplam solar 3.0, site basina 1.5")
print("=" * 74)
print("mpc2_test import ediliyor...\n")

import mpc2_test as T  # noqa: E402

# ---- ayni olcekleme, ayni kanitlar (toplam-3.0 kosumuyla birebir) -----------
OLD = T.SOLAR_PEAK
if abs(OLD - 1.25) > 1e-9:
    sys.exit(f"[HATA] mpc2_test.SOLAR_PEAK beklenen 1.25 degil: {OLD}. DUR.")
TARGET_SITE = 1.5
T.solar *= TARGET_SITE / OLD
T.SOLAR_PEAK = TARGET_SITE
mx = float(np.max(T.solar[:T.SPLIT]))
print(f"KANIT (1) gercek solar site-tepesi: {mx:.4f}  (hedef {TARGET_SITE})")
if abs(mx - TARGET_SITE) > 0.02:
    sys.exit("[HATA] olcekleme tutmadi, DUR.")
print("  -> tuttu. (perfect mod tahmin matrisi KULLANMAZ; fc olcegi konu disi)\n")

REF = {"naive": 3.09, "NWP": 2.76, "tek-batarya naive": 11.28}
print("KUNYE + REFERANSLAR (ayni kurulum, onceki log'lardan):")
for k, v in REF.items():
    print(f"  {k:22s}: %{v:.2f}")
print()

print("--- perfect-koordineli kosuluyor...")
t0 = time.time()
rate, cnt, sat = T.run("perfect", True)
print(f"    perfect: ihlal {rate:.2f}% | adet {cnt} | doygun {sat} "
      f"| sure {(time.time()-t0)/60:.0f} dk\n")

print("=" * 74)
print("SONUC (iki batarya, koordineli, toplam solar 3.0)")
print("=" * 74)
print(f"  {'kosul':>12s} {'ihlal%':>8s}")
print(f"  {'naive':>12s} {REF['naive']:8.2f}")
print(f"  {'NWP':>12s} {REF['NWP']:8.2f}")
print(f"  {'perfect':>12s} {rate:8.2f}   <- TABAN (bu kosum)")

gap = REF["NWP"] - rate
print(f"\n  bilgi boslugu = NWP - perfect = {REF['NWP']:.2f} - {rate:.2f} "
      f"= {gap:+.2f} puan")
print(f"  doygunluk: {cnt} ihlalin {sat}'i batarya-dolu aninda "
      f"({100*sat/max(cnt,1):.0f}%)")

print("\nON-KAYITLI HUKUM (<author> onayladi):")
if gap < 1.0:
    print(f"  bosluk {gap:.2f} < 1.0 puan -> KAPI KAPALI.")
    print("  Iyi tahmin (NWP) zaten tabana yakin; ogrenmenin/koordinasyonun")
    print("  kapatabilecegi anlamli bosluk YOK. Kalan ihlal fiziksel taban")
    print("  (kapasite siniri). Rapora: 'iki bataryada da mesele kapasite;")
    print("  ogrenme boslugu olcumle kapali' diye yazilir.")
else:
    print(f"  bosluk {gap:.2f} >= 1.0 puan -> KAPI ACIK ADAY.")
    print("  Tahminin kapatamadigi gercek bir bosluk var. Bu, ogrenmenin")
    print("  MUMKUN OLABILECEGINI gosterir - KAZANACAGINI DEGIL. Sonraki adim")
    print("  (ayri karar): kucuk DRL pilotu (Kapi 2) bu boslugun en az yarisini")
    print("  kapatiyor mu? Kapatmiyorsa yine kapanir..")
print()