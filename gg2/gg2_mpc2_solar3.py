#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_mpc2_solar3.py - IKI-BATARYA, GERCEKCI SOLAR (3.0), adil bilgi. SON GEDIK.

NEDEN: gru kosumu (solar 1.25) kapiyi kapatti (%0.41), AMA o solar seviyesi
tek-batarya deneylerinin yarisi bile degil -> kolaylastirilmis problem.
Hakem gedigi: "gercekci solarla da mi cozuyor?" Bu script iki kosumla kapatir:
  1) solar 3.0 + naive tahmin (tek-batarya fair-MPC ile AYNI kosullar, tek fark
     batarya sayisi -> dogrudan kiyas: 11.28% tek batarya vs ? iki batarya)
  2) solar 3.0 + NWP tahmin (en iyi bilgiyle de kontrol -> sizinti sifir)

YONTEM: mpc2_test.run()'i KOPYALAMADAN kullanamayiz cunku SOLAR_PEAK modul
sabitine gomulu. Cozum: mpc2_test import edilir, modul dizileri yerinde
yeniden olceklenir (solar 1.25 -> 3.0), solar_fc de ayni oranla olceklenir
(forecast_matrix solar'i 3.0 tepeye gore uretilmisti -> naive icin dogrudan
M2.solar_fc = f["solar"] kullanilabilir; NWP icin nwp npz'si zaten 3.0 olcekli).

KUNYE:
  script          : mpc2_test.run (import; solar dizileri 3.0'a olceklendi)
  voltaj penceresi: runpp, tahminli girdiyle
  solar tahmini   : kosum 1 = daily-naive | kosum 2 = NWP cap-norm
  SOLAR_PEAK      : 3.0 (tek-batarya deneyleriyle AYNI)
  dilim           : 13800:16229

ON-KAYITLI HUKUM (<author> onayladi):
  gru-koordineli(3.0) < %2  -> kapi KESIN KAPALI (gercekci solar + adil bilgi)
  %2 - %8                   -> ara bolge; NWP kosumu belirleyici
  > %8                      -> kapi ACIK; koordinasyon GG-2 adayi
  Kiyas cipasi: tek-batarya fair MPC ayni kosullarda %11.28 idi. Iki batarya
  bunun cok altina inerse "depolama kapasitesi problemi cozuyor" sonucu cikar.

CALISTIR (once mpc2_test.py'nin main-guard'li surumunun yerinde oldugundan emin ol):
  python gg2/gg2_mpc2_solar3.py 2>&1 | tee gg2/mpc2_solar3.log
Sure: ~2 x 50 dk.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 74)
print("IKI-BATARYA, SOLAR 3.0, ADIL BILGI - son gedik kapatma")
print("=" * 74)
print("mpc2_test import ediliyor...\n")

import mpc2_test as T  # noqa: E402

# ---- solar dizilerini 3.0 tepeye yeniden olcekle (modul icinde, yerinde) ----
OLD, NEW = T.SOLAR_PEAK, 3.0
if abs(OLD - 1.25) > 1e-9:
    sys.exit(f"[HATA] mpc2_test.SOLAR_PEAK beklenen 1.25 degil: {OLD} - dosya degismis, dur.")
T.solar *= NEW / OLD                # gercek seri: tepe 1.25 -> 3.0
T.SOLAR_PEAK = NEW
# naive tahmin: forecast_matrix zaten 3.0-tepe olcekli uretilmisti (tek-batarya
# deneyleri onu kullaniyor). mpc2_test icindeki solar_fc de ayni dosyadan geliyor
# -> OLCEK KONTROLU sart: tahmin ort / gercek ort ~ 1 olmali.
r = float(np.nanmean(T.solar_fc[T.SPLIT:T.N-1, 0]) / np.nanmean(T.solar[T.SPLIT:T.N-1]))
print(f"OLCEK KONTROLU (naive fc / gercek, test dilimi, 1-adim): {r:.3f}")
if not (0.7 <= r <= 1.3):
    sys.exit("[HATA] naive tahmin olcegi gercekle uyusmuyor - olcekleme varsayimi yanlis. DUR.")
print("  -> olcekler uyumlu.\n")

nwp_path = Path("gg2/data/nwp_solar_fc_capnorm.npz")
if not nwp_path.exists():
    sys.exit(f"[HATA] bulunamadi: {nwp_path}")
nwp_solar = np.load(nwp_path)["solar"]          # (N,K), 3.0-tepe olcekli
if nwp_solar.shape != T.solar_fc.shape:
    sys.exit(f"[HATA] NWP sekli {nwp_solar.shape} != naive {T.solar_fc.shape}")

KUNYE = {
    "script": "mpc2_test.run (import; solar 3.0'a olceklendi)",
    "voltaj penceresi": "runpp, tahminli girdiyle",
    "SOLAR_PEAK": T.SOLAR_PEAK,
    "dilim": f"{T.SPLIT}:{T.N-1}",
    "kiyas cipasi": "tek-batarya fair MPC ayni kosulda %11.28",
}
print("KONTROLCU KUNYESI:")
for k, v in KUNYE.items():
    print(f"  {k:18s}: {v}")
print()

results = {}
for tag, fc in (("naive", None), ("NWP", nwp_solar)):
    if fc is not None:
        T.solar_fc = fc                          # tek degisken: solar tahmini
    print(f"--- gru-koordineli, solar 3.0, tahmin={tag} kosuluyor...")
    t0 = time.time()
    rate, cnt, sat = T.run("gru", True)
    results[tag] = (rate, cnt, sat)
    print(f"    {tag}: ihlal {rate:.2f}% | adet {cnt} | doygun {sat} "
          f"| sure {(time.time()-t0)/60:.0f} dk\n")

print("=" * 74)
print("SONUC (iki batarya, koordineli, solar 3.0)")
print("=" * 74)
print(f"  {'tahmin':>8s} {'ihlal%':>8s} {'adet':>6s} {'doygun':>7s}")
for tag, (rate, cnt, sat) in results.items():
    print(f"  {tag:>8s} {rate:8.2f} {cnt:6d} {sat:7d}")
print(f"  {'tek-batarya naive (cipa)':>26s}   11.28   [kunye: ayni kosul, 1 batarya]")

rk = results["naive"][0]
print("\nON-KAYITLI HUKUM (<author> onayladi):")
if rk < 2.0:
    print(f"  naive-koordineli %{rk:.2f} < %2 -> KAPI KESIN KAPALI.")
    print("  Gercekci solar + adil (kusurlu) bilgiyle bile iki batarya problemi")
    print("  cozuyor. Sorun koordinasyon degil DEPOLAMA KAPASITESIYMIS:")
    print("  ayni kosulda tek batarya %11.28, iki batarya %{:.2f}.".format(rk))
    print("  DRL'in ogrenebilecegi koordinasyon boslugu yok. Rapora boyle yazilir.")
elif rk <= 8.0:
    print(f"  naive-koordineli %{rk:.2f}, %2-%8 arasi -> ARA BOLGE.")
    print(f"  NWP kosumu belirleyici: NWP %{results['NWP'][0]:.2f} < %2 ise iyi")
    print("  bilgiyle kapaniyor demektir (kapi 'iyi tahmin sartiyla kapali');")
    print("  NWP de araliktaysa hukum yok, ek olcum gerekir.")
else:
    print(f"  naive-koordineli %{rk:.2f} > %8 -> KAPI ACIK.")
    print("  Iki batarya + gercekci solar MPC'yi zorluyor; koordinasyon GG-2 adayi.")
print()