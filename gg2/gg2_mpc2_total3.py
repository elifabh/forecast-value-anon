#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_mpc2_total3.py - IKI-BATARYA, TOPLAM SOLAR 3.0 (site basina 1.5). DUZELTILMIS.

ONCEKI IKI KOSUM NEDEN GECERSIZDI (kayit icin):
  - mpc2_test.py solari IKI barada enjekte eder: net.sgen.p_mw=[s,s,w]
    (bus17'ye s, bus14'e s). Yani toplam solar = 2*s.
  - Dunku 0.41 kosumu: s tepe 1.25 -> toplam 2.5; AMA tahminler 3.0-tepe
    olcekliydi (forecast_matrix) -> MPC'ye ~2.4x sisik tahmin -> asiri temkin.
  - Bugunku 22.15 kosumu: s tepe 3.0 -> TOPLAM 6.0 = tek-batarya senaryosunun
    iki kati -> farkli sebeke, kiyas gecersiz.
  Ikisi de kunye kuralini ihlal etti. Bu script ikisini de duzeltir.

BU KOSUMUN KUNYESI (tek-batarya fair MPC ile ayni TOPLAM kosullar):
  toplam solar tepe : 3.0 MW  (site basina s=1.5; [s,s,w] -> 1.5+1.5)
  tahmin olcegi     : site basina 1.5-tepe (naive matrisi 0.5x; NWP matrisi 0.5x
                      -- her iki matris de 3.0-tepe uretilmisti, kanitli)
  batarya           : 2 x (6 MWh / 3 MW) = tek-batarya senaryosunun 2 kati depolama
  voltaj penceresi  : runpp, tahminli girdiyle (gru) | dilim 13800:16229
  kiyas cipasi      : tek-batarya naive %11.28 (toplam solar ayni, depolama yarisi)

OLCEK KANITLARI (kosum oncesi, otomatik; tutmazsa DURUR):
  (1) gercek: max(T.solar[:SPLIT]) == 1.5 (+/- kayan nokta)
  (2) naive : fc(1-adim) ort / gercek ort in [0.7, 1.3]
  (3) NWP   : ayni oran kontrolu

ON-KAYITLI HUKUM (<author> onayladi, 17 Tem):
  naive-koordineli < %2   -> kapi KESIN KAPALI: ayni toplam solarda 2. batarya
                             problemi cozuyor -> mesele KAPASITE, koordinasyon degil
  %2 - %8                 -> ara bolge; NWP kosumu belirleyici
  > %8                    -> kapi ACIK (gecerli kunyeyle)

CALISTIR:
  python gg2/gg2_mpc2_total3.py 2>&1 | tee gg2/mpc2_total3.log
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
print("IKI-BATARYA, TOPLAM SOLAR 3.0 (site basina 1.5) - DUZELTILMIS KOSUM")
print("=" * 74)
print("mpc2_test import ediliyor...\n")

import mpc2_test as T  # noqa: E402

# ---- 1) gercek solar: site basina tepe 1.5 (toplam [s,s] = 3.0) ------------
OLD = T.SOLAR_PEAK
if abs(OLD - 1.25) > 1e-9:
    sys.exit(f"[HATA] mpc2_test.SOLAR_PEAK beklenen 1.25 degil: {OLD}. Dosya degismis, DUR.")
TARGET_SITE = 1.5
T.solar *= TARGET_SITE / OLD
T.SOLAR_PEAK = TARGET_SITE
mx = float(np.max(T.solar[:T.SPLIT]))
print(f"KANIT (1) gercek solar site-tepesi: {mx:.4f}  (hedef {TARGET_SITE})")
if abs(mx - TARGET_SITE) > 0.02:
    sys.exit("[HATA] olcekleme tutmadi, DUR.")

# ---- 2) tahmin matrisleri: 3.0-tepe -> 1.5-tepe (0.5x) ---------------------
naive_15 = T.solar_fc * (TARGET_SITE / 3.0)
nwp_path = Path("gg2/data/nwp_solar_fc_capnorm.npz")
if not nwp_path.exists():
    sys.exit(f"[HATA] bulunamadi: {nwp_path}")
nwp_15 = np.load(nwp_path)["solar"] * (TARGET_SITE / 3.0)
if nwp_15.shape != naive_15.shape:
    sys.exit(f"[HATA] sekil uyusmazligi: NWP {nwp_15.shape} vs naive {naive_15.shape}")

sl_t = slice(T.SPLIT, T.N - 1)
for tag, m in (("naive", naive_15), ("NWP", nwp_15)):
    r = float(np.nanmean(m[sl_t, 0]) / np.nanmean(T.solar[sl_t]))
    print(f"KANIT ({'2' if tag=='naive' else '3'}) {tag:5s} fc/gercek orani: {r:.3f}")
    if not (0.7 <= r <= 1.3):
        sys.exit(f"[HATA] {tag} olcegi gercekle uyusmuyor ({r:.3f}), DUR.")
print("  -> uc kanit da tuttu.\n")

print("KONTROLCU KUNYESI:")
print(f"  toplam solar tepe : 3.0 MW (site basina {TARGET_SITE})")
print(f"  depolama          : 2 x 6 MWh (tek-batarya senaryosunun 2 kati)")
print(f"  dilim             : {T.SPLIT}:{T.N-1} | mod: gru (adil) | koordineli")
print(f"  kiyas cipasi      : tek-batarya naive %11.28 (ayni toplam solar)\n")

results = {}
for tag, m in (("naive", naive_15), ("NWP", nwp_15)):
    T.solar_fc = m                                  # tek degisken: tahmin
    print(f"--- gru-koordineli, tahmin={tag} kosuluyor...")
    t0 = time.time()
    rate, cnt, sat = T.run("gru", True)
    results[tag] = (rate, cnt, sat)
    print(f"    {tag}: ihlal {rate:.2f}% | adet {cnt} | doygun {sat} "
          f"| sure {(time.time()-t0)/60:.0f} dk\n")

print("=" * 74)
print("SONUC (iki batarya, koordineli, TOPLAM solar 3.0)")
print("=" * 74)
print(f"  {'tahmin':>8s} {'ihlal%':>8s} {'adet':>6s} {'doygun':>7s}")
for tag, (rate, cnt, sat) in results.items():
    print(f"  {tag:>8s} {rate:8.2f} {cnt:6d} {sat:7d}")
print(f"  {'tek-batarya naive (cipa)':>26s}   11.28")

rk = results["naive"][0]
print("\nON-KAYITLI HUKUM (<author> onayladi):")
if rk < 2.0:
    print(f"  naive-koordineli %{rk:.2f} < %2 -> KAPI KESIN KAPALI.")
    print("  Ayni toplam solarda ikinci batarya problemi cozuyor:")
    print(f"  tek batarya %11.28 -> iki batarya %{rk:.2f}.")
    print("  Mesele KOORDINASYON degil DEPOLAMA KAPASITESI. DRL'in ogrenecegi")
    print("  koordinasyon boslugu yok. Rapora bu cumleyle yazilir.")
elif rk <= 8.0:
    print(f"  naive-koordineli %{rk:.2f}, %2-%8 arasi -> ARA BOLGE.")
    print(f"  NWP belirleyici: NWP %{results['NWP'][0]:.2f} < %2 ise")
    print("  'iyi tahmin sartiyla kapali'; degilse hukum yok, ek olcum gerekir.")
else:
    print(f"  naive-koordineli %{rk:.2f} > %8 -> KAPI ACIK (gecerli kunyeyle).")
    print("  Ayni toplam solar + iki kat depolama + adil bilgi MPC'yi hala")
    print("  zorluyor -> koordinasyon/yerlestirme sorusu GG-2 adayi olur.")
print("\nNOT: dunku 0.41 (solar 1.25 + sisik tahmin) ve bugunku 22.15 (toplam 6.0)")
print("kosumlari GECERSIZ sayilir; kayitlara 'kunye hatasi' olarak gecer.")
print()