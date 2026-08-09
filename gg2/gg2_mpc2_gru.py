#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_mpc2_gru.py - IKI-BATARYA testi, ADIL BILGI (gru) modunda.

NEDEN: mpc2_test.py'nin devire gecen sonucu (%0.49) PERFECT-INFO satiriydi -
MPC'ye gercek gelecek verilmisti. GG-2'de ogrendik: bilgi kalitesi her seyi
degistiriyor (naive 11.28 vs oracle 9.14). "Multi-agent'a yer yok" hukmu bu
yuzden ASKIDA. Bu kosum adil cevabi verir.

NE YAPAR: mpc2_test.py'nin run() fonksiyonunu AYNEN kullanir (kod kopyalanmadi,
import edildi -> davranis birebir). Sadece gru modunu kosar:
  1) gru + koordineli (solve_window2, iki batarya birlikte)
  2) gru + bagimsiz  (solve_window x2, ayri ayri)

DIKKAT - BILINEN SINIRLAR (hukumde yazili olacak):
  - SOLAR_PEAK=1.25 (tek-batarya deneylerinde 3.0'di) -> bu KOLAYLASTIRILMIS
    problem. Cikan hukum bu solar seviyesine BAGLIDIR.
  - mpc2_test.py import'u modul seviyesinde veri/sebeke kurar; kosum baslatmaz
    (run() cagrilari if-main altinda degil ama print+for dongusu MODUL SEVIYESINDE!)
    -> KONTROL EDILDI: mpc2_test.py'de for dongusu modul seviyesinde DEGILSE
       import guvenli. DEGILSE bu sarmalayici calismaz; o durumda mpc2_test.py
       icindeki for dongusunu if __name__ blogu icine tasiyip yeniden dene.

ON-KAYITLI HUKUM (<author> onayladi):
  gru-koordineli ihlal < %2  -> adil bilgide bile MPC cozuyor; kapi KAPALI
                                (bu solar seviyesinde)
  %2 - %8                    -> ara bolge; hukum YOK, solar 3.0 ile tekrar gerekir
  > %8                       -> adil bilgi MPC'yi ciddi zorluyor; koordinasyon
                                kapisi YENIDEN ACIK, GG-2 adayi
  Ek: |koordineli - bagimsiz| < 0.5 puan ise koordinasyonun kendisi onemsiz.

CALISTIR:
  python gg2/gg2_mpc2_gru.py 2>&1 | tee gg2/mpc2_gru.log
Sure: 2 kosum x (2429 adim x ~49 runpp/adim) -> uzun, gece isi. Ara ilerleme
mpc2_test.run() icinden her 400 adimda basilir.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 74)
print("IKI-BATARYA ADIL-BILGI (gru) KOSUMU")
print("=" * 74)
print("mpc2_test import ediliyor (veri+sebeke kurulumu, ~10 sn)...")
print("UYARI: import sirasinda 'calisiyor...' satirlari akmaya baslarsa")
print("mpc2_test.py'nin for dongusu modul seviyesindedir -> Ctrl+C ile durdur,")
print("o donguyu if __name__=='__main__': altina tasi, tekrar dene.\n")

import mpc2_test as T   # noqa: E402  (kurulum + run() buradan)

print("\nimport tamam. Kosumlar basliyor.\n")

KUNYE = {
    "script": "mpc2_test.run (birebir import)",
    "voltaj penceresi": "runpp, tahminli girdiyle (gru modda)",
    "solar tahmini": "forecast_matrix (daily-naive) - ADIL",
    "SOLAR_PEAK": T.SOLAR_PEAK,
    "dilim": f"{T.SPLIT}:{T.N-1}",
}
print("KONTROLCU KUNYESI:")
for k, v in KUNYE.items():
    print(f"  {k:18s}: {v}")
print(f"\nON-KAYIT REFERANSLARI (devirden):")
print(f"  perfect-koordineli: ~%0.49 (kusurlu test, kiyas icin)")
print(f"  tek-batarya fair MPC (solar 3.0): %11.28 - AMA solar farkli, KIYASLANMAZ")
print()

results = {}
for coord in (True, False):
    name = f"gru-{'koordineli' if coord else 'bagimsiz'}"
    print(f"--- {name} kosuluyor...")
    t0 = time.time()
    rate, cnt, sat = T.run("gru", coord)
    dt_min = (time.time() - t0) / 60
    results[name] = (rate, cnt, sat)
    print(f"    {name}: ihlal {rate:.2f}% | adet {cnt} | doygun {sat} | sure {dt_min:.0f} dk\n")

print("=" * 74)
print("SONUC")
print("=" * 74)
print(f"  {'kosul':>18s} {'ihlal%':>8s} {'adet':>6s} {'doygun':>7s}")
for name, (rate, cnt, sat) in results.items():
    print(f"  {name:>18s} {rate:8.2f} {cnt:6d} {sat:7d}")
print(f"  {'perfect-koord (ref)':>18s} {'~0.49':>8s}")

rk = results["gru-koordineli"][0]
rb = results["gru-bagimsiz"][0]
print("\nON-KAYITLI HUKUM (kosumdan once <author> onayladi):")
if rk < 2.0:
    print(f"  gru-koordineli %{rk:.2f} < %2 -> adil bilgide bile MPC koordinasyonu")
    print("  cozuyor. Multi-agent kapisi KAPALI (bu solar seviyesinde, 1.25).")
elif rk <= 8.0:
    print(f"  gru-koordineli %{rk:.2f}, %2-%8 arasi -> ARA BOLGE. Hukum YOK.")
    print("  Guclu solarla (3.0) tekrar gerekir; kapi ne acik ne kapali.")
else:
    print(f"  gru-koordineli %{rk:.2f} > %8 -> adil bilgi MPC'yi ciddi zorluyor.")
    print("  Koordinasyon kapisi YENIDEN ACIK. GG-2 adayi olarak degerlendirilir.")

d = abs(rk - rb)
print(f"\n  koordineli-bagimsiz farki: {d:.2f} puan "
      f"-> koordinasyonun kendisi {'ONEMSIZ (<0.5)' if d < 0.5 else 'ETKILI (>=0.5)'}")
print("\nNOT: Bu sonuc SOLAR_PEAK=1.25 kosulundadir; tek-batarya deneyleriyle")
print("dogrudan kiyaslanamaz (kunye farki). Rapora bu sinirla yazilir.")
print()

