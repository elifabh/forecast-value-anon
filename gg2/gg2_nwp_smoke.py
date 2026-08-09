#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_nwp_smoke.py - ADIM 1: Open-Meteo "Previous Runs" duman testi.

NEDEN: Hicbir sey insa etmeden once en ucuz varsayimlari dogrula:
  (1) endpoint calisiyor mu
  (2) 'shortwave_radiation_previous_day1' degisken adi dogru mu
  (3) start_date/end_date kabul ediliyor mu  <- BILMIYORUZ, API soylesin
  (4) veri bos mu (null orani)
  (5) buyukluk mantikli mi (Haziran, Irlanda, ogle UTC ~400-700 W/m2)

NE YAPMAZ: dosya yazmaz, GG-1/GG-2 verisine dokunmaz. Sadece ekrana basar.

KRITIK AYRIM (bu testin varlik sebebi):
  archive-api  (/v1/archive)      = ERA5 reanalizi  = "ne oldu"   -> ORACLE, YASAK
  previous-runs (_previous_day1)  = 24 saat once tahmin = "ne bekleniyordu" -> DOGRU

Calistir (GG-1 kok klasorunden):  python gg2/gg2_nwp_smoke.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://previous-runs-api.open-meteo.com/v1/forecast"

# Cork civari (Irlanda'da sebeke-olcekli solar'in yogun oldugu bolgelerden biri).
# Duman testi icin tek nokta yeterli; tam cekimde birden fazla nokta kullanilacak.
LAT, LON = 51.90, -8.47

# <author>'in joint penceresi icinde, gunes aktif bir hafta (2025-05-20 -> 2026-05-31 araligi)
START, END = "2025-06-15", "2025-06-21"

VARS = "shortwave_radiation,shortwave_radiation_previous_day1"


def fetch(params: dict) -> dict:
    url = BASE + "?" + urllib.parse.urlencode(params)
    print(f"\nGET {url}\n")
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:500]
        print(f"[HTTP {e.code}] {body}")
        return {"_http_error": e.code, "_body": body}
    except Exception as e:
        print(f"[AG HATASI] {type(e).__name__}: {e}")
        return {"_error": str(e)}


def summarize(js: dict, label: str) -> bool:
    """Tek bir cevabi ozetle. True donerse bu yol kullanilabilir."""
    if "_http_error" in js or "_error" in js:
        print(f"  {label}: BASARISIZ")
        return False
    if "hourly" not in js:
        print(f"  {label}: 'hourly' yok -> cevap: {str(js)[:300]}")
        return False

    h = js["hourly"]
    times = h.get("time", [])
    print(f"  {label}: OK | {len(times)} saatlik satir | {times[0]} -> {times[-1]}"
          if times else f"  {label}: satir yok")

    ok = True
    for v in VARS.split(","):
        if v not in h:
            print(f"    {v:42s} -> KOLON YOK (degisken adi yanlis olabilir)")
            ok = False
            continue
        vals = h[v]
        nn = [x for x in vals if x is not None]
        nullpct = 100.0 * (len(vals) - len(nn)) / max(len(vals), 1)
        if not nn:
            print(f"    {v:42s} -> TAMAMEN NULL ({nullpct:.0f}% null)")
            ok = False
            continue
        print(f"    {v:42s} -> null %{nullpct:5.1f} | min {min(nn):6.1f} "
              f"| max {max(nn):6.1f} | ort {sum(nn)/len(nn):6.1f} W/m2")

    # Ogle saati capasi: her iki degisken de gerceklikte ~400-700 W/m2 tepe yapmali
    if ok and times:
        d1 = h.get("shortwave_radiation_previous_day1", [])
        noon = [(t, v) for t, v in zip(times, d1) if t.endswith("T12:00") and v is not None]
        if noon:
            print(f"    ogle (12:00 UTC) day1-tahmin degerleri: "
                  + ", ".join(f"{v:.0f}" for _, v in noon))
    return ok


def main() -> None:
    print("=" * 72)
    print("ADIM 1 - Open-Meteo Previous Runs duman testi")
    print(f"nokta: {LAT}, {LON}  | pencere: {START} -> {END}  | tz: UTC")
    print("=" * 72)

    # --- Yol A: start_date/end_date destekleniyor mu? (bilmiyoruz)
    print("\n--- YOL A: start_date / end_date ile")
    a = fetch({
        "latitude": LAT, "longitude": LON,
        "hourly": VARS,
        "start_date": START, "end_date": END,
        "timezone": "UTC",
    })
    a_ok = summarize(a, "YOL A")

    # --- Yol B: past_days ile (dokumandaki ornek bu yolu kullaniyor)
    print("\n--- YOL B: past_days=7 ile (gecmise ne kadar gidebiliyor?)")
    b = fetch({
        "latitude": LAT, "longitude": LON,
        "hourly": VARS,
        "past_days": 7, "forecast_days": 1,
        "timezone": "UTC",
    })
    b_ok = summarize(b, "YOL B")

    # --- Hukum
    print("\n" + "=" * 72)
    print("HUKUM")
    print("=" * 72)
    if a_ok:
        print("  YOL A calisiyor -> tarih araligiyla toplu cekim yapilabilir. Adim 2'ye gec.")
    elif b_ok:
        print("  YOL A calismiyor, YOL B calisiyor -> past_days ile sinirliyiz.")
        print("  past_days tavani ~92 gun ise ~370 gunluk pencere CEKILEMEZ.")
        print("  Bu durumda alternatif: Single Runs API (ECMWF IFS HRES, Mart 2024'ten).")
    else:
        print("  IKISI DE BASARISIZ -> yukaridaki hata mesajini yapistir, birlikte bakalim.")
        print("  (Degisken adi, model destegi veya tarih araligi sorunu olabilir.)")
    print()


if __name__ == "__main__":
    main()