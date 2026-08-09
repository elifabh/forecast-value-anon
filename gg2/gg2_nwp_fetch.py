#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_nwp_fetch.py - ADIM 2: Open-Meteo Previous Runs'tan tam pencere radyasyon cekimi.

NE CEKIYOR (her nokta icin):
  ghi_p1_<nokta> : shortwave_radiation_previous_day1
                   = gecerlilik aninda 24 SAAT ONCE tahmin edilmis GHI  <- TESTIN OZNESI
  ghi_p0_<nokta> : shortwave_radiation
                   = en taze model kosumu (~analiz kalitesi)            <- SADECE TANI

KRITIK: /v1/archive (ERA5 reanalizi = "ne oldu") KULLANILMIYOR. O oracle olurdu.

NEDEN day0 da cekiliyor: sonra "kalan hata NWP'nin gecikmesinden mi, bizim
radyasyon->MW donusumumuzden mi" ayrilabilsin diye. Tahminci SADECE day1 kullanir.

TARIHLER KODA GOMULU DEGIL: data/aligned_30min_joint.csv'den okunur.

CIKTI: gg2/data/nwp_ghi_roi.csv  (saatlik, UTC)
       Mevcut dosya varsa EZILMEZ (--force gerekir).

Calistir (GG-1 kok klasorunden):  python gg2/gg2_nwp_fetch.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import timedelta
from pathlib import Path

import pandas as pd

BASE = "https://previous-runs-api.open-meteo.com/v1/forecast"

# ROI uzerine yayilmis 6 nokta. Sebeke-olcekli solar'in TAM yerini bildigimizi
# iddia ETMIYORUZ; bunlar uzamsal ornekleme. Agirliklari model train'de ogrenir.
POINTS = {
    "cork":     (51.90, -8.47),
    "wexford":  (52.34, -6.46),
    "dublin":   (53.35, -6.26),
    "midlands": (53.42, -7.94),
    "galway":   (53.27, -9.05),
    "limerick": (52.66, -8.63),
}

VARS = ["shortwave_radiation", "shortwave_radiation_previous_day1"]
CHUNK_DAYS = 92          # uzun araliklarda API'yi zorlamamak icin
PAD_DAYS = 3             # 30-dk hizalama/interpolasyon icin kenar payi
SLEEP_S = 1.0            # nazik olalim (ucretsiz servis)


def _die(msg: str) -> None:
    sys.exit(f"[HATA] {msg}")


def fetch_chunk(lat: float, lon: float, d0: str, d1: str) -> pd.DataFrame:
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(VARS),
        "start_date": d0, "end_date": d1,
        "timezone": "UTC",
    }
    url = BASE + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            js = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        _die(f"HTTP {e.code} ({d0}->{d1}): {e.read().decode('utf-8', 'replace')[:300]}")
    except Exception as e:
        _die(f"ag hatasi ({d0}->{d1}): {type(e).__name__}: {e}")

    if "hourly" not in js:
        _die(f"cevapta 'hourly' yok ({d0}->{d1}): {str(js)[:300]}")
    h = js["hourly"]
    for v in VARS:
        if v not in h:
            _die(f"kolon yok: {v} ({d0}->{d1})")

    df = pd.DataFrame({"datetime_utc": pd.to_datetime(h["time"], utc=True)})
    df["ghi_p0"] = h[VARS[0]]
    df["ghi_p1"] = h[VARS[1]]
    return df


def fetch_point(name: str, lat: float, lon: float, d0, d1) -> pd.DataFrame:
    parts, cur = [], d0
    while cur <= d1:
        end = min(cur + timedelta(days=CHUNK_DAYS - 1), d1)
        s0, s1 = cur.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        parts.append(fetch_chunk(lat, lon, s0, s1))
        print(f"    {s0} -> {s1}  ({len(parts[-1])} satir)")
        cur = end + timedelta(days=1)
        time.sleep(SLEEP_S)
    df = pd.concat(parts, ignore_index=True).drop_duplicates("datetime_utc")
    return df.rename(columns={"ghi_p0": f"ghi_p0_{name}", "ghi_p1": f"ghi_p1_{name}"})


def main() -> None:
    ap = argparse.ArgumentParser(description="GG-2 Adim 2: NWP GHI cekimi.")
    ap.add_argument("--joint", default="data/aligned_30min_joint.csv",
                    help="tarih araligi buradan okunur (SADECE OKUNUR)")
    ap.add_argument("--out", default="gg2/data/nwp_ghi_roi.csv")
    ap.add_argument("--force", action="store_true", help="mevcut ciktiyi ez")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        _die(f"cikti zaten var, ezilmedi: {out}  (--force ile ez)")

    jp = Path(args.joint)
    if not jp.exists():
        _die(f"bulunamadi: {jp}")
    j = pd.read_csv(jp, index_col=0, parse_dates=True)
    t0, t1 = j.index[0], j.index[-1]
    print("=" * 72)
    print("ADIM 2 - NWP GHI cekimi (Previous Runs, day1 = 24h once tahmin)")
    print(f"joint pencere : {t0} -> {t1}  ({len(j)} adim)")
    d0 = (t0 - timedelta(days=PAD_DAYS)).date()
    d1 = (t1 + timedelta(days=PAD_DAYS)).date()
    print(f"cekim araligi : {d0} -> {d1}  (+/-{PAD_DAYS} gun pay)")
    print(f"nokta sayisi  : {len(POINTS)}")
    print("=" * 72)

    merged = None
    for name, (lat, lon) in POINTS.items():
        print(f"\n  [{name}] {lat}, {lon}")
        dfp = fetch_point(name, lat, lon, d0, d1)
        merged = dfp if merged is None else merged.merge(dfp, on="datetime_utc", how="outer")

    merged = merged.sort_values("datetime_utc").reset_index(drop=True)

    # --- dogrulama capalari (CSV'ye bakmadan once burada gorulsun)
    print("\n" + "=" * 72)
    print("DOGRULAMA")
    print("=" * 72)
    print(f"  satir: {len(merged)} | {merged.datetime_utc.iloc[0]} -> {merged.datetime_utc.iloc[-1]}")
    beklenen = int((pd.Timestamp(d1) - pd.Timestamp(d0)).total_seconds() / 3600) + 24
    print(f"  beklenen ~{beklenen} saatlik satir")
    bad = False
    for name in POINTS:
        for tag in ("p1", "p0"):
            c = f"ghi_{tag}_{name}"
            s = merged[c]
            nullpct = 100.0 * s.isna().mean()
            print(f"    {c:22s} null %{nullpct:5.1f} | max {s.max():6.1f} | ort {s.mean():6.1f}")
            if nullpct > 5.0:
                print(f"      ^ UYARI: null orani yuksek")
                bad = True
            if not (500 <= (s.max() if pd.notna(s.max()) else 0) <= 1100):
                print(f"      ^ UYARI: yaz tepesi 500-1100 W/m2 araliginda degil")
                bad = True

    # gece kontrolu: 00:00 UTC'de radyasyon ~0 olmali
    night = merged[merged.datetime_utc.dt.hour == 0]
    nmax = night[[f"ghi_p1_{n}" for n in POINTS]].max().max()
    print(f"  gece (00:00 UTC) day1 max: {nmax:.1f} W/m2  (0'a yakin olmali)")
    if nmax > 20:
        print("    ^ UYARI: gece radyasyonu sifir degil -> zaman dilimi suphesi")
        bad = True

    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"\n  yazildi: {out}")
    print("\n  " + ("BAZI UYARILAR VAR - yukariyi oku." if bad else "Capalar tuttu. Adim 3'e gec."))


if __name__ == "__main__":
    main()