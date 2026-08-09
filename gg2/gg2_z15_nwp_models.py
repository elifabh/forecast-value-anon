#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_z15_nwp_models.py - Z15 ADIM 1: farkli hava modelleriyle NWP GHI cekimi.

NEDEN
  gg2_nwp_fetch.py 'models' parametresi GECIRMIYOR -> Open-Meteo varsayilani
  'best_match' kullanildi. best_match tek bir hava modeli DEGIL, konuma gore
  otomatik model secimidir ve zamanla degisebilir (tekrarlanabilirlik zayifligi).
  Makalede "tek deterministik NWP modeli" yazmak YANLIS olurdu.

  Bu script ISIMLI modellerle ayni veriyi ceker, boylece
  "1.48 puanlik etki hangi hava modelini sectigimize bagli mi?" sorusu olculur.

ON-KAYIT (koda girmeden yazildi, sonuca bakilmadan)
  Olculecek : ana boru hattinin NWP kolundaki ihlal orani, UC hava modeli
              girdisiyle: best_match (mevcut, %9.80), ECMWF, GFS.
  KRITIK    : UC MODELIN HEPSI RAPORLANIR. En iyisi SECILMEZ.
              Secim yapmak test blogunda karar almak olur = sizinti.
              Bir model naive'i yenemezse o da aynen raporlanir.
  Esik      : uc modelin ihlal yayilimi (max-min) vs 0.72 puan
              (naive-NWP farkinin bootstrap SE'si).
  HUKUM (1) : yayilim < 0.72 -> "Etki hava modeli secimine duyarsiz;
              uc bagimsiz NWP merkezinde de ayni buyuklukte."
  HUKUM (2) : yayilim >= 0.72 -> "Etkinin buyuklugu hava modeline bagli.
              Bu bir SINIRDIR ve makalede aralik olarak yazilir."
  Ara durum yok. Sebep uydurulmaz.

BU SCRIPT NE YAPAR
  1) DUMAN TESTI: her model icin 3 gunluk tek nokta cekimi. Model adi yanlissa
     veya arsivde yoksa saniyeler icinde anlasilir, tam cekim baslamaz.
  2) TAM CEKIM: 6 nokta x tum pencere, model basina ayri CSV.
  3) CIPA: null orani, yaz tepesi 500-1100 W/m2, gece ~0.

  gg2/data/nwp_ghi_roi.csv (best_match) UZERINE YAZILMAZ. Dokunulmaz.

CALISTIR (GG-1 kok klasorunden, ~15-25 dk):
  python gg2/gg2_z15_nwp_models.py 2>&1 | tee gg2/z15_nwp_models.log
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

# gg2_nwp_fetch.py ile BIREBIR ayni noktalar. Degistirilmedi.
POINTS = {
    "cork":     (51.90, -8.47),
    "wexford":  (52.34, -6.46),
    "dublin":   (53.35, -6.26),
    "midlands": (53.42, -7.94),
    "galway":   (53.27, -9.05),
    "limerick": (52.66, -8.63),
}

VARS = ["shortwave_radiation", "shortwave_radiation_previous_day1"]
CHUNK_DAYS = 92
PAD_DAYS = 3
SLEEP_S = 1.0

# Iki BAGIMSIZ merkez. best_match ile birlikte uc girdi olur.
DEFAULT_MODELS = "ecmwf_ifs025,gfs_seamless"


def _die(msg: str) -> None:
    sys.exit("[HATA] " + msg)


def call_api(lat: float, lon: float, d0: str, d1: str, model: str):
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(VARS),
        "start_date": d0, "end_date": d1,
        "timezone": "UTC",
    }
    if model:
        params["models"] = model
    url = BASE + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_chunk(lat, lon, d0, d1, model) -> pd.DataFrame:
    try:
        js = call_api(lat, lon, d0, d1, model)
    except urllib.error.HTTPError as e:
        _die("HTTP %d [%s] (%s->%s): %s"
             % (e.code, model, d0, d1, e.read().decode("utf-8", "replace")[:300]))
    except Exception as e:
        _die("ag hatasi [%s] (%s->%s): %s: %s" % (model, d0, d1, type(e).__name__, e))

    if "hourly" not in js:
        _die("cevapta 'hourly' yok [%s]: %s" % (model, str(js)[:300]))
    h = js["hourly"]
    for v in VARS:
        if v not in h:
            _die("kolon yok: %s [%s]" % (v, model))
    df = pd.DataFrame({"datetime_utc": pd.to_datetime(h["time"], utc=True)})
    df["ghi_p0"] = h[VARS[0]]
    df["ghi_p1"] = h[VARS[1]]
    return df


def smoke(model: str, d1) -> bool:
    """3 gunluk tek nokta cekimi. Model adi / arsiv kapsami kontrolu."""
    lat, lon = POINTS["dublin"]
    a = (pd.Timestamp(d1) - timedelta(days=5)).strftime("%Y-%m-%d")
    b = (pd.Timestamp(d1) - timedelta(days=3)).strftime("%Y-%m-%d")
    print("  [duman] %-18s %s -> %s ..." % (model, a, b), end=" ", flush=True)
    try:
        js = call_api(lat, lon, a, b, model)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:200]
        print("HTTP %d  -> %s" % (e.code, body))
        return False
    except Exception as e:
        print("%s: %s" % (type(e).__name__, e))
        return False
    if "hourly" not in js or VARS[1] not in js.get("hourly", {}):
        print("day1 kolonu yok")
        return False
    s = pd.Series(js["hourly"][VARS[1]], dtype="float64")
    nn = int(s.notna().sum())
    if nn == 0:
        print("TUM DEGERLER BOS -> bu model bu tarihte arsivde yok")
        return False
    print("OK (%d dolu / %d, max %.0f W/m2)" % (nn, len(s), s.max()))
    return True


def fetch_point(name, lat, lon, d0, d1, model) -> pd.DataFrame:
    parts, cur = [], d0
    while cur <= d1:
        end = min(cur + timedelta(days=CHUNK_DAYS - 1), d1)
        s0, s1 = cur.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        parts.append(fetch_chunk(lat, lon, s0, s1, model))
        print("      %s -> %s  (%d satir)" % (s0, s1, len(parts[-1])))
        cur = end + timedelta(days=1)
        time.sleep(SLEEP_S)
    df = pd.concat(parts, ignore_index=True).drop_duplicates("datetime_utc")
    return df.rename(columns={"ghi_p0": "ghi_p0_" + name, "ghi_p1": "ghi_p1_" + name})


def verify(merged: pd.DataFrame, model: str) -> bool:
    print("\n  DOGRULAMA [%s]" % model)
    print("    satir %d | %s -> %s"
          % (len(merged), merged.datetime_utc.iloc[0], merged.datetime_utc.iloc[-1]))
    bad = False
    for name in POINTS:
        c = "ghi_p1_" + name
        s = merged[c]
        nullpct = 100.0 * s.isna().mean()
        mx = s.max() if pd.notna(s.max()) else 0.0
        flag = ""
        if nullpct > 5.0:
            flag += " <- NULL YUKSEK"
            bad = True
        if not (500 <= mx <= 1100):
            flag += " <- TEPE ARALIK DISI"
            bad = True
        print("    %-22s null %%%5.1f | max %6.1f | ort %6.1f%s"
              % (c, nullpct, mx, s.mean(), flag))
    night = merged[merged.datetime_utc.dt.hour == 0]
    nmax = night[["ghi_p1_" + n for n in POINTS]].max().max()
    print("    gece (00:00 UTC) max: %.1f W/m2  (0'a yakin olmali)" % nmax)
    if nmax > 20:
        print("      ^ UYARI: zaman dilimi suphesi")
        bad = True
    return not bad


def main() -> None:
    ap = argparse.ArgumentParser(description="Z15 Adim 1: isimli hava modelleriyle NWP cekimi.")
    ap.add_argument("--joint", default="data/aligned_30min_joint.csv")
    ap.add_argument("--models", default=DEFAULT_MODELS,
                    help="virgulle ayrilmis Open-Meteo model adlari")
    ap.add_argument("--outdir", default="gg2/data")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke-only", action="store_true",
                    help="sadece duman testi yap, tam cekim yapma")
    a = ap.parse_args()

    jp = Path(a.joint)
    if not jp.exists():
        _die("bulunamadi: %s  (kok klasorden mi calistiriyorsun?)" % jp)
    j = pd.read_csv(jp, index_col=0, parse_dates=True)
    t0, t1 = j.index[0], j.index[-1]
    d0 = (t0 - timedelta(days=PAD_DAYS)).date()
    d1 = (t1 + timedelta(days=PAD_DAYS)).date()

    models = [m.strip() for m in a.models.split(",") if m.strip()]

    print("=" * 74)
    print("Z15 ADIM 1 - ISIMLI HAVA MODELLERIYLE NWP GHI CEKIMI")
    print("=" * 74)
    print("  joint pencere : %s -> %s (%d adim)" % (t0, t1, len(j)))
    print("  cekim araligi : %s -> %s (+/-%d gun pay)" % (d0, d1, PAD_DAYS))
    print("  modeller      : %s" % ", ".join(models))
    print("  nokta sayisi  : %d" % len(POINTS))
    print("  NOT: gg2/data/nwp_ghi_roi.csv (best_match) DOKUNULMAZ.")

    # -------------------------------------------------------- duman testi
    print("\n" + "=" * 74)
    print("DUMAN TESTI (model adi + arsiv kapsami)")
    print("=" * 74)
    ok_models = [m for m in models if smoke(m, d1)]
    if not ok_models:
        print("\n  HICBIR MODEL GECMEDI. Tam cekim yapilmadi.")
        print("  Model adlari icin: https://open-meteo.com/en/docs")
        sys.exit(2)
    if len(ok_models) < len(models):
        print("\n  UYARI: gecemeyen modeller atlandi: %s"
              % ", ".join(m for m in models if m not in ok_models))
    if a.smoke_only:
        print("\n  --smoke-only: burada duruldu.")
        return

    # ---------------------------------------------------------- tam cekim
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written, failed = [], []

    for model in ok_models:
        out = outdir / ("nwp_ghi_roi_%s.csv" % model)
        if out.exists() and not a.force:
            print("\n  [%s] cikti zaten var, atlandi: %s  (--force ile ez)" % (model, out))
            continue
        print("\n" + "=" * 74)
        print("TAM CEKIM - %s" % model)
        print("=" * 74)
        merged = None
        for name, (lat, lon) in POINTS.items():
            print("    [%s] %.2f, %.2f" % (name, lat, lon))
            dfp = fetch_point(name, lat, lon, d0, d1, model)
            merged = dfp if merged is None else merged.merge(dfp, on="datetime_utc", how="outer")
        merged = merged.sort_values("datetime_utc").reset_index(drop=True)
        good = verify(merged, model)
        merged.to_csv(out, index=False)
        print("\n    yazildi: %s  %s" % (out, "(capalar tuttu)" if good else "(UYARI VAR)"))
        (written if good else failed).append(model)

    print("\n" + "=" * 74)
    print("OZET")
    print("=" * 74)
    print("  capalari tutan : %s" % (", ".join(written) if written else "-"))
    print("  uyarili        : %s" % (", ".join(failed) if failed else "-"))
    print("\n  Ciktiyi Claude'a yapistir. Sonraki adim (tahmin matrisi + MPC)")
    print("  bu cikti gorulmeden BASLATILMAZ.\n")


if __name__ == "__main__":
    main()