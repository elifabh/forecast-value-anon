#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_v1_ghi_provenance.py (v2) - DISKTEKI GHI SUTUNLARI NE?

gg2_nwp_fetch.py'den okunan gercekler:
  POINTS = cork(51.90,-8.47) wexford(52.34,-6.46) dublin(53.35,-6.26)
           midlands(53.42,-7.94) galway(53.27,-9.05) limerick(52.66,-8.63)
  VARS iki degisken iceriyor: shortwave_radiation VE
  shortwave_radiation_previous_day1 -> CSV'de IKI SET sutun olmali.

IKI SORU
  S1  Hangi CSV sutunu neye karsilik geliyor?
      (A) previous_day1 = GUN-ONCESI tahmin   [dogru olan]
      (C) current run   = AYNI GUN tahmini    [zayif; gun-oncesi degil]
      (B) archive/ERA5  = REANALIZ            [olumcul olan]
  S2  Tahminci HANGISINI kullaniyor?
      gg2_nwp_build_fc.py vb. kaynagindan OKUNUR, tahmin edilmez.

HUKUM (on-kayit, sonuca bakilmadan)
  kullanilan sutun -> (A)  : GECTI, manset nedenselligi saglam
  kullanilan sutun -> (B)  : OLUMCUL, NWP kolu gizli oracle, YAZIM DURUR
  kullanilan sutun -> (C)  : CIDDI, "day-ahead" ifadesi yanlis; manset kalir
                             ama iddia zayiflar ve acikca duzeltilir
  hicbiri                  : DUR, hukum verilmez

CALISTIR (kok klasorden, ~1 dk, internet gerekir):
  python gg2/gg2_v1_ghi_provenance.py 2>&1 | tee gg2/v1_ghi_provenance.log
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NDAYS = 21
TIGHT = 0.98

POINTS = {
    "cork": (51.90, -8.47), "wexford": (52.34, -6.46), "dublin": (53.35, -6.26),
    "midlands": (53.42, -7.94), "galway": (53.27, -9.05), "limerick": (52.66, -8.63),
}


def die(m):
    sys.exit("\n[DUR] " + m + "\n")


def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def pull(base, key, d0, d1):
    la = ",".join("%.4f" % v[0] for v in POINTS.values())
    lo = ",".join("%.4f" % v[1] for v in POINTS.values())
    j = get(base % (la, lo, d0, d1, key))
    js = j if isinstance(j, list) else [j]
    if len(js) != len(POINTS):
        die("API %d nokta dondurdu, %d bekleniyordu." % (len(js), len(POINTS)))
    out = {}
    for nm, k in zip(POINTS, js):
        h = k.get("hourly", {})
        if key not in h:
            die("cevapta '%s' yok (nokta %s)" % (key, nm))
        out[nm] = pd.Series(h[key], dtype="float64",
                            index=pd.to_datetime(h["time"], utc=True))
    return out


FC = ("https://previous-runs-api.open-meteo.com/v1/forecast"
      "?latitude=%s&longitude=%s&start_date=%s&end_date=%s&hourly=%s&timezone=UTC")
AR = ("https://archive-api.open-meteo.com/v1/archive"
      "?latitude=%s&longitude=%s&start_date=%s&end_date=%s&hourly=%s&timezone=UTC")


def main():
    print("=" * 78)
    print("V1 v2 - GHI SUTUN PROVENANSI")
    print("=" * 78)

    print("\n[S2] TAHMINCININ KULLANDIGI SUTUN (koddan okunuyor)")
    used = set()
    for fn in ("gg2/gg2_nwp_build_fc.py", "gg2/gg2_z15_build_models.py",
               "gg2/gg2_z19_noweather.py", "gg2/gg2_nwp_season.py",
               "gg2/gg2_verify.py"):
        p = ROOT / fn
        if not p.exists():
            print("  %-34s YOK" % fn)
            continue
        t = p.read_text(encoding="utf-8", errors="replace")
        hits = sorted(set(re.findall(r"ghi[_a-zA-Z0-9]*", t)))
        print("  %-34s %s" % (fn, hits if hits else "(ghi gecmiyor)"))
        used.update(h for h in hits if any(n in h.lower() for n in POINTS))
    print("  -> nokta-bazli kullanilan sutunlar: %s" % (sorted(used) or "BELIRSIZ"))

    cands = sorted((ROOT / "gg2" / "data").glob("nwp_ghi*.csv"))
    if not cands:
        die("gg2/data/nwp_ghi*.csv yok.")
    csv = cands[0]
    df = pd.read_csv(csv)
    tcol = next((c for c in df.columns if c.lower() in
                 ("time", "datetime", "datetime_utc", "date", "timestamp")),
                df.columns[0])
    df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df.dropna(subset=[tcol]).set_index(tcol).sort_index()
    num = [c for c in df.columns if df[c].dtype.kind in "fi"]
    print("\n[CSV] %s | %d satir | %s -> %s"
          % (csv.name, len(df), df.index[0], df.index[-1]))
    print("  sayisal sutunlar (%d): %s" % (len(num), num))

    mid = len(df) // 2
    w = df.iloc[mid: mid + NDAYS * 24]
    d0 = w.index[0].strftime("%Y-%m-%d")
    d1 = w.index[-1].strftime("%Y-%m-%d")
    print("  pencere: %s -> %s (%d satir)" % (d0, d1, len(w)))

    print("\n  (A) previous_day1 (GUN-ONCESI tahmin) cekiliyor ...")
    A = pull(FC, "shortwave_radiation_previous_day1", d0, d1)
    print("  (C) current run (AYNI GUN tahmini) cekiliyor ...")
    C = pull(FC, "shortwave_radiation", d0, d1)
    print("  (B) archive / ERA5 (REANALIZ) cekiliyor ...")
    B = pull(AR, "shortwave_radiation", d0, d1)

    print("\n" + "=" * 78)
    print("SUTUN BAZINDA KARSILASTIRMA   (corr / MAE W/m2)")
    print("=" * 78)
    print("  %-24s %-9s %17s %17s %17s"
          % ("CSV sutunu", "nokta", "A gun-oncesi", "C ayni gun", "B reanaliz"))
    verdict = {}
    for c in num:
        nm = next((n for n in POINTS if n in c.lower()), None)
        if nm is None:
            print("  %-24s %-9s (nokta eslesmedi, atlandi)" % (c, "-"))
            continue
        s = w[c].astype(float)
        row, sc = [], {}
        for lab, D in (("A", A), ("C", C), ("B", B)):
            ix = s.index.intersection(D[nm].index)
            if len(ix) < 100:
                row.append("       yetersiz")
                continue
            cc = float(np.corrcoef(s[ix], D[nm][ix])[0, 1])
            mm = float(np.abs(s[ix] - D[nm][ix]).mean())
            sc[lab] = cc
            row.append("%8.4f / %6.1f" % (cc, mm))
        while len(row) < 3:
            row.append("       yetersiz")
        print("  %-24s %-9s %17s %17s %17s" % (c, nm, row[0], row[1], row[2]))
        if sc:
            verdict[c] = (max(sc, key=sc.get), sc)

    print("\n" + "=" * 78)
    print("HUKUM")
    print("=" * 78)
    tag = {"A": "GUN-ONCESI TAHMIN", "C": "AYNI GUN TAHMINI", "B": "REANALIZ"}
    for c, (best, sc) in verdict.items():
        ok = "yapisiyor" if sc[best] >= TIGHT else "ZAYIF eslesme"
        print("  %-24s -> %-18s (corr %.4f, %s)" % (c, tag[best], sc[best], ok))

    if not verdict:
        die("Hicbir sutun karsilastirilamadi.")

    if not used:
        print("\n  [S2] Tahmincinin kullandigi sutun koddan cikarilamadi.")
        print("  Yukaridaki tabloya bak; hangi sutunun kullanildigini soyle.")
        print("  HUKUM ancak o zaman verilir.")
        return

    key = [c for c in verdict if c in used or any(u.lower() in c.lower() for u in used)]
    if not key:
        print("\n  [S2] kullanilan sutunlar %s CSV ile eslesmedi -> DUR." % sorted(used))
        return
    bs = [verdict[c][0] for c in key]
    print("\n  kullanilan sutunlar: %s" % key)
    if all(b == "A" for b in bs):
        print("\n  -> GECTI. Tahminci GUN-ONCESI sutunlari kullaniyor.")
        print("     Manset nedenselligi SAGLAM.")
    elif any(b == "B" for b in bs):
        print("\n  -> **OLUMCUL**. Kullanilan sutun REANALIZ ile ortusuyor.")
        print("     NWP kolu bir tahmin degil, gizli oracle'dir. YAZIM DURUR.")
    elif any(b == "C" for b in bs):
        print("\n  -> **CIDDI**. Kullanilan sutun AYNI GUN tahmini.")
        print("     'day-ahead' ifadesi YANLIS ve duzeltilir. Manset kalir,")
        print("     iddia zayiflar; makalede acikca yazilir.")
    else:
        print("\n  -> BELIRSIZ. Hukum verilmez.")
    print()


if __name__ == "__main__":
    main()