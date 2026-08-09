#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_mpc2_wcost.py - IZOLASYON: ekonomik agirlik, tahmin etkisini aciklar mi?

SORUN
  Ana boru hattinda tahmin etkisi COZULUYOR : 1.54 +/- 0.72 puan (2.1 SE)
  Iki-batarya boru hattinda COZULMUYOR       : 0.70 +/- 0.45 (1 bat), 0.33 +/- 0.45 (2 bat)
  Iki kurulum UC yerde farkli:
    (a) ekonomik agirlik : w_econ = 1.0   vs  W_COST = 1e-3
    (b) ufuk voltaj modeli: LinDistFlow farki  vs  her adim tam runpp
    (c) solar yerlesimi   : tek bara 3.0  vs  iki bara 1.5 + 1.5
  Hangisinin sorumlu oldugu OLCULMEDI.

BU SCRIPT (a)'YI IZOLE EDER -- tek dugme.
  gg2_mpc2_evidence.py'nin naive1 / nwp1 kollarini AYNEN tekrar kosar; tek fark:
  opf_expert.W_COST  1e-3 -> 1.0
  (b) ve (c) SABIT kalir. Yani cikan fark yalnizca ekonomik agirliktandir.

  Neden 1-batarya kollari: opf_expert2 (iki-batarya cozucu) w_econ argumani
  almiyor, W_COST'u sabit. Tek-batarya kollarinda opf_expert kullanildigi icin
  patch temiz calisir ve kiyas 1-batarya cifti uzerinden kurulur.

ON-KAYIT (kod yazilmadan once, duz Turkce)
  Olculecek : naive1 - nwp1 ihlal farki, w_econ = 1.0 altinda.
  Referans  : ayni fark w_econ = 1e-3 altinda = 0.70 +/- 0.45 puan (1.6 SE)
  Esik      : 2 SE (her yerdeki kural)

  HUKUM:
   (1) fark >= 2 SE olursa  -> "ekonomik agirlik farki, iki boru hatti arasindaki
       tahmin-etkisi farkini ACIKLIYOR." Kapsam notu olculmus cevapla degisir.
   (2) fark < 2 SE kalirsa  -> "ekonomik agirlik TEK BASINA aciklamiyor; kalan
       aday ufuk modeli ve/veya solar yerlesimi. Bu calismada ayrilmadi."
       Bu da gecerli bir sonuctur; caveat cumlesi aynen kalir, ama artik bir
       adayi ELEDIGIMIZI yazariz.
   (3) Ara durum yok. Sebep UYDURULMAZ.

  NOT: ihlal SEVIYELERI de degisecektir (w_econ buyudukce kontrolcu ekonomiyi
  onemser, voltaj baskisi azalir). Seviye kiyasi YAPILMAZ; ilgilendigimiz tek
  sey naive - nwp FARKI ve onun SE'si.

CALISTIR (kok klasorden, ~1.6 saat):
  python gg2/gg2_mpc2_wcost.py 2>&1 | tee gg2/mpc2_wcost.log

Bitmis kollari atlamak icin: --resume
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # kok klasor
sys.path.insert(0, _HERE)                    # gg2/ klasoru

import gg2_mpc2_evidence as E  # noqa: E402  (run_traj / summarize / boot_paired)

TARGET_SITE = 1.5
NEW_WCOST = 1.0
REF = {"gap_1bat": 0.70, "se_1bat": 0.45}
OUTDIR = Path("gg2/data")


def _die(m: str) -> None:
    sys.exit("[HATA] " + m)


def main() -> None:
    ap = argparse.ArgumentParser(description="w_econ izolasyonu (1-batarya kollari).")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--nwp", default="gg2/data/nwp_solar_fc_capnorm.npz")
    ap.add_argument("--wcost", type=float, default=NEW_WCOST)
    args = ap.parse_args()

    print("=" * 74)
    print("IZOLASYON - EKONOMIK AGIRLIK  (tek dugme: opf_expert.W_COST)")
    print("=" * 74)
    print("mpc2_test import ediliyor...\n")

    import mpc2_test as T   # noqa: E402
    import opf_expert as OE  # noqa: E402

    # ---- olcekleme (evidence kosumuyla BIREBIR ayni, uc kanit) -------------
    old = T.SOLAR_PEAK
    if abs(old - 1.25) > 1e-9:
        _die("mpc2_test.SOLAR_PEAK beklenen 1.25 degil: %s. DUR." % old)
    T.solar = T.solar * (TARGET_SITE / old)
    T.SOLAR_PEAK = TARGET_SITE
    mx = float(np.max(T.solar[:T.SPLIT]))
    print("KANIT (1) gercek solar site-tepesi: %.4f  (hedef %.2f)" % (mx, TARGET_SITE))
    if abs(mx - TARGET_SITE) > 0.02:
        _die("olcekleme tutmadi, DUR.")

    naive_15 = T.solar_fc * (TARGET_SITE / 3.0)
    p = Path(args.nwp)
    if not p.exists():
        _die("bulunamadi: %s" % p)
    nwp_15 = np.load(p)["solar"] * (TARGET_SITE / 3.0)
    if nwp_15.shape != naive_15.shape:
        _die("sekil uyusmazligi.")
    sl_t = slice(T.SPLIT, T.N - 1)
    for k, (tag, m) in enumerate((("naive", naive_15), ("NWP", nwp_15))):
        r = float(np.nanmean(m[sl_t, 0]) / np.nanmean(T.solar[sl_t]))
        print("KANIT (%d) %-5s fc/gercek orani: %.3f" % (k + 2, tag, r))
        if not (0.7 <= r <= 1.3):
            _die("%s olcegi uyusmuyor (%.3f), DUR." % (tag, r))
    print("  -> uc kanit da tuttu.\n")

    # ---- TEK DUGME ---------------------------------------------------------
    old_w = OE.W_COST
    OE.W_COST = float(args.wcost)
    print("TEK DEGISKEN: opf_expert.W_COST  %g -> %g" % (old_w, OE.W_COST))
    if abs(OE.W_COST - old_w) < 1e-12:
        _die("W_COST degismedi; izolasyonun anlami kalmaz.")
    print("SABIT KALAN: ufuk = her adim tam runpp | solar = iki bara 1.5+1.5")
    print("             safe layer = SafeLayer2 | dilim %d:%d\n" % (T.SPLIT, T.N - 1))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    FC = {"naive": naive_15, "nwp": nwp_15}
    res, tag_w = {}, ("w%g" % OE.W_COST).replace(".", "p")

    for arm, fckey in (("naive1", "naive"), ("nwp1", "nwp")):
        out = OUTDIR / ("mpc2_ev_%s_%s.npz" % (arm, tag_w))
        if args.resume and out.exists():
            z = np.load(out)
            a = {k: z[k] for k in z.files}
            print("--- %s: mevcut npz yuklendi (%s)" % (arm, out))
        else:
            T.solar_fc = FC[fckey]
            print("--- %s kosuluyor  (1 batarya, W_COST=%g, fc=%s)"
                  % (arm, OE.W_COST, fckey))
            t0 = time.time()
            a = E.run_traj(T, "gru", 1, arm)
            np.savez(out, **a)
            print("    yazildi: %s  (%.0f dk)" % (out, (time.time() - t0) / 60))

        s = E.summarize(a, T, 1)
        res[arm] = s
        print("    ihlal %.2f%% (%d adet: %d asiri, %d dusuk) | kar %.0f EUR | dolu %.1f%%\n"
              % (s["viol_pct"], s["n_viol"], s["n_ov"], s["n_uv"], s["kar"],
                 s["ov_full_pct"]))

    # ---- sonuc -------------------------------------------------------------
    print("=" * 74)
    print("SONUC - 1 batarya, W_COST = %g" % OE.W_COST)
    print("=" * 74)
    print("  %-10s %8s %10s %10s" % ("kol", "ihlal%", "kar(EUR)", "adet"))
    for arm in ("naive1", "nwp1"):
        s = res[arm]
        print("  %-10s %8.2f %10.0f %10d" % (arm, s["viol_pct"], s["kar"], s["n_viol"]))

    m, se = E.boot_paired(res["naive1"]["viol"], res["nwp1"]["viol"])
    m, se = m * 100, se * 100
    k = abs(m) / se if se > 0 else float("nan")
    mk, sek = E.boot_paired(res["naive1"]["profit_step"], res["nwp1"]["profit_step"])
    n = res["naive1"]["n"]
    mk, sek = mk * n, sek * n
    kk = abs(mk) / sek if sek > 0 else float("nan")

    print("\n  TAHMIN ETKISI (naive - NWP), gunluk blok bootstrap 10.000:")
    print("    ihlal : %+.2f +/- %.2f puan  (%.1f SE)" % (m, se, k))
    print("    kar   : %+.0f +/- %.0f EUR    (%.1f SE)" % (mk, sek, kk))
    print("\n  REFERANS (ayni kol cifti, W_COST=1e-3):")
    print("    ihlal : %+.2f +/- %.2f puan  (1.6 SE)" % (REF["gap_1bat"], REF["se_1bat"]))
    print("    ana boru hatti (w_econ=1.0, tek bara solar, ldf ufuk): +1.54 +/- 0.72 (2.1 SE)")

    print("\n  ON-KAYITLI HUKUM:")
    if k >= 2.0:
        print("    %.1f SE >= 2 -> EKONOMIK AGIRLIK ACIKLIYOR." % k)
        print("    Dusuk W_COST'ta kontrolcu zaten voltaj-baskin davraniyor;")
        print("    tahminin katabilecegi bilgi orada kisitli. Kapsam notu (2.4)")
        print("    olculmus cevapla degistirilir.")
    else:
        print("    %.1f SE < 2 -> ekonomik agirlik TEK BASINA ACIKLAMIYOR." % k)
        print("    Bir aday ELENDI. Kalan adaylar: ufuk voltaj modeli ve solar")
        print("    yerlesimi. Bu calismada ayrilmadi -> caveat cumlesi kalir,")
        print("    ama artik 'bir adayi test ettik ve elendi' diye yazilir.")
    print("\n  NOT: ihlal SEVIYELERI w_econ ile degisir; seviye kiyasi YAPILMAZ.")
    print("  Ilgilenilen tek sey naive - NWP FARKI ve SE'sidir.\n")


if __name__ == "__main__":
    main()