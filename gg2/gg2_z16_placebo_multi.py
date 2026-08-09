#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_z16_placebo_multi.py - Z16: plasebo testi COKLU KAYDIRMA ile.

NEDEN
  Mevcut plasebo TEK noktadir: NWP matrisi 7 gun geri kaydirilmis, ihlal
  %11.16 cikmis (naive %11.28'e yakin, NWP %9.80'e degil). Bu iyi bir sonuc
  AMA tek kaydirma. Tek noktada sansli/sanssiz bir hava dizisine denk gelmis
  olabiliriz. Hakem "neden 7 gun?" diye sorar ve hakli olur.

  Bu script kaydirmayi DORT farkli uzunlukta tekrarlar. Kaydirma uzunlugunun
  secimi sonuca bakilmadan sabitlenmistir: 3, 7, 14, 21 gun.
  7 zaten kosuldu -> CIPA olarak yeniden uretilir.

TASARIM
  Sadece GERIYE kaydirma kullanilir (i-shift). Ileriye kaydirma gelecekteki
  tahmini simdiye tasirdi; sonuc kotu ciksa bile "gelecege baktin" itirazina
  yol acar. Gereksiz risk, alinmiyor.

  Kaydirilmis tahmin: ayni tahminci, ayni istatistik, ayni gunluk sekil,
  YANLIS hava. Yani "iyilesme tahminin ICERIGINDEN mi geliyor" sorusunu sorar.

ON-KAYIT (koda girmeden, sonuclara bakilmadan yazildi)
  Olculecek : her kaydirma icin ihlal orani.
  Esik      : naive (%11.28) ile NWP (%9.80) arasindaki orta nokta = %10.54.
              Plasebo bu esigin USTUNDE ise naive tarafindadir.
  HUKUM (1) : DORT kaydirmanin hepsi > %10.54
              -> "Kazanc tahmin iceriginden geliyor; dort bagimsiz kaydirma
                 uzunlugunda dogrulandi."
  HUKUM (2) : herhangi biri <= %10.54
              -> O kaydirma AYNEN raporlanir ve bir UYARIDIR. Kazancin bir
                 kismi tahmin iceriginden degil, tahminin istatistiksel
                 sekilinden geliyor olabilir. Aciklama UYDURULMAZ.
  CIPA      : shift=7 -> %11.16 (+/- 0.15) uretmeli. Uretmezse hukum yok.

CALISTIR (GG-1 kok klasorunden). Her kaydirma ~15 dk.
  Tek pencerede hepsi (~60 dk):
    python gg2/gg2_z16_placebo_multi.py 2>&1 | tee gg2/z16_placebo.log
  Iki pencerede paralel (~30 dk):
    python gg2/gg2_z16_placebo_multi.py --shifts 7,3  2>&1 | tee gg2/z16_a.log
    python gg2/gg2_z16_placebo_multi.py --shifts 14,21 2>&1 | tee gg2/z16_b.log
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

DEFAULT_SHIFTS = "3,7,14,21"
ANCHOR_SHIFT = 7
ANCHOR_VIOL = 11.16
ANCHOR_TOL = 0.15

NAIVE_VIOL = 11.28
NWP_VIOL = 9.80
MIDPOINT = 0.5 * (NAIVE_VIOL + NWP_VIOL)     # 10.54


def _die(m: str) -> None:
    sys.exit("[HATA] " + m)


def load_verify():
    """gg2_verify.py'yi dosya yolundan yukler.

    run_block KOPYALANMAZ - orijinali kullanilir. Kopyalanirsa zamanla
    ayrisirlar ve sessiz hata olur. Modul seviyesinde M ve D yuklenir
    (veri okur, diske yazmaz) - gg2_nwp_mpc_fair.py ile ayni davranis.
    """
    p = os.path.join(HERE, "gg2_verify.py")
    if not os.path.exists(p):
        _die("bulunamadi: " + p)
    spec = importlib.util.spec_from_file_location("gg2_verify", p)
    if spec is None or spec.loader is None:
        _die("gg2_verify.py yuklenemedi")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("run_block", "PLACEBO_SHIFT"):
        if not hasattr(mod, fn):
            _die("gg2_verify icinde %s yok -> dosya degismis" % fn)
    if int(mod.PLACEBO_SHIFT) != 7 * 48:
        _die("gg2_verify.PLACEBO_SHIFT %d, beklenen %d -> tasarim degismis"
             % (int(mod.PLACEBO_SHIFT), 7 * 48))
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description="Z16: coklu kaydirmali plasebo.")
    ap.add_argument("--nwp", default="gg2/data/nwp_solar_fc_capnorm.npz")
    ap.add_argument("--shifts", default=DEFAULT_SHIFTS,
                    help="virgulle ayrilmis GUN cinsinden kaydirmalar")
    ap.add_argument("--outdir", default="gg2/data")
    a = ap.parse_args()

    p = Path(a.nwp)
    if not p.exists():
        _die("bulunamadi: %s" % p)

    V = load_verify()
    M = V.M

    try:
        days = [int(x.strip()) for x in a.shifts.split(",") if x.strip()]
    except ValueError:
        _die("--shifts sayilardan olusmali, ornek: 3,7,14,21")
    if any(d <= 0 for d in days):
        _die("kaydirmalar pozitif olmali (sadece geriye kaydirma)")

    nwp = np.load(p)["solar"]
    T0, T1 = M.SPLIT, M.N - 1
    n = T1 - T0

    print("=" * 74)
    print("Z16 - COKLU KAYDIRMALI PLASEBO")
    print("=" * 74)
    print("  tahmin matrisi : %s  shape %s" % (a.nwp, str(nwp.shape)))
    print("  dilim          : %d:%d (%d adim)" % (T0, T1, n))
    print("  kaydirmalar    : %s gun" % ", ".join(str(d) for d in days))
    print("  ON-KAYIT esigi : %%%.2f  (naive %.2f ile NWP %.2f arasi orta nokta)"
          % (MIDPOINT, NAIVE_VIOL, NWP_VIOL))
    print("  CIPA           : shift=7 -> %%%.2f (+/- %.2f)" % (ANCHOR_VIOL, ANCHOR_TOL))
    print("  her kaydirma ~15 dk.")

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    res = {}
    for d in days:
        shift = d * 48
        # kaydirma test dilimine tasmiyor mu? (0'a clip olursa tahmin tekrarlanir)
        if T0 - shift < 0:
            _die("kaydirma %d gun test diliminin basini asiyor" % d)
        idx = np.clip(np.arange(len(nwp)) - shift, 0, len(nwp) - 1)
        placebo = nwp[idx]
        print("\n" + "=" * 74)
        print("KAYDIRMA %d GUN (%d adim) kosuluyor..." % (d, shift))
        print("=" * 74)
        r = V.run_block(placebo, T0, T1, "plasebo%d" % d)
        res[d] = r
        op = outdir / ("placebo_shift%d.npz" % d)
        np.savez(op, mn=r["mn"], mx=r["mx"], p=r["P"], soc=r["SOC"],
                 start=T0, done=n, shift_days=d)
        print("  ihlal %.2f%%  |  kar %.0f EUR   -> %s" % (r["V"], r["kar"], op))

    # --------------------------------------------------------------- ozet
    print("\n" + "=" * 74)
    print("OZET")
    print("=" * 74)
    print("  %-22s %9s %11s   %s" % ("kol", "ihlal %", "kar EUR", "taraf"))
    print("  " + "-" * 62)
    print("  %-22s %9.2f %11.1f   %s" % ("naive (Tablo 1)", NAIVE_VIOL, 9681.0, "-"))
    for d in sorted(res):
        r = res[d]
        side = "naive tarafi" if r["V"] > MIDPOINT else "NWP TARAFI <- UYARI"
        print("  %-22s %9.2f %11.1f   %s" % ("plasebo %d gun" % d, r["V"], r["kar"], side))
    print("  %-22s %9.2f %11.1f   %s" % ("GERCEK NWP", NWP_VIOL, 10248.0, "-"))

    # --------------------------------------------------------------- cipa
    print("\n" + "=" * 74)
    print("CIPA")
    print("=" * 74)
    if ANCHOR_SHIFT in res:
        got = res[ANCHOR_SHIFT]["V"]
        ok = abs(got - ANCHOR_VIOL) <= ANCHOR_TOL
        print("  shift=7  beklenen %.2f | olculen %.2f | fark %+.2f  -> %s"
              % (ANCHOR_VIOL, got, got - ANCHOR_VIOL, "GECTI" if ok else "KALDI"))
        if not ok:
            print("  KALDI -> boru hatti supheli. HICBIR HUKUM VERILMEZ.")
            print("  Ciktiyi Claude'a yapistir, sebep bulunmadan devam etme.")
    else:
        print("  shift=7 bu kosumda yok -> cipa DEGERLENDIRILEMEDI.")
        print("  Hukum ancak shift=7 iceren kosumla birlikte verilir.")

    # -------------------------------------------------------------- hukum
    print("\n" + "=" * 74)
    print("ON-KAYITLI HUKUM")
    print("=" * 74)
    below = [d for d in sorted(res) if res[d]["V"] <= MIDPOINT]
    if not below:
        print("  Tum kaydirmalar esigin (%%%.2f) USTUNDE." % MIDPOINT)
        print("  -> HUKUM (1): Kazanc tahmin ICERIGINDEN geliyor;")
        print("     %d bagimsiz kaydirma uzunlugunda dogrulandi." % len(res))
    else:
        print("  Esigin ALTINDA kalan kaydirma(lar): %s"
              % ", ".join("%d gun (%%%.2f)" % (d, res[d]["V"]) for d in below))
        print("  -> HUKUM (2): UYARI. Kazancin bir kismi tahmin iceriginden degil,")
        print("     tahminin istatistiksel seklinden geliyor olabilir.")
        print("     Bu aynen raporlanir. ACIKLAMA UYDURULMAZ.")
    print("\n  NOT: bu bir SEVIYE kiyasidir. Hata cubuklari icin yorungeler")
    print("  gg2/data/placebo_shift*.npz olarak kaydedildi; bootstrap ayri kosulur.\n")


if __name__ == "__main__":
    main()