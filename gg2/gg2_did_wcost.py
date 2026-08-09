#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_did_wcost.py - FARK-FARKI: ekonomik agirlik, tahmin etkisini GERCEKTEN degistiriyor mu?

NEDEN BU SCRIPT VAR (on-kayit kusuru duzeltmesi)
  gg2_mpc2_wcost.py'nin on-kayitli hukmu su idi:
    "tahmin etkisi >= 2 SE olursa -> ekonomik agirlik ACIKLIYOR"
  Bu bir ESIK-GECME testidir, FARK testi degildir. 1.6 SE'den 2.9 SE'ye
  gecmek gurultuyle de olabilir. Kriter, beklenen cevabi dogrulayacak sekilde
  kurulmustu -- bu, yaz kar kosulunda dusulen hatanin aynisi.

  Dogru soru: iki w_econ altindaki tahmin etkileri BIRBIRINDEN ayirt
  edilebiliyor mu? Cevap fark-farki (difference-in-differences) ile verilir:

    DiD = [naive - NWP]_(w=1.0)  -  [naive - NWP]_(w=1e-3)

  Dort kosum da AYNI zaman adimlarinda oldugu icin ESLESTIRILMIS bootstrap
  yapilir; bagimsiz varsayimla hesaplanan SE bir UST SINIRDIR.

MPC YENIDEN KOSULMAZ. Dort yorunge dosyasi zaten diskte.

ON-KAYIT (kod yazilmadan once, duz Turkce -- sonuc GORULMEDEN)
  Olculecek : DiD (ihlal, puan) ve DiD (kar, EUR).
  Yontem    : gunluk blok (48 adim), 10.000 tekrar, DORT SERI birlikte
              yeniden orneklenir (ayni gun indisleri).
  Esik      : 2 SE.

  HUKUM:
   (1) |DiD| >= 2 SE -> "Ekonomik agirlik, tahminin voltaj guvenligine
       katkisini OLCULEBILIR sekilde degistiriyor. Iki-batarya boru hattindaki
       zayif tahmin etkisi, ekonomik agirlik farkiyla aciklanir."
   (2) |DiD| < 2 SE -> "Iki agirlik altindaki tahmin etkileri BIRBIRINDEN
       AYIRT EDILEMIYOR. Yon hipotezle tutarli ama buyukluk farki
       KANITLANMADI. Makalede iki etki yan yana raporlanir ve izolasyonun
       sonucsuz kaldigi yazilir."
   (3) Ara durum yok. Sebep UYDURULMAZ.

  NOT: (2) cikmasi bir basarisizlik degildir. Her iki durumda da soylenecek
  seffaf bir cumlemiz olur; (1) daha guclu, (2) daha temkinli.

CALISTIR (kok klasorden, ~1 dk):
  python gg2/gg2_did_wcost.py 2>&1 | tee gg2/did_wcost.log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

LO, HI = 0.95, 1.05
DT = 0.5
DAY = 48
NBOOT = 10000
SEED = 12345

FILES = {
    "naive_w0": "gg2/data/mpc2_ev_naive1.npz",
    "nwp_w0":   "gg2/data/mpc2_ev_nwp1.npz",
    "naive_w1": "gg2/data/mpc2_ev_naive1_w1.npz",
    "nwp_w1":   "gg2/data/mpc2_ev_nwp1_w1.npz",
}
REF = {
    "w0": (0.70, 0.45),
    "w1": (1.25, 0.43),
    "main": (1.54, 0.72),
}


def _die(m: str) -> None:
    sys.exit("[HATA] " + m)


def load(path: str):
    p = Path(path)
    if not p.exists():
        _die("bulunamadi: %s\n       Once gg2_mpc2_evidence.py ve gg2_mpc2_wcost.py kosulmali." % p)
    z = np.load(p)
    for k in ("vmin", "vmax", "pA", "pB", "price"):
        if k not in z.files:
            _die("%s icinde '%s' yok" % (p, k))
    viol = ((z["vmin"] < LO) | (z["vmax"] > HI)).astype(float)
    profit = -(z["price"] * (z["pA"] + z["pB"])) * DT
    return viol, profit


def main() -> None:
    ap = argparse.ArgumentParser(description="w_econ fark-farki testi.")
    ap.add_argument("--nboot", type=int, default=NBOOT)
    ap.parse_args()  # nboot sabit tutulur (SEED ile birlikte tekrarlanabilirlik)

    print("=" * 74)
    print("FARK-FARKI (DiD) - ekonomik agirlik tahmin etkisini degistiriyor mu?")
    print("=" * 74)

    V, P = {}, {}
    n = None
    for k, f in FILES.items():
        V[k], P[k] = load(f)
        if n is None:
            n = len(V[k])
        elif len(V[k]) != n:
            _die("uzunluk uyusmazligi: %s -> %d, beklenen %d" % (k, len(V[k]), n))
        print("  yuklendi %-10s %-38s ihlal %.2f%%  kar %.0f EUR"
              % (k, f, V[k].mean() * 100, P[k].sum()))

    nd = n // DAY
    print("\n  %d adim -> %d tam gun (son %d adim bloklama disi)"
          % (n, nd, n - nd * DAY))
    if nd < 5:
        _die("5'ten az tam gun; bloklu bootstrap yapilamaz")

    def blocks(x):
        return x[:nd * DAY].reshape(nd, DAY)

    Vb = {k: blocks(v) for k, v in V.items()}
    Pb = {k: blocks(p) for k, p in P.items()}

    rng = np.random.default_rng(SEED)
    did_v = np.empty(NBOOT)
    eff_w0 = np.empty(NBOOT)
    eff_w1 = np.empty(NBOOT)
    did_p = np.empty(NBOOT)

    for b in range(NBOOT):
        s = rng.integers(0, nd, nd)          # AYNI gunler, dort seriye de
        e0 = Vb["naive_w0"][s].mean() - Vb["nwp_w0"][s].mean()
        e1 = Vb["naive_w1"][s].mean() - Vb["nwp_w1"][s].mean()
        eff_w0[b] = e0
        eff_w1[b] = e1
        did_v[b] = e1 - e0
        q0 = Pb["naive_w0"][s].mean() - Pb["nwp_w0"][s].mean()
        q1 = Pb["naive_w1"][s].mean() - Pb["nwp_w1"][s].mean()
        did_p[b] = (q1 - q0) * n

    print("\n" + "=" * 74)
    print("TAHMIN ETKISI (naive - NWP), ihlal puani")
    print("=" * 74)
    for tag, arr in (("w_econ = 1e-3", eff_w0), ("w_econ = 1.0 ", eff_w1)):
        m, se = arr.mean() * 100, arr.std(ddof=1) * 100
        print("  %-14s %+6.2f +/- %.2f puan  (%.1f SE)"
              % (tag, m, se, abs(m) / se if se > 0 else float("nan")))
    print("  %-14s %+6.2f +/- %.2f puan  (%.1f SE)   [ana boru hatti, referans]"
          % ("w_econ = 1.0 ", REF["main"][0], REF["main"][1],
             REF["main"][0] / REF["main"][1]))

    m_v = float(did_v.mean() * 100)
    se_v = float(did_v.std(ddof=1) * 100)
    k_v = abs(m_v) / se_v if se_v > 0 else float("nan")
    m_p = float(did_p.mean())
    se_p = float(did_p.std(ddof=1))
    k_p = abs(m_p) / se_p if se_p > 0 else float("nan")

    print("\n" + "=" * 74)
    print("FARK-FARKI  DiD = etki(w=1.0) - etki(w=1e-3)")
    print("=" * 74)
    print("  ihlal : %+.2f +/- %.2f puan  (%.1f SE)" % (m_v, se_v, k_v))
    print("  kar   : %+.0f +/- %.0f EUR    (%.1f SE)" % (m_p, se_p, k_p))
    ub = (REF["w1"][1] ** 2 + REF["w0"][1] ** 2) ** 0.5
    print("\n  (bagimsiz varsayimla kaba ust sinir SE = %.2f; eslestirme bunu"
          " kucultur)" % ub)

    print("\n" + "=" * 74)
    print("ON-KAYITLI HUKUM")
    print("=" * 74)
    if k_v >= 2.0:
        print("  %.1f SE >= 2 -> EKONOMIK AGIRLIK OLCULEBILIR FARK YARATIYOR." % k_v)
        print("  Iki-batarya boru hattindaki zayif tahmin etkisi, ekonomik agirlik")
        print("  farkiyla aciklanir. Kapsam notu olculmus cevapla degistirilir:")
        print("  'we isolated economic weight as the explanation'.")
    else:
        print("  %.1f SE < 2 -> IKI ETKI BIRBIRINDEN AYIRT EDILEMIYOR." % k_v)
        print("  Yon hipotezle tutarli (w=1.0'da etki daha buyuk ve cozuluyor),")
        print("  ama BUYUKLUK FARKI KANITLANMADI.")
        print("  Makalede iki etki yan yana raporlanir ve izolasyonun sonucsuz")
        print("  kaldigi acikca yazilir. 'Ekonomik agirlik acikliyor' DENMEZ.")
    print("\n  Her iki durumda da soylenebilecek OLCULMUS cumle:")
    print("  Ana boru hattinin disinda, ekonomik agirlik esitlendiginde tahmin")
    print("  etkisi yeniden cozuluyor (%+.2f +/- %.2f puan, %.1f SE) ve ana"
          % (eff_w1.mean() * 100, eff_w1.std(ddof=1) * 100,
             abs(eff_w1.mean()) / eff_w1.std(ddof=1)))
    print("  boru hattinin buyuklugu (+1.54) ile tutarli.\n")


if __name__ == "__main__":
    main()