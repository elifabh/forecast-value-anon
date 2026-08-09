#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_z17_bias.py - Z17: 3-gunluk plasebo anomalisinin MEKANIZMA testi.

ANOMALI
  plac3'un tahmin becerisi YOK (alpha 1.285, kuyruk 1.313 - ikisi de naive'den
  kotu) ama ihlali %10.21 (naive %11.28). Fark 1.6 SE -> cozulmuyor, yani
  fark KANITLANMIS bile degil. Yine de mekanizma aranmadan "bilinmiyor"
  denmez. Bu script arar.

HIPOTEZ (koda girmeden yazildi)
  Ihlallerin %100'u ASIRI voltaj (DEVIR 3.10). Asiri voltaj batarya doluyken
  ve solar yuksekken olusur. Solar'i OLDUGUNDAN FAZLA tahmin eden bir
  kontrolcu "oglen cok solar gelecek" der, bataryayi onceden bosaltir, oglene
  daha cok sarj boslugu birakir -> daha az ihlal.

  Yani DOGRULUK degil YANLILIK (bias) belirleyici olabilir. Genel MAE'si
  kotu ama sistematik olarak YUKARI yanli bir tahmin, voltaj guvenligi icin
  ise yarayabilir.

UC KATMANDA OLCULUR (hikaye degil, zincir)
  [1] GIRDI    : tahminin solar-saat isaretli hatasi (pred - truth).
                 Pozitif = fazla tahmin = koruyucu.
  [2] DAVRANIS : solar saatlerde GERCEKLESEN SOC (yorungeden).
                 Dusuk SOC = cok bosluk = koruyucu.
  [3] SONUC    : ihlal orani.

ON-KAYIT (sonuclara bakilmadan)
  HUKUM (A): yanlilik ile ihlal orani alti kol arasinda TERS yonde gidiyor
             VE plac3'un solar-saat SOC'u naive'den DUSUK ise
             -> mekanizma OLCULDU. Makaleye "ihlal azalmasi sadece tahmin
                dogrulugunun fonksiyonu degil; yanlilik da rol oynuyor"
                olarak girer.
  HUKUM (B): siralamalar uyusmuyorsa
             -> mekanizma REDDEDILDI. "Bilinmiyor" yazilir, aciklama
                UYDURULMAZ. Ikinci teste gecilir (1,2,4,5 gun kaydirma).
  HUKUM (C): yanlilik siralamasi uyuyor ama SOC uymuyorsa (veya tersi)
             -> zincir KOPUK. Mekanizma DOGRULANMADI, kismi kanit olarak
                bile yazilmaz.

  Siralama karsilastirmasi Spearman rank korelasyonu ile raporlanir.
  ALTI nokta ile rho'nun kendi hata cubugu genistir; rho tek basina KANIT
  DEGILDIR, sadece yon gostergesidir. Karar plac3-naive ikilisinin
  yon tutarliligina bakilarak verilir.

  CIPA: gercek NWP alpha = 0.799 cikmali.

CALISTIR (~1 dk, MPC KOSMAZ):
  python gg2/gg2_z17_bias.py 2>&1 | tee gg2/z17_bias.log
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mpc_econ_voltage_pareto as M

SP = 3.0
SOLAR_H = (10, 16)
REF_ALPHA = 0.799
TOL_ALPHA = 0.002
SHIFTS = (3, 7, 14, 21)

TRAJ = {
    "NWP":    "gg2/nwp_fair_traj_nwp.npz",
    "plac3":  "gg2/data/placebo_shift3.npz",
    "plac7":  "gg2/data/placebo_shift7.npz",
    "plac14": "gg2/data/placebo_shift14.npz",
    "plac21": "gg2/data/placebo_shift21.npz",
    "naive":  "gg2/nwp_fair_traj_naive.npz",
}


def _die(m: str) -> None:
    sys.exit("[HATA] " + m)


def spearman(x, y):
    """Basit Spearman rho (scipy'siz)."""
    def rank(v):
        o = np.argsort(np.argsort(np.asarray(v, dtype=float)))
        return o.astype(float)
    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Z17: plasebo anomalisinin mekanizma testi.")
    ap.add_argument("--joint", default="data/aligned_30min_joint.csv")
    ap.add_argument("--fcmat", default="data/forecast_matrix.npz")
    ap.add_argument("--nwp", default="gg2/data/nwp_solar_fc_capnorm.npz")
    a = ap.parse_args()

    for p in (a.joint, a.fcmat, a.nwp):
        if not Path(p).exists():
            _die("bulunamadi: %s" % p)

    df = pd.read_csv(a.joint, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    N = len(df)
    y = (df.solar_mw / df.iloc[:M.SPLIT].solar_mw.max() * SP).to_numpy()
    hour = df.index.hour.to_numpy()
    naive_m = np.load(a.fcmat)["solar"]
    K = int(naive_m.shape[1])
    nwp_m = np.load(a.nwp)["solar"]

    idx = np.minimum(np.arange(N)[:, None] + np.arange(1, K + 1)[None, :], N - 1)
    truth = y[idx]
    hr = hour[idx]
    sl = slice(M.SPLIT, N - 1)

    mats = {"NWP": nwp_m, "naive": naive_m}
    for d in SHIFTS:
        sh = d * 48
        ii = np.clip(np.arange(len(nwp_m)) - sh, 0, len(nwp_m) - 1)
        mats["plac%d" % d] = nwp_m[ii]

    print("=" * 78)
    print("Z17 - 3-GUNLUK PLASEBO ANOMALISININ MEKANIZMA TESTI")
    print("=" * 78)
    print("  test blogu %d:%d | solar saat %d-%d UTC" % (M.SPLIT, N - 1, *SOLAR_H))

    # -------------------------------------------------------- [1] GIRDI
    def layer1(fc):
        err = (fc[sl] - truth[sl]).ravel()
        h = hr[sl].ravel()
        mm = (h >= SOLAR_H[0]) & (h <= SOLAR_H[1])
        e = err[mm]
        return (float(e.mean()),                  # isaretli yanlilik
                float(np.abs(e).mean()),          # MAE (solar saat)
                100.0 * float((e > 0).mean()))    # fazla tahmin orani %

    a_ref = None
    mnall = np.abs(naive_m[sl] - truth[sl]).mean()
    print("\n" + "=" * 78)
    print("[1] GIRDI - tahminin solar-saat isaretli hatasi")
    print("=" * 78)
    print("  pozitif yanlilik = FAZLA tahmin = koruyucu (hipotez)")
    print("  %-8s %10s %10s %12s %8s" % ("kol", "yanlilik", "MAE(sol)", "fazla tah.%", "alpha"))
    print("  " + "-" * 54)
    L1 = {}
    for t in ["NWP", "plac3", "plac7", "plac14", "plac21", "naive"]:
        b, mae_s, over = layer1(mats[t])
        al = float(np.abs(mats[t][sl] - truth[sl]).mean() / mnall)
        L1[t] = dict(bias=b, mae=mae_s, over=over, alpha=al)
        if t == "NWP":
            a_ref = al
        print("  %-8s %+10.4f %10.4f %12.1f %8.3f" % (t, b, mae_s, over, al))

    if abs(a_ref - REF_ALPHA) > TOL_ALPHA:
        print("\n  CIPA KALDI: NWP alpha %.3f, beklenen %.3f -> hukum yok."
              % (a_ref, REF_ALPHA))
        sys.exit(2)
    print("\n  CIPA: NWP alpha %.3f = %.3f  -> GECTI" % (a_ref, REF_ALPHA))

    # ----------------------------------------------------- [2] DAVRANIS
    print("\n" + "=" * 78)
    print("[2] DAVRANIS - solar saatlerde GERCEKLESEN SOC (yorungeden)")
    print("=" * 78)
    print("  dusuk SOC = cok sarj boslugu = koruyucu (hipotez)")
    L2, L3 = {}, {}
    hsl = hour[M.SPLIT:N - 1]
    sm = (hsl >= SOLAR_H[0]) & (hsl <= SOLAR_H[1])
    print("  %-8s %11s %11s %13s %11s" % ("kol", "SOC ort", "SOC solar", "dolu%(>5.5)", "ihlal %"))
    print("  " + "-" * 58)
    for t in ["NWP", "plac3", "plac7", "plac14", "plac21", "naive"]:
        p = TRAJ[t]
        if not Path(p).exists():
            _die("yorunge yok: %s" % p)
        z = np.load(p)
        d = int(z["done"])
        soc = z["soc"][:d]
        v = ((z["mn"] < M.LO) | (z["mx"] > M.HI))[:d]
        m_ = sm[:d]
        L2[t] = dict(soc=float(soc.mean()), soc_sol=float(soc[m_].mean()),
                     full=100.0 * float((soc[m_] > 5.5).mean()))
        L3[t] = 100.0 * float(v.mean())
        print("  %-8s %11.3f %11.3f %13.1f %11.2f"
              % (t, L2[t]["soc"], L2[t]["soc_sol"], L2[t]["full"], L3[t]))

    # -------------------------------------------------------- siralama
    order = ["NWP", "plac3", "plac7", "plac14", "plac21", "naive"]
    bias = [L1[t]["bias"] for t in order]
    socs = [L2[t]["soc_sol"] for t in order]
    viol = [L3[t] for t in order]

    print("\n" + "=" * 78)
    print("SIRALAMA TUTARLILIGI (Spearman rho, alti nokta - yon gostergesi)")
    print("=" * 78)
    r_bv = spearman(bias, viol)
    r_bs = spearman(bias, socs)
    r_sv = spearman(socs, viol)
    print("  yanlilik vs ihlal   rho = %+.3f   (hipotez: NEGATIF)" % r_bv)
    print("  yanlilik vs SOC     rho = %+.3f   (hipotez: NEGATIF)" % r_bs)
    print("  SOC vs ihlal        rho = %+.3f   (hipotez: POZITIF)" % r_sv)
    print("\n  UYARI: alti nokta ile rho'nun hata cubugu genistir.")
    print("  rho KANIT DEGILDIR, yon gostergesidir. Karar asagida.")

    # ---------------------------------------------------------- hukum
    print("\n" + "=" * 78)
    print("ON-KAYITLI HUKUM - plac3 vs naive ikilisi")
    print("=" * 78)
    db = L1["plac3"]["bias"] - L1["naive"]["bias"]
    ds = L2["plac3"]["soc_sol"] - L2["naive"]["soc_sol"]
    dv = L3["plac3"] - L3["naive"]
    print("  yanlilik farki (plac3 - naive) = %+.4f   %s"
          % (db, "plac3 DAHA FAZLA tahmin ediyor" if db > 0 else "plac3 DAHA AZ tahmin ediyor"))
    print("  solar SOC farki               = %+.4f   %s"
          % (ds, "plac3 DAHA BOS" if ds < 0 else "plac3 DAHA DOLU"))
    print("  ihlal farki                   = %+.2f pp %s"
          % (dv, "(plac3 daha az ihlal)" if dv < 0 else "(plac3 daha cok ihlal)"))

    chain = (db > 0) and (ds < 0) and (dv < 0)
    dir_ok = (r_bv < 0) and (r_sv > 0)
    print()
    if chain and dir_ok:
        print("  HUKUM (A): ZINCIR TAM. Fazla tahmin -> dusuk SOC -> az ihlal.")
        print("    -> Mekanizma OLCULDU. Makaleye girer:")
        print("       'ihlal azalmasi sadece tahmin dogrulugunun fonksiyonu degil;")
        print("        tahminin yanliligi da rol oynuyor.'")
        print("    -> KATKI 3'u (sentetik supurme yaniltiyor) GUCLENDIRIR.")
    elif chain and not dir_ok:
        print("  HUKUM (C): plac3-naive ikilisi tutuyor ama alti kol genelinde")
        print("    yon tutarli DEGIL. Zincir KOPUK -> mekanizma DOGRULANMADI.")
        print("    Kismi kanit olarak bile YAZILMAZ. Ikinci teste gecilir.")
    else:
        print("  HUKUM (B): ZINCIR KOPUK. Mekanizma REDDEDILDI.")
        print("    'Bilinmiyor' yazilir. ACIKLAMA UYDURULMAZ.")
        print("    Ikinci test: 1,2,4,5 gun kaydirma -> 3 gun ozel mi,")
        print("    yoksa kisa-kaydirma bolgesi sistematik olarak mi farkli.")
    print()


if __name__ == "__main__":
    main()