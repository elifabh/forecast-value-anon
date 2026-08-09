#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_z1_drl_nwp.py - Z1: DRL politikasina MPC'ye verdigimiz NWP tahminini ver.

SORUN (esit avantaj kuralinin ihlali)
  GG-2'de MPC'nin solar tahmini daily-naive'den NWP cap-norm'a YUKSELTILDI.
  DRL politikasi naive (gru matrisi) ile kaldi. Hakemin ilk sorusu bu olur:
  "MPC'ye iyi tahmin verdiniz, DRL'e vermediniz; karsilastirma adil mi?"

BU SCRIPT NE YAPAR
  Egitilmis politikayi (models/sac_actor_best_gru.pt) YENIDEN EGITMEDEN,
  sadece gozlemindeki SOLAR tahminini NWP cap-norm ile degistirip yeniden
  degerlendirir. sweep_forecast.py'nin altyapisi birebir kullanilir.
  wind / dmult / price tahminleri DEGISMEZ -- tek degisken solar.

  Yeniden egitim YOK. Bu bir "politika bu bilgiyi kullanabiliyor mu" testidir,
  "NWP ile egitilse ne olurdu" testi DEGILDIR. Bu sinir metne yazilir.

CIPA (once bu tutmali)
  kosul "gru" -> beklenen %8.65 (sweep_forecast.py log'u). Sapma > 0.15 puan
  ise DUR: altyapi eskisini uretmiyorsa yeni kol yorumlanamaz.

OLCEK KANITLARI (kosumdan once, tutmazsa DURUR)
  (1) NWP matrisi satir sayisi N ile ayni mi
  (2) NWP ufku Hfc ile ayni mi (degilse kesilir ve RAPOR EDILIR)
  (3) NWP'nin test dilimindeki ortalamasi, mevcut solar_fc'nin ayni dilimdeki
      ortalamasina orani 0.7-1.3 arasinda mi (birim/olcek uyusmazligi yakalar)

ON-KAYIT (kod yazilmadan once, duz Turkce)
  Olculecek : DRL ihlal orani, gru vs NWP. Bootstrap: gunluk blok 48 adim,
              10.000 tekrar, eslestirilmis.
  Esik      : 2 SE.
  Referans  : sweep_forecast sonuclari -- perfect 8.52 | gru 8.65 | noise 8.60 |
              persist 8.52 | shuffle(cop) 8.52. Aralik 0.13; seed yayilimi
              +/-0.14. Yani politika simdiye kadar tahmin ICERIGINE duyarsiz.

  HUKUM:
   (1) |fark| < 2 SE -> "Politika, MPC'ye 1.54 puan kazandiran ayni tahmin
       yukseltmesinden olculebilir fayda saglamiyor. Esit-bilgi itirazi
       cevaplandi: DRL'e ayni bilgi VERILDI ve kullanmadi."
   (2) fark <= -2 SE (NWP daha iyi) -> "Politika tahmini KULLANIYOR; GG-1'deki
       'forecast-insensitive' sonucu daraltilmali. Ana tabloya DRL-NWP satiri
       eklenir."
   (3) fark >= +2 SE (NWP daha kotu) -> "Dagitim disi girdi politikayi bozuyor.
       Bu bir robustluk bulgusudur; oldugu gibi raporlanir, sebep UYDURULMAZ."

CALISTIR (kok klasorden, ~2-5 dk):
  python gg2/gg2_z1_drl_nwp.py 2>&1 | tee gg2/z1_drl_nwp.log
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SPLIT = 13800
LO, HI = 0.95, 1.05
DAY = 48
NBOOT = 10000
SEED = 12345
ANCHOR_GRU = 8.65
ANCHOR_TOL = 0.15


def _die(m: str) -> None:
    sys.exit("[HATA] " + m)


def boot_paired(x: np.ndarray, y: np.ndarray, nboot: int = NBOOT):
    n = len(x)
    nd = n // DAY
    if nd < 5:
        return float("nan"), float("nan")
    xd = x[:nd * DAY].reshape(nd, DAY)
    yd = y[:nd * DAY].reshape(nd, DAY)
    rng = np.random.default_rng(SEED)
    d = np.empty(nboot)
    for b in range(nboot):
        s = rng.integers(0, nd, nd)
        d[b] = xd[s].mean() - yd[s].mean()
    return float(np.mean(d)), float(np.std(d, ddof=1))


def main() -> None:
    ap = argparse.ArgumentParser(description="Z1: DRL politikasina NWP solar tahmini.")
    ap.add_argument("--nwp", default="gg2/data/nwp_solar_fc_capnorm.npz")
    ap.add_argument("--model", default="models/sac_actor_best_gru.pt")
    ap.add_argument("--force-continue", action="store_true")
    args = ap.parse_args()

    for p in (args.nwp, args.model, "data/aligned_30min_joint.csv"):
        if not Path(p).exists():
            _die("bulunamadi: " + p)

    import torch  # noqa: E402
    from env_gridguard import GridGuardEnv  # noqa: E402
    from sac_agent import Actor  # noqa: E402

    df = pd.read_csv("data/aligned_30min_joint.csv", index_col=0, parse_dates=True)
    N = len(df)
    T0, T1 = SPLIT, N - 1

    print("=" * 74)
    print("Z1 - DRL politikasi + NWP solar tahmini (yeniden egitim YOK)")
    print("=" * 74)

    ref = GridGuardEnv(seed=0, forecast="gru")
    Hfc = ref.H_fc
    base_solar_fc = np.array(ref.solar_fc, dtype=float)
    print("  env: N=%d | H_fc=%d | test dilimi %d:%d" % (N, Hfc, T0, T1))

    # ---------------------------------------------------------- olcek kanitlari
    z = np.load(args.nwp)
    nwp = np.array(z["solar"], dtype=float)
    print("\n  KANIT (1) NWP sekli %s | env solar_fc sekli %s"
          % (nwp.shape, base_solar_fc.shape))
    if nwp.shape[0] != base_solar_fc.shape[0]:
        _die("satir sayisi uyusmuyor (%d vs %d). DUR."
             % (nwp.shape[0], base_solar_fc.shape[0]))

    if nwp.shape[1] != Hfc:
        if nwp.shape[1] < Hfc:
            _die("NWP ufku %d < env ufku %d. Doldurma UYDURMA olur. DUR."
                 % (nwp.shape[1], Hfc))
        print("  KANIT (2) NWP ufku %d > env ufku %d -> ilk %d sutun kullanilir"
              % (nwp.shape[1], Hfc, Hfc))
        nwp = nwp[:, :Hfc]
    else:
        print("  KANIT (2) ufuk esit (%d)" % Hfc)

    sl = slice(T0, T1)
    a_old = float(np.nanmean(base_solar_fc[sl, 0]))
    a_new = float(np.nanmean(nwp[sl, 0]))
    r = a_new / a_old if a_old else float("nan")
    print("  KANIT (3) test dilimi ort (ufuk 1): mevcut %.4f | NWP %.4f | oran %.3f"
          % (a_old, a_new, r))
    if not (0.7 <= r <= 1.3):
        _die("olcek uyusmuyor (%.3f). Env birimi ile NWP birimi ayni degil. DUR." % r)
    print("  -> uc kanit da tuttu.\n")

    # ---------------------------------------------------------------- politika
    actor = Actor(ref.obs_dim)
    actor.load_state_dict(torch.load(args.model))
    actor.eval()

    def policy(o):
        return float(actor.act_mean(torch.tensor(o[None]))[0, 0].detach())

    def run(kind: str):
        """sweep_forecast.py ile BIREBIR ayni dongu; tek fark solar_fc kaynagi."""
        e = GridGuardEnv(seed=0, forecast="gru")
        if kind == "nwp":
            e.solar_fc = nwp.copy()          # TEK degisken
        elif kind != "gru":
            _die("bilinmeyen kosul: " + kind)
        o = e.reset(start=T0)
        mn, mx = [], []
        while e.i < T1:
            o, rw, done, ex = e.step(policy(o))
            mn.append(ex["Vmin"])
            mx.append(ex["Vmax"])
            if done and e.i < T1:
                s = e.soc
                o = e.reset(start=e.i)
                e.soc = s
                o = e._obs(e.i)

        mn = np.array(mn)
        mx = np.array(mx)
        viol = ((mn < LO) | (mx > HI)).astype(float)
        return viol

    print("  --- gru (CIPA) kosuluyor...")
    v_gru = run("gru")
    rate_gru = float(v_gru.mean() * 100)
    d = abs(rate_gru - ANCHOR_GRU)
    print("      gru: %.2f%% (%d adet, %d adim)"
          % (rate_gru, int(v_gru.sum()), len(v_gru)))
    print("      CIPA: beklenen %.2f | fark %.2f puan -> %s"
          % (ANCHOR_GRU, d, "TUTTU" if d <= ANCHOR_TOL else "TUTMADI"))
    if d > ANCHOR_TOL and not args.force_continue:
        _die("cipa TUTMADI. Altyapi sweep_forecast sonucunu uretmiyor.\n"
             "       Sebep bulunmadan NWP kolu yorumlanmaz. --force-continue ile gecilir.")

    print("\n  --- nwp kosuluyor...")
    v_nwp = run("nwp")
    rate_nwp = float(v_nwp.mean() * 100)
    print("      nwp: %.2f%% (%d adet)" % (rate_nwp, int(v_nwp.sum())))

    # ------------------------------------------------------------------ sonuc
    m, se = boot_paired(v_gru, v_nwp)
    m, se = m * 100, se * 100
    k = abs(m) / se if se > 0 else float("nan")

    print("\n" + "=" * 74)
    print("SONUC - DRL politikasi, tek degisken solar tahmini")
    print("=" * 74)
    print("  %-10s %8s" % ("kosul", "ihlal%"))
    print("  %-10s %8.2f" % ("gru", rate_gru))
    print("  %-10s %8.2f" % ("nwp", rate_nwp))
    print("\n  fark (gru - nwp) = %+.2f +/- %.2f puan  (%.1f SE)" % (m, se, k))
    print("\n  REFERANSLAR:")
    print("    sweep_forecast: perfect 8.52 | gru 8.65 | noise 8.60 | persist 8.52 |")
    print("                    shuffle(cop) 8.52   -> aralik 0.13, seed yayilimi +/-0.14")
    print("    MPC'de ayni tahmin yukseltmesi: 11.28 -> 9.80 (1.54 +/- 0.72, 2.1 SE)")

    print("\n  ON-KAYITLI HUKUM:")
    if k < 2.0:
        print("    %.1f SE < 2 -> POLITIKA BU BILGIYI KULLANMIYOR." % k)
        print("    Esit-bilgi itirazi cevaplandi: DRL'e MPC ile AYNI tahmin verildi,")
        print("    olculebilir fayda saglamadi. Makaleye tek cumleyle girer.")
    elif m > 0:
        print("    %.1f SE >= 2, NWP DAHA IYI -> politika tahmini KULLANIYOR." % k)
        print("    GG-1'deki 'forecast-insensitive' sonucu DARALTILMALI.")
        print("    Ana tabloya DRL-NWP satiri eklenir.")
    else:
        print("    %.1f SE >= 2, NWP DAHA KOTU -> dagitim disi girdi politikayi" % k)
        print("    bozuyor. Robustluk bulgusu olarak oldugu gibi raporlanir.")
        print("    SEBEP UYDURULMAZ.")

    print("\n  SINIR (metne yazilacak): politika NWP ile YENIDEN EGITILMEDI.")
    print("  Bu test 'mevcut politika bu bilgiyi kullanabiliyor mu' sorusunu")
    print("  cevaplar; 'NWP ile egitilse ne olurdu' sorusunu CEVAPLAMAZ.")
    print("  Ayrica tek model dosyasi (%s), cok-seed degil.\n" % args.model)


if __name__ == "__main__":
    main()