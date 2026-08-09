#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_mpc2_evidence.py - IKI-BATARYA KANIT KOSUMU (Z4+Z5+Z6+Z7 + doygunluk ayrimi)

NEDEN BU SCRIPT VAR
  Devirdeki iki-batarya sayilari (3.09 / 2.76 / 1.98) dogru olcuKldu, ama uc
  eksigi var:
    Z4  kar yok            -> mpc2_test.run sadece ihlal donduruyor
    Z5  hata cubugu yok    -> 0.78 puanlik bosluk 1.0 esigine yakin
    Z6  depth yok          -> GG-1 tablosunda var, burada yok
    Z7  kunye kiyaslanmadi -> tek-batarya boru hatti ile ayarlar farkli
  Ayrica KOD OKUMASINDA BULUNAN IKI SORUN:

  (a) "ihlallerin %100'u doygunlukta" TOTOLOJI.
      safe_layer2.project: status="saturated" <=> viol>0 <=> (Vmin<LO or Vmax>HI)
      mpc2_test ihlal sayimi da tam bu kosul. Yani sayi olculmedi, tanim geregi
      cikti. Bu script GERCEK ayrimi kaydeder: ihlal aninda batarya basliklari
      (hiA/hiB, loA/loB) ENERJI siniri yuzunden mi yoksa GUC siniri (P_MAX)
      yuzunden mi bagliydi.

  (b) IKI BORU HATTI FARKLI AYARDA.
      tek-batarya (gg2_nwp_season) : ufuk = LinDistFlow farki, w_econ = 1.0
      iki-batarya (mpc2_test)      : ufuk = her adim tam runpp, W_COST = 1e-3
      Yani 11.28 -> 3.09 dususunun ne kadari 2. bataryadan, ne kadari ayar
      farkindan bilinmiyor. 4. KOL bunu cozer: AYNI boru hattinda tek batarya.
      (opf_expert.W_COST = opf_expert2.W_COST = 1e-3 -> kol icinde ayar sabit.)

DORT KOL (hepsi: toplam solar 3.0, dilim 13800:16229, koordineli)
  1 naive   , 2 batarya   -> CIPA, beklenen 3.09
  2 NWP     , 2 batarya   -> beklenen 2.76
  3 perfect , 2 batarya   -> beklenen 1.98 (taban)
  4 NWP     , 1 batarya   -> Z7, beklenen deger YOK (on-kayit: hukum yok, olcum)

CIPA KURALI: kol 1 once kosar. |ihlal - 3.09| > 0.15 puan ise DURUR.
  Sebep: yeni dongu eski davranisi birebir uretmiyorsa kalan 3 kol bosa gider.
  Bilerek devam etmek icin --force-continue.

ON-KAYIT (kod yazilmadan once, <author> onayladi - secenek C)
  * "Kapi kapali/acik" dili KULLANILMAZ. Raporlanan cumle:
      "bilgi boslugu = NWP - perfect = X.XX +/- SE puan"
  * Bootstrap: gunluk blok (48 adim), 10.000 tekrar, eslestirilmis.
  * |fark| < 2 SE -> yorumlanmaz.
  * Doygunluk: ASIRI-VOLTAJ ihlallerinde sarj basligi enerji-sinirli miydi?
    Bu bir OLCUM, hukum degil. Hangi sayi cikarsa oldugu gibi yazilir.
  * Kol 4 icin beklenti yok: batarya sayisi etkisi ne cikarsa raporlanir.

CIKTI
  gg2/data/mpc2_ev_<kol>.npz   her kol icin yorunge (kolda basari -> hemen yazilir)
  ekrana: ihlal, kar, depth, doygunluk ayrimi, bootstrap

CALISTIR (kok klasorden, GECE ISI ~3.5 saat):
  python gg2/gg2_mpc2_evidence.py 2>&1 | tee gg2/mpc2_evidence.log

Tek kol tekrar etmek icin:  --arms nwp1
Bitmis kollari atlamak icin: --resume   (npz varsa yeniden kosmaz)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------- sabitler
TARGET_SITE = 1.5          # site basina tepe -> [s,s,w] iki barada = toplam 3.0
ANCHOR = {"naive2": 3.09, "nwp2": 2.76, "perfect2": 1.98}
ANCHOR_TOL = 0.15          # puan
DAY = 48                   # 30-dk adim -> 1 gun
NBOOT = 10000
SEED = 12345
OUTDIR = Path("gg2/data")

ARMS = {
    # etiket    : (mode,      n_batt, fc_anahtari)
    "naive2":    ("gru",      2, "naive"),
    "nwp2":      ("gru",      2, "nwp"),
    "perfect2":  ("perfect",  2, None),
    "nwp1":      ("gru",      1, "nwp"),
    "naive1":    ("gru",      1, "naive"),
}
ORDER = ["naive2", "nwp2", "perfect2", "nwp1", "naive1"]


def _die(m: str) -> None:
    sys.exit("[HATA] " + m)


# ---------------------------------------------------------------- kosum
def run_traj(T, mode: str, n_batt: int, label: str):
    """mpc2_test.run ile AYNI mantik; ek olarak yorunge kaydeder."""
    T0, T1 = T.SPLIT, T.N - 1
    M = T1 - T0
    socA = socB = T.CAP / 2

    a = {k: np.zeros(M) for k in
         ("vmin", "vmax", "pA", "pB", "socA", "socB",
          "loA", "hiA", "loB", "hiB", "price")}
    status = np.zeros(M, dtype=np.int8)   # 0 agent_ok, 1 projected, 2 saturated

    t_start = time.time()
    for j, i in enumerate(range(T0, T1)):
        Hh = min(T.H, T.N - i)
        vmb = np.zeros((Hh, 33))
        pr = np.zeros(Hh)
        for t in range(Hh):
            s, w, dm, pq = T.fcast(i, t, mode)
            vmb[t] = T.vm_at(s, w, dm, 0.0, 0.0)
            pr[t] = pq

        if n_batt == 2:
            pA_plan, pB_plan, _, _, _ = T.solve_window2(
                vmb, T.RA, T.RB, T.S_base, pr, socA0=socA, socB0=socB)
            pA = float(pA_plan[0])
            pB = float(pB_plan[0])
        else:
            pA_plan, _, _ = T.solve_window(vmb, T.RA, T.S_base, pr, soc0=socA)
            pA = float(pA_plan[0])
            pB = 0.0

        pA = float(np.clip(pA, -T.P_MAX, T.P_MAX))
        pB = float(np.clip(pB, -T.P_MAX, T.P_MAX))

        loA = max(-T.P_MAX, -socA * T.ETA / T.dt)
        hiA = min(T.P_MAX, (T.CAP - socA) / (T.ETA * T.dt))
        if n_batt == 2:
            loB = max(-T.P_MAX, -socB * T.ETA / T.dt)
            hiB = min(T.P_MAX, (T.CAP - socB) / (T.ETA * T.dt))
        else:
            loB = hiB = 0.0        # B bataryasi yok

        # gercek sebeke durumu (tahmin degil)
        s0, w0, dm0, _ = T.fcast(i, 0, mode)
        T.vm_at(s0, w0, dm0, 0.0, 0.0)

        # kayit: karar ANINDAKI durum (guncellemeden ONCE)
        a["socA"][j] = socA
        a["socB"][j] = socB
        a["loA"][j] = loA
        a["hiA"][j] = hiA
        a["loB"][j] = loB
        a["hiB"][j] = hiB
        a["price"][j] = T.price[i]

        pA, pB, inf = T.sl.project(pA, pB, loA, hiA, loB, hiB)
        a["vmin"][j] = inf["Vmin"]
        a["vmax"][j] = inf["Vmax"]
        a["pA"][j] = pA
        a["pB"][j] = pB
        status[j] = {"agent_ok": 0, "projected_exact": 1}.get(inf["status"], 2)

        socA = min(max(socA + (T.ETA * pA * T.dt if pA >= 0 else pA * T.dt / T.ETA), 0), T.CAP)
        if n_batt == 2:
            socB = min(max(socB + (T.ETA * pB * T.dt if pB >= 0 else pB * T.dt / T.ETA), 0), T.CAP)

        if (j + 1) % 400 == 0:
            el = (time.time() - t_start) / 60.0
            print("    %s: %d/%d  (%.0f dk, kalan ~%.0f dk)"
                  % (label, j + 1, M, el, el * (M - j - 1) / (j + 1)), flush=True)

    a["status"] = status.astype(float)
    return a


# ---------------------------------------------------------------- metrikler
def summarize(a: dict, T, n_batt: int) -> dict:
    lo, hi = T.LO, T.HI
    vmin, vmax = a["vmin"], a["vmax"]
    ov = vmax > hi
    uv = vmin < lo
    viol = ov | uv

    # kar: p>0 = sarj (satin alma) -> kar = -(fiyat * p) * dt
    ptot = a["pA"] + a["pB"]
    kar = float(-(a["price"] * ptot).sum() * T.dt)

    # depth: p.u. asim toplami (GG-1 tanimi AYRICA dogrulanmali - notu asagida)
    exceed = np.maximum(0.0, vmax - hi) + np.maximum(0.0, lo - vmin)
    depth_sum = float(exceed.sum())
    depth_mean_v = float(exceed[viol].mean()) if viol.any() else 0.0

    # --- doygunluk AYRIMI (totoloji degil, olcum)
    eps = 1e-6
    # asiri voltaj -> daha cok SARJ gerek -> sarj basligi (hi) bakilir
    capA_e = a["hiA"] < T.P_MAX - eps          # A sarj basligi ENERJI-sinirli
    capA_0 = a["hiA"] < 0.05                   # A pratikte DOLU
    if n_batt == 2:
        capB_e = a["hiB"] < T.P_MAX - eps
        capB_0 = a["hiB"] < 0.05
        both_e = capA_e & capB_e
        both_0 = capA_0 & capB_0
    else:
        both_e, both_0 = capA_e, capA_0
    # dusuk voltaj -> DESARJ gerek -> desarj basligi (lo) bakilir
    floA_e = a["loA"] > -T.P_MAX + eps
    if n_batt == 2:
        floB_e = a["loB"] > -T.P_MAX + eps
        bothf_e = floA_e & floB_e
    else:
        bothf_e = floA_e

    n_ov, n_uv = int(ov.sum()), int(uv.sum())
    return {
        "n": len(vmin),
        "viol_pct": float(viol.mean() * 100),
        "n_viol": int(viol.sum()),
        "n_ov": n_ov,
        "n_uv": n_uv,
        "kar": kar,
        "depth_sum": depth_sum,
        "depth_mean_viol": depth_mean_v,
        "sat_status_pct": float((a["status"] >= 2).mean() * 100),
        "ov_energy_pct": float(both_e[ov].mean() * 100) if n_ov else float("nan"),
        "ov_full_pct": float(both_0[ov].mean() * 100) if n_ov else float("nan"),
        "uv_energy_pct": float(bothf_e[uv].mean() * 100) if n_uv else float("nan"),
        "viol": viol.astype(float),
        "profit_step": -(a["price"] * ptot) * T.dt,
    }


def boot_paired(x: np.ndarray, y: np.ndarray, nboot: int = NBOOT) -> tuple:
    """Gunluk-blok eslestirilmis bootstrap: mean(x) - mean(y)."""
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


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Iki-batarya kanit kosumu.")
    ap.add_argument("--arms", default=",".join(ORDER),
                    help="virgul ile: naive2,nwp2,perfect2,nwp1")
    ap.add_argument("--resume", action="store_true", help="npz varsa atla")
    ap.add_argument("--force-continue", action="store_true",
                    help="cipa tutmasa da devam et")
    ap.add_argument("--nwp", default="gg2/data/nwp_solar_fc_capnorm.npz")
    args = ap.parse_args()

    want = [x.strip() for x in args.arms.split(",") if x.strip()]
    for w in want:
        if w not in ARMS:
            _die("bilinmeyen kol: " + w)

    print("=" * 74)
    print("IKI-BATARYA KANIT KOSUMU  (Z4 kar + Z5 hata cubugu + Z6 depth + Z7 kunye)")
    print("=" * 74)
    print("mpc2_test import ediliyor...\n")
    import mpc2_test as T  # noqa: E402

    # ---- olcekleme (total3/floor ile BIREBIR ayni) -----------------------
    old = T.SOLAR_PEAK
    if abs(old - 1.25) > 1e-9:
        _die("mpc2_test.SOLAR_PEAK beklenen 1.25 degil: %s. Dosya degismis, DUR." % old)
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
        _die("sekil uyusmazligi: NWP %s vs naive %s" % (nwp_15.shape, naive_15.shape))

    sl_t = slice(T.SPLIT, T.N - 1)
    for k, (tag, m) in enumerate((("naive", naive_15), ("NWP", nwp_15))):
        r = float(np.nanmean(m[sl_t, 0]) / np.nanmean(T.solar[sl_t]))
        print("KANIT (%d) %-5s fc/gercek orani: %.3f" % (k + 2, tag, r))
        if not (0.7 <= r <= 1.3):
            _die("%s olcegi gercekle uyusmuyor (%.3f), DUR." % (tag, r))
    print("  -> uc kanit da tuttu.\n")

    print("KUNYE: toplam solar 3.0 (site %.1f) | 2 x 6 MWh/3 MW | dilim %d:%d"
          % (TARGET_SITE, T.SPLIT, T.N - 1))
    print("       planlayici W_COST = 1e-3 (opf_expert ve opf_expert2 AYNI)")
    print("       ufuk voltaji = her adim tam runpp | safe layer = SafeLayer2\n")

    FC = {"naive": naive_15, "nwp": nwp_15, None: None}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    res = {}
    raw = {}

    for arm in ORDER:
        if arm not in want:
            continue
        mode, nb, fckey = ARMS[arm]
        out = OUTDIR / ("mpc2_ev_%s.npz" % arm)

        if args.resume and out.exists():
            z = np.load(out)
            a = {k: z[k] for k in z.files}
            print("--- %s: mevcut npz yuklendi (%s)" % (arm, out))
        else:
            if fckey is not None:
                T.solar_fc = FC[fckey]      # TEK degisken: tahmin matrisi
            print("--- %s kosuluyor  (mode=%s, batarya=%d, fc=%s)"
                  % (arm, mode, nb, fckey))
            t0 = time.time()
            a = run_traj(T, mode, nb, arm)
            np.savez(out, **a)
            print("    yazildi: %s  (%.0f dk)" % (out, (time.time() - t0) / 60))

        s = summarize(a, T, nb)
        res[arm] = s
        raw[arm] = a
        print("    ihlal %.2f%% (%d adet: %d asiri, %d dusuk) | kar %.0f EUR"
              % (s["viol_pct"], s["n_viol"], s["n_ov"], s["n_uv"], s["kar"]))

        if arm in ANCHOR:
            d = abs(s["viol_pct"] - ANCHOR[arm])
            print("    CIPA: beklenen %.2f | fark %.2f puan -> %s"
                  % (ANCHOR[arm], d, "TUTTU" if d <= ANCHOR_TOL else "TUTMADI"))
            if arm == "naive2" and d > ANCHOR_TOL and not args.force_continue:
                _die("cipa TUTMADI. Yeni dongu eski davranisi uretmiyor.\n"
                     "       Kalan kollar kosulmadi. Sebep bulunmadan devam edilmez.\n"
                     "       Bilerek devam icin: --force-continue")
        print()

    # ------------------------------------------------------------- tablolar
    print("=" * 74)
    print("TABLO 1 - IHLAL / KAR / DEPTH")
    print("=" * 74)
    print("  %-10s %8s %10s %10s %10s" % ("kol", "ihlal%", "kar(EUR)", "depth_sum", "adet"))
    for arm in ORDER:
        if arm in res:
            s = res[arm]
            print("  %-10s %8.2f %10.0f %10.2f %10d"
                  % (arm, s["viol_pct"], s["kar"], s["depth_sum"], s["n_viol"]))
    print("\n  NOT (Z6): depth_sum = sum(max(0,Vmax-HI) + max(0,LO-Vmin)) p.u.")
    print("  GG-1 tablosundaki 'Depth' bu tanimla AYNI MI, GG-1 script'inden")
    print("  dogrulanmadan tek tabloda birlestirilmez.")

    print("\n" + "=" * 74)
    print("TABLO 2 - DOYGUNLUK AYRIMI  (totolojinin yerine gecen OLCUM)")
    print("=" * 74)
    print("  %-10s %9s %14s %12s %14s"
          % ("kol", "status%", "asiri:enerji%", "asiri:dolu%", "dusuk:enerji%"))
    for arm in ORDER:
        if arm in res:
            s = res[arm]
            print("  %-10s %9.1f %14.1f %12.1f %14.1f"
                  % (arm, s["sat_status_pct"], s["ov_energy_pct"],
                     s["ov_full_pct"], s["uv_energy_pct"]))
    print("\n  status%  : safe_layer 'saturated' orani -> ihlal tanimiyla AYNI (totoloji)")
    print("  enerji%  : ihlal aninda basligin ENERJI siniri yuzunden bagli oldugu oran")
    print("             (guc siniri P_MAX degil). Iki bataryada DA bagli ise sayilir.")
    print("  dolu%    : baslik < 0.05 MW, yani batarya pratikte DOLU")

    # ------------------------------------------------------------- bootstrap
    print("\n" + "=" * 74)
    print("TABLO 3 - BOOTSTRAP (gunluk blok %d adim, %d tekrar, eslestirilmis)"
          % (DAY, NBOOT))
    print("=" * 74)
    pairs = [
        ("bilgi boslugu (ihlal)", "nwp2", "perfect2", "viol", 100.0),
        ("naive - NWP (ihlal)", "naive2", "nwp2", "viol", 100.0),
        ("batarya sayisi (ihlal)", "nwp1", "nwp2", "viol", 100.0),
        ("1bat: naive - NWP (ihlal)", "naive1", "nwp1", "viol", 100.0),
        ("naive: 1bat - 2bat (ihlal)", "naive1", "naive2", "viol", 100.0),
        ("naive - NWP (kar)", "naive2", "nwp2", "profit_step", None),
        ("1bat - 2bat (kar)", "nwp1", "nwp2", "profit_step", None),
    ]
    print("  %-24s %10s %8s %8s  %s" % ("karsilastirma", "fark", "SE", "SE kat", "hukum"))
    for name, A, B, key, scale in pairs:
        if A not in res or B not in res:
            continue
        x, y = res[A][key], res[B][key]
        m, se = boot_paired(x, y)
        if scale is not None:
            m, se = m * scale, se * scale
            unit = "puan"
        else:
            n = res[A]["n"]
            m, se = m * n, se * n      # adim ortalamasi -> toplam EUR
            unit = "EUR"
        k = abs(m) / se if se and se == se and se > 0 else float("nan")
        verdict = "YORUMLANMAZ (<2 SE)" if k < 2 else "GERCEK (>=2 SE)"
        print("  %-24s %+10.2f %8.2f %8.1f  %s  [%s]" % (name, m, se, k, verdict, unit))

    if "nwp2" in res and "perfect2" in res:
        m, se = boot_paired(res["nwp2"]["viol"], res["perfect2"]["viol"])
        print("\n  RAPORLANACAK CUMLE (on-kayit, secenek C):")
        print("    bilgi boslugu = NWP - perfect = %.2f +/- %.2f puan"
              % (m * 100, se * 100))
        print("    'kapi kapali/acik' dili KULLANILMAZ.")

    print("\n" + "=" * 74)
    print("Z7 - BATARYA SAYISI ETKISI (ayni boru hatti, ayni ayar)")
    print("=" * 74)
    if "nwp1" in res and "nwp2" in res:
        print("  1 batarya (NWP) : %.2f%%" % res["nwp1"]["viol_pct"])
        print("  2 batarya (NWP) : %.2f%%" % res["nwp2"]["viol_pct"])
        print("  fark            : %.2f puan" % (res["nwp1"]["viol_pct"] - res["nwp2"]["viol_pct"]))
        print("\n  Bu, tek degisken batarya sayisi olan TEK temiz kiyas.")
        print("  Diger boru hattindaki 11.28%% ile DOGRUDAN kiyaslanamaz:")
        print("  ufuk modeli (LinDistFlow farki vs tam runpp) ve ekonomik agirlik")
        print("  (w_econ=1.0 vs W_COST=1e-3) farkli. Makalede bu ayrim yazilir.")
    else:
        print("  (nwp1 ve nwp2 kollari birlikte kosulmadi)")
    print()


if __name__ == "__main__":
    main()