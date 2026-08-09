#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_z14_seed_fc.py - Z14 ADIM 1: tahminci seed'i basina NWP tahmin matrisi uretir.

NEDEN
  Makalenin ana sayisi (%11.28 -> %9.80) TEK tahminci seed'i (42) ile uretildi.
  Z10 tahmincinin kendisinin seed'e duyarsiz oldugunu gosterdi (alpha yayilimi
  0.018 < 0.02) AMA kontrol deneyi (MPC) uc seed uzerinde TEKRARLANMADI.
  Z10'un kendi on-kaydinda "MPC yeniden kosulmaz (pahali)" yaziyordu; simdi
  kosuyoruz, dolayisiyla bu YENI bir deneydir ve YENI on-kayit gerektirir.

BU SCRIPT NE YAPAR
  Seed 1 ve 7 icin cap-norm NWP tahmin matrisini uretir ve diske yazar.
  Seed 42 zaten var (nwp_solar_fc_capnorm.npz) - UZERINE YAZILMAZ.
  MPC KOSMAZ. Sadece tahmin matrisi uretir. MPC ayri komutla kosulur.

  Model SECIMI YAPILMAZ. Z9'da validation'da secildi (lgbm), degistirilmiyor.
  Mevcut npz'den okunur; lgbm degilse script durur (Ridge'de seed yoktur).

IKI CIPA (ikisi de gecmeden dosya yazilmaz)
  CIPA A - matris: seed 42 ile yeniden uretilen matris, diskteki
           nwp_solar_fc_capnorm.npz ile ayni mi? (tol 1e-4, float32 yuvarlamasi)
  CIPA B - alpha: uc seed'in TEST alpha'lari Z10 log'undaki degerleri
           uretmeli:  seed 42 -> 0.799 | seed 1 -> 0.781 | seed 7 -> 0.795
           (tol 0.002)

  Herhangi biri kalirsa: HICBIR dosya yazilmaz, hukum verilmez, durulur.

ON-KAYIT (Z14 - ADIM 2 icin, koda girmeden onaylandi)
  Olculecek : ana boru hattinin NWP kolundaki ihlal oraninin uc tahminci
              seed'i (42, 1, 7) uzerindeki YAYILIMI (max - min).
  Esik      : 0.72 puan -- naive-NWP farkinin bootstrap SE'si.
              Gerekce: ana iddiadaki 1.54 puanlik etkinin hata cubugu budur;
              seed yayilimi bunun altindaysa etki seed gurultusunun disindadir.
  HUKUM (1) : yayilim < 0.72 -> "Ana sonuc tahminci seed'ine duyarsiz.
              Makalede uc seed'in ortalamasi +/- yayilimi raporlanir."
  HUKUM (2) : yayilim >= 0.72 -> "Ana sonuc tahminci seed'ine DUYARLI.
              Bu bir SINIRDIR; 1.54 puanlik etki tek seed'e bagli olarak
              sunulamaz ve makalede boyle yazilir."
  Ara durum yok. Sebep uydurulmaz. "Bilmiyorum" gecerli cevaptir.

CALISTIR (GG-1 kok klasorunden, ~5 dk):
  python gg2/gg2_z14_seed_fc.py 2>&1 | tee gg2/z14_seed_fc.log
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*valid feature names.*")

SEEDS = (42, 1, 7)
Z10_ALPHA = {42: 0.799, 1: 0.781, 7: 0.795}   # Z10 log'undan, cipa B
TOL_MAT = 1e-4
TOL_ALPHA = 0.002
SOLAR_H = (10, 16)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _die(m: str) -> None:
    sys.exit("[HATA] " + m)


def load_build_fc():
    """gg2_nwp_build_fc.py'yi dosya yolundan yukler.

    Sebep: ghi30 ve make_X fonksiyonlarini KOPYALAMIYORUZ, orijinalini
    kullaniyoruz. Kopyalarsak zamanla ayrisirlar ve sessiz hata olur.
    Modul seviyesinde sadece import ve def var (main() korumali) -> yan etki yok.
    """
    p = os.path.join(HERE, "gg2_nwp_build_fc.py")
    if not os.path.exists(p):
        _die("bulunamadi: " + p)
    spec = importlib.util.spec_from_file_location("gg2_nwp_build_fc", p)
    if spec is None or spec.loader is None:
        _die("gg2_nwp_build_fc.py yuklenemedi")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("ghi30", "make_X"):
        if not hasattr(mod, fn):
            _die("gg2_nwp_build_fc icinde %s yok -> dosya degismis" % fn)
    return mod


def fit_lgbm_seed(X: np.ndarray, y: np.ndarray, tr: slice, seed: int) -> np.ndarray:
    """gg2_nwp_build_fc.fit_pred'in lgbm dali ile BIREBIR ayni.
    TEK fark: random_state disaridan verilir."""
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                          min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                          random_state=seed, verbose=-1)
    m.fit(X[tr], y[tr])
    return m.predict(X)


def solar_mae_pair(fc: np.ndarray, naive: np.ndarray, truth: np.ndarray,
                   hr: np.ndarray, sl: slice):
    """build_fc'deki m() ile ayni: ortalama MAE ve solar-saat q95."""
    def _m(err, h):
        a = np.abs(err).ravel()
        mm = (h.ravel() >= SOLAR_H[0]) & (h.ravel() <= SOLAR_H[1])
        return float(a.mean()), float(np.quantile(a[mm], 0.95))
    mn, qn = _m(naive[sl] - truth[sl], hr[sl])
    mw, qw = _m(fc[sl] - truth[sl], hr[sl])
    return mn, qn, mw, qw


def main() -> None:
    ap = argparse.ArgumentParser(description="Z14 Adim 1: seed basina NWP tahmin matrisi.")
    ap.add_argument("--joint", default="data/aligned_30min_joint.csv")
    ap.add_argument("--fcmat", default="data/forecast_matrix.npz")
    ap.add_argument("--ghi", default="gg2/data/nwp_ghi_roi.csv")
    ap.add_argument("--ref", default="gg2/data/nwp_solar_fc_capnorm.npz",
                    help="seed 42 referans matrisi (CIPA A)")
    ap.add_argument("--outdir", default="gg2/data")
    ap.add_argument("--force", action="store_true", help="mevcut seed dosyalarini ez")
    a = ap.parse_args()

    for p in (a.joint, a.fcmat, a.ghi, a.ref):
        if not Path(p).exists():
            _die("bulunamadi: " + p + "  (kok klasorden mi calistiriyorsun?)")

    B = load_build_fc()
    SP, SPLIT, CAP_WIN, LAG = B.SP, B.SPLIT, B.CAP_WIN, B.LAG

    print("=" * 74)
    print("Z14 ADIM 1 - SEED BASINA NWP TAHMIN MATRISI")
    print("=" * 74)
    print("  seed'ler: %s" % ", ".join(map(str, SEEDS)))
    print("  SP=%.1f SPLIT=%d CAP_WIN=%d LAG=%d  (gg2_nwp_build_fc'den okundu)"
          % (SP, SPLIT, CAP_WIN, LAG))

    # ------------------------------------------------ referans (Z9 karari)
    ref = np.load(a.ref, allow_pickle=True)
    ref_model = str(ref["model"])
    ref_fc = ref["solar"]
    print("\n  referans: %s | model=%s | shape=%s | a_mean=%.3f"
          % (a.ref, ref_model, ref_fc.shape, float(ref["a_mean"])))
    if ref_model != "lgbm":
        _die("referans model '%s' -- Ridge'de random_state yoktur, seed deneyi "
             "anlamsizdir. Z14 burada durur." % ref_model)

    # ------------------------------------------------------------- veri
    df = pd.read_csv(a.joint, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    N = len(df)
    y = (df.solar_mw / df.iloc[:SPLIT].solar_mw.max() * SP).to_numpy()
    hour = df.index.hour.to_numpy()
    naive = np.load(a.fcmat)["solar"]
    K = int(naive.shape[1])
    print("  joint %d adim | K=%d | test dilimi %d:%d" % (N, K, SPLIT, N - 1))

    if ref_fc.shape != (N, K):
        _die("referans matris sekli %s, beklenen (%d,%d) -> veri degismis"
             % (str(ref_fc.shape), N, K))

    g1 = B.ghi30(df, pd.read_csv(a.ghi, parse_dates=["datetime_utc"]), "p1")
    s = pd.Series(y, index=df.index)
    cap = (s.rolling(CAP_WIN, min_periods=48).max().shift(LAG)
             .bfill().clip(lower=0.05).to_numpy())
    yn = y / cap
    X = B.make_X(df, g1, cap)
    print("  ozellik matrisi: %s" % str(X.shape))

    idx = np.minimum(np.arange(N)[:, None] + np.arange(1, K + 1)[None, :], N - 1)
    truth = y[idx]
    hr = hour[idx]
    sl = slice(SPLIT, N - 1)

    # ------------------------------------------------------- seed dongusu
    print("\n" + "=" * 74)
    print("EGITIM (fit 0:%d, model=lgbm, model SECIMI YAPILMIYOR)" % SPLIT)
    print("=" * 74)

    built = {}
    alphas = {}
    for sd in SEEDS:
        pred = np.clip(fit_lgbm_seed(X, yn, slice(0, SPLIT), sd), 0.0, None) * cap
        fc = pred[idx].astype(np.float32)
        mn, qn, mw, qw = solar_mae_pair(fc.astype(np.float64), naive, truth, hr, sl)
        built[sd] = fc
        alphas[sd] = mw / mn
        print("  seed %-3d  MAE %.4f | alpha %.3f | alpha_kuyruk %.3f"
              % (sd, mw, mw / mn, qw / qn))

    # ------------------------------------------------------------ CIPA A
    print("\n" + "=" * 74)
    print("CIPA A - seed 42 matrisi diskteki referansi yeniden uretiyor mu?")
    print("=" * 74)
    d = float(np.max(np.abs(built[42].astype(np.float64) - ref_fc.astype(np.float64))))
    print("  max mutlak fark = %.3e   (tolerans %.0e)" % (d, TOL_MAT))
    ok_a = d <= TOL_MAT
    print("  -> %s" % ("GECTI" if ok_a else "KALDI"))

    # ------------------------------------------------------------ CIPA B
    print("\n" + "=" * 74)
    print("CIPA B - alpha'lar Z10 log'undaki degerleri uretiyor mu?")
    print("=" * 74)
    ok_b = True
    for sd in SEEDS:
        exp = Z10_ALPHA[sd]
        got = alphas[sd]
        good = abs(got - exp) <= TOL_ALPHA
        ok_b = ok_b and good
        print("  seed %-3d  beklenen %.3f | olculen %.3f | fark %+.4f  -> %s"
              % (sd, exp, got, got - exp, "OK" if good else "SAPMA"))
    print("  -> %s" % ("GECTI" if ok_b else "KALDI"))

    if not (ok_a and ok_b):
        print("\n" + "=" * 74)
        print("DUR. Cipa gecmedi -> HICBIR DOSYA YAZILMADI, hukum verilmez.")
        print("Sebep bulunmadan devam etme. Ciktiyi Claude'a yapistir.")
        print("=" * 74)
        sys.exit(2)

    # ------------------------------------------------------------- yazim
    print("\n" + "=" * 74)
    print("YAZIM (seed 42 ATLANIR - zaten var, uzerine yazilmaz)")
    print("=" * 74)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for sd in SEEDS:
        if sd == 42:
            print("  seed 42 -> atlandi (%s zaten var)" % a.ref)
            continue
        op = outdir / ("nwp_solar_fc_capnorm_s%d.npz" % sd)
        if op.exists() and not a.force:
            print("  seed %-3d -> ZATEN VAR, ezilmedi: %s  (--force ile ez)" % (sd, op))
            continue
        mn, qn, mw, qw = solar_mae_pair(built[sd].astype(np.float64), naive, truth, hr, sl)
        np.savez(op, solar=built[sd], model="lgbm", a_mean=mw / mn,
                 a_tail=qw / qn, K=K, split=SPLIT, seed=sd)
        written.append(str(op))
        print("  seed %-3d -> yazildi: %s" % (sd, op))

    # ------------------------------------------------------- sonraki adim
    print("\n" + "=" * 74)
    print("ADIM 2 - MPC KOSUMU (bu script kosmaz, sen kosarsin)")
    print("=" * 74)
    print("  Seed 42 kolu ZATEN VAR: gg2/nwp_fair_traj_nwp.npz (%9.80 / 10248 EUR)")
    print("  Kalan iki kol (SIRAYLA, her biri saatler surebilir, --resume var):")
    print()
    for sd in (1, 7):
        print("  python gg2/gg2_nwp_mpc_fair.py \\")
        print("      --nwp gg2/data/nwp_solar_fc_capnorm_s%d.npz \\" % sd)
        print("      --out gg2/nwp_fair_s%d.npz 2>&1 | tee gg2/nwp_fair_s%d.log" % (sd, sd))
        print()
    print("  Kesilirse ayni komuta --resume ekle.")
    print("  Iki kosum bitince uc ihlal orani karsilastirilir; hukum yukaridaki")
    print("  on-kayitta (0.72 puan esigi) yazili.\n")
    if written:
        print("  yazilan dosyalar: %s" % ", ".join(written))
    print()


if __name__ == "__main__":
    main()