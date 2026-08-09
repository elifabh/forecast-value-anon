#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_z15_build_models.py - Z15 ADIM 2: hava modeli basina cap-norm tahmin matrisi.

BAGLAM
  Adim 1'de iki ISIMLI hava modeli cekildi (ecmwf_ifs025, gfs_seamless).
  Mevcut tahmin best_match ile uretilmisti (alpha 0.799, ihlal %9.80).
  Bu script ucunu de AYNI boru hattindan gecirir.

TEK DEGISKEN = HAVA MODELI
  Sabit tutulanlar: cap-norm donusumu (Z9 karari), lgbm, seed 42, ozellik seti,
  fit dilimi 0:13800, 6 uzamsal nokta, hepsi gg2_nwp_build_fc'den okunur.
  Model SINIFI yeniden secilmez (ridge vs lgbm). Sebep: secim yapmak yeni bir
  karar katmani ekler; biz sadece hava modelinin etkisini olcuyoruz.
  Bu bir tercih, makalede boyle yazilir.

CIPA (gecmezse hicbir dosya yazilmaz)
  best_match GHI dosyasindan uretilen matris, diskteki
  nwp_solar_fc_capnorm.npz ile AYNI olmali (tol 1e-4) ve alpha 0.799 vermeli.
  Yani boru hattini once bilinen sayiyla dogruluyoruz, sonra yeni girdi veriyoruz.

ON-KAYIT (Adim 3 icin; Adim 1'de yazildi, degistirilmedi)
  Uc modelin de ihlal orani raporlanir. En iyisi SECILMEZ.
  Esik: uc modelin ihlal yayilimi vs 0.72 puan.
  (1) yayilim < 0.72 -> "Etki hava modeli secimine duyarsiz."
  (2) yayilim >= 0.72 -> "Etkinin buyuklugu hava modeline bagli; SINIRDIR."
  Bir model naive'i yenemezse aynen raporlanir, gizlenmez.

CALISTIR (GG-1 kok klasorunden, ~5 dk):
  python gg2/gg2_z15_build_models.py 2>&1 | tee gg2/z15_build_models.log
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

SEED = 42
TOL_MAT = 1e-4
TOL_ALPHA = 0.002
REF_ALPHA = 0.799
SOLAR_H = (10, 16)

HERE = os.path.dirname(os.path.abspath(__file__))

# tag -> GHI csv. Ilk kayit CIPA'dir, sirasi degistirilmez.
SOURCES = [
    ("best_match",   "gg2/data/nwp_ghi_roi.csv"),
    ("ecmwf_ifs025", "gg2/data/nwp_ghi_roi_ecmwf_ifs025.csv"),
    ("gfs_seamless", "gg2/data/nwp_ghi_roi_gfs_seamless.csv"),
]


def _die(m: str) -> None:
    sys.exit("[HATA] " + m)


def load_build_fc():
    """gg2_nwp_build_fc.py'yi dosya yolundan yukler (kopyalamiyoruz)."""
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


def fit_lgbm(X, y, tr, seed):
    """gg2_nwp_build_fc.fit_pred'in lgbm dali ile BIREBIR ayni."""
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                          min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                          random_state=seed, verbose=-1)
    m.fit(X[tr], y[tr])
    return m.predict(X)


def mae_pair(fc, naive, truth, hr, sl):
    def _m(err, h):
        a = np.abs(err).ravel()
        mm = (h.ravel() >= SOLAR_H[0]) & (h.ravel() <= SOLAR_H[1])
        return float(a.mean()), float(np.quantile(a[mm], 0.95))
    mn, qn = _m(naive[sl] - truth[sl], hr[sl])
    mw, qw = _m(fc[sl] - truth[sl], hr[sl])
    return mn, qn, mw, qw


def main() -> None:
    ap = argparse.ArgumentParser(description="Z15 Adim 2: hava modeli basina tahmin matrisi.")
    ap.add_argument("--joint", default="data/aligned_30min_joint.csv")
    ap.add_argument("--fcmat", default="data/forecast_matrix.npz")
    ap.add_argument("--ref", default="gg2/data/nwp_solar_fc_capnorm.npz")
    ap.add_argument("--outdir", default="gg2/data")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    for p in (a.joint, a.fcmat, a.ref):
        if not Path(p).exists():
            _die("bulunamadi: %s  (kok klasorden mi calistiriyorsun?)" % p)
    for tag, g in SOURCES:
        if not Path(g).exists():
            _die("GHI dosyasi yok: %s  (Adim 1'i kostun mu?)" % g)

    B = load_build_fc()
    SP, SPLIT, CAP_WIN, LAG = B.SP, B.SPLIT, B.CAP_WIN, B.LAG

    print("=" * 74)
    print("Z15 ADIM 2 - HAVA MODELI BASINA CAP-NORM TAHMIN MATRISI")
    print("=" * 74)
    print("  TEK DEGISKEN: hava modeli. lgbm/seed42/cap-norm/fit 0:%d SABIT." % SPLIT)

    ref = np.load(a.ref, allow_pickle=True)
    ref_fc = ref["solar"]
    print("  referans: %s | model=%s | a_mean=%.3f"
          % (a.ref, str(ref["model"]), float(ref["a_mean"])))

    df = pd.read_csv(a.joint, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    N = len(df)
    y = (df.solar_mw / df.iloc[:SPLIT].solar_mw.max() * SP).to_numpy()
    hour = df.index.hour.to_numpy()
    naive = np.load(a.fcmat)["solar"]
    K = int(naive.shape[1])
    print("  joint %d adim | K=%d | test %d:%d" % (N, K, SPLIT, N - 1))

    s = pd.Series(y, index=df.index)
    cap = (s.rolling(CAP_WIN, min_periods=48).max().shift(LAG)
             .bfill().clip(lower=0.05).to_numpy())
    yn = y / cap

    idx = np.minimum(np.arange(N)[:, None] + np.arange(1, K + 1)[None, :], N - 1)
    truth = y[idx]
    hr = hour[idx]
    sl = slice(SPLIT, N - 1)

    print("\n" + "=" * 74)
    print("EGITIM")
    print("=" * 74)
    built, alphas, tails = {}, {}, {}
    for tag, gpath in SOURCES:
        g = B.ghi30(df, pd.read_csv(gpath, parse_dates=["datetime_utc"]), "p1")
        X = B.make_X(df, g, cap)
        pred = np.clip(fit_lgbm(X, yn, slice(0, SPLIT), SEED), 0.0, None) * cap
        fc = pred[idx].astype(np.float32)
        mn, qn, mw, qw = mae_pair(fc.astype(np.float64), naive, truth, hr, sl)
        built[tag] = fc
        alphas[tag] = mw / mn
        tails[tag] = qw / qn
        beat = "naive YENILDI" if mw < mn else "naive YENILEMEDI"
        print("  %-14s X%s | MAE %.4f | alpha %.3f | kuyruk %.3f | %s"
              % (tag, str(X.shape), mw, mw / mn, qw / qn, beat))

    # ---------------------------------------------------------------- CIPA
    print("\n" + "=" * 74)
    print("CIPA - best_match boru hattini yeniden uretiyor mu?")
    print("=" * 74)
    d = float(np.max(np.abs(built["best_match"].astype(np.float64) - ref_fc.astype(np.float64))))
    da = abs(alphas["best_match"] - REF_ALPHA)
    print("  matris max fark = %.3e  (tolerans %.0e)" % (d, TOL_MAT))
    print("  alpha           = %.3f  beklenen %.3f  fark %+.4f (tolerans %.3f)"
          % (alphas["best_match"], REF_ALPHA, alphas["best_match"] - REF_ALPHA, TOL_ALPHA))
    ok = (d <= TOL_MAT) and (da <= TOL_ALPHA)
    print("  -> %s" % ("GECTI" if ok else "KALDI"))

    if not ok:
        print("\n" + "=" * 74)
        print("DUR. Cipa gecmedi -> HICBIR DOSYA YAZILMADI, hukum verilmez.")
        print("Ciktiyi Claude'a yapistir, sebep bulunmadan devam etme.")
        print("=" * 74)
        sys.exit(2)

    # --------------------------------------------------------------- yazim
    print("\n" + "=" * 74)
    print("YAZIM (best_match ATLANIR - zaten var)")
    print("=" * 74)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for tag, _ in SOURCES:
        if tag == "best_match":
            print("  %-14s -> atlandi (%s zaten var)" % (tag, a.ref))
            continue
        op = outdir / ("nwp_solar_fc_capnorm_%s.npz" % tag)
        if op.exists() and not a.force:
            print("  %-14s -> ZATEN VAR, ezilmedi: %s  (--force ile ez)" % (tag, op))
            continue
        np.savez(op, solar=built[tag], model="lgbm", a_mean=alphas[tag],
                 a_tail=tails[tag], K=K, split=SPLIT, seed=SEED, nwp_model=tag)
        print("  %-14s -> yazildi: %s" % (tag, op))

    # ------------------------------------------------------- sonraki adim
    print("\n" + "=" * 74)
    print("ADIM 3 - MPC KOSUMU (bu script kosmaz)")
    print("=" * 74)
    print("  best_match kolu ZATEN VAR: %9.80 / 10248 EUR")
    print("  Kalan iki kol (ayri pencerelerde ayni anda kosabilir, ~12 dk):\n")
    for tag, _ in SOURCES[1:]:
        print("  python gg2/gg2_nwp_mpc_fair.py \\")
        print("      --nwp gg2/data/nwp_solar_fc_capnorm_%s.npz \\" % tag)
        print("      --out gg2/nwp_fair_%s.npz 2>&1 | tee gg2/nwp_fair_%s.log" % (tag, tag))
        print()
    print("  alpha ozeti (test blogu):")
    for tag, _ in SOURCES:
        print("    %-14s alpha %.3f | kuyruk %.3f" % (tag, alphas[tag], tails[tag]))
    print()


if __name__ == "__main__":
    main()