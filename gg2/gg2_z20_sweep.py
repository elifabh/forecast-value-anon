#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_z20_sweep.py - Z20 ARAMA FAZI: hangi hata istatistigi ihlali sirali ongoruyor?

SORU
  17 kolun her biri icin solar tahmin hatasinin 14 farkli ozeti hesaplanir.
  Her ozetin, kollarin IHLAL ORANI siralamasiyla ne kadar ortustugu olculur.
  Standart MAE bunu yapabiliyor mu, yoksa baska bir istatistik daha mi iyi?

  Bu, DEVIR bolum 4 KATKI 2'nin testidir.

NEDEN DONGUSEL DEGIL
  Hata istatistikleri SADECE (tahmin, gercek) ciftinden hesaplanir.
  Yorunge, voltaj, SOC, safe layer HICBIRI girmiyor. Ihlal orani ayri
  kaynaktan (yorunge dosyalari) geliyor. Iki taraf birbirine bakmiyor.

HIZALAMA CIPASI (gecmezse hicbir sey hesaplanmaz)
  Tahmin matrisinin sutun-ufuk esleslmesi ve alpha penceresi BILINMIYOR.
  Tahmin edilmez: 32 kombinasyon denenir, 14 BILINEN alpha'yi yeniden
  ureten secilir.
    - 6 alpha dosyalarin icinde saklidir (anahtar 'a_mean')
    - 8 plasebo alpha'si DEVIR 3.4 tablosundan
    - naive alpha = 1.000 tanim geregi
  En iyi kombinasyonun maksimum hatasi ALPHA_TOL'u asarsa script DURUR.

ON-KAYIT (sonuclara BAKILMADAN yazildi)
  A1 : |rho| >= 0.70 olan her istatistik ADAYDIR.
  A2 : ANA TEST -> en iyi adayin |rho|'su MAE'nin |rho|'sunu kol-uzeri
       bootstrap ile 2 SE'den fazla asiyor mu?
       ASIYORSA  : "MAE voltaj degerini en iyi siralayan istatistik degil"
                   yazilabilir (yaz blogu dogrulamasina bagli).
       ASMIYORSA : yazilamaz. MAE ile en iyi aday AYIRT EDILEMIYOR.
  A3 : MAE'nin kendi |rho|'su >= 0.70 ise bu da bir BULGUDUR:
       fiyat ekseninde bilinen uyusmazlik voltaj ekseninde GORULMUYOR.
       Daha zayif ama raporlanabilir bir sonuctur.
  A4 : DUYARLILIK -> oracle kolu cikarilinca hukum degisiyorsa,
       sonuc tek bir noktaya baglidir ve BOYLE YAZILIR.
  A5 : uc tahminci seed'i (bm, s1, s7) neredeyse ayni noktadir.
       Etkin n 17'den kucuktur; bu SINIR OLARAK yazilir.

  BU SCRIPT HICBIR SEYI DOGRULAMAZ. Dogrulama yaz blogunda ayri kosulur.
  Yaz blogunda tutmayan istatistik KULLANILMAZ.

CALISTIR (kok klasorden, ~2 dk):
  python gg2/gg2_z20_sweep.py 2>&1 | tee gg2/z20_sweep.log
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mpc_econ_voltage_pareto as M

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALPHA_TOL = 0.010          # hizalama cipasi toleransi
NBOOT_ARMS = 10000
SEED = 20250802
RHO_CAND = 0.70

FC_NWP = "gg2/data/nwp_solar_fc_capnorm.npz"

# tag -> (tahmin matrisi kaynagi, yorunge dosyasi, BILINEN alpha or None)
# alpha kaynagi: 'file' = npz icindeki a_mean | sayi = DEVIR 3.4 | None = anchor degil
ARMS = [
    ("naive",   "NAIVE",                                        "gg2/nwp_fair_traj_naive.npz",         1.000),
    ("oracle",  "ORACLE",                                       "gg2/nwp_fair_traj_oracle.npz",        0.000),
    ("bm",      FC_NWP,                                         "gg2/nwp_fair_traj_nwp.npz",           "file"),
    ("s1",      "gg2/data/nwp_solar_fc_capnorm_s1.npz",         "gg2/nwp_fair_s1_nwp.npz",             "file"),
    ("s7",      "gg2/data/nwp_solar_fc_capnorm_s7.npz",         "gg2/nwp_fair_s7_nwp.npz",             "file"),
    ("ecmwf",   "gg2/data/nwp_solar_fc_capnorm_ecmwf_ifs025.npz", "gg2/nwp_fair_ecmwf_ifs025_nwp.npz", "file"),
    ("gfs",     "gg2/data/nwp_solar_fc_capnorm_gfs_seamless.npz", "gg2/nwp_fair_gfs_seamless_nwp.npz", "file"),
    ("cal",     "gg2/data/nwp_solar_fc_cal.npz",                "gg2/nwp_fair_cal_nwp.npz",            "file"),
    ("calcap",  "gg2/data/nwp_solar_fc_calcap.npz",             "gg2/nwp_fair_calcap_nwp.npz",         "file"),
    ("p1",      "SHIFT:1",  "gg2/data/placebo_shift1.npz",  1.161),
    ("p2",      "SHIFT:2",  "gg2/data/placebo_shift2.npz",  1.287),
    ("p3",      "SHIFT:3",  "gg2/data/placebo_shift3.npz",  1.285),
    ("p4",      "SHIFT:4",  "gg2/data/placebo_shift4.npz",  1.349),
    ("p5",      "SHIFT:5",  "gg2/data/placebo_shift5.npz",  1.324),
    ("p7",      "SHIFT:7",  "gg2/data/placebo_shift7.npz",  1.461),
    ("p14",     "SHIFT:14", "gg2/data/placebo_shift14.npz", 1.291),
    ("p21",     "SHIFT:21", "gg2/data/placebo_shift21.npz", 1.636),
]


def _die(m):
    sys.exit("\n[DUR] " + m + "\n")


def _p(rel_):
    q = ROOT / rel_
    if not q.exists():
        _die("dosya yok: %s" % rel_)
    return q


# ---------------------------------------------------------------- siralama
def _rank(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    # baglari ortala
    i = 0
    xs = x[order]
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def kendall(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = len(a)
    c = d = 0
    for i in range(n - 1):
        da = a[i + 1:] - a[i]
        db = b[i + 1:] - b[i]
        s = np.sign(da) * np.sign(db)
        c += int((s > 0).sum())
        d += int((s < 0).sum())
    return (c - d) / float(c + d) if (c + d) > 0 else 0.0


# ------------------------------------------------------------ yukleyiciler
def load_fc(src, N, K):
    if src == "NAIVE":
        return np.asarray(M.solar_fc, dtype=np.float64)
    if src == "ORACLE":
        return None                      # hizalama secildikten sonra kurulur
    if src.startswith("SHIFT:"):
        d = int(src.split(":")[1])
        base = np.asarray(np.load(_p(FC_NWP))["solar"], dtype=np.float64)
        sh = d * 48
        idx = np.clip(np.arange(base.shape[0]) - sh, 0, base.shape[0] - 1)
        return base[idx]                 # gg2_z16_placebo_multi.py sat.142-143
    z = np.load(_p(src))
    return np.asarray(z["solar"], dtype=np.float64)


def stored_alpha(src):
    if src in ("NAIVE", "ORACLE") or src.startswith("SHIFT:"):
        return None
    z = np.load(_p(src))
    return float(z["a_mean"]) if "a_mean" in z else None


def load_viol(rel_):
    z = np.load(_p(rel_))
    done = int(z["done"])
    v = ((z["mn"][:done] < M.LO) | (z["mx"][:done] > M.HI)).astype(np.float64)
    return 100.0 * v.mean(), done


def build_truth(off, N, K):
    j = np.arange(N)[:, None] + np.arange(K)[None, :] + off
    return M.solar[np.minimum(j, N - 1)]


# ------------------------------------------------------------- metrikler
def metrics(FC, TR, W0, W1, ncol):
    f = FC[W0:W1, :ncol]
    t = TR[W0:W1, :ncol]
    e = f - t
    ae = np.abs(e)
    day = t > 0.01
    out = {}
    out["MAE"] = float(ae.mean())
    out["RMSE"] = float(np.sqrt((e * e).mean()))
    out["BIAS"] = float(e.mean())
    out["ABSBIAS"] = float(abs(e.mean()))
    out["MAE_day"] = float(ae[day].mean()) if day.any() else 0.0
    if day.any():
        thr = np.percentile(t[day], 80.0)
        hi = t >= thr
        out["MAE_top20"] = float(ae[hi].mean()) if hi.any() else 0.0
    else:
        out["MAE_top20"] = 0.0
    out["UNDER"] = float(np.maximum(0.0, -e).mean())     # tahmin GERCEKTEN AZ
    out["OVER"] = float(np.maximum(0.0, e).mean())       # tahmin GERCEKTEN COK
    out["P90"] = float(np.percentile(ae, 90.0))
    out["P99"] = float(np.percentile(ae, 99.0))
    out["MAXERR"] = float(ae.max())
    # gunluk tepe (sutun 0 serisi uzerinden)
    f0 = FC[W0:W1, 0]
    t0 = TR[W0:W1, 0]
    nd = len(f0) // 48
    if nd >= 1:
        fm = f0[:nd * 48].reshape(nd, 48).max(axis=1)
        tm = t0[:nd * 48].reshape(nd, 48).max(axis=1)
        out["PEAK_MAE"] = float(np.abs(fm - tm).mean())
        out["PEAK_BIAS"] = float((fm - tm).mean())
    else:
        out["PEAK_MAE"] = out["PEAK_BIAS"] = 0.0
    out["1-CORR"] = 1.0 - (float(np.corrcoef(f0, t0)[0, 1])
                           if f0.std() > 0 and t0.std() > 0 else 1.0)
    out["1-TAU"] = 1.0 - spearman(f0, t0)
    return out


METRIC_NAMES = ["MAE", "RMSE", "BIAS", "ABSBIAS", "MAE_day", "MAE_top20",
                "UNDER", "OVER", "P90", "P99", "MAXERR", "PEAK_MAE",
                "PEAK_BIAS", "1-CORR", "1-TAU"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=NBOOT_ARMS)
    a = ap.parse_args()

    print("=" * 78)
    print("Z20 ARAMA FAZI - hangi hata istatistigi ihlali sirali ongoruyor?")
    print("=" * 78)
    N = len(M.solar)
    K = M.solar_fc.shape[1]
    print("  N=%d  K=%d  SPLIT=%d  LO=%.3f HI=%.3f" % (N, K, M.SPLIT, M.LO, M.HI))

    # ------------------------------------------------- tahminler + ihlaller
    FCS, VIOL, DONE, KNOWN = {}, {}, {}, {}
    for tag, src, traj, al in ARMS:
        FCS[tag] = load_fc(src, N, K)
        VIOL[tag], DONE[tag] = load_viol(traj)
        if al == "file":
            KNOWN[tag] = stored_alpha(src)
        elif al is not None:
            KNOWN[tag] = float(al)
    nmin = min(DONE.values())
    print("  17 kol yuklendi. en kisa yorunge = %d adim" % nmin)
    print("  bilinen alpha cipasi: %d kol" % len([k for k in KNOWN if KNOWN[k] is not None]))

    # ------------------------------------------------------ hizalama cipasi
    print("\n" + "=" * 78)
    print("HIZALAMA CIPASI - 32 kombinasyon, 14 bilinen alpha ile secim")
    print("=" * 78)
    WINDOWS = [("test2429", M.SPLIT, M.SPLIT + 2429),
               ("test2400", M.SPLIT, M.SPLIT + 2400),
               ("split-son", M.SPLIT, N),
               ("tam", 0, N)]
    best = None
    for off in (0, 1):
        TR = build_truth(off, N, K)
        for wname, w0, w1 in WINDOWS:
            for ncol in (K, 1):
                for mask in ("all", "day"):
                    tr = TR[w0:w1, :ncol]
                    m = (tr > 0.01) if mask == "day" else np.ones_like(tr, bool)
                    if not m.any():
                        continue
                    base = np.abs(FCS["naive"][w0:w1, :ncol] - tr)[m].mean()
                    if base <= 0:
                        continue
                    err = []
                    for tag in KNOWN:
                        if KNOWN[tag] is None or FCS[tag] is None:
                            continue
                        al = np.abs(FCS[tag][w0:w1, :ncol] - tr)[m].mean() / base
                        err.append(abs(al - KNOWN[tag]))
                    if not err:
                        continue
                    sc = max(err)
                    if best is None or sc < best[0]:
                        best = (sc, off, wname, w0, w1, ncol, mask)
    sc, off, wname, W0, W1, NCOL, MASK = best
    print("  en iyi: offset=%d  pencere=%s [%d:%d]  sutun=%s  maske=%s"
          % (off, wname, W0, W1, "hepsi" if NCOL == K else str(NCOL), MASK))
    print("  14 bilinen alpha'da MAKSIMUM hata = %.5f  (tolerans %.3f)" % (sc, ALPHA_TOL))
    if sc > ALPHA_TOL:
        _die("HIZALAMA BULUNAMADI (max hata %.5f > %.3f).\n"
             "      Sutun-ufuk esleslmesi veya alpha penceresi denenen 32\n"
             "      kombinasyonun disinda. Z20 DURDU - hicbir metrik hesaplanmadi.\n"
             "      Gereken: gg2_z16_alpha.py icindeki alpha hesabi." % (sc, ALPHA_TOL))
    print("  -> GECTI. Bu hizalama ile devam ediliyor.")

    TR = build_truth(off, N, K)
    FCS["oracle"] = TR.copy()

    print("\n  alpha yeniden uretimi:")
    tr_ = TR[W0:W1, :NCOL]
    msk = (tr_ > 0.01) if MASK == "day" else np.ones_like(tr_, bool)
    base = np.abs(FCS["naive"][W0:W1, :NCOL] - tr_)[msk].mean()
    print("  %-8s %10s %10s %8s" % ("kol", "bilinen", "olculen", "fark"))
    for tag, _s, _t, _a in ARMS:
        al = np.abs(FCS[tag][W0:W1, :NCOL] - tr_)[msk].mean() / base
        kn = KNOWN.get(tag)
        print("  %-8s %10s %10.4f %8s"
              % (tag, ("%.4f" % kn) if kn is not None else "-", al,
                 ("%+.4f" % (al - kn)) if kn is not None else "-"))

    # --------------------------------------------------------- metrikler
    print("\n" + "=" * 78)
    print("METRIKLER (17 kol x 15 istatistik) ve IHLAL")
    print("=" * 78)
    Mx = {t: metrics(FCS[t], TR, W0, W1, NCOL) for t in FCS}
    tags = [t for t, _s, _tr, _a in ARMS]
    print("  %-8s %7s " % ("kol", "ihlal%") + " ".join("%9s" % m for m in METRIC_NAMES))
    for t in tags:
        print("  %-8s %7.2f " % (t, VIOL[t])
              + " ".join("%9.4f" % Mx[t][m] for m in METRIC_NAMES))

    # ------------------------------------------------------------- rho
    y = np.array([VIOL[t] for t in tags])
    rng = np.random.default_rng(SEED)
    n = len(tags)
    draws = rng.integers(0, n, size=(a.nboot, n))

    def rho_of(mname, idx=None):
        x = np.array([Mx[t][mname] for t in tags])
        if idx is None:
            return spearman(x, y)
        return spearman(x[idx], y[idx])

    print("\n" + "=" * 78)
    print("SPEARMAN rho (istatistik vs ihlal orani), 17 kol")
    print("=" * 78)
    rows = []
    for mname in METRIC_NAMES:
        r = rho_of(mname)
        tau = kendall(np.array([Mx[t][mname] for t in tags]), y)
        rows.append((mname, r, tau))
    rows_sorted = sorted(rows, key=lambda z: -abs(z[1]))
    print("  %-10s %8s %8s   %s" % ("istatistik", "rho", "tau", "durum"))
    for mname, r, tau in rows_sorted:
        st = "ADAY" if abs(r) >= RHO_CAND else ""
        if mname == "MAE":
            st = (st + " <- MAE").strip()
        print("  %-10s %+8.3f %+8.3f   %s" % (mname, r, tau, st))

    r_mae = dict((m, r) for m, r, _t in rows)["MAE"]
    cands = [(m, r) for m, r, _t in rows_sorted if abs(r) >= RHO_CAND and m != "MAE"]

    # ------------------------------------------------ A2: MAE vs en iyi aday
    print("\n" + "=" * 78)
    print("ON-KAYITLI HUKUMLER")
    print("=" * 78)
    print("  A3  MAE'nin kendi rho'su = %+.3f" % r_mae)
    if abs(r_mae) >= RHO_CAND:
        print("      -> MAE ihlali SIRALIYOR. Fiyat ekseninde bilinen uyusmazlik")
        print("         voltaj ekseninde GORULMUYOR. Bu daha zayif ama")
        print("         raporlanabilir bir sonuctur.")
    else:
        print("      -> MAE esigin ALTINDA. Uyusmazlik voltaj ekseninde de var.")

    if not cands:
        print("  A2  |rho| >= %.2f olan baska istatistik YOK." % RHO_CAND)
        print("      -> KATKI 2 icin aday yok. Makale KATKI 1+3 ile gider.")
    else:
        bm_, br_ = cands[0]
        xb = np.array([Mx[t][bm_] for t in tags])
        xm = np.array([Mx[t]["MAE"] for t in tags])
        d = np.array([abs(spearman(xb[i], y[i])) - abs(spearman(xm[i], y[i]))
                      for i in draws])
        dm, ds = float(d.mean()), float(d.std(ddof=1))
        k = abs(dm) / ds if ds > 0 else float("inf")
        print("  A2  en iyi aday = %s  (rho %+.3f) vs MAE (rho %+.3f)" % (bm_, br_, r_mae))
        print("      |rho| farki = %+.3f +/- %.3f   (%.1f SE)" % (dm, ds, k))
        if k >= 2.0 and dm > 0:
            print("      -> AYRISIYOR. 'MAE en iyi siralayici degil' YAZILABILIR,")
            print("         ANCAK yaz blogu dogrulamasina baglidir.")
        else:
            print("      -> AYIRT EDILEMIYOR (%.1f SE). YAZILAMAZ." % k)

    # ---------------------------------------------------- A4 duyarlilik
    print("\n  A4  DUYARLILIK - oracle kolu cikarilinca:")
    idx_no = np.array([i for i, t in enumerate(tags) if t != "oracle"])
    print("      %-10s %8s %8s" % ("istatistik", "17 kol", "16 kol"))
    for mname, r, _t in rows_sorted[:6]:
        print("      %-10s %+8.3f %+8.3f" % (mname, r, rho_of(mname, idx_no)))

    print("\n  A5  SINIR: bm/s1/s7 neredeyse ayni noktadir; etkin n < 17.")
    print("      Bu makalede SINIR olarak yazilir.")
    print("\n  BU SCRIPT HICBIR SEYI DOGRULAMADI. Yaz blogu ayri kosulur.\n")


if __name__ == "__main__":
    main()