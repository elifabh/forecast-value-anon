#!/usr/bin/env python3
"""
GG-2 / DEVIR v8 BOLUM 8 - IS 1b
mv_oberrhein: ZAMAN SERISI uzerinde k kalibrasyonu (termal barindirma olcutu).

Kosum (GG-1 KOK klasorunden, gg2/ icinden DEGIL):
    python gg2/gg2_ob2_kcal.py 2>&1 | tee ob2_kcal.log

ASCII only. Batarya YOK, kontrolcu YOK - sadece sebeke fizigi olculur.

-------------------------------------------------------------------------------
NEDEN BU SCRIPT
-------------------------------------------------------------------------------
gg2_ob1_ksweep.py TEK ANLIK GORUNTU olctu (tepe PV, sabit yuk). Ama:
  1. `mpc_econ_voltage_pareto.py` solar'i EGITIM penceresi tepesine normalize
     ediyor (`solar = df.solar_mw / train.solar_mw.max() * SP`). Test blogunda
     solar bunu ~1.25 kat asiyor. Yani statik tablodaki k, test blogunda
     fiilen 1.25k gibi davraniyor -> statik tabloya BAKILAMAZ.
  2. Talep carpani dmult(t) zamanla degisiyor; termal en kotu adim, tepe PV ile
     dusuk talebin CAKISTIGI adimdir, tek bir anlik goruntu bunu goremez.
Bu script ayni dmult/solar dizilerini `mpc_econ_voltage_pareto`'dan IMPORT eder
(yeniden uretmez) ve degerlendirme penceresinin HER adiminda AC yuk akisi kosar.

-------------------------------------------------------------------------------
KILITLI KARARLAR (bu kosumdan ONCE sabitlendi)
-------------------------------------------------------------------------------
K1. RUZGAR YOK. mv_oberrhein dagitik PV iceriyor, ruzgar icermiyor. Ruzgar tum
    kollarda ORTAK girdi oldugu icin cikarilmasi naive-NWP kiyasini bozmaz.
K2. Batarya bu scriptte YOK. Olcut sebeke barindirma kapasitesidir; batarya
    desarjinin termal etkisi olcut DEGIL, naive kolundan sonra ayrica olculur.
K3. Yuk olcekleme: gonderilen `load.scaling=0.6` UZERINE dmult(t). `LS` YOK.
K4. sgen.scaling=1.0; PV(t) = k * p0 * solar_norm(t),
    solar_norm = M.solar / M.SP  (egitim penceresi tepesi = 1.0).
K5. Degerlendirme penceresi = test blogu, M.SPLIT .. M.N-1 (case33bw ile AYNI).
K6. Ihlaller SADECE bilesen 0'in 20 kV baralarinda sayilir.

-------------------------------------------------------------------------------
ON-KAYIT (kosumdan ONCE yazildi)
-------------------------------------------------------------------------------
OLCUT (fiziksel, sonuc DEGIL):
    k* = degerlendirme penceresinin trafonun IHRACAT yaptigi (p_hv < 0) HER
         adiminda trafo yuklenmesi <= 100.0 %  VE  max hat yuklenmesi <= 100.0 %
         kalmasini saglayan EN BUYUK k.
Bu bir PV BARINDIRMA olcutudur. Ihlal orani HEDEFLENMEZ, k* bulunduktan sonra
OLCULUR ve cikan ne ise raporlanir.

NEDEN SADECE IHRACAT YONU: bilesen 0'in yuku dmult=1.0'da 20.27 MW, trafo
25 MVA -> taban %81. Irlanda talep carpani ~1.23'u astiginda trafo PV OLMADAN
da %100'u asiyor. Bu, Alman fider yukunun Irlanda talep sekliyle modulasyonunun
bir SONUCU, PV'nin degil. Ithalat yonu ayrica ve acikca raporlanir; olcut
PV'nin yarattigi yone baglanir. k=0 referans kosumu bu tabani olcer.

Olcut bu kosumdan sonra GEVSETILMEYECEK. Kisa sureli acil trafo asimina izin
verme secenegi SIMDIDEN reddedilmistir.

Tek istisna: k*'da kontrolsuz asiri voltaj adimi SIFIR cikarsa fider bu deney
icin kullanilamaz ve oyle yazilir.

BAGIMSIZ CAPRAZ KONTROL (olcut DEGIL, sadece teyit):
    case33bw'de PV_tepe / yuk = 3.0 / 1.86 = 1.61.
    mv_oberrhein'de ayni orani veren k = 20.274 * 1.61 / 12.1657 = 2.69.
    k* bu civara duserse iki fider yapisal olarak eslesmis olur.
    Dusmezse ZORLANMAZ; olcut termal kalir.

CIPA (uc adet, hepsi gecmeden hukum verilmez):
    A1  M.SPLIT == 13800 ve degerlendirme penceresi == 2429 adim
    A2  test blogunda HAM M.solar: ortalama ~0.864 MW, tepe ~3.749 MW
        (DEVIR v8 bolum 3.3 ve 14. vaka)
    A3  mv_oberrhein statik: sgen p_mw = 4.0*p0, load.scaling=0.6
        -> vmax 1.08218 / 38 ihlal barasi  (gg2_ob1_ksweep.py ile birebir)
-------------------------------------------------------------------------------
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top


# ----------------------------------------------------------------------------
# SABITLER
# ----------------------------------------------------------------------------
V_HI = 1.05
V_LO = 0.95
MV_KV = 20.0
THERM_LIMIT = 100.0          # trafo ve hat icin, yuzde

K_GRID = [1.00, 1.50, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50]
BISECT_ITERS = 3             # kaba izgaradan sonra k* icin ikiye bolme

ANCHOR_SPLIT = 13800
ANCHOR_NSTEP = 2429
ANCHOR_SOLAR_MEAN = 0.864     # MW, HAM M.solar uzerinde (normalize DEGIL)
ANCHOR_SOLAR_PEAK = 3.749     # MW, HAM M.solar uzerinde
ANCHOR_SOLAR_TOL = 0.005
ANCHOR_STATIC_K = 4.0
ANCHOR_STATIC_VMAX = 1.08218
ANCHOR_STATIC_NOVER = 38
ANCHOR_STATIC_TOL = 5e-4

PROGRESS_EVERY = 500

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "data", "ob2_kcal.csv")
OUT_NPZ = os.path.join(HERE, "data", "ob2_kcal_traj.npz")


def banner(t):
    print("")
    print("=" * 78)
    print(t)
    print("=" * 78)


def sub(t):
    print("")
    print("--- " + t + " " + "-" * max(0, 70 - len(t)))


# ----------------------------------------------------------------------------
# 0. KUNYE
# ----------------------------------------------------------------------------
banner("0. KUNYE / ORTAM")
print("pandapower : %s" % pp.__version__)
print("numpy      : %s" % np.__version__)
print("pandas     : %s" % pd.__version__)
print("python     : %s" % sys.version.split()[0])
try:
    import numba  # noqa: F401
    USE_NUMBA = True
    print("numba      : kurulu (runpp numba=True)")
except Exception:
    USE_NUMBA = False
    print("numba      : YOK (runpp numba=False, daha yavas, sonuc AYNI)")
print("V_HI=%.3f  V_LO=%.3f  THERM_LIMIT=%.1f%%" % (V_HI, V_LO, THERM_LIMIT))
print("K_GRID=%s  BISECT_ITERS=%d" % (K_GRID, BISECT_ITERS))


# ----------------------------------------------------------------------------
# 1. SERILERI IMPORT ET (yeniden URETME - cipa budur)
# ----------------------------------------------------------------------------
banner("1. SERILER - mpc_econ_voltage_pareto'dan IMPORT")
print("NOT: bu import cvxpy + case33bw + forecast_matrix yukler, yavas olabilir.")
print("     Kok klasorden calistirilmazsa 'data/aligned_30min_joint.csv'")
print("     bulunamaz ve HATA verir - bu beklenen davranistir.")
try:
    import mpc_econ_voltage_pareto as M
except Exception as exc:
    print("")
    print("[HATA] import basarisiz: %s: %s" % (type(exc).__name__, exc))
    print("Scripti GG-1 KOK klasorunden calistir:")
    print("    cd ~/Desktop/New\\ folder && python gg2/gg2_ob2_kcal.py")
    sys.exit(1)

T0 = int(M.SPLIT)
T1 = int(M.N) - 1
NSTEP = T1 - T0
SOLAR_NORM = np.asarray(M.solar, dtype=float) / float(M.SP)
PEAK_NORM = ANCHOR_SOLAR_PEAK / float(M.SP)   # test blogu tepe / egitim tepe
DMULT = np.asarray(M.dmult, dtype=float)

print("")
print("M.N        : %d" % M.N)
print("M.SPLIT    : %d" % M.SPLIT)
print("M.SP       : %.3f MW (case33bw solar tepe olcegi)" % M.SP)
print("M.LS       : %.3f  (case33bw yuk yarilama - BURADA KULLANILMAZ)" % M.LS)
print("pencere    : %d:%d  (%d adim)" % (T0, T1, NSTEP))
print("solar_norm : egitim tepe %.4f | test ort %.4f | test tepe %.4f"
      % (SOLAR_NORM[:T0].max(), SOLAR_NORM[T0:T1].mean(), SOLAR_NORM[T0:T1].max()))
print("dmult      : test min %.4f | ort %.4f | maks %.4f"
      % (DMULT[T0:T1].min(), DMULT[T0:T1].mean(), DMULT[T0:T1].max()))
print("NaN kontrol: solar_norm %d | dmult %d"
      % (int(np.isnan(SOLAR_NORM[T0:T1]).sum()), int(np.isnan(DMULT[T0:T1]).sum())))


# ----------------------------------------------------------------------------
# 2. CIPA A1 + A2
# ----------------------------------------------------------------------------
banner("2. CIPA A1 + A2 - SERI KIMLIGI")
A1 = bool(T0 == ANCHOR_SPLIT and NSTEP == ANCHOR_NSTEP)
print("A1 pencere : SPLIT=%d (beklenen %d), adim=%d (beklenen %d) -> %s"
      % (T0, ANCHOR_SPLIT, NSTEP, ANCHOR_NSTEP, "GECTI" if A1 else "KALDI"))

_sm = float(np.asarray(M.solar, dtype=float)[T0:T1].mean())
_sp = float(np.asarray(M.solar, dtype=float)[T0:T1].max())
A2 = bool(abs(_sm - ANCHOR_SOLAR_MEAN) <= ANCHOR_SOLAR_TOL
          and abs(_sp - ANCHOR_SOLAR_PEAK) <= ANCHOR_SOLAR_TOL)
print("A2 solar   : ort %.4f MW (beklenen %.4f) | tepe %.4f MW (beklenen %.4f) -> %s"
      % (_sm, ANCHOR_SOLAR_MEAN, _sp, ANCHOR_SOLAR_PEAK, "GECTI" if A2 else "KALDI"))
if not A2:
    print("     NOT: DEVIR v8 bolum 3.3'teki 0.864 ve 14. vakadaki 3.749")
    print("     sayilari referans alindi. Tutmuyorsa DURUP bana yaz.")


# ----------------------------------------------------------------------------
# 3. mv_oberrhein KURULUMU
# ----------------------------------------------------------------------------
banner("3. mv_oberrhein KURULUMU")
NET = pn.mv_oberrhein()
NET.sgen["scaling"] = 1.0
SGEN_P0 = NET.sgen.p_mw.values.copy()
SGEN_Q0 = NET.sgen.q_mvar.values.copy()
LOAD_SCALING0 = NET.load.scaling.values.copy()

_mg = top.create_nxgraph(NET, respect_switches=True)
_comps = [sorted(int(b) for b in c) for c in top.connected_components(_mg)]
_comps.sort(key=len, reverse=True)
C0_ALL = [b for b in _comps[0] if bool(NET.bus.at[b, "in_service"])]
C0 = [b for b in C0_ALL if float(NET.bus.at[b, "vn_kv"]) == MV_KV]
TIDX = int(NET.trafo[NET.trafo.lv_bus.isin(C0_ALL)].index[0])
LINES0 = list(NET.line[NET.line.from_bus.isin(C0_ALL)
                       & NET.line.to_bus.isin(C0_ALL)].index)
PV_NOM_C0 = float(NET.sgen[NET.sgen.bus.isin(C0_ALL)].p_mw.sum())
LOAD_C0 = float((NET.load[NET.load.bus.isin(C0_ALL)].p_mw
                 * NET.load[NET.load.bus.isin(C0_ALL)].scaling).sum())

print("bilesen 0  : %d bara | olcum kumesi %d bara (%.0f kV)"
      % (len(C0_ALL), len(C0), MV_KV))
print("trafo idx  : %d  sn=%.1f MVA" % (TIDX, float(NET.trafo.at[TIDX, "sn_mva"])))
print("hat        : %d adet" % len(LINES0))
print("PV nominal : %.4f MW (bilesen 0, 93 sgen)" % PV_NOM_C0)
print("yuk        : %.4f MW (bilesen 0, scaling=0.6, dmult=1)" % LOAD_C0)
print("ruzgar     : YOK (karar K1)")
print("batarya    : YOK (karar K2)")

_kx = LOAD_C0 * 1.61 / PV_NOM_C0
print("")
print("capraz kontrol referansi: case33bw PV/yuk = 1.61 -> esdeger k = %.2f" % _kx)


def solve_step(k, s_norm, dm):
    """Tek adim AC yuk akisi.
    Donen: (yakinsadi, vmax, vmin, bmax, trafo%, hat%, p_hv_mw)."""
    NET.sgen["p_mw"] = k * SGEN_P0 * s_norm
    NET.sgen["q_mvar"] = SGEN_Q0
    NET.load["scaling"] = LOAD_SCALING0 * dm
    try:
        pp.runpp(NET, algorithm="nr", init="auto", calculate_voltage_angles=True,
                 max_iteration=100, numba=USE_NUMBA)
    except Exception:
        return False, np.nan, np.nan, -1, np.nan, np.nan, np.nan
    v = NET.res_bus.loc[C0, "vm_pu"]
    if bool(v.isna().any()):
        return False, np.nan, np.nan, -1, np.nan, np.nan, np.nan
    return (True, float(v.max()), float(v.min()), int(v.idxmax()),
            float(NET.res_trafo.at[TIDX, "loading_percent"]),
            float(NET.res_line.loc[LINES0, "loading_percent"].max()),
            float(NET.res_trafo.at[TIDX, "p_hv_mw"]))


# ----------------------------------------------------------------------------
# 4. CIPA A3 - STATIK YENIDEN URETIM
# ----------------------------------------------------------------------------
banner("4. CIPA A3 - gg2_ob1_ksweep.py ILE STATIK ESLESME")
_ok, _vmax, _vmin, _bmax, _tr, _ln, _phv = solve_step(ANCHOR_STATIC_K, 1.0, 1.0)
_nover = int((NET.res_bus.loc[C0, "vm_pu"] > V_HI).sum())
A3 = bool(_ok and abs(_vmax - ANCHOR_STATIC_VMAX) <= ANCHOR_STATIC_TOL
          and _nover == ANCHOR_STATIC_NOVER)
print("k=%.1f, solar_norm=1.0, dmult=1.0" % ANCHOR_STATIC_K)
print("  vmax  : %.5f  (beklenen %.5f, fark %.5f)"
      % (_vmax, ANCHOR_STATIC_VMAX, abs(_vmax - ANCHOR_STATIC_VMAX)))
print("  ihlal : %d bara (beklenen %d)" % (_nover, ANCHOR_STATIC_NOVER))
print("  trafo : %.1f%%   hat: %.1f%%" % (_tr, _ln))
print("A3 -> %s" % ("GECTI" if A3 else "KALDI"))

banner("CIPA OZETI")
print("A1 %s | A2 %s | A3 %s"
      % ("GECTI" if A1 else "KALDI", "GECTI" if A2 else "KALDI",
         "GECTI" if A3 else "KALDI"))
if not (A1 and A2 and A3):
    print("")
    print("EN AZ BIR CIPA KALDI. Kalibrasyon YAPILMAZ - ciktiyi bana yapistir.")
    sys.exit(2)
print("Uc cipa da gecti. Kalibrasyona gecilir.")


# ----------------------------------------------------------------------------
# 5. ZAMAN SERISI KOSUMU
# ----------------------------------------------------------------------------
TRAJ = {}


def run_series(k, label=""):
    """Degerlendirme penceresinde her adim AC yuk akisi. Donen: ozet dict."""
    vmax = np.full(NSTEP, np.nan)
    vmin = np.full(NSTEP, np.nan)
    trl = np.full(NSTEP, np.nan)
    lnl = np.full(NSTEP, np.nan)
    phv = np.full(NSTEP, np.nan)
    bmx = np.full(NSTEP, -1, dtype=int)
    bad = 0
    t0 = time.time()
    for j in range(NSTEP):
        i = T0 + j
        ok, vh, vl, bm, tr, ln, ph = solve_step(k, float(SOLAR_NORM[i]),
                                                 float(DMULT[i]))
        if not ok:
            bad += 1
            continue
        vmax[j] = vh
        vmin[j] = vl
        trl[j] = tr
        lnl[j] = ln
        phv[j] = ph
        bmx[j] = bm
        if (j + 1) % PROGRESS_EVERY == 0:
            el = time.time() - t0
            rate = (j + 1) / max(el, 1e-9)
            print("    k=%.3f%s  %d/%d  (%.0f adim/dk, kalan ~%.1f dk)"
                  % (k, label, j + 1, NSTEP, rate * 60,
                     (NSTEP - j - 1) / max(rate, 1e-9) / 60), flush=True)

    good = ~np.isnan(vmax)
    ng = int(good.sum())
    exp = good & (phv < 0.0)          # trafo IHRACAT yapiyor -> PV baskin
    imp = good & (phv >= 0.0)
    ne, ni = int(exp.sum()), int(imp.sum())

    def _mx(arr, msk):
        return float(arr[msk].max()) if int(msk.sum()) else np.nan

    res = {
        "k": k,
        "n_step": NSTEP,
        "n_ok": ng,
        "n_fail": bad,
        "n_export": ne,
        "n_import": ni,
        "tr_max_exp": _mx(trl, exp),
        "ln_max_exp": _mx(lnl, exp),
        "tr_max_imp": _mx(trl, imp),
        "ln_max_imp": _mx(lnl, imp),
        "tr_argmax_step": int(T0 + int(np.where(exp)[0][int(np.argmax(trl[exp]))]))
        if ne else -1,
        "tr_max_all": float(np.nanmax(trl)) if ng else np.nan,
        "vmax_max": float(np.nanmax(vmax)) if ng else np.nan,
        "vmin_min": float(np.nanmin(vmin)) if ng else np.nan,
        "over_pct": float((vmax[good] > V_HI).mean() * 100) if ng else np.nan,
        "under_pct": float((vmin[good] < V_LO).mean() * 100) if ng else np.nan,
        "any_viol_pct": float(((vmax[good] > V_HI) | (vmin[good] < V_LO)).mean() * 100)
        if ng else np.nan,
        "bus147_share": float((bmx[good & (bmx >= 0)] == 147).mean() * 100)
        if ng else np.nan,
        "thermal_ok": bool(ne > 0 and _mx(trl, exp) <= THERM_LIMIT
                           and _mx(lnl, exp) <= THERM_LIMIT),
        "secs": round(time.time() - t0, 1),
    }
    TRAJ["vmax_k%.3f" % k] = vmax
    TRAJ["trl_k%.3f" % k] = trl
    TRAJ["lnl_k%.3f" % k] = lnl
    TRAJ["phv_k%.3f" % k] = phv
    return res


def show(r):
    print("  k=%5.3f | IHRACAT trafo %6.1f%% hat %6.1f%% (%d adim) | "
          "ithalat trafo %6.1f%% | asiri %5.2f%% | dusuk %5.2f%% | vmax %.4f | "
          "%s | %.0f s"
          % (r["k"], r["tr_max_exp"], r["ln_max_exp"], r["n_export"],
             r["tr_max_imp"], r["over_pct"], r["under_pct"], r["vmax_max"],
             "TERMAL OK" if r["thermal_ok"] else "TERMAL ASIM",
             r["secs"]), flush=True)


banner("5a. REFERANS KOSUMU - k=0 (PV YOK)")
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
print("Sebekenin PV OLMADAN termal tabani. Ithalat yonu asimini burada olcuyoruz.")
R0 = run_series(0.0, label=" (k=0 ref)")
print("  ithalat trafo en yuksek : %.1f%%   (adim sayisi %d)"
      % (R0["tr_max_imp"], R0["n_import"]))
print("  ihracat adimi sayisi    : %d" % R0["n_export"])
print("  asiri voltaj            : %.2f%%   dusuk voltaj: %.2f%%"
      % (R0["over_pct"], R0["under_pct"]))
if R0["tr_max_imp"] > THERM_LIMIT:
    print("")
    print("  NOT: trafo PV OLMADAN da yuzde 100'u asiyor. Bu, Alman fider")
    print("  yukunun Irlanda talep sekliyle modulasyonundan gelir, PV'den DEGIL.")
    print("  Makalede acikca yazilacak. Olcut ihracat yonune baglidir.")

banner("5b. KABA IZGARA - ZAMAN SERISI")
print("Her k icin %d adim AC yuk akisi. Bu uzun surer." % NSTEP)
RES = [R0]
for _k in K_GRID:
    print("")
    print("  --> k=%.2f baslatiliyor (PV tepe %.2f MW = k*1.2497*%.4f)"
          % (_k, _k * PEAK_NORM * PV_NOM_C0, PV_NOM_C0))
    _r = run_series(_k)
    RES.append(_r)
    show(_r)
    pd.DataFrame(RES).to_csv(OUT_CSV, index=False)


# ----------------------------------------------------------------------------
# 6. IKIYE BOLME
# ----------------------------------------------------------------------------
banner("6. k* ICIN IKIYE BOLME")
_cand = [r for r in RES if r["k"] > 0.0]
_okk = [r["k"] for r in _cand if r["thermal_ok"]]
_bad = [r["k"] for r in _cand if not r["thermal_ok"]]

if not _okk:
    print("Izgaradaki HICBIR k termal olcutu saglamadi (en kucuk k=%.2f bile)."
          % min(K_GRID))
    print("k* < %.2f. Izgara asagi genisletilmeli - DUR ve bana yaz." % min(K_GRID))
    KSTAR = None
elif not _bad:
    print("Izgaradaki HER k termal olcutu sagladi (en buyuk k=%.2f dahil)."
          % max(K_GRID))
    print("k* > %.2f. Izgara yukari genisletilmeli - DUR ve bana yaz." % max(K_GRID))
    KSTAR = None
else:
    lo, hi = max(_okk), min(_bad)
    print("baslangic araligi: k_ok=%.3f  k_asim=%.3f" % (lo, hi))
    for _it in range(BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        print("")
        print("  --> ikiye bolme %d/%d, k=%.3f" % (_it + 1, BISECT_ITERS, mid))
        _r = run_series(mid, label=" (bisect)")
        RES.append(_r)
        show(_r)
        pd.DataFrame(RES).to_csv(OUT_CSV, index=False)
        if _r["thermal_ok"]:
            lo = mid
        else:
            hi = mid
    KSTAR = lo
    print("")
    print("k* = %.3f   (cozunurluk %.3f)" % (KSTAR, hi - lo))


# ----------------------------------------------------------------------------
# 7. TABLO + HUKUM
# ----------------------------------------------------------------------------
banner("7. TAM TABLO")
DF = pd.DataFrame(RES).sort_values("k")
print("IHRACAT yonu = trafo p_hv < 0. Termal olcut SADECE bu sutunlara bakar.")
print("")
print("%-7s %-9s %-10s %-9s %-10s %-8s %-8s %-8s %-8s %s"
      % ("k", "PV_tepe", "ihr_trafo%", "ihr_hat%", "ith_trafo%", "ihr_adim",
         "asiri%", "dusuk%", "vmax", "termal"))
for _, r in DF.iterrows():
    print("%-7.3f %-9.2f %-10.1f %-9.1f %-10.1f %-8d %-8.2f %-8.2f %-8.4f %s"
          % (r["k"], r["k"] * PEAK_NORM * PV_NOM_C0, r["tr_max_exp"],
             r["ln_max_exp"], r["tr_max_imp"], r["n_export"], r["over_pct"],
             r["under_pct"], r["vmax_max"],
             ("REF" if r["k"] == 0.0 else ("OK" if r["thermal_ok"] else "ASIM"))))

banner("8. HUKUM")
if KSTAR is None:
    print("HUKUM VERILMEDI - izgara araligi yetersiz (bkz. bolum 6).")
else:
    _s = DF[np.isclose(DF["k"], KSTAR)].iloc[0]
    print("k*                       : %.3f" % KSTAR)
    print("PV tepe (test blogu)     : %.2f MW  (bilesen 0 yuku %.2f MW)"
          % (KSTAR * PEAK_NORM * PV_NOM_C0, LOAD_C0))
    print("PV_tepe / yuk            : %.2f   (case33bw: 1.61)"
          % (KSTAR * PEAK_NORM * PV_NOM_C0 / LOAD_C0))
    print("IHRACAT trafo en yuksek  : %.1f%%  (adim %d, %d ihracat adimi)"
          % (_s["tr_max_exp"], _s["tr_argmax_step"], _s["n_export"]))
    print("IHRACAT hat en yuksek    : %.1f%%" % _s["ln_max_exp"])
    print("ithalat trafo en yuksek  : %.1f%%  (PV disi, k=0'da %.1f%%)"
          % (_s["tr_max_imp"], R0["tr_max_imp"]))
    print("")
    print("KONTROLSUZ (batarya yok) ihlal oranlari:")
    print("  asiri voltaj  : %.2f%%" % _s["over_pct"])
    print("  dusuk voltaj  : %.2f%%" % _s["under_pct"])
    print("  herhangi biri : %.2f%%" % _s["any_viol_pct"])
    print("  vmax barasi 147 olan adimlarin orani: %.1f%%" % _s["bus147_share"])
    print("")
    print("Kiyas (case33bw, DEVIR v8): kontrolsuz do-nothing ~%22-23,")
    print("naive MPC + safe layer %11.28. Oran kabaca yariya iniyor.")
    print("")
    if _s["over_pct"] == 0.0:
        print("ON-KAYITLI ISTISNA ATESLENDI: k*'da asiri voltaj adimi SIFIR.")
        print("mv_oberrhein termal fizibil bir PV kurulumunda olculebilir asiri")
        print("voltaj URETMIYOR. Ikinci fider plani yeniden degerlendirilmeli.")
    else:
        print("Olcut karsilandi. k*=%.3f kilitlenir; ihlal orani HEDEFLENMEDI," % KSTAR)
        print("olculdu. Sonraki is: VM_BASE + Rcol on-hesabi bu k*'da.")

    print("")
    print("SINIR: bu oran BATARYASIZ ve KONTROLSUZ ust sinirdir. Naive MPC +")
    print("safe layer kolunda daha dusuk cikacaktir. Batarya desarjinin trafo")
    print("uzerindeki etkisi bu olcute GIRMEDI, naive kolundan sonra olculecek.")


# ----------------------------------------------------------------------------
# 9. KAYIT
# ----------------------------------------------------------------------------
banner("9. KAYIT")
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
DF.to_csv(OUT_CSV, index=False)
np.savez_compressed(OUT_NPZ, start=T0, n_step=NSTEP,
                    kstar=(KSTAR if KSTAR is not None else np.nan), **TRAJ)
print("yazildi: %s  (%d satir)" % (OUT_CSV, len(DF)))
print("yazildi: %s  (%d dizi)" % (OUT_NPZ, len(TRAJ)))
print("")
print("CIPA A1/A2/A3 : GECTI / GECTI / GECTI")
print("k*            : %s" % ("%.3f" % KSTAR if KSTAR is not None else "BELIRLENEMEDI"))
print("")
print("BITTI.")