#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_nwp_mpc_fair.py - ADIM 5: NWP tahmini, RAPORDAKI ADIL MPC uzerinde.

NEDEN BU SCRIPT: Adim 4b NWP'yi opf_expert kontrolcusunde test etti (10.62 -> 9.18).
Ama makalenin Tablo 1'indeki "Fair MPC" BASKA bir kontrolcu (9.681 EUR @ %11.28).
Duvar Tablo 1'de tanimli; duvari BASKA bir kontrolcude yikmak hakemin ilk sorusu olur.
Bu script duvari kendi tanimlandigi kontrolcude test eder.

KONTROLCU KUNYESI (her sayi bu dort alanla kaydedilir):
  script          : diag_mpc_forecastvolt.fc_windows_ldf  (BIREBIR ayni fonksiyon mantigi)
  voltaj penceresi: VM_BASE[i] (gercek simdi, mesru) + LDF farki (TAHMINDEN)
  solar tahmini   : --use-naive ise forecast_matrix (naive) | degilse NWP cap-norm
  w_econ          : 1.0   (diag_mpc_forecastvolt ile birebir)
TEK DEGISKEN solar tahminidir. Digerleri sabit -> kiyas temiz.

ON-KAYIT (kosumdan ONCE):
  DOGRULAMA : --use-naive  ->  %11.28 ve ~9681 EUR uretmeli (Tablo 1).
              Uretmezse boru hatti supheli, HICBIR hukum verilmez.
  ESIK      : duvar yikildi sayilmasi icin ihlal < %11.28 belirgin olcude dusmeli.
              Adim 4b'deki formulle: bosluğun >=%50'si kapanmali.
              (Bu kontrolcunun solar-oracle ucu bilinmiyor -> once --use-oracle ile
               olculur, sonra esik hesaplanir. Esik SONUCA BAKILMADAN sabitlenir.)

KESINTIYE DAYANIKLI: her 400 adimda ara kayit; --resume ile devam.

CALISTIR (GG-1 kok klasorunden, SIRAYLA):
  python gg2/gg2_nwp_mpc_fair.py --use-naive     # 1) dogrulama: %11.28 cikmali
  python gg2/gg2_nwp_mpc_fair.py --use-oracle    # 2) alt sinir: solar-oracle ucu
  python gg2/gg2_nwp_mpc_fair.py                 # 3) asil kosum: NWP
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOT: bu iki import modul seviyesinde veri yukler (CSV/npz/pandapower net).
# Diske YAZMAZ - kontrol edildi. Yavas olabilir, normal.
import mpc_econ_voltage_pareto as M
import diag_mpc_forecastvolt as D

W_ECON = 1.0            # diag_mpc_forecastvolt ile BIREBIR
CKPT_EVERY = 400
TABLE1_VIOL = 11.28     # on-kayit: Tablo 1 fair MPC
TABLE1_PROFIT = 9681.0


def _die(m: str) -> None:
    sys.exit(f"[HATA] {m}")


def fc_windows_ldf(i, Hh, solar_fc):
    """diag_mpc_forecastvolt.fc_windows_ldf ile BIREBIR ayni.
    TEK fark: solar_fc disaridan verilir (naive / oracle / NWP)."""
    v_now = D.ldf_v(M.solar[i], M.wind[i], M.dmult[i])
    vmb = np.empty((Hh, D.n))
    vmb[0] = M.VM_BASE[i]
    for t in range(1, Hh):
        kk = min(t - 1, solar_fc.shape[1] - 1)
        v_fut = D.ldf_v(solar_fc[i, kk], M.wind_fc[i, kk], M.dmult_fc[i, kk])
        vmb[t] = M.VM_BASE[i] + (v_fut - v_now)
    pr = np.empty(Hh)
    pr[0] = M.price[i]
    for t in range(1, Hh):
        kk = min(t - 1, M.price_fc.shape[1] - 1)
        pr[t] = M.price_fc[i, kk]
    return vmb, pr


def main() -> None:
    ap = argparse.ArgumentParser(description="GG-2 Adim 5: NWP uzerinde adil MPC.")
    ap.add_argument("--nwp", default="gg2/data/nwp_solar_fc_capnorm.npz")
    ap.add_argument("--out", default="gg2/nwp_fair_traj.npz")
    ap.add_argument("--resume", action="store_true")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--use-naive", action="store_true",
                   help="DOGRULAMA: naive tahmin -> %%11.28 / ~9681 EUR uretmeli")
    g.add_argument("--use-oracle", action="store_true",
                   help="ALT SINIR: mukemmel solar (GERCEKCI DEGIL, sadece esik icin)")
    a = ap.parse_args()

    T0, T1 = M.SPLIT, M.N - 1
    n = T1 - T0
    K = M.solar_fc.shape[1]

    # --- solar tahmini secimi (TEK degisken)
    if a.use_naive:
        idx = None
        solar_fc = M.solar_fc
        tag, suffix = "daily-naive (DOGRULAMA)", "_naive"
    elif a.use_oracle:
        idx = np.minimum(np.arange(M.N)[:, None] + np.arange(1, K + 1)[None, :], M.N - 1)
        solar_fc = M.solar[idx]                    # mukemmel solar
        tag, suffix = "solar-ORACLE (alt sinir)", "_oracle"
    else:
        p = Path(a.nwp)
        if not p.exists():
            _die(f"bulunamadi: {p}  (Adim 4a'yi calistirdin mi?)")
        z = np.load(p)
        solar_fc = z["solar"]
        tag, suffix = f"NWP cap-norm ({z['model']}, alpha={float(z['a_mean']):.3f})", "_nwp"

    if solar_fc.shape != M.solar_fc.shape:
        _die(f"sekil uyusmuyor: {solar_fc.shape} vs {M.solar_fc.shape}")

    out = Path(str(a.out).replace(".npz", suffix + ".npz"))

    print("=" * 74)
    print("ADIM 5 - KONTROLCU KUNYESI")
    print("=" * 74)
    print(f"  script           : diag_mpc_forecastvolt.fc_windows_ldf (birebir)")
    print(f"  voltaj penceresi : VM_BASE[i] + LDF farki (TAHMINDEN)  [adil]")
    print(f"  solar tahmini    : {tag}")
    print(f"  w_econ           : {W_ECON}")
    print(f"  dilim            : {T0}:{T1} ({n} adim) | H={M.H} | K={K}")
    print(f"  ON-KAYIT (Tablo 1 fair MPC): %{TABLE1_VIOL:.2f} | {TABLE1_PROFIT:.0f} EUR")
    print("=" * 74)

    mn = np.zeros(n); mx = np.zeros(n); P = np.zeros(n); SOC = np.zeros(n)
    soc = M.CAP / 2
    j0, nsat = 0, 0

    if a.resume and out.exists():
        c = np.load(out); done = int(c["done"])
        if done >= n:
            print("  ara kayit tamamlanmis -> sadece ozet.\n")
            mn, mx, P, SOC = c["mn"], c["mx"], c["p"], c["soc"]; j0 = n
        else:
            mn[:done], mx[:done] = c["mn"][:done], c["mx"][:done]
            P[:done], SOC[:done] = c["p"][:done], c["soc"][:done]
            soc = float(SOC[done - 1]); j0 = done
            print(f"  DEVAM: {done}/{n}, soc={soc:.3f}\n")
    elif out.exists():
        _die(f"cikti zaten var: {out}  (--resume ile devam, ya da sil)")

    t0 = time.time()
    for j in range(j0, n):
        i = T0 + j
        Hh = min(M.H, M.N - i)
        vmb, pr = fc_windows_ldf(i, Hh, solar_fc)
        p_plan, _ = M.solve_window(vmb, pr, soc, W_ECON)
        p_plan = float(np.clip(p_plan, -M.P_MAX, M.P_MAX))
        p_lo = max(-M.P_MAX, -soc * M.ETA / M.dt)
        p_hi = min(M.P_MAX, (M.CAP - soc) / (M.ETA * M.dt))
        p_plan = float(np.clip(p_plan, p_lo, p_hi))
        M.set_grid(M.solar[i], M.wind[i], M.dmult[i], 0.0)
        p_safe, sinfo = M.sl.project(p_plan, p_lo, p_hi)
        if sinfo["status"] == "saturated":
            nsat += 1
        mn[j] = sinfo["Vmin"]; mx[j] = sinfo["Vmax"]; P[j] = p_safe
        soc = min(max(soc + (M.ETA * p_safe * M.dt if p_safe >= 0 else p_safe * M.dt / M.ETA),
                      0), M.CAP)
        SOC[j] = soc
        if (j + 1) % CKPT_EVERY == 0:
            el = time.time() - t0
            r = (j + 1 - j0) / max(el, 1e-9)
            print(f"  {j+1}/{n}  ({r*60:.0f} adim/dk, kalan ~{(n-j-1)/max(r,1e-9)/60:.0f} dk)",
                  flush=True)
            np.savez(out, mn=mn, mx=mx, p=P, soc=SOC, start=T0, done=j + 1)

    np.savez(out, mn=mn, mx=mx, p=P, soc=SOC, start=T0, done=n)

    V = ((mn < M.LO) | (mx > M.HI)).mean() * 100
    kar = float(-(M.price[T0:T1] * P).sum() * M.dt)

    print("\n" + "=" * 74)
    print(f"SONUC - {tag}")
    print("=" * 74)
    print(f"  ihlal {V:.2f}%  |  kar {kar:.1f} EUR  |  saturated {nsat}")
    print(f"  yazildi: {out}")

    if a.use_naive:
        dv, dk = abs(V - TABLE1_VIOL), abs(kar - TABLE1_PROFIT)
        print("\n  DOGRULAMA HUKMU:")
        if dv <= 0.15 and dk <= 100:
            print(f"    GECTI - Tablo 1 yeniden uretildi ({V:.2f}% vs {TABLE1_VIOL}%, "
                  f"{kar:.0f} vs {TABLE1_PROFIT:.0f} EUR).")
            print("    Boru hatti dogrulandi. --use-oracle ve asil kosuma gecilebilir.")
        else:
            print(f"    KALDI - Tablo 1 uretilemedi (fark: {dv:.2f} puan, {dk:.0f} EUR).")
            print("    Sebep bulunmadan HICBIR hukum verilmez. Devam etme.")
    else:
        print(f"\n  kiyas (ayni kontrolcu, tek degisken solar tahmini):")
        print(f"    naive  : %{TABLE1_VIOL:.2f} | {TABLE1_PROFIT:.0f} EUR  [Tablo 1]")
        print(f"    bu kosum: %{V:.2f} | {kar:.0f} EUR")
        if not a.use_oracle:
            print("\n    Esik icin --use-oracle sonucunu bekle; esik SONUCA BAKILMADAN")
            print("    hesaplanir: naive - 0.5*(naive - oracle).")
    print()


if __name__ == "__main__":
    main()