#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_nwp_mpc.py - ADIM 4b: GERCEK NWP solar tahminiyle MPC. KESIN CEVAP.

SORU: Hava-tabanli (NWP) gun-oncesi solar tahmin, MPC'nin ihlal duvarini yikar mi?

NEDEN BU SCRIPT VAR: gg2_solar_alpha_sweep.py "hata TEKDUZE %X kuculurse ne olur"
sorusunu cevapladi (fc_a = truth + alpha*(fc-truth) -> her hata alpha ile carpilir,
kuyruk dahil). Ama NWP'nin persistence'a ustunlugu tekduze DEGIL. Yani supurme
vekili olcuyordu, sorunun kendisini degil. Bu script sorunun kendisini olcer:
gercek NWP tahminini MPC'ye verir.

FARK: supurmeyle BIREBIR AYNI (ayni sabitler, ayni dongu, ayni safe layer, ayni
solve_window, ayni T0/T1). TEK degisiklik: fc_a yerine gercek NWP matrisi.

ON-KAYIT (deneyden ONCE yazildi, degistirilmez):
  ESIK   : ihlal <= 9.68%  (GAP_TARGET = W_REF - 0.5*(W_REF - 8.73), supurmeyle ayni)
  GECER  : NWP duvari yikar -> Aday C YASAR, yon acik
  KALIR  : NWP duvari yikamaz -> Aday C OLUR, ve bu sefer vekille degil,
           gercek tahminle olculmus olur
  BOSLUK : tek deterministik model (ensemble degil); 6 nokta uzamsal ornekleme;
           radyasyon->MW donusumu kendi hatasini katar. Adim 4a'da tahminci
           naive'i YENDIGI icin (alpha<1) bu boslugun hukmu tasima riski dustu.

KESINTIYE DAYANIKLI: her 400 adimda ara kayit; --resume ile kaldigi yerden devam.

CALISTIR (GG-1 kok klasorunden):
  python gg2/gg2_nwp_mpc.py            # bastan
  python gg2/gg2_nwp_mpc.py --resume   # kesildiyse devam
CIKTI: gg2/nwp_mpc_traj.npz  (GG-1 dosyalarina YAZILMAZ)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandapower as pp
import pandapower.networks as nw

from opf_expert import solve_window, H, CAP, dt, ETA
from safe_layer import SafeLayer

# --- gg2_solar_alpha_sweep.py ile BIREBIR AYNI sabitler
LS = 0.5
SP, WP = 3.0, 2.0
SB, WB, BB = 17, 32, 17
P_MAX = 3.0
SPLIT = 13800
LO, HI = 0.95, 1.05
W_REF = 10.62                              # alpha=1 ucu (on-kayit)
V_SOLAR_OR = 8.69                          # alpha=0 ucu (on-kayit)
GAP_TARGET = W_REF - 0.5 * (W_REF - 8.73)  # 9.685 -> bosluğun yarisi
CKPT_EVERY = 400

Rcol_G = None
S_base_G = None


def _die(m: str) -> None:
    sys.exit(f"[HATA] {m}")


def main() -> None:
    global Rcol_G, S_base_G
    ap = argparse.ArgumentParser(description="GG-2 Adim 4b: gercek NWP + MPC.")
    ap.add_argument("--joint", default="data/aligned_30min_joint.csv")
    ap.add_argument("--fcmat", default="data/forecast_matrix.npz")
    ap.add_argument("--nwp", default="gg2/data/nwp_solar_fc_capnorm.npz")
    ap.add_argument("--out", default="gg2/nwp_mpc_traj.npz")
    ap.add_argument("--resume", action="store_true", help="ara kayittan devam et")
    ap.add_argument("--use-naive", action="store_true",
                    help="DOGRULAMA MODU: NWP yerine daily-naive kullan. "
                         "Sonuc W_REF (10.62%%) cikmali; cikmazsa boru hatti supheli.")
    a = ap.parse_args()

    for p in (a.joint, a.fcmat, a.nwp):
        if not Path(p).exists():
            _die(f"bulunamadi: {p}  (Adim 4a'yi calistirdin mi?)")

    df = pd.read_csv(a.joint, index_col=0, parse_dates=True)
    N = len(df)
    train = df.iloc[:SPLIT]
    solar = (df.solar_mw / train.solar_mw.max() * SP).to_numpy()
    wind = (df.wind_mw / train.wind_mw.max() * WP).to_numpy()
    dmult = (df.demand_mw / train.demand_mw.mean()).to_numpy()
    price = df.price_eur_mwh.to_numpy()

    d = np.load("data/opf_base_vm.npz")
    Rcol_G = d["Rcol"]
    S_base_G = float(d["S_base"])

    f = np.load(a.fcmat)
    wind_fc, dmult_fc, price_fc = f["wind"], f["dmult"], f["price"]
    naive_fc = f["solar"]

    z = np.load(a.nwp)
    if a.use_naive:
        solar_fc = naive_fc                  # DOGRULAMA: W_REF'i yeniden uretmeli
        tag = "daily-naive (DOGRULAMA MODU)"
    else:
        solar_fc = z["solar"]                # <-- TEK DEGISIKLIK: gercek NWP
        tag = "GERCEK NWP (cap-norm, day1)"
    K = solar_fc.shape[1]
    if solar_fc.shape != naive_fc.shape:
        _die(f"sekil uyusmuyor: NWP {solar_fc.shape} vs naive {naive_fc.shape}")

    T0, T1 = SPLIT, N - 1
    n = T1 - T0

    print("=" * 72)
    print("ADIM 4b - " + tag)
    if not a.use_naive:
        print(f"  tahminci : {z['model']} | alpha_gercek {float(z['a_mean']):.3f} "
              f"| alpha_kuyruk {float(z['a_tail']):.3f}")
    print(f"  dilim    : {T0}:{T1}  ({n} adim) | K={K} | H={H}")
    print(f"  ON-KAYIT : esik <= {GAP_TARGET:.2f}%  (W_REF {W_REF:.2f} / "
          f"solar-oracle {V_SOLAR_OR:.2f})")
    print("=" * 72)

    net = nw.case33bw()
    base_p = net.load.p_mw.copy()
    base_q = net.load.q_mvar.copy()
    pp.create_sgen(net, SB, p_mw=0.0, name="solar")
    pp.create_sgen(net, WB, p_mw=0.0, name="wind")
    st = pp.create_storage(net, bus=BB, p_mw=0.0, max_e_mwh=CAP, name="batt")
    sl = SafeLayer(net, st, Rcol_G, S_base_G)

    def vm_at(s, w, dm, p_batt):
        net.load.p_mw = base_p * dm * LS
        net.load.q_mvar = base_q * dm * LS
        net.sgen.p_mw = [s, w]
        net.storage.p_mw = p_batt
        pp.runpp(net)
        return net.res_bus.vm_pu.to_numpy()

    out = Path(a.out if not a.use_naive else str(a.out).replace(".npz", "_naive.npz"))
    mn = np.zeros(n); mx = np.zeros(n); P = np.zeros(n); SOC = np.zeros(n)
    soc = CAP / 2
    j0 = 0
    bad = 0

    if a.resume and out.exists():
        c = np.load(out)
        done = int(c["done"])
        if done >= n:
            print("  ara kayit zaten tamamlanmis -> sadece ozet basiliyor.\n")
            mn, mx, P, SOC = c["mn"], c["mx"], c["p"], c["soc"]
            j0 = n
        else:
            mn[:done], mx[:done] = c["mn"][:done], c["mx"][:done]
            P[:done], SOC[:done] = c["p"][:done], c["soc"][:done]
            soc = float(SOC[done - 1]); j0 = done
            print(f"  DEVAM: {done}/{n} adim ara kayitta, soc={soc:.3f}\n")
    elif out.exists():
        _die(f"cikti zaten var: {out}  (--resume ile devam et, ya da dosyayi sil)")

    t_start = time.time()
    for j in range(j0, n):
        i = T0 + j
        Hh = min(H, N - i)
        vmb = np.zeros((Hh, 33)); pr = np.zeros(Hh)
        for t in range(Hh):
            if t == 0:
                s, w, dm, pq = solar[i], wind[i], dmult[i], price[i]
            else:
                kk = min(t - 1, K - 1)
                s = solar_fc[i, kk]
                w, dm, pq = wind_fc[i, kk], dmult_fc[i, kk], price_fc[i, kk]
            vmb[t] = vm_at(s, w, dm, 0.0); pr[t] = pq
        p_plan, _, info = solve_window(vmb, Rcol_G, S_base_G, pr, soc0=soc)
        if info["status"] not in ("optimal", "optimal_inaccurate"):
            bad += 1
        p = float(np.clip(p_plan[0], -P_MAX, P_MAX))
        p_lo = max(-P_MAX, -soc * ETA / dt)
        p_hi = min(P_MAX, (CAP - soc) / (ETA * dt))
        vm_at(solar[i], wind[i], dmult[i], 0.0)
        p_safe, sinfo = sl.project(p, p_lo, p_hi)
        mn[j] = sinfo["Vmin"]; mx[j] = sinfo["Vmax"]; P[j] = p_safe
        soc = min(max(soc + (ETA * p_safe * dt if p_safe >= 0 else p_safe * dt / ETA), 0), CAP)
        SOC[j] = soc
        if (j + 1) % CKPT_EVERY == 0:
            el = time.time() - t_start
            rate = (j + 1 - j0) / max(el, 1e-9)
            eta = (n - j - 1) / max(rate, 1e-9)
            print(f"  {j+1}/{n}  ({rate*60:.0f} adim/dk, kalan ~{eta/60:.0f} dk)")
            np.savez(out, mn=mn, mx=mx, p=P, soc=SOC, start=T0, done=j + 1)

    np.savez(out, mn=mn, mx=mx, p=P, soc=SOC, start=T0, done=n)

    V = ((mn < LO) | (mx > HI)).mean() * 100
    kar = float(-(price[T0:T1] * P).sum() * dt)

    print("\n" + "=" * 72)
    print("SONUC")
    print("=" * 72)
    print(f"  {'alpha=1.00 (daily-naive, W_REF)':38s} {W_REF:6.2f}%   [on-kayit]")
    print(f"  {tag:38s} {V:6.2f}%   <-- BU KOSUM")
    print(f"  {'alpha=0.00 (solar-oracle)':38s} {V_SOLAR_OR:6.2f}%   [on-kayit]")
    print(f"  {'ON-KAYITLI ESIK':38s} {GAP_TARGET:6.2f}%")
    print(f"\n  kar {kar:.1f} EUR | kirli solve {bad}")

    if a.use_naive:
        d = abs(V - W_REF)
        print("\n  DOGRULAMA HUKMU:")
        if d <= 0.15:
            print(f"    GECTI - naive {V:.2f}% ~= W_REF {W_REF:.2f}% (fark {d:.2f} puan).")
            print("    Boru hatti dogrulandi. NWP kosumunun 9.18% sonucu GUVENILIR.")
            print(f"    Bu kosumun kari ({kar:.0f} EUR) = NWP kosumuyla kiyaslanacak taban.")
        else:
            print(f"    KALDI - naive {V:.2f}% != W_REF {W_REF:.2f}% (fark {d:.2f} puan).")
            print("    Boru hatti W_REF'i yeniden uretemiyor -> NWP sonucu SUPHELI.")
            print("    Sebep bulunmadan hicbir hukum verilmez.")
        print()
        return

    print("\n  HUKUM:")
    if V <= GAP_TARGET:
        print("    GECTI - gercek NWP tahmini duvari YIKTI.")
        print("    Aday C YASIYOR. Supurmenin 'oldur' hukmu YANLISTI (vekil hatasi).")
    else:
        print("    KALDI - gercek NWP tahmini duvari YIKAMADI.")
        print(f"    Naive {W_REF:.2f}% -> NWP {V:.2f}% (esik {GAP_TARGET:.2f}%).")
        print("    Aday C OLUR. Bu sefer vekille degil, GERCEK tahminle olculdu.")
    print(f"\n  Bosluğun kapanan orani: "
          f"{100*(W_REF-V)/(W_REF-V_SOLAR_OR):.1f}%  (hedef: 50%)")
    print()


if __name__ == "__main__":
    main()