#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_oracle_test.py — Aday C, basamak 1 karari.
Mevcut yorungelerden V_oracle ve W'yi GG-1 resmi metrigiyle hesaplar.
SALT-OKUNUR: hicbir dosyaya yazmaz. GG-1 kok klasorunde calistir:
    python gg2/gg2_oracle_test.py
"""
import numpy as np
import pandas as pd

LO, HI, dt = 0.95, 1.05, 0.5   # final_eval.py ile birebir ayni
F = 8.5                        # fiziksel ihlal tabani (%), DEVIR/on-kayit

def rate_and_profit(npz_path, price_full):
    z = np.load(npz_path)
    mn, mx, p = z["mn"], z["mx"], z["p"]
    start, done = int(z["start"]), int(z["done"])
    viol = (mn < LO) | (mx > HI)
    rate = viol.mean() * 100.0
    pr = price_full[start:start + len(p)]
    cost = float((pr * p).sum() * dt)      # dusuk iyi
    return rate, -cost, len(p), done

def main():
    df = pd.read_csv("data/aligned_30min_joint.csv", index_col=0, parse_dates=True)
    price = df.price_eur_mwh.to_numpy()

    V_or, prof_or, n_or, done_or = rate_and_profit("data/mpc_perfect_safe_traj.npz", price)
    W,    prof_w,  n_w,  done_w  = rate_and_profit("data/mpc_gru_safe_traj.npz",    price)

    print("Basamak 1 — oracle on-testi (mevcut yorungelerden)")
    print("-" * 60)
    print(f"koşu tamlik: oracle done={done_or}/{n_or} | gru done={done_w}/{n_w}")
    if done_or != n_or or done_w != n_w:
        print("!! UYARI: yorunge yarim kalmis gorunuyor — sonuc yorumlanmaz.")
    print(f"W (MPC-gru+safe)      ihlal: {W:6.2f}%   kar: {prof_w:10.1f} EUR")
    print(f"V_oracle (MPC-perf+safe) ihlal: {V_or:6.2f}%   kar: {prof_or:10.1f} EUR")
    print(f"F (fiziksel taban)         : {F:6.2f}%")
    print("-" * 60)

    if W - F <= 0:
        print("!! W <= F: gercek-tahminli MPC zaten tabanda; 'duvar' bosluğu yok.")
        print("   ON-KAYITLI KARAR: ADAY C OLUR (kapatilacak bosluk yok).")
        return

    gap_closure = (W - V_or) / (W - F)
    print(f"bosluk kapatma orani (W-V_or)/(W-F): {gap_closure:5.2f}  (esik: >=0.80)")
    print(f"V_oracle mutlak esik kontrolu: {V_or:.2f}%  (esik: <=10.00%)")

    if prof_or < prof_w:
        print("!! AKIL SAGLIGI UYARISI: oracle kari < gru kari — kurulum incelenmeli,")
        print("   karar verilmeden once bu aciklanmali.")

    if (V_or > 10.0) or (gap_closure < 0.80):
        print("KARAR: OLDUR — oracle bile duvari yikamiyor; hicbir gercek")
        print("tahminci yikamaz. Aday C oldu, siradaki adaya gecilir.")
    else:
        print("KARAR: GECTI — bosluk fiziksel olarak var; basamak 2'ye")
        print("(oracle kantil bantlari + daralan hata supurmesi) gecilir.")

if __name__ == "__main__":
    main()