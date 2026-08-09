#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_solar_alpha_sweep.py — Basamak 2b: daralan-hata supurmesi.
Sentetik solar tahmini: fc_a = truth + alpha*(fc - truth), alpha=0.75/0.50/0.25.
Geri kalani mpc_gru_safe.py ile birebir ayni.
GG-1 KOK klasorunden calistir:  python gg2/gg2_solar_alpha_sweep.py
Ciktilar: gg2/mpc_solar_a{75,50,25}_traj.npz  (GG-1 dosyalarina yazilmaz)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import pandapower as pp, pandapower.networks as nw
from opf_expert import solve_window, H, CAP, dt, ETA
from safe_layer import SafeLayer

LS=0.5; SP,WP=3.0,2.0; SB,WB,BB=17,32,17
P_MAX=3.0; SPLIT=13800
LO,HI=0.95,1.05
W_REF=10.62; V_SOLAR_OR=8.69            # uclar (on-kayit, basamak 1/2a)
GAP_TARGET=W_REF-0.5*(W_REF-8.73)       # 9.685 -> bosluğun yarisi
ALPHAS=[0.75,0.50,0.25]

def run_alpha(alpha, df, arrs, fc, net_parts):
    solar,wind,dmult,price=arrs
    solar_fc,wind_fc,dmult_fc,price_fc=fc
    net,base_p,base_q,sl=net_parts
    N=len(df); T0=SPLIT; T1=N-1
    K=solar_fc.shape[1]
    # sentetik solar tahmini (hedef zaman i+k+1 hizalamasi, mpc_gru_safe ile ayni)
    idx=np.minimum(np.arange(N)[:,None]+np.arange(1,K+1)[None,:],N-1)
    truth=solar[idx]                                   # (N,K)
    fc_a=truth+alpha*(solar_fc-truth)
    mae_o=np.abs(solar_fc[T0:T1]-truth[T0:T1]).mean()
    mae_a=np.abs(fc_a[T0:T1]-truth[T0:T1]).mean()
    print(f"\n== alpha={alpha}: solar MAE {mae_o:.4f} -> {mae_a:.4f} "
          f"(oran {mae_a/mae_o:.2f}, beklenen {alpha})")

    def vm_at(s,w,dm,p_batt):
        net.load.p_mw=base_p*dm*LS; net.load.q_mvar=base_q*dm*LS
        net.sgen.p_mw=[s,w]; net.storage.p_mw=p_batt
        pp.runpp(net); return net.res_bus.vm_pu.to_numpy()

    out=f"gg2/mpc_solar_a{int(alpha*100)}_traj.npz"
    mn=np.zeros(T1-T0); mx=np.zeros(T1-T0); P=np.zeros(T1-T0); SOC=np.zeros(T1-T0)
    soc=CAP/2; bad=0
    for j,i in enumerate(range(T0,T1)):
        Hh=min(H,N-i)
        vmb=np.zeros((Hh,33)); pr=np.zeros(Hh)
        for t in range(Hh):
            if t==0: s,w,dm,pq=solar[i],wind[i],dmult[i],price[i]
            else:
                kk=min(t-1,K-1)
                s=fc_a[i,kk]
                w,dm,pq=wind_fc[i,kk],dmult_fc[i,kk],price_fc[i,kk]
            vmb[t]=vm_at(s,w,dm,0.0); pr[t]=pq
        p_plan,_,info=solve_window(vmb,Rcol_G,S_base_G,pr,soc0=soc)
        if info["status"] not in ("optimal","optimal_inaccurate"): bad+=1
        p=float(np.clip(p_plan[0],-P_MAX,P_MAX))
        p_lo=max(-P_MAX,-soc*ETA/dt); p_hi=min(P_MAX,(CAP-soc)/(ETA*dt))
        vm_at(solar[i],wind[i],dmult[i],0.0)
        p_safe,sinfo=sl.project(p,p_lo,p_hi)
        mn[j]=sinfo["Vmin"]; mx[j]=sinfo["Vmax"]; P[j]=p_safe
        soc=min(max(soc+(ETA*p_safe*dt if p_safe>=0 else p_safe*dt/ETA),0),CAP); SOC[j]=soc
        if (j+1)%400==0:
            print(f"  a{alpha}: {j+1}/{T1-T0}")
            np.savez(out,mn=mn,mx=mx,p=P,soc=SOC,start=T0,done=j+1)
    np.savez(out,mn=mn,mx=mx,p=P,soc=SOC,start=T0,done=T1-T0)
    V=((mn<LO)|(mx>HI)).mean()*100
    kar=float(-(price[T0:T1]*P).sum()*dt)
    print(f"alpha={alpha}: ihlal {V:.2f}% | kar {kar:.1f} EUR | kirli solve {bad}")
    return V,kar

def main():
    global Rcol_G,S_base_G
    df=pd.read_csv("data/aligned_30min_joint.csv",index_col=0,parse_dates=True)
    train=df.iloc[:SPLIT]
    solar=(df.solar_mw/train.solar_mw.max()*SP).to_numpy()
    wind =(df.wind_mw /train.wind_mw.max()*WP).to_numpy()
    dmult=(df.demand_mw/train.demand_mw.mean()).to_numpy()
    price=df.price_eur_mwh.to_numpy()
    d=np.load("data/opf_base_vm.npz"); Rcol_G=d["Rcol"]; S_base_G=float(d["S_base"])
    f=np.load("data/forecast_matrix.npz")
    fc=(f["solar"],f["wind"],f["dmult"],f["price"])
    net=nw.case33bw(); base_p=net.load.p_mw.copy(); base_q=net.load.q_mvar.copy()
    pp.create_sgen(net,SB,p_mw=0.0,name="solar"); pp.create_sgen(net,WB,p_mw=0.0,name="wind")
    st=pp.create_storage(net,bus=BB,p_mw=0.0,max_e_mwh=CAP,name="batt")
    sl=SafeLayer(net,st,Rcol_G,S_base_G)

    res={}
    for a in ALPHAS:
        res[a]=run_alpha(a,df,(solar,wind,dmult,price),fc,(net,base_p,base_q,sl))

    print("\n===== EGRI (ihlal % vs alpha) =====")
    print(f"alpha=1.00 : {W_REF:.2f}%  (W, mevcut tahmin)")
    for a in ALPHAS: print(f"alpha={a:.2f} : {res[a][0]:.2f}%  | kar {res[a][1]:.0f} EUR")
    print(f"alpha=0.00 : {V_SOLAR_OR:.2f}%  (solar-oracle)")
    print(f"hedef: V(alpha) <= {GAP_TARGET:.2f}%  (bosluğun yarisi)")
    ok=[a for a in ALPHAS if res[a][0]<=GAP_TARGET]
    if ok:
        a_star=max(ok)
        print(f"alpha* = {a_star}  (hedefi tutturan EN BUYUK alpha)")
        if a_star>=0.50:
            print("KARAR: GECTI — gereken hata azaltmasi NWP literatur menzilinde")
            print("(%35-50 azaltma gerceklesebilir). Aday C, Kapi 0'a hak kazandi.")
        else:
            print("KARAR: OLDUR — hedef ancak alpha<0.50 ile tutuyor; literatur")
            print("menzili disinda. Aday C oldu.")
    else:
        print("KARAR: OLDUR — supurulen hicbir alpha hedefi tutturamadi.")

if __name__ == "__main__":
    main()