#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_mpc_solar_oracle.py — Basamak 2a: DEGISKEN AYRISTIRMA.
mpc_gru_safe.py ile BIREBIR ayni; TEK fark: solar tahmin penceresi
gerceklesen degerlerden alinir (solar-oracle). Wind/talep/fiyat GRU-matris.
GG-1 KOK klasorunden calistir:  python gg2/gg2_mpc_solar_oracle.py
Cikti: gg2/mpc_solar_oracle_traj.npz  (GG-1 dosyalarina yazmaz)
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
W_REF=10.62; V_ORACLE_REF=8.73          # basamak 1 sonuclari (on-kayit)
OUT="gg2/mpc_solar_oracle_traj.npz"

def main():
    df=pd.read_csv("data/aligned_30min_joint.csv",index_col=0,parse_dates=True)
    N=len(df); train=df.iloc[:SPLIT]
    dmult=(df.demand_mw/train.demand_mw.mean()).to_numpy()
    solar=(df.solar_mw/train.solar_mw.max()*SP).to_numpy()
    wind =(df.wind_mw /train.wind_mw.max()*WP).to_numpy()
    price=df.price_eur_mwh.to_numpy()

    d=np.load("data/opf_base_vm.npz"); Rcol=d["Rcol"]; S_base=float(d["S_base"])
    f=np.load("data/forecast_matrix.npz")
    wind_fc,dmult_fc,price_fc=f["wind"],f["dmult"],f["price"]   # solar_fc KULLANILMAZ

    net=nw.case33bw(); base_p=net.load.p_mw.copy(); base_q=net.load.q_mvar.copy()
    pp.create_sgen(net,SB,p_mw=0.0,name="solar"); pp.create_sgen(net,WB,p_mw=0.0,name="wind")
    st=pp.create_storage(net,bus=BB,p_mw=0.0,max_e_mwh=CAP,name="batt")
    sl=SafeLayer(net, st, Rcol, S_base)

    def vm_at(s,w,dm,p_batt):
        net.load.p_mw=base_p*dm*LS; net.load.q_mvar=base_q*dm*LS
        net.sgen.p_mw=[s,w]; net.storage.p_mw=p_batt
        pp.runpp(net); return net.res_bus.vm_pu.to_numpy()

    T0=SPLIT; T1=N-1
    mn=np.zeros(T1-T0); mx=np.zeros(T1-T0); P=np.zeros(T1-T0); SOC=np.zeros(T1-T0)
    soc=CAP/2; bad=0; stat={}
    for j,i in enumerate(range(T0,T1)):
        Hh=min(H,N-i)
        vmb=np.zeros((Hh,33)); pr=np.zeros(Hh)
        for t in range(Hh):
            if t==0: s,w,dm,pq=solar[i],wind[i],dmult[i],price[i]
            else:
                kk=min(t-1,wind_fc.shape[1]-1)
                s = solar[min(i+t,N-1)]                 # <-- TEK FARK: solar ORACLE
                w,dm,pq=wind_fc[i,kk],dmult_fc[i,kk],price_fc[i,kk]
            vmb[t]=vm_at(s,w,dm,0.0); pr[t]=pq
        p_plan,_,info=solve_window(vmb,Rcol,S_base,pr,soc0=soc)
        if info["status"] not in ("optimal","optimal_inaccurate"): bad+=1
        p=float(np.clip(p_plan[0],-P_MAX,P_MAX))
        p_lo=max(-P_MAX,-soc*ETA/dt); p_hi=min(P_MAX,(CAP-soc)/(ETA*dt))
        vm_at(solar[i],wind[i],dmult[i],0.0)
        p_safe,sinfo=sl.project(p, p_lo, p_hi)
        stat[sinfo["status"]]=stat.get(sinfo["status"],0)+1
        mn[j]=sinfo["Vmin"]; mx[j]=sinfo["Vmax"]; P[j]=p_safe
        soc=min(max(soc+(ETA*p_safe*dt if p_safe>=0 else p_safe*dt/ETA),0),CAP); SOC[j]=soc
        if (j+1)%200==0:
            print(f"  {j+1}/{T1-T0} {df.index[i]} soc={soc:.2f} Vmax={mx[j]:.4f}")
            np.savez(OUT,mn=mn,mx=mx,p=P,soc=SOC,start=T0,done=j+1)
    np.savez(OUT,mn=mn,mx=mx,p=P,soc=SOC,start=T0,done=T1-T0)

    V_s=((mn<LO)|(mx>HI)).mean()*100
    kar=float(-(price[T0:T1]*P).sum()*dt)
    pay=(W_REF-V_s)/(W_REF-V_ORACLE_REF)
    print(f"\nMPC solar-ORACLE + safe: ihlal {V_s:.2f}% | kar {kar:.1f} EUR | kirli solve: {bad}")
    print(f"statu: {stat}")
    print(f"solar payi = (W - V_s)/(W - V_oracle) = ({W_REF} - {V_s:.2f})/({W_REF} - {V_ORACLE_REF}) = {pay:.2f}")
    if pay < 0.50:
        print("KARAR: OLDUR — duvarin cogu solar-disi hatadan; NWP solar tahmini")
        print("duvari yikamaz. Aday C oldu.")
    else:
        print("KARAR: GECTI — duvar agirlikla solar kaynakli; basamak 2b'ye")
        print("(daralan-hata supurmesi) gecilir.")

if __name__ == "__main__":
    main()