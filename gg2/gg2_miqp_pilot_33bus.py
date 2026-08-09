#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_miqp_pilot_33bus.py — KAPI 1 / Adim 2: cozucu borusu dogrulama (KILL DEGIL).
GG-1 33-bus LinDistFlow fizigini odunc alir; uzerine AYRIK cihazlar ekler:
  - 1 OLTC (tamsayi kademe, taban-gerilim ofseti)
  - NCAP kapasitor (ikili 0/1, reaktif enjeksiyon)
  - 2 batarya (surekli)
Receding-horizon MIQP'yi pyscipopt ile kurar, ADIM BASINA cozum suresi + gap olcer.
GG-1 KOK klasorunden calistir:  python gg2/gg2_miqp_pilot_33bus.py
Cikti: gg2/miqp_pilot_33bus_result.txt  (GG-1 dosyalarina yazmaz)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import pandapower as pp, pandapower.networks as nw
import networkx as nx
from pyscipopt import Model, quicksum

# ---- sabitler (opf_expert.py ile ayni) ----
P_MAX, CAP, dt = 3.0, 6.0, 0.5
RTE = 0.90; ETA = RTE**0.5
LO, HI, MARGIN = 0.95, 1.05, 0.005
LOAD_SCALE = 0.5; SOLAR_PEAK, WIND_PEAK = 3.0, 2.0
SOLAR_BUS, WIND_BUS = 17, 32
BATT_BUSES = [17, 32]          # 2 batarya (pilot)
CAP_BUSES  = [17, 32]          # 2 kapasitor
CAP_MVAR   = 0.30              # her kapasitor adimi (MVAr)
OLTC_STEPS = 8                 # +/- kademe sayisi
OLTC_DV    = 0.00625           # kademe basina taban-gerilim degisimi (pu) ~ %0.625
H = 12                         # pilot ufuk (kisa; boru testi)
NWIN = 40                      # kac pencere olculecek
SOC_TARGET = CAP/2
W_SLACK, W_COST, W_THRU, W_TERM = 1e4, 1e-3, 1e-2, 1e-1
OUT = "gg2/miqp_pilot_33bus_result.txt"

def build_RX():
    net = nw.case33bw()
    base_p, base_q = net.load.p_mw.copy(), net.load.q_mvar.copy()
    S_base = net.sn_mva; V_base = net.bus.vn_kv.iloc[0]; Z_base = V_base**2/S_base
    slack = int(net.ext_grid.bus.iloc[0]); n = len(net.bus)
    G = nx.Graph(); line_r={}; line_x={}
    for idx, ln in net.line.iterrows():
        if not ln.in_service: continue
        line_r[idx]=ln.r_ohm_per_km*ln.length_km/Z_base
        line_x[idx]=ln.x_ohm_per_km*ln.length_km/Z_base
        G.add_edge(int(ln.from_bus), int(ln.to_bus), lid=idx)
    path_lines={slack:set()}
    for b in net.bus.index:
        if b==slack: continue
        sp=nx.shortest_path(G,slack,b)
        path_lines[b]={G[sp[i]][sp[i+1]]['lid'] for i in range(len(sp)-1)}
    R=np.zeros((n,n)); X=np.zeros((n,n))
    for j in net.bus.index:
        for kk in net.bus.index:
            c=path_lines[j]&path_lines[kk]
            R[j,kk]=sum(line_r[l] for l in c); X[j,kk]=sum(line_x[l] for l in c)
    return net, base_p, base_q, S_base, R, X, n, slack

def vm_base_row(net, base_p, base_q, S_base, R, X, n, dm, solar, wind):
    """cihazsiz (batarya=0, kapasitor=0, OLTC=0) LinDistFlow taban gerilimi."""
    net.load.p_mw = base_p*dm*LOAD_SCALE; net.load.q_mvar = base_q*dm*LOAD_SCALE
    p=np.zeros(n); q=np.zeros(n)
    for _,ld in net.load.iterrows(): b=int(ld.bus); p[b]-=ld.p_mw; q[b]-=ld.q_mvar
    p[SOLAR_BUS]+=solar; p[WIND_BUS]+=wind
    p/=S_base; q/=S_base
    v2 = 1.0 + 2.0*(R@p + X@q)
    return np.sqrt(np.clip(v2,1e-6,None)), R, X

def solve_miqp(vm0, R, X, n, price_w, soc0):
    """
    Bir pencere icin MIQP.
    Gerilim (lineer yaklasim, v^2 uzerinden):
      v2[t,b] = vm0[t,b]^2
                + 2*sum_batt (R[b,bus]/S_base)*p_batt   (aktif -> R kanali)
                + 2*sum_cap  (X[b,bus]/S_base)*Qcap*z   (reaktif -> X kanali)
                + 2*tap*OLTC_DV                          (taban ofseti, pu^2 lineerize)
    NOT: S_base disari carpani vm_base_row'da p,q normalize edildigi icin
         burada R,X zaten pu; batt/cap MW/MVAr -> S_base ile bolerek ekliyoruz.
    """
    Hh = vm0.shape[0]
    m = Model("win"); m.hideOutput()
    S_base = 1.0  # R,X pu; batt/cap'i asagida S_base_real ile bolecegiz
    # --- degiskenler ---
    pch = {(t,i): m.addVar(lb=0, ub=P_MAX) for t in range(Hh) for i in range(len(BATT_BUSES))}
    pdi = {(t,i): m.addVar(lb=0, ub=P_MAX) for t in range(Hh) for i in range(len(BATT_BUSES))}
    soc = {(t,i): m.addVar(lb=0, ub=CAP)   for t in range(Hh+1) for i in range(len(BATT_BUSES))}
    z   = {(t,c): m.addVar(vtype="B")      for t in range(Hh) for c in range(len(CAP_BUSES))}
    tap = {t: m.addVar(vtype="I", lb=-OLTC_STEPS, ub=OLTC_STEPS) for t in range(Hh)}
    shi = {(t,b): m.addVar(lb=0) for t in range(Hh) for b in range(n)}
    slo = {(t,b): m.addVar(lb=0) for t in range(Hh) for b in range(n)}

    for i in range(len(BATT_BUSES)):
        m.addCons(soc[0,i]==soc0)
    S_real = 10.0  # case33bw sn_mva; asagida gercek deger ile guncellenecek
    return m, dict(pch=pch,pdi=pdi,soc=soc,z=z,tap=tap,shi=shi,slo=slo,Hh=Hh)

def main():
    net, base_p, base_q, S_base, R, X, n, slack = build_RX()
    df = pd.read_csv("data/aligned_30min_joint.csv", index_col=0, parse_dates=True)
    N = len(df)
    dmult=(df.demand_mw/df.demand_mw.mean()).to_numpy()
    solar=(df.solar_mw/df.solar_mw.max()*SOLAR_PEAK).to_numpy()
    wind =(df.wind_mw /df.wind_mw.max()*WIND_PEAK).to_numpy()
    price=df.price_eur_mwh.to_numpy()
    Bbus=[int(b) for b in BATT_BUSES]; Cbus=[int(b) for b in CAP_BUSES]
    nb=len(Bbus); nc=len(Cbus)

    # en sikintili pencerelerden basla (max taban gerilimi yuksek yerler)
    scan = pd.read_csv("data/feeder_voltage_scan.csv", index_col=0, parse_dates=True)
    order = np.argsort(scan.maxV.to_numpy())[::-1]
    starts = [int(o) for o in order[:NWIN] if o+H < N]

    lines=[]
    def log(s):
        print(s); lines.append(s)

    log(f"MIQP PILOT 33-bus | H={H} | bat={nb} cap={nc} OLTC(+-{OLTC_STEPS}) | cozucu=SCIP")
    log(f"olculen pencere: {len(starts)} | (KILL DEGIL: boru/formulasyon dogrulama)")
    log("-"*64)

    times=[]; gaps=[]; slacks=[]; feas=0
    for w,i0 in enumerate(starts):
        # pencere taban gerilimleri
        vm0=np.zeros((H,n))
        for t in range(H):
            i=min(i0+t,N-1)
            v,_,_=vm_base_row(net,base_p,base_q,S_base,R,X,n,dmult[i],solar[i],wind[i])
            vm0[t]=v
        pr=price[i0:i0+H]
        soc0=CAP/2

        m=Model("win"); m.hideOutput()
        pch={(t,i):m.addVar(lb=0,ub=P_MAX) for t in range(H) for i in range(nb)}
        pdi={(t,i):m.addVar(lb=0,ub=P_MAX) for t in range(H) for i in range(nb)}
        soc={(t,i):m.addVar(lb=0,ub=CAP)   for t in range(H+1) for i in range(nb)}
        z  ={(t,c):m.addVar(vtype="B")     for t in range(H) for c in range(nc)}
        tap={t:m.addVar(vtype="I",lb=-OLTC_STEPS,ub=OLTC_STEPS) for t in range(H)}
        shi={(t,b):m.addVar(lb=0) for t in range(H) for b in range(n)}
        slo={(t,b):m.addVar(lb=0) for t in range(H) for b in range(n)}

        for i in range(nb): m.addCons(soc[0,i]==soc0)
        for t in range(H):
            for i in range(nb):
                m.addCons(soc[t+1,i]==soc[t,i]+ETA*pch[t,i]*dt - pdi[t,i]*dt/ETA)
            for b in range(n):
                # v^2 lineer yaklasim: taban + batt(R kanali) + cap(X kanali) + OLTC ofset
                vb = vm0[t,b]*vm0[t,b]
                batt_term = quicksum(2.0*(R[b,Bbus[i]]/S_base)*(pdi[t,i]-pch[t,i])/S_base
                                     for i in range(nb))
                cap_term  = quicksum(2.0*(X[b,Cbus[c]]/S_base)*(CAP_MVAR/S_base)*z[t,c]
                                     for c in range(nc))
                oltc_term = 2.0*OLTC_DV*tap[t]
                V2 = vb + batt_term + cap_term + oltc_term
                m.addCons(V2 <= (HI-MARGIN)**2 + shi[t,b])
                m.addCons(V2 >= (LO+MARGIN)**2 - slo[t,b])

        obj = (W_SLACK*quicksum(shi[t,b]+slo[t,b] for t in range(H) for b in range(n))
               + W_COST*quicksum(pr[t]*(pch[t,i]-pdi[t,i])*dt for t in range(H) for i in range(nb))
               + W_THRU*quicksum(pch[t,i]+pdi[t,i] for t in range(H) for i in range(nb))
               + W_TERM*quicksum(soc[H,i] for i in range(nb)))  # basit terminal
        m.setObjective(obj,"minimize")
        m.setParam("limits/time", 60.0)

        t0=time.perf_counter()
        m.optimize()
        el=time.perf_counter()-t0
        st=m.getStatus()
        gap=m.getGap() if st in ("optimal","gaplimit","timelimit") else float("nan")
        if st in ("optimal","bestsollimit","gaplimit","timelimit") and m.getNSols()>0:
            feas+=1
            sl=sum(m.getVal(shi[t,b])+m.getVal(slo[t,b]) for t in range(H) for b in range(n))
        else:
            sl=float("nan")
        times.append(el); gaps.append(gap); slacks.append(sl)
        if w<5 or (w+1)%10==0:
            log(f"  pencere {w+1:3d}/{len(starts)} | {st:10s} | {el*1000:7.1f} ms | "
                f"gap {gap*100 if gap==gap else float('nan'):5.2f}% | slack {sl:.4f}")

    tarr=np.array(times)*1000
    log("-"*64)
    log(f"cozulen (feasible): {feas}/{len(starts)}")
    log(f"cozum suresi (ms): medyan {np.median(tarr):.1f} | p95 {np.percentile(tarr,95):.1f} | max {tarr.max():.1f}")
    g=np.array(gaps); g=g[~np.isnan(g)]
    if len(g): log(f"gap (%): medyan {np.median(g)*100:.3f} | max {g.max()*100:.3f}")
    log("")
    log("YORUM: bu bir BORU testidir (33-bus kucuk). Beklenti: ms mertebesi, gap~0.")
    log("Anlam: formulasyon kuruluyor + SCIP cozuyor. KILL kriteri 123-bus'ta isler.")
    log("Sonraki: ayni formulasyonu ~120-bus sinifa tasi, medyan/p95 sure + gap egrisi.")

    with open(OUT,"w",encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nyazildi: {OUT}")

if __name__=="__main__":
    main()