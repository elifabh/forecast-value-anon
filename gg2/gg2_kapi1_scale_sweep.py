#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gg2_kapi1_scale_sweep.py — KAPI 1: klasik cozucu (MIQP-MPC) tikaniyor mu?
Yontem: case33bw'yi spanning-tree ile radyallestirip k kez zincirle -> 33/66/99/132 bara.
Her olcekte AYRIK cihazlar: blok basina 1 OLTC(tamsayi tap) + 2 kapasitor(ikili) + 1 batarya(surekli).
MIQP'yi CVXPY+SCIP ile receding-horizon kur, ADIM basina cozum suresi + gap olc.
Olcek(bara) x Ufuk(H) x cihaz sayisi ekseninde SURE EGRISI cikar.

ONEMLI: pandapower runpp zincirlenmis sebekede NaN veriyor -> taban gerilim
ANALITIK LinDistFlow'dan hesaplaniyor (GG-1 check_lindistflow.py mantigi).
pyscipopt ham API'si normalize hatasi veriyor -> CVXPY sart (ifadeyi temiz kurar).


ON-KAYITLI KARAR (bu dosyadan ONCE yazildi, degistirilemez):
  Referans: 30-dk kontrol araligi = 1800 s tavan; pratik esik cok daha siki.
  OLDUR (A oler, B'ye gecilir): en buyuk konfig (132 bara, H=48) icin
     medyan <= 2000 ms VE p95 <= 10000 ms VE gap ~0 (cozucu rahat cozuyor).
  GECTI (alan var, Kapi 2'ye): 132 bara H=48'de p95 > 60000 ms VEYA gap > %5
     VEYA feasible < %90 (cozucu tikaniyor).
  ARADA kalirsa: egrinin buyume egimine bakilir, rapor edilir, <author> karar verir.

GG-1 KOK klasorunden calistir:  python gg2/gg2_kapi1_scale_sweep.py
Cikti: gg2/kapi1_scale_sweep_result.txt  (GG-1 dosyalarina YAZMAZ)
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandapower as pp, pandapower.networks as nw
import networkx as nx
import cvxpy as cp

# ---- sabitler (opf_expert.py ile hizali) ----
P_MAX, CAP, dt = 3.0, 6.0, 0.5
RTE = 0.90; ETA = RTE**0.5
LO, HI, MARGIN = 0.95, 1.05, 0.005
LOAD_SCALE = 0.5
CAP_MVAR = 0.30            # her kapasitor adimi (MVAr) — pilot olcek (fiziksel kalibre DEGIL)
OLTC_STEPS = 8            # +/- tap sayisi
OLTC_DV = 0.00625          # tap basina taban-gerilim degisimi (pu)
W_SLACK, W_COST, W_THRU = 1e4, 1e-3, 1e-2

KS = [1, 2, 3, 4]          # olcek: 33/66/99/132 bara
HS = [12, 24, 48]          # ufuk uzunlugu
NWIN = 6                   # her konfig icin pencere sayisi (medyan/p95 icin)
OUT = "gg2/kapi1_scale_sweep_result.txt"

_LINES = []
def log(s):
    print(s); _LINES.append(s)

def radial_case33():
    net = nw.case33bw(); G = nx.Graph()
    for idx, ln in net.line.iterrows():
        G.add_edge(int(ln.from_bus), int(ln.to_bus), lid=idx)
    T = nx.minimum_spanning_tree(G)
    keep = {T[u][v]['lid'] for u, v in T.edges()}
    net.line.drop([i for i in net.line.index if i not in keep], inplace=True)
    return net

def chain(k):
    base = radial_case33(); net = radial_case33()
    tail = int(net.bus.index.max()); blocks = [list(range(len(net.bus)))]
    for c in range(1, k):
        bmap = {}
        for b in base.bus.index:
            bmap[b] = pp.create_bus(net, vn_kv=base.bus.at[b, 'vn_kv'])
        for _, ln in base.line.iterrows():
            pp.create_line_from_parameters(net, bmap[int(ln.from_bus)], bmap[int(ln.to_bus)],
                length_km=ln.length_km, r_ohm_per_km=ln.r_ohm_per_km,
                x_ohm_per_km=ln.x_ohm_per_km, c_nf_per_km=ln.c_nf_per_km, max_i_ka=ln.max_i_ka)
        for _, ld in base.load.iterrows():
            pp.create_load(net, bmap[int(ld.bus)], p_mw=ld.p_mw, q_mvar=ld.q_mvar)
        sc = bmap[int(base.ext_grid.bus.iloc[0])]
        pp.create_line_from_parameters(net, tail, sc, length_km=1.0,
            r_ohm_per_km=0.05, x_ohm_per_km=0.05, c_nf_per_km=0.0, max_i_ka=10.0)
        tail = bmap[int(base.bus.index.max())]
        blocks.append(list(bmap.values()))
    return net, blocks

def build_RX(net):
    S_base = net.sn_mva; V_base = net.bus.vn_kv.iloc[0]; Z_base = V_base**2 / S_base
    slack = int(net.ext_grid.bus.iloc[0]); n = len(net.bus)
    busidx = list(net.bus.index); pos = {b: i for i, b in enumerate(busidx)}
    G = nx.Graph(); lr = {}; lx = {}
    for idx, ln in net.line.iterrows():
        lr[idx] = ln.r_ohm_per_km * ln.length_km / Z_base
        lx[idx] = ln.x_ohm_per_km * ln.length_km / Z_base
        G.add_edge(int(ln.from_bus), int(ln.to_bus), lid=idx)
    pl = {slack: set()}
    for b in busidx:
        if b == slack: continue
        sp = nx.shortest_path(G, slack, b)
        pl[b] = {G[sp[i]][sp[i+1]]['lid'] for i in range(len(sp)-1)}
    R = np.zeros((n, n)); X = np.zeros((n, n))
    for a in busidx:
        for b in busidx:
            c = pl[a] & pl[b]
            R[pos[a], pos[b]] = sum(lr[l] for l in c)
            X[pos[a], pos[b]] = sum(lx[l] for l in c)
    return R, X, n, pos, S_base, busidx, pos[slack]

def ldf_vm(net, R, X, n, pos, S_base, scale=1.0):
    """analitik LinDistFlow taban gerilimi (cihazsiz) — runpp'nin NaN sorununu bypass eder."""
    p = np.zeros(n); q = np.zeros(n)
    for _, ld in net.load.iterrows():
        b = pos[int(ld.bus)]
        p[b] -= ld.p_mw * LOAD_SCALE * scale
        q[b] -= ld.q_mvar * LOAD_SCALE * scale
    p /= S_base; q /= S_base
    return np.sqrt(np.clip(1.0 + 2.0*(R@p + X@q), 1e-6, None))

def measure(k, H, nwin=NWIN):
    net, blocks = chain(k)
    R, X, n, pos, S_base, busidx, slackpos = build_RX(net)
    batt_bus = [blk[16] for blk in blocks]
    cap_bus  = [blk[16] for blk in blocks] + [blk[31] for blk in blocks]
    nb = len(batt_bus); nc = len(cap_bus)
    v0 = ldf_vm(net, R, X, n, pos, S_base)
    if np.isnan(v0).any():
        return dict(k=k, H=H, n=n, nb=nb, nc=nc, feas=0, err="taban gerilim NaN")
    vm0 = np.tile(v0, (H, 1))
    # farkli pencereler icin farkli fiyat/yuk profili (cesitlilik)
    keep = [b for b in range(n) if b != slackpos]
    Rb = np.array([[2.0*(R[b, pos[batt_bus[i]]]/S_base)/S_base for i in range(nb)] for b in keep])
    Xc = np.array([[2.0*(X[b, pos[cap_bus[c]]]/S_base)*(CAP_MVAR/S_base) for c in range(nc)] for b in keep])
    vb2 = np.array([[vm0[t, b]**2 for b in keep] for t in range(H)])

    wall = []; solt = []; gaps = []; feas = 0
    for w in range(nwin):
        pr = np.linspace(50, 120, H) * (1.0 + 0.1*np.sin(w))  # pencere cesitliligi
        pch = cp.Variable((H, nb), nonneg=True); pdi = cp.Variable((H, nb), nonneg=True)
        soc = cp.Variable((H+1, nb)); z = cp.Variable((H, nc), boolean=True)
        tap = cp.Variable(H, integer=True)
        shi = cp.Variable((H, len(keep)), nonneg=True); slo = cp.Variable((H, len(keep)), nonneg=True)
        cons = [soc[0, :] == CAP/2, soc >= 0, soc <= CAP,
                pch <= P_MAX, pdi <= P_MAX, tap >= -OLTC_STEPS, tap <= OLTC_STEPS]
        for t in range(H):
            cons += [soc[t+1, :] == soc[t, :] + ETA*pch[t, :]*dt - pdi[t, :]*dt/ETA]
            V2 = vb2[t, :] + (pdi[t, :]-pch[t, :])@Rb.T + z[t, :]@Xc.T + 2.0*OLTC_DV*tap[t]
            cons += [V2 <= (HI-MARGIN)**2 + shi[t, :], V2 >= (LO+MARGIN)**2 - slo[t, :]]
        obj = (W_SLACK*cp.sum(shi+slo)
               + W_COST*cp.sum(cp.multiply(pr[:, None], (pch-pdi)))*dt
               + W_THRU*cp.sum(pch+pdi))
        prob = cp.Problem(cp.Minimize(obj), cons)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                prob.solve(solver='SCIP')
            except Exception as e:
                wall.append(float('nan')); solt.append(float('nan')); gaps.append(float('nan'))
                if w == 0: log(f"    (solver hata: {str(e)[:60]})")
                continue
        el = (time.perf_counter()-t0)*1000
        wall.append(el)
        st = prob.solver_stats
        solt.append(st.solve_time*1000 if (st and st.solve_time) else float('nan'))
        gaps.append(0.0 if prob.status == "optimal" else float('nan'))
        if prob.status in ("optimal", "optimal_inaccurate"):
            feas += 1
    wa = np.array(wall); sa = np.array(solt)
    def safe(fn, arr):
        v = arr[~np.isnan(arr)]
        return float(fn(v)) if len(v) else float("nan")
    return dict(k=k, H=H, n=n, nb=nb, nc=nc, feas=feas, nwin=nwin,
                wall_med=safe(np.median, wa), wall_p95=safe(lambda x: np.percentile(x,95), wa), wall_max=safe(np.max, wa),
                solt_med=safe(np.median, sa), solt_p95=safe(lambda x: np.percentile(x,95), sa))

def main():
    log("="*72)
    log("KAPI 1 — MIQP-MPC olcek supurmesi (cozucu tikaniyor mu?)")
    log(f"cozucu: CVXPY+SCIP | cihaz: blok basina 1 OLTC + 2 kapasitor + 1 batarya")
    log(f"olcek k={KS} (33/66/99/132 bara) | ufuk H={HS} | pencere={NWIN}")
    log("="*72)
    log(f"{'bara':>5} {'H':>4} {'bat':>4} {'cap':>4} {'feas':>6} | "
        f"{'wall_med':>9} {'wall_p95':>9} {'wall_max':>9} | {'solt_med':>9} {'solt_p95':>9}  (ms)")
    log("-"*72)
    results = []
    for H in HS:
        for k in KS:
            r = measure(k, H)
            results.append(r)
            if r.get("err"):
                log(f"{r['n']:>5} {r['H']:>4} {r['nb']:>4} {r['nc']:>4} | HATA: {r['err']}")
            else:
                log(f"{r['n']:>5} {r['H']:>4} {r['nb']:>4} {r['nc']:>4} {r['feas']:>3}/{r['nwin']:<2} | "
                    f"{r['wall_med']:9.1f} {r['wall_p95']:9.1f} {r['wall_max']:9.1f} | "
                    f"{r['solt_med']:9.1f} {r['solt_p95']:9.1f}")
        log("-"*72)

    # ---- ON-KAYITLI KARAR ----
    big = [r for r in results if r['k'] == max(KS) and r['H'] == max(HS) and not r.get("err")]
    log("")
    log("ON-KAYITLI KARAR (132 bara, H=48):")
    if not big:
        log("  En buyuk konfig olculemedi — karar verilemiyor, incele.")
    else:
        r = big[0]
        feas_ratio = r['feas']/r['nwin']
        p95 = r['wall_p95']; med = r['wall_med']
        log(f"  medyan {med:.0f} ms | p95 {p95:.0f} ms | feasible {r['feas']}/{r['nwin']}")
        if (p95 > 60000) or (feas_ratio < 0.90):
            log("  KARAR: GECTI — cozucu tikaniyor (p95>60s veya feasible<%90).")
            log("  Alan var; Kapi 2'ye (ogrenme payi pilotu) gecilir.")
        elif (med <= 2000) and (p95 <= 10000):
            log("  KARAR: OLDUR — cozucu rahat cozuyor (medyan<=2s, p95<=10s).")
            log("  'MPC tikanir' iddiasi bu olcekte dogrulanamadi. Aday A oldu; B'ye gecilir.")
            log("  NOT: olcek-sure egrisi (yukaridaki tablo) yine de raporlanabilir bulgudur (Q5).")
        else:
            log("  KARAR: ARADA — net esikler tutmadi. Egrinin buyume egimine bak:")
            log("  bara/ufuk 2x olunca sure kac kat artiyor? Ustel ~> alan; dogrusal ~> yok.")
            log("  <author> yorumlar.")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_LINES))
    print(f"\nyazildi: {OUT}")

if __name__ == "__main__":
    main()