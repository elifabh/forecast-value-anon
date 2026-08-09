# safe_layer.py  (FINAL v2: storage convention + correct saturation label)
# Convention: p>0 = CHARGE (consumption, LOWERS the voltage), p<0 = DISCHARGE (RAISES the voltage)
import numpy as np
import pandapower as pp
from pandapower.auxiliary import LoadflowNotConverged

V_LO, V_HI, MARGIN = 0.95, 1.05, 0.005
TOL = 5e-4   # MW

class SafeLayer:
    """Storage conv.: p>0=charge (V drops). Projection verified by a real AC solve."""
    def __init__(self, net, storage_idx, Rcol, S_base):
        self.net = net; self.si = storage_idx
        self.Rcol = Rcol; self.S_base = S_base
        self.last_runpp = 0

    def _vac(self, p):
        self.last_runpp += 1
        self.net.storage.at[self.si, "p_mw"] = float(p)
        try:
            pp.runpp(self.net)
            return self.net.res_bus.vm_pu.min(), self.net.res_bus.vm_pu.max()
        except LoadflowNotConverged:
            return None

    def _root(self, x0, f0, x1, f1, fval, tol=TOL, itmax=10):
        """Regula falsi: f decreasing, root inside the bracket f(x0)>0>=f(x1)."""
        for _ in range(itmax):
            if abs(x1 - x0) < tol: break
            xm = x1 - f1*(x1-x0)/(f1-f0) if f1 != f0 else 0.5*(x0+x1)
            xm = min(max(xm, min(x0,x1)+1e-9), max(x0,x1)-1e-9)
            fm = fval(xm)
            if fm is None: fm = np.inf
            if fm > 0: x0, f0 = xm, fm
            else:      x1, f1 = xm, fm
        return x1

    def project(self, p_agent, p_lo, p_hi):
        self.last_runpp = 0
        Vhi_m, Vlo_m = V_HI - MARGIN, V_LO + MARGIN
        pa = float(np.clip(p_agent, p_lo, p_hi))

        r = self._vac(pa)
        if r and r[0] >= V_LO and r[1] <= V_HI:
            return pa, dict(status="agent_ok", Vmin=r[0], Vmax=r[1], runpp=self.last_runpp)

        over  = (r is None) or (r[1] > V_HI)
        under = (r is not None) and (r[0] < V_LO)

        if over and not under:
            rh = self._vac(p_hi)
            if rh is None or rh[1] > Vhi_m:
                if rh is not None and rh[0] >= V_LO:
                    res = round(max(0, rh[1]-V_HI), 4)
                    st = "saturated" if res > 0 else "projected_margin"
                    return p_hi, dict(status=st, Vmin=rh[0], Vmax=rh[1],
                                      residual_pu=res, runpp=self.last_runpp)
                return self._scan(p_lo, p_hi)
            f_pa = (np.inf if r is None else r[1] - Vhi_m)
            a = self._root(pa, f_pa, p_hi, rh[1]-Vhi_m,
                           lambda p: (lambda q: None if q is None else q[1]-Vhi_m)(self._vac(p)))
            p_safe = float(np.clip(pa, a, p_hi))
            rr = self._vac(p_safe)
            if rr and rr[0] >= V_LO and rr[1] <= V_HI:
                return p_safe, dict(status="projected_exact", Vmin=rr[0], Vmax=rr[1],
                                    bound=round(a,3), runpp=self.last_runpp)
            return self._scan(p_lo, p_hi)

        if under and not over:
            rl = self._vac(p_lo)
            if rl is None or rl[0] < Vlo_m:
                if rl is not None and rl[1] <= V_HI:
                    res = round(max(0, V_LO-rl[0]), 4)
                    st = "saturated" if res > 0 else "projected_margin"
                    return p_lo, dict(status=st, Vmin=rl[0], Vmax=rl[1],
                                      residual_pu=res, runpp=self.last_runpp)
                return self._scan(p_lo, p_hi)
            f_pa = (-np.inf if r is None else r[0] - Vlo_m)
            b = self._root(pa, -f_pa, p_lo, -(rl[0]-Vlo_m),
                           lambda p: (lambda q: np.inf if q is None else Vlo_m - q[0])(self._vac(p)))
            p_safe = float(np.clip(pa, p_lo, b))
            rr = self._vac(p_safe)
            if rr and rr[0] >= V_LO and rr[1] <= V_HI:
                return p_safe, dict(status="projected_exact", Vmin=rr[0], Vmax=rr[1],
                                    bound=round(b,3), runpp=self.last_runpp)
            return self._scan(p_lo, p_hi)

        return self._scan(p_lo, p_hi)

    def _scan(self, p_lo, p_hi):
        best = None
        for p in np.linspace(p_lo, p_hi, 13):
            r = self._vac(p)
            if r is None: continue
            viol = max(0, r[1]-V_HI) + max(0, V_LO-r[0])
            if best is None or viol < best[1]: best = (p, viol, r)
        st = "saturated" if best[1] > 0 else "projected_margin"
        return best[0], dict(status=st, Vmin=best[2][0], Vmax=best[2][1],
                             residual_pu=round(best[1],4), runpp=self.last_runpp)

# ---------------- SELF-TEST ----------------
if __name__ == "__main__":
    import pandas as pd, pandapower.networks as nw, networkx as nx
    LS=0.5; SP,WP=3.0,2.0; SB,WB,BB=17,32,17; P_MAX,CAP=3.0,6.0

    net=nw.case33bw(); base_p,base_q=net.load.p_mw.copy(),net.load.q_mvar.copy()
    S_base=net.sn_mva; Z_base=net.bus.vn_kv.iloc[0]**2/S_base
    slack=int(net.ext_grid.bus.iloc[0]); n=len(net.bus)
    G=nx.Graph(); lr={}
    for idx,ln in net.line.iterrows():
        if ln.in_service:
            lr[idx]=ln.r_ohm_per_km*ln.length_km/Z_base
            G.add_edge(int(ln.from_bus),int(ln.to_bus),lid=idx)
    ps={slack:set()}
    for b in net.bus.index:
        if b!=slack:
            sp=nx.shortest_path(G,slack,b); ps[b]={G[sp[i]][sp[i+1]]['lid'] for i in range(len(sp)-1)}
    Rcol=np.array([sum(lr[l] for l in (ps[j]&ps[BB])) for j in range(n)])

    df=pd.read_csv("data/aligned_30min_joint.csv",index_col=0,parse_dates=True)
    dmult=(df.demand_mw/df.demand_mw.mean()).to_numpy()
    solar_f=(df.solar_mw/df.solar_mw.max()*SP).to_numpy()
    wind_f =(df.wind_mw /df.wind_mw.max()*WP).to_numpy()
    pp.create_sgen(net,SB,p_mw=0.0,name="solar")
    pp.create_sgen(net,WB,p_mw=0.0,name="wind")
    st=pp.create_storage(net,bus=BB,p_mw=0.0,max_e_mwh=CAP,name="batt")

    def set_grid(i):
        net.load.p_mw=base_p*dmult[i]*LS; net.load.q_mvar=base_q*dmult[i]*LS
        net.sgen.p_mw=[solar_f[i],wind_f[i]]

    scan=pd.read_csv("data/feeder_voltage_scan.csv",index_col=0,parse_dates=True)
    i_over=int(np.argmax(scan.maxV.to_numpy())); i_under=int(np.argmin(scan.minV.to_numpy()))
    i_med=int(np.argsort(scan.maxV.to_numpy())[len(scan)//2])
    sl=SafeLayer(net, st, Rcol, S_base)

    cases=[("median, agent p=0 (safe)",              i_med,   0.0, -P_MAX, P_MAX),
           ("worst-over, agent p=-2 (wrong way)",    i_over, -2.0, -P_MAX, P_MAX),
           ("worst-under, agent p=+1 (wrong way)",   i_under, 1.0, -P_MAX, P_MAX),
           ("worst-over, low SOC: charge<=+1",       i_over,  0.0, -P_MAX, 1.0)]
    for name,i,pa,pl,ph in cases:
        set_grid(i)
        p_safe,info=sl.project(pa,pl,ph)
        extra=info.get("bound",info.get("residual_pu",""))
        print(f"{name}\n  agent={pa:+.2f} -> safe={p_safe:+.3f} | "
              f"V={info['Vmin']:.4f}/{info['Vmax']:.4f} | {info['status']} | {extra} | runpp={info['runpp']}\n")