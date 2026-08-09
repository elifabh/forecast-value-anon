# diag_mpc_forecastvolt.py -- FAIR MPC: base voltage FROM THE FORECAST (delta-LinDistFlow).
# vm_fc[t] = VM_BASE[i] (the real present state, legitimate) + LDF difference (forecast future).
# Expectation: the result lands BETWEEN persistence(7415 @ 19.1%) and perfect-volt(10363 @ 8.7%).
import numpy as np
import networkx as nx
import pandapower.networks as nw
import mpc_econ_voltage_pareto as M

W_ECON = 1.0

# --- R, X matrices (LinDistFlow; a generalisation of the Rcol method) ---
net0 = nw.case33bw()
Z_base = net0.bus.vn_kv.iloc[0]**2 / net0.sn_mva
slack = int(net0.ext_grid.bus.iloc[0]); n = len(net0.bus)
G = nx.Graph(); lr={}; lx={}
for idx, ln in net0.line.iterrows():
    if ln.in_service:
        lr[idx] = ln.r_ohm_per_km*ln.length_km/Z_base
        lx[idx] = ln.x_ohm_per_km*ln.length_km/Z_base
        G.add_edge(int(ln.from_bus), int(ln.to_bus), lid=idx)
ps = {}
for b in net0.bus.index:
    if b == slack: ps[b] = set()
    else:
        sp = nx.shortest_path(G, slack, b)
        ps[b] = {G[sp[i]][sp[i+1]]['lid'] for i in range(len(sp)-1)}
R = np.zeros((n,n)); X = np.zeros((n,n))
for j in range(n):
    for b in range(n):
        c = ps[j] & ps[b]
        R[j,b] = sum(lr[l] for l in c)
        X[j,b] = sum(lx[l] for l in c)

# spread the loads onto the bus axis
Pl_bus = np.zeros(n); Ql_bus = np.zeros(n)
for li, row in net0.load.iterrows():
    Pl_bus[int(row.bus)] += float(M.base_p[li])
    Ql_bus[int(row.bus)] += float(M.base_q[li])

def ldf_v(s, w, dm):
    """LinDistFlow approximate voltage profile (not absolute, to be used for DIFFERENCES)."""
    P = Pl_bus*dm*M.LS; Q = Ql_bus*dm*M.LS      # consumption (+) lowers the voltage
    P[M.SB] -= s; P[M.WB] -= w                   # generation = negative consumption
    return -(R@P + X@Q)/M.S_base

def fc_windows_ldf(i, Hh):
    v_now = ldf_v(M.solar[i], M.wind[i], M.dmult[i])
    vmb = np.empty((Hh, n)); vmb[0] = M.VM_BASE[i]
    for t in range(1, Hh):
        kk = min(t-1, M.solar_fc.shape[1]-1)
        v_fut = ldf_v(M.solar_fc[i,kk], M.wind_fc[i,kk], M.dmult_fc[i,kk])
        vmb[t] = M.VM_BASE[i] + (v_fut - v_now)
    pr = np.empty(Hh); pr[0] = M.price[i]
    for t in range(1, Hh):
        kk = min(t-1, M.price_fc.shape[1]-1); pr[t] = M.price_fc[i,kk]
    return vmb, pr

if __name__ == "__main__":
    # --- sanity: are the forecast scales the same as the real ones? ---
    T0, T1 = M.SPLIT, M.N-1
    sl_t = slice(T0, T1)
    print("SANITY (test-slice means, 1-step forecast vs real):")
    print(f"  solar: real {np.nanmean(M.solar[sl_t]):.3f} | fc {np.nanmean(M.solar_fc[sl_t,0]):.3f}")
    print(f"  wind : real {np.nanmean(M.wind[sl_t]):.3f}  | fc {np.nanmean(M.wind_fc[sl_t,0]):.3f}")
    print(f"  dmult: real {np.nanmean(M.dmult[sl_t]):.3f} | fc {np.nanmean(M.dmult_fc[sl_t,0]):.3f}")
    print("  (if the pairs are not close, STOP and do not use the result)\n")
    
    soc = M.CAP/2; tot=0.0; nn=0; nv=0; nsat=0
    for j, i in enumerate(range(T0, T1)):
        if (j+1)%200==0: print(f"  ... {j+1}/{T1-T0}", flush=True)
        Hh = min(M.H, M.N-i)
        vmb, pr = fc_windows_ldf(i, Hh)
        p_plan,_ = M.solve_window(vmb, pr, soc, W_ECON)
        p_plan = float(np.clip(p_plan,-M.P_MAX,M.P_MAX))
        p_lo = max(-M.P_MAX,-soc*M.ETA/M.dt); p_hi = min(M.P_MAX,(M.CAP-soc)/(M.ETA*M.dt))
        p_plan = float(np.clip(p_plan,p_lo,p_hi))
        M.set_grid(M.solar[i],M.wind[i],M.dmult[i],0.0)
        p_safe,sinfo = M.sl.project(p_plan,p_lo,p_hi)
        if sinfo["status"]=="saturated": nsat+=1
        if sinfo["Vmax"]>M.HI or sinfo["Vmin"]<M.LO: nv+=1
        tot += -M.price[i]*p_safe*M.dt
        soc = min(max(soc+(M.ETA*p_safe*M.dt if p_safe>=0 else p_safe*M.dt/M.ETA),0),M.CAP)
        nn+=1
    
    print(f"\nMPC-FORECASTVOLT (forecast base) | gain {tot:.1f} EUR | violations {nv/nn*100:.2f}% | saturated {nsat}")
    print(f"benchmark: perfect-volt 10363 @ 8.69% | persistence 7415 @ 19.14% | DRL 7524 @ 8.56%")