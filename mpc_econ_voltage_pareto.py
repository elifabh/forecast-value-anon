# mpc_econ_voltage_pareto.py -- STEP 0: voltage-aware economic MPC, PARETO FRONT
#
# Purpose: build a FAIR opponent for the DRL. The MPC solves voltage WITHIN THE PLAN
# as a LinDistFlow constraint (opf_expert logic) + an economic objective. The resulting
# action passes through the real-AC safe layer (symmetric with the DRL: the planner
# avoids + the safe layer guarantees).
#
# METHOD (standard in the literature, weighted-sum + Pareto front):
#   We DO NOT PICK a single "best weight". We SWEEP the economy-voltage weight ratio.
#   Each weight -> one (gain, violation) point -> a Pareto front.
#   The DRL is then placed ON TOP of this front:
#     DRL outside/above the front -> DRL is Pareto-dominant, it genuinely beats the MPC.
#     DRL not above the front     -> DRL is worth no more than a choice of weight, no structural edge.
#
# Single battery at bus17 (single for now; multi-battery depending on the result). Test slice.

import numpy as np, pandas as pd
import cvxpy as cp, warnings
import pandapower as pp, pandapower.networks as nw
from safe_layer import SafeLayer

CAP, dt = 6.0, 0.5
P_MAX = 3.0
RTE = 0.90; ETA = RTE**0.5
SPLIT = 13800
H = 48
LO, HI, MARGIN = 0.95, 1.05, 0.005
SOC_TARGET = CAP/2

LS = 0.5
SP, WP = 3.0, 2.0
SB, WB, BB = 17, 32, 17

df = pd.read_csv("data/aligned_30min_joint.csv", index_col=0, parse_dates=True)
N = len(df)
price = df.price_eur_mwh.to_numpy()
train = df.iloc[:SPLIT]
dmult = (df.demand_mw / train.demand_mw.mean()).to_numpy()
solar = (df.solar_mw / train.solar_mw.max() * SP).to_numpy()
wind  = (df.wind_mw  / train.wind_mw.max()  * WP).to_numpy()

f = np.load("data/forecast_matrix.npz")
price_fc = f["price"]
solar_fc = f["solar"]            # IMPERFECT solar forecast (same information as the DRL -- fair)
wind_fc  = f["wind"]
dmult_fc = f["dmult"]

d = np.load("data/opf_base_vm.npz", allow_pickle=True)
Rcol = d["Rcol"]; S_base = float(d["S_base"])
VM_BASE = d["vm"]   # (N, 33) pre-computed base voltages (battery=0) -> no runpp needed

net = nw.case33bw()
base_p = net.load.p_mw.copy(); base_q = net.load.q_mvar.copy()
pp.create_sgen(net, SB, p_mw=0.0, name="solar")
pp.create_sgen(net, WB, p_mw=0.0, name="wind")
st = pp.create_storage(net, bus=BB, p_mw=0.0, max_e_mwh=CAP, name="batt")
sl = SafeLayer(net, st, Rcol, S_base)

def set_grid(s, w, dm, p_batt):
    net.load.p_mw = base_p * dm * LS
    net.load.q_mvar = base_q * dm * LS
    net.sgen.p_mw = [s, w]
    net.storage.p_mw = p_batt

# --- voltage-aware economic OPF (opf_expert logic + economy weight as a parameter) ---
def solve_window(vm_base, price_w, soc0, w_econ):
    """
    vm_base: (Hh, n) base voltages with battery=0 (forecast-based)
    LinDistFlow: V(t) ~ vm_base[t] - k[t]*p_store[t]
    Objective: W_SLACK*voltage_violation + w_econ*economy + W_THRU + W_TERM
    Large w_econ -> economy dominates, small -> voltage dominates -> Pareto sweep.
    """
    Hh, n = vm_base.shape
    k = Rcol[None,:] / (S_base * np.maximum(vm_base, 1e-3))
    pchg = cp.Variable(Hh, nonneg=True)
    pdis = cp.Variable(Hh, nonneg=True)
    s_hi = cp.Variable((Hh, n), nonneg=True)
    s_lo = cp.Variable((Hh, n), nonneg=True)
    soc  = cp.Variable(Hh+1)
    p_store = pchg - pdis

    W_SLACK, W_THRU, W_TERM = 1e4, 1e-2, 1e-1
    cons = [soc[0]==soc0, pchg<=P_MAX, pdis<=P_MAX, soc>=0, soc<=CAP]
    for t in range(Hh):
        V = vm_base[t] - cp.multiply(k[t], p_store[t])
        cons += [V <= (HI-MARGIN) + s_hi[t],
                 V >= (LO+MARGIN) - s_lo[t],
                 soc[t+1] == soc[t] + ETA*pchg[t]*dt - pdis[t]*dt/ETA]
    obj = (W_SLACK*cp.sum(s_hi+s_lo)
           + w_econ*cp.sum(cp.multiply(price_w, p_store))*dt
           + W_THRU*cp.sum(pchg+pdis)
           + W_TERM*cp.abs(soc[Hh]-SOC_TARGET))
    prob = cp.Problem(cp.Minimize(obj), cons)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.solve(solver=cp.CLARABEL)
        if prob.status not in ("optimal","optimal_inaccurate"):
            prob.solve(solver=cp.ECOS)
    pv = p_store.value
    return (pv[0] if pv is not None else 0.0), prob.status

def fc_windows(i, Hh):
    """base-voltage window (from the pre-computed VM_BASE) + price window.
       NO runpp -> very fast. The voltage window is the real base (a strong break
       for the MPC), the price window is an IMPERFECT forecast (price_fc) -> the
       real tension is in price timing."""
    end = min(i+Hh, len(VM_BASE))
    vmb = VM_BASE[i:end]
    if vmb.shape[0] < Hh:   # overflow past the end of the window
        vmb = np.vstack([vmb, np.repeat(vmb[-1:], Hh-vmb.shape[0], axis=0)])
    pr = np.empty(Hh); pr[0] = price[i]
    for t in range(1, Hh):
        kk = min(t-1, price_fc.shape[1]-1); pr[t] = price_fc[i,kk]
    return vmb, pr

# --- full test run for a single weight ---
def run_one(w_econ):
    T0, T1 = SPLIT, N-1; soc = CAP/2
    total_cost = 0.0
    mn = np.full(T1-T0, np.nan); mx = np.full(T1-T0, np.nan)
    n_sat = 0
    for j, i in enumerate(range(T0, T1)):
        if (j+1) % 500 == 0:
            print(f"      ... w={w_econ:.0e} step {j+1}/{T1-T0}", flush=True)
        Hh = min(H, N-i)
        vmb, pr = fc_windows(i, Hh)
        p_plan, _ = solve_window(vmb, pr, soc, w_econ)
        p_plan = float(np.clip(p_plan, -P_MAX, P_MAX))
        p_lo = max(-P_MAX, -soc*ETA/dt)
        p_hi = min(P_MAX, (CAP-soc)/(ETA*dt))
        p_plan = float(np.clip(p_plan, p_lo, p_hi))
        # real grid + safe layer (symmetric with the DRL)
        set_grid(solar[i], wind[i], dmult[i], 0.0)
        p_safe, sinfo = sl.project(p_plan, p_lo, p_hi)
        if sinfo["status"] == "saturated": n_sat += 1
        mn[j] = sinfo["Vmin"]; mx[j] = sinfo["Vmax"]
        total_cost += price[i]*p_safe*dt        # cost at the REAL price (after safe)
        soc = min(max(soc + (ETA*p_safe*dt if p_safe>=0 else p_safe*dt/ETA), 0), CAP)
        if (j+1) % 200 == 0:
            print(f"      ... {j+1}/{T1-T0} steps", flush=True)
    vr = ((mn < LO) | (mx > HI)).mean() * 100
    return -total_cost, vr, n_sat

# ============================================================
if __name__ == "__main__":
    T0, T1 = SPLIT, N-1; days = (T1-T0)//48
    print("STEP 0: voltage-aware economic MPC -- PARETO FRONT (single battery bus17)\n")
    print(f"  test slice: {T1-T0} steps (~{days} days) | with safe layer (DRL-symmetric)")
    print(f"  method: sweep of the economy weight (w_econ) -> each point (gain, violation)\n")

    # coarse sweep: voltage-dominant (small w_econ) -> economy-dominant (large w_econ)
    # W_SLACK=1e4 fixed. Log-sweep w_econ around 1e-3 (the opf_expert original).
    w_list = [100.0, 1000.0]
    print(f"  {'w_econ':>8} | {'gain(EUR)':>12} | {'viol%':>7} | {'saturated':>9}")
    print("  " + "-"*45)
    results = []
    for w in w_list:
        print(f"  w_econ={w:.0e} computing...", flush=True)
        g, vr, ns = run_one(w)
        results.append((w, g, vr, ns))
        print(f"  {w:8.0e} | {g:12.1f} | {vr:6.2f}% | {ns:9d}", flush=True)
        np.save("data/mpc_pareto_partial.npy", np.array(results, dtype=object))

    print("\n  " + "="*60)
    print("  PARETO FRONT (how do violations change as the gain rises?):")
    print("  " + "="*60)
    print(f"  {'w_econ':>8} | {'gain':>10} | {'viol%':>7}")
    for w, g, vr, ns in sorted(results, key=lambda r: r[1]):
        print(f"  {w:8.0e} | {g:10.0f} | {vr:6.2f}%")
    print("\n  benchmark: pure-arbitrage MPC (no voltage) 24277 | DRL pure-arbitrage ~16500")
    print("  INTERPRETATION:")
    print("   - low w_econ: few violations but low gain (voltage-dominant end)")
    print("   - high w_econ: high gain but many violations (economy-dominant end)")
    print("   - this curve is the DRL's TARGET. If the DRL can rise ABOVE it (more gain at")
    print("     the same violations OR fewer violations at the same gain) -> DRL beats MPC.")
    print("   - if the DRL stays below the curve -> little room for DRL with one battery, move to multi-battery.")
