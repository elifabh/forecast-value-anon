# mpc_ob.py -- PARAMETRIC (feeder-independent) versions of
# mpc_econ_voltage_pareto.solve_window and diag_mpc_forecastvolt.ldf_v.
#
# WHY PARAMETRIC: in their original form both functions depend on case33bw
# constants (Rcol, S_base, P_MAX, CAP, LS, SB, WB) at MODULE LEVEL. Rather than
# copying and modifying them for mv_oberrhein, I parameterised the same LOGIC
# through arguments.
# ANCHOR: when called with the case33bw constants they MUST give EXACTLY the
# same result as the original functions (gg2_ob7_ldf_time.py tests this).
#
# The weights (W_SLACK, W_THRU, W_TERM) and the horizon logic were NOT CHANGED.

import warnings

import numpy as np
import networkx as nx
import cvxpy as cp

W_SLACK, W_THRU, W_TERM = 1e4, 1e-2, 1e-1


# ---------------------------------------------------------------------------
# LinDistFlow R / X matrices
# ---------------------------------------------------------------------------
def build_rx(net, root, buses, z_base, open_lines=None, restrict=None):
    """Generalisation of the construction in diag_mpc_forecastvolt.py.

    net         : pandapower network
    root        : root of the path sets (the ext_grid bus for case33bw,
                  the 20 kV supply bus 319 for mv_oberrhein)
    buses       : list of buses forming the ROW/COLUMN axis of the matrix
    z_base      : vn_kv**2 / sn_mva
    open_lines  : indices of lines separated by an OPEN switch (removed from
                  the graph). Empty for case33bw; 6 of them for mv_oberrhein.
    restrict    : set of buses allowed when building the graph (None = all)

    Returns: R, X  (len(buses) x len(buses)), and the graph used.
    """
    open_lines = set() if open_lines is None else set(int(x) for x in open_lines)
    G = nx.Graph()
    lr, lx = {}, {}
    for idx, ln in net.line.iterrows():
        if not bool(ln.in_service) or int(idx) in open_lines:
            continue
        fb, tb = int(ln.from_bus), int(ln.to_bus)
        if restrict is not None and (fb not in restrict or tb not in restrict):
            continue
        lr[idx] = float(ln.r_ohm_per_km) * float(ln.length_km) / z_base
        lx[idx] = float(ln.x_ohm_per_km) * float(ln.length_km) / z_base
        G.add_edge(fb, tb, lid=idx)

    ps = {}
    for b in buses:
        if b == root:
            ps[b] = set()
            continue
        sp = nx.shortest_path(G, root, b)
        ps[b] = {G[sp[t]][sp[t + 1]]["lid"] for t in range(len(sp) - 1)}

    n = len(buses)
    R = np.zeros((n, n))
    X = np.zeros((n, n))
    for a, ba in enumerate(buses):
        for b, bb in enumerate(buses):
            common = ps[ba] & ps[bb]
            R[a, b] = sum(lr[l] for l in common)
            X[a, b] = sum(lx[l] for l in common)
    return R, X, G


def ldf_v(R, X, P, Q, s_base):
    """LinDistFlow approximate voltage DEVIATION (not absolute - for DIFFERENCES).
    P, Q: NET CONSUMPTION on the bus axis (generation enters as negative
    consumption)."""
    return -(R @ P + X @ Q) / s_base


# ---------------------------------------------------------------------------
# MPC window
# ---------------------------------------------------------------------------
def solve_window(vm_base, price_w, soc0, w_econ, Rcol, s_base, p_max, cap,
                 eta, dt, hi, lo, margin, soc_target):
    """EXACTLY the same logic as mpc_econ_voltage_pareto.solve_window.
    The only difference: the constants come from the ARGUMENTS rather than from
    module level."""
    Hh, n = vm_base.shape
    k = Rcol[None, :] / (s_base * np.maximum(vm_base, 1e-3))
    pchg = cp.Variable(Hh, nonneg=True)
    pdis = cp.Variable(Hh, nonneg=True)
    s_hi = cp.Variable((Hh, n), nonneg=True)
    s_lo = cp.Variable((Hh, n), nonneg=True)
    soc = cp.Variable(Hh + 1)
    p_store = pchg - pdis

    cons = [soc[0] == soc0, pchg <= p_max, pdis <= p_max, soc >= 0, soc <= cap]
    for t in range(Hh):
        V = vm_base[t] - cp.multiply(k[t], p_store[t])
        cons += [V <= (hi - margin) + s_hi[t],
                 V >= (lo + margin) - s_lo[t],
                 soc[t + 1] == soc[t] + eta * pchg[t] * dt - pdis[t] * dt / eta]
    obj = (W_SLACK * cp.sum(s_hi + s_lo)
           + w_econ * cp.sum(cp.multiply(price_w, p_store)) * dt
           + W_THRU * cp.sum(pchg + pdis)
           + W_TERM * cp.abs(soc[Hh] - soc_target))
    prob = cp.Problem(cp.Minimize(obj), cons)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.solve(solver=cp.CLARABEL)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            prob.solve(solver=cp.ECOS)
    pv = p_store.value
    return (pv[0] if pv is not None else 0.0), prob.status