"""
GG-2 / ROUND v8 SECTION 8 - TASK 1
mv_oberrhein: k sweep with sgen.scaling=1.0 + diagnosis of the k=8 monotonicity
anomaly.

Run (from inside the gg2/ folder):
    python gg2_ob1_ksweep.py 2>&1 | tee ob1_ksweep.log

ASCII only. It only MEASURES; it changes nothing permanent.

-------------------------------------------------------------------------------
PRE-REGISTRATION + HONESTY NOTE
-------------------------------------------------------------------------------
This pre-registration was NOT written blind. Claude made an exploratory run in
its own sandbox (pandapower 3.5.4, Linux) and formulated the hypotheses only
afterwards. This is a DIAGNOSTIC experiment, not a VERDICT experiment; no claim
in the paper rests on it. The decision rules were nevertheless fixed before the
code ran and are evaluated MECHANICALLY. The record = the run on <author>'s
machine (the spec sheet = their versions).

H1 (NUMERICAL): if the spread between the vmax values of DIFFERENT algorithms
    that converge at the same k is > EPS_AGREE -> the anomaly comes from the
    numerical solution.
    Solver set: nr(auto), nr(flat), iwamoto_nr, fdbx, fdxb.
H2 (REACTIVE): if the sum of sgen |q_mvar| is > 0 AND the mode that holds q
    fixed is monotone while the mode that scales q is not -> the anomaly comes
    from the sgen reactive power.
    (If the total q = 0 this test is INVALID and is written up as such.)
H3a (TRAFO): if vmax is NOT monotone in the baseline sweep but IS monotone over
    the same k range once the trafo impedance is reduced by a factor of 10 ->
    the trafo series impedance.
H3b (LINE): the same test for the line impedance.
H0 (NONE): if none of them fires -> "THE CAUSE WAS NOT MEASURED". NO EXPLANATION
    IS FABRICATED.

ANCHOR (the ROUND v8 section 8 table; sgen.scaling=1.0, shipped load):
    k=0 -> vmax 1.0288 / 0 violating buses
    k=1 -> vmax 1.0365 / 0
    k=4 -> vmax 1.0822 / 38
    k=8 -> vmax 1.0789 / 26
If these are not reproduced within ANCHOR_TOL the script prints a WARNING and
work does not continue.

MEASUREMENT SET: the 20 kV buses of component 0. The 110 kV slack bus (318) is
EXCLUDED (its voltage is fixed by definition; counting it as a violation would
be meaningless). This does not change the anchor numbers - that is checked.
-------------------------------------------------------------------------------
"""

import os
import io
import sys
import copy
import time
import contextlib

import numpy as np
import pandas as pd

import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top


# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------
V_HI = 1.05
V_LO = 0.95
MV_KV = 20.0

EPS_AGREE = 1e-3      # solver agreement threshold (pu)
EPS_MONO = 1e-4       # monotonicity tolerance (pu)
ANCHOR_TOL = 5e-4     # anchor tolerance (pu)
MAX_ITER = 1000

K_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5,
          3.75, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# A multiplier ON TOP OF the shipped load.scaling = 0.6. 1.0 = no change.
# 0.7 and 0.5 are FOR DIAGNOSIS ONLY, they are NOT a design decision
# (ROUND v8 section 8 decision 3: "no LS"). Since the demand multiplier in the
# time series will change the load anyway, this axis is informational.
LOAD_MULTS = [1.0, 0.7, 0.5]

# Cross-check solvers: (label, algorithm, runpp kwargs)
SOLVERS = [
    ("nr_auto", "nr", dict(init="auto", calculate_voltage_angles=True)),
    ("nr_flat", "nr", dict(init="flat", calculate_voltage_angles=False)),
    ("iwamoto", "iwamoto_nr", dict(init="auto", calculate_voltage_angles=True)),
    ("fdbx", "fdbx", dict()),
    ("fdxb", "fdxb", dict()),
]

ANCHOR = {0.0: (1.0288, 0), 1.0: (1.0365, 0), 4.0: (1.0822, 38), 8.0: (1.0789, 26)}

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "data", "ob1_ksweep.csv")


def banner(txt):
    print("")
    print("=" * 78)
    print(txt)
    print("=" * 78)


def sub(txt):
    print("")
    print("--- " + txt + " " + "-" * max(0, 70 - len(txt)))


# ----------------------------------------------------------------------------
# 0. SPEC SHEET
# ----------------------------------------------------------------------------
banner("0. SPEC SHEET / ENVIRONMENT")
print("pandapower : %s" % pp.__version__)
print("numpy      : %s" % np.__version__)
print("pandas     : %s" % pd.__version__)
print("python     : %s" % sys.version.split()[0])
try:
    import numba  # noqa: F401
    USE_NUMBA = True
    print("numba      : installed (runpp numba=True)")
except Exception:
    USE_NUMBA = False
    print("numba      : MISSING (runpp numba=False, slower, result IDENTICAL)")
print("V_HI=%.3f V_LO=%.3f EPS_AGREE=%.1e EPS_MONO=%.1e ANCHOR_TOL=%.1e MAX_ITER=%d"
      % (V_HI, V_LO, EPS_AGREE, EPS_MONO, ANCHOR_TOL, MAX_ITER))


# ----------------------------------------------------------------------------
# 1. INVENTORY
# ----------------------------------------------------------------------------
banner("1. GRID INVENTORY (mv_oberrhein, default scenario)")
BASE = pn.mv_oberrhein()

print("shipped sgen.scaling    : %s" % sorted(set(float(x) for x in BASE.sgen.scaling)))
print("shipped load.scaling    : %s" % sorted(set(float(x) for x in BASE.load.scaling)))
print("bus=%d line=%d trafo=%d sgen=%d load=%d ext_grid=%d switch=%d"
      % (len(BASE.bus), len(BASE.line), len(BASE.trafo), len(BASE.sgen),
         len(BASE.load), len(BASE.ext_grid), len(BASE.switch)))
NCTRL = len(BASE.controller) if "controller" in BASE else 0
print("controller count        : %d  (0 => NO automatic tap changer)" % NCTRL)
print("all sgen in_service     : %s" % bool(BASE.sgen.in_service.all()))
print("sgen p_mw total (nom)   : %.6f MW" % BASE.sgen.p_mw.sum())
print("sgen q_mvar total       : %.6f MVAr" % BASE.sgen.q_mvar.sum())
print("sgen q_mvar min / max   : %.6f / %.6f"
      % (BASE.sgen.q_mvar.min(), BASE.sgen.q_mvar.max()))

BASE.sgen["scaling"] = 1.0
print("")
print("APPLIED: sgen.scaling = 1.0   (load.scaling NOT TOUCHED)")

SGEN_P0 = BASE.sgen.p_mw.values.copy()
SGEN_Q0 = BASE.sgen.q_mvar.values.copy()
LOAD_SCALING0 = BASE.load.scaling.values.copy()


# ----------------------------------------------------------------------------
# 2. TOPOLOGY
# ----------------------------------------------------------------------------
banner("2. TOPOLOGY")
_mg = top.create_nxgraph(BASE, respect_switches=True)
_comps = [sorted(int(b) for b in c) for c in top.connected_components(_mg)]
_comps.sort(key=len, reverse=True)
for _i, _c in enumerate(_comps[:5]):
    _eg = [int(b) for b in BASE.ext_grid.bus.values if int(b) in _c]
    print("component %d: %3d buses, ext_grid bus %s" % (_i, len(_c), _eg))

C0_ALL = [b for b in _comps[0] if bool(BASE.bus.at[b, "in_service"])]
C0 = [b for b in C0_ALL if float(BASE.bus.at[b, "vn_kv"]) == MV_KV]
print("")
print("SELECTED component 0 : %d in-service buses" % len(C0_ALL))
print("MEASUREMENT SET      : %d buses (only %.0f kV; HV slack excluded)"
      % (len(C0), MV_KV))

_sg0 = BASE.sgen[BASE.sgen.bus.isin(C0_ALL)]
_ld0 = BASE.load[BASE.load.bus.isin(C0_ALL)]
PV_NOM_C0 = float(_sg0.p_mw.sum())
print("component 0 sgen     : %d of them, nominal %.4f MW" % (len(_sg0), PV_NOM_C0))
print("component 0 load     : %d of them, scaled %.4f MW"
      % (len(_ld0), float((_ld0.p_mw * _ld0.scaling).sum())))

_tr = BASE.trafo[BASE.trafo.lv_bus.isin(C0_ALL)]
TIDX = int(_tr.index[0])
print("component 0 trafo    : idx %d  %d->%d  sn=%.1f MVA  vk=%.3f%%  vkr=%.3f%%"
      "  tap=%s  step=%.2f%%"
      % (TIDX, int(_tr.hv_bus.iloc[0]), int(_tr.lv_bus.iloc[0]),
         float(_tr.sn_mva.iloc[0]), float(_tr.vk_percent.iloc[0]),
         float(_tr.vkr_percent.iloc[0]), str(_tr.tap_pos.iloc[0]),
         float(_tr.tap_step_percent.iloc[0])))

LINES0 = list(BASE.line[BASE.line.from_bus.isin(C0_ALL)
                        & BASE.line.to_bus.isin(C0_ALL)].index)
print("component 0 lines    : %d of them" % len(LINES0))


# ----------------------------------------------------------------------------
# 3. RUN ENGINE
# ----------------------------------------------------------------------------
def make_net(k, dmult=1.0, mode="p", trafo_z_div=1.0, line_z_div=1.0):
    """mode 'p'  : scaling=1, p_mw=k*p0, q_mvar=q0   (q is NOT SCALED)
       mode 'pq' : scaling=k                          (p and q together)"""
    n = copy.deepcopy(BASE)
    if mode == "p":
        n.sgen["scaling"] = 1.0
        n.sgen["p_mw"] = k * SGEN_P0
        n.sgen["q_mvar"] = SGEN_Q0
    elif mode == "pq":
        n.sgen["p_mw"] = SGEN_P0
        n.sgen["q_mvar"] = SGEN_Q0
        n.sgen["scaling"] = k
    else:
        raise ValueError("unknown mode: %s" % mode)

    n.load["scaling"] = LOAD_SCALING0 * dmult

    if trafo_z_div != 1.0:
        n.trafo["sn_mva"] = n.trafo["sn_mva"] * trafo_z_div
    if line_z_div != 1.0:
        n.line["r_ohm_per_km"] = n.line["r_ohm_per_km"] / line_z_div
        n.line["x_ohm_per_km"] = n.line["x_ohm_per_km"] / line_z_div
    return n


def solve(n, algorithm="nr", kwargs=None):
    """Silent solve. Returns: (converged, error_name, iterations)."""
    kw = dict(algorithm=algorithm, max_iteration=MAX_ITER, numba=USE_NUMBA)
    if kwargs:
        kw.update(kwargs)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            pp.runpp(n, **kw)
        conv = bool(n["converged"])
    except Exception as exc:
        return False, type(exc).__name__, np.nan
    it = np.nan
    try:
        it = int(n["_ppc"]["iterations"])
    except Exception:
        pass
    return conv, "", it


NAN_KEYS = ["vmax", "bmax", "vmin", "bmin", "n_over", "n_under", "n_nan",
            "tr_load", "tr_p_hv", "tr_q_hv", "ln_load"]


def measure(n):
    v = n.res_bus.loc[C0, "vm_pu"]
    n_nan = int(v.isna().sum())
    v = v.dropna()
    m = {"vmax": float(v.max()), "bmax": int(v.idxmax()),
         "vmin": float(v.min()), "bmin": int(v.idxmin()),
         "n_over": int((v > V_HI).sum()), "n_under": int((v < V_LO).sum()),
         "n_nan": n_nan}
    try:
        t = n.res_trafo.loc[TIDX]
        m["tr_load"] = float(t.loading_percent)
        m["tr_p_hv"] = float(t.p_hv_mw)
        m["tr_q_hv"] = float(t.q_hv_mvar)
    except Exception:
        m["tr_load"] = m["tr_p_hv"] = m["tr_q_hv"] = np.nan
    try:
        m["ln_load"] = float(n.res_line.loc[LINES0, "loading_percent"].max())
    except Exception:
        m["ln_load"] = np.nan
    return m


def run_point(k, dmult=1.0, mode="p", trafo_z_div=1.0, line_z_div=1.0,
              cross_check=False):
    n = make_net(k, dmult, mode, trafo_z_div, line_z_div)
    conv, err, it = solve(n, "nr", dict(init="auto", calculate_voltage_angles=True))
    row = {"k": k, "dmult": dmult, "mode": mode, "trafo_z_div": trafo_z_div,
           "line_z_div": line_z_div, "conv": conv, "iters": it, "err": err}
    row.update(measure(n) if conv else {key: np.nan for key in NAN_KEYS})

    if cross_check:
        vals = []
        for tag, alg, kw in SOLVERS:
            n2 = make_net(k, dmult, mode, trafo_z_div, line_z_div)
            c2, _e2, _i2 = solve(n2, alg, kw)
            val = float(n2.res_bus.loc[C0, "vm_pu"].max()) if c2 else np.nan
            row["vmax_" + tag] = val
            if not np.isnan(val):
                vals.append(val)
        row["n_solv"] = len(vals)
        row["spread"] = (max(vals) - min(vals)) if len(vals) >= 2 else np.nan
    return row


# ----------------------------------------------------------------------------
# 4. ANCHOR
# ----------------------------------------------------------------------------
banner("4. ANCHOR - REPRODUCING THE ROUND v8 SECTION 8 TABLE")
print("(mode='p', dmult=1.0, measurement set = the 20 kV buses of component 0)")
print("%-6s %-12s %-13s %-9s %-8s %-8s %s"
      % ("k", "vmax_expect", "vmax_measured", "diff", "viol_exp", "viol_obs", "status"))
ANCHOR_OK = True
for _k in sorted(ANCHOR):
    _ev, _en = ANCHOR[_k]
    _r = run_point(_k, 1.0, "p")
    _d = abs(_r["vmax"] - _ev) if _r["conv"] else float("nan")
    _good = bool(_r["conv"] and (_d <= ANCHOR_TOL) and (_r["n_over"] == _en))
    ANCHOR_OK = ANCHOR_OK and _good
    print("%-6.1f %-12.4f %-13.5f %-9.5f %-8d %-8d %s"
          % (_k, _ev, _r["vmax"], _d, _en, _r["n_over"],
             "PASS" if _good else "FAIL"))
print("")
if ANCHOR_OK:
    print("ANCHOR: PASSED - the pipeline reproduces the known numbers.")
else:
    print("ANCHOR: FAILED. There is a version/installation difference.")
    print("DO NOT CONTINUE - paste the output to me exactly as it is.")


# ----------------------------------------------------------------------------
# 5. MAIN SWEEP
# ----------------------------------------------------------------------------
banner("5. MAIN SWEEP - k x load multiplier x mode")
print("The cross-solver check runs ONLY on the reference sweep (dmult=1.0, mode='p').")
_rows = []
_t0 = time.time()
for _dm in LOAD_MULTS:
    for _mode in ["p", "pq"]:
        _cc = bool(_dm == 1.0 and _mode == "p")
        sub("dmult=%.2f (load.scaling=%.3f)  mode=%s%s"
            % (_dm, 0.6 * _dm, _mode, "  [cross-check ON]" if _cc else ""))
        print("%-6s %-5s %-6s %-10s %-6s %-10s %-6s %-7s %-7s %-8s %-8s %s"
              % ("k", "conv", "iter", "vmax", "bus", "vmin", "bus",
                 "n>1.05", "n<0.95", "trafo%", "line%", "spread"))
        for _k in K_GRID:
            _r = run_point(_k, _dm, _mode, cross_check=_cc)
            _rows.append(_r)
            if not _r["conv"]:
                print("%-6.2f %-5s DID NOT CONVERGE (%s)" % (_k, "N", _r["err"]))
                continue
            _sp = ("%.2e" % _r["spread"]) if _cc and not np.isnan(_r["spread"]) else "-"
            print("%-6.2f %-5s %-6s %-10.5f %-6d %-10.5f %-6d %-7d %-7d "
                  "%-8.1f %-8.1f %s"
                  % (_k, "Y", str(_r["iters"]), _r["vmax"], _r["bmax"],
                     _r["vmin"], _r["bmin"], _r["n_over"], _r["n_under"],
                     _r["tr_load"], _r["ln_load"], _sp))
print("")
print("sweep duration %.1f s, %d points" % (time.time() - _t0, len(_rows)))
DF = pd.DataFrame(_rows)


# ----------------------------------------------------------------------------
# 6. COUNTERFACTUAL IMPEDANCE TESTS
# ----------------------------------------------------------------------------
banner("6. COUNTERFACTUAL - IMPEDANCE TESTS (dmult=1.0, mode='p')")
_cfr = []
for _tag, _tz, _lz in [("CF1_trafo_Z_div10", 10.0, 1.0),
                       ("CF2_line_Z_div10", 1.0, 10.0)]:
    sub(_tag)
    print("%-6s %-5s %-10s %-10s %-7s %-8s %s"
          % ("k", "conv", "vmax", "vmin", "n>1.05", "trafo%", "line%"))
    for _k in K_GRID:
        _r = run_point(_k, 1.0, "p", trafo_z_div=_tz, line_z_div=_lz)
        _r["cf"] = _tag
        _cfr.append(_r)
        if _r["conv"]:
            print("%-6.2f %-5s %-10.5f %-10.5f %-7d %-8.1f %.1f"
                  % (_k, "Y", _r["vmax"], _r["vmin"], _r["n_over"],
                     _r["tr_load"], _r["ln_load"]))
        else:
            print("%-6.2f %-5s DID NOT CONVERGE (%s)" % (_k, "N", _r["err"]))
DF_CF = pd.DataFrame(_cfr)


# ----------------------------------------------------------------------------
# 7. VERDICT
# ----------------------------------------------------------------------------
def is_monotone(df):
    d = df[df["conv"] == True].sort_values("k")          # noqa: E712
    if len(d) < 3:
        return None, float("nan"), float("nan")
    dv = np.diff(d["vmax"].values)
    worst = float(dv.min())
    kbad = float(d["k"].values[1:][int(np.argmin(dv))])
    return bool(worst >= -EPS_MONO), worst, kbad


banner("7. VERDICT - PRE-REGISTERED DECISION RULES")
REF = DF[(DF["dmult"] == 1.0) & (DF["mode"] == "p")]
REF_PQ = DF[(DF["dmult"] == 1.0) & (DF["mode"] == "pq")]
MONO_P, W_P, KB_P = is_monotone(REF)
MONO_PQ, W_PQ, KB_PQ = is_monotone(REF_PQ)
print("Baseline sweep (dmult=1.0):")
print("  mode='p'  monotone=%s  worst_step=%+.5f pu  (k=%.2f)" % (MONO_P, W_P, KB_P))
print("  mode='pq' monotone=%s  worst_step=%+.5f pu  (k=%.2f)" % (MONO_PQ, W_PQ, KB_PQ))

sub("H1 NUMERICAL")
print("%-6s %s" % ("k", "  ".join("%-10s" % t for t, _a, _k in SOLVERS) + "  spread"))
for _, _r in REF.sort_values("k").iterrows():
    _cells = []
    for _t, _a, _kw in SOLVERS:
        _v = _r.get("vmax_" + _t, np.nan)
        _cells.append("%-10s" % ("NONE" if np.isnan(_v) else "%.6f" % _v))
    _sp = _r.get("spread", np.nan)
    print("%-6.2f %s  %s" % (_r["k"], "  ".join(_cells),
                             "-" if np.isnan(_sp) else "%.2e" % _sp))
_cc = REF[REF["n_solv"] >= 2]
MAX_SPREAD = float(_cc["spread"].max()) if len(_cc) else float("nan")
print("")
print("LARGEST vmax spread among converging solvers        : %.3e pu" % MAX_SPREAD)
print("EPS_AGREE                                           : %.3e pu" % EPS_AGREE)
H1 = bool(MAX_SPREAD > EPS_AGREE)
print("did H1 fire: %s" % H1)
print("NOTE: a solver failing to converge is NOT H1 ON ITS OWN;")
print("      the CONVERGING solutions must DISAGREE.")

sub("H2 REACTIVE")
QSUM = float(np.abs(SGEN_Q0).sum())
print("sgen |q_mvar| total: %.6f MVAr" % QSUM)
if QSUM == 0.0:
    print("q is ZERO at every sgen -> the 'p' and 'pq' modes build the SAME grid.")
    print("H2 CANNOT BE TESTED (invalid). did H2 fire: False")
    H2 = False
else:
    H2 = bool((MONO_P is True) and (MONO_PQ is False))
    print("did H2 fire: %s" % H2)

sub("H3 GRID LOADABILITY")
CF1 = DF_CF[DF_CF["cf"] == "CF1_trafo_Z_div10"]
CF2 = DF_CF[DF_CF["cf"] == "CF2_line_Z_div10"]
MONO_CF1, W_CF1, _ = is_monotone(CF1)
MONO_CF2, W_CF2, _ = is_monotone(CF2)
print("CF1 trafo Z/10 : monotone=%s  worst_step=%+.5f pu" % (MONO_CF1, W_CF1))
print("CF2 line  Z/10 : monotone=%s  worst_step=%+.5f pu" % (MONO_CF2, W_CF2))
H3A = bool((MONO_P is False) and (MONO_CF1 is True))
H3B = bool((MONO_P is False) and (MONO_CF2 is True))
print("did H3a (trafo) fire: %s" % H3A)
print("did H3b (line)  fire: %s" % H3B)

sub("SUPPORTING MEASUREMENTS (facts, NOT an explanation)")
print("%-6s %-9s %-9s %-9s %-10s %-9s %s"
      % ("k", "vmax", "vmin", "trafo%", "trafo_p_hv", "trafo_q_hv", "line%"))
for _, _r in REF[REF["conv"] == True].sort_values("k").iterrows():   # noqa: E712
    print("%-6.2f %-9.5f %-9.5f %-9.1f %-10.2f %-9.2f %.1f"
          % (_r["k"], _r["vmax"], _r["vmin"], _r["tr_load"],
             _r["tr_p_hv"], _r["tr_q_hv"], _r["ln_load"]))

sub("CONCLUDING SENTENCE")
if H1:
    print("VERDICT: the anomaly is NUMERICAL. Converging solutions differ by %.2e pu" % MAX_SPREAD)
    print("(threshold %.2e). The high-k points are NOT trustworthy." % EPS_AGREE)
elif H2:
    print("VERDICT: the anomaly comes from scaling the sgen REACTIVE POWER.")
    print("The pipeline must scale ONLY p_mw (mode='p').")
elif H3A or H3B:
    _w = []
    if H3A:
        _w.append("trafo series impedance")
    if H3B:
        _w.append("line impedance")
    print("VERDICT: the anomaly comes from GRID LOADABILITY (%s)." % " and ".join(_w))
    print("The drop in vmax is not a measurement error; monotonicity returns once")
    print("the impedance is reduced. High k IS NOT A VALID OPERATING POINT.")
else:
    print("VERDICT: THE CAUSE WAS NOT MEASURED. No pre-registered hypothesis fired.")
    print("NO EXPLANATION IS WRITTEN. Further measurement is required.")


# ----------------------------------------------------------------------------
# 8. k SELECTION TABLES
# ----------------------------------------------------------------------------
banner("8. k SELECTION TABLES")
print("Criteria: (a) is there over-voltage  (b) is it thermally defensible")
print("          (trafo <= 100 and line <= 100)  (c) vmin >= %.2f" % V_LO)

sub("8a. BY LOAD MULTIPLIER (mode='p')")
OK_BY_DM = {}
for _dm in LOAD_MULTS:
    _b = DF[(DF["dmult"] == _dm) & (DF["mode"] == "p")
            & (DF["conv"] == True)].sort_values("k")      # noqa: E712
    _ok = []
    for _, _r in _b.iterrows():
        if (_r["n_over"] > 0 and _r["tr_load"] <= 100.0
                and _r["ln_load"] <= 100.0 and _r["vmin"] >= V_LO):
            _ok.append(float(_r["k"]))
    OK_BY_DM[_dm] = _ok
    print("dmult=%.2f -> usable k: %s"
          % (_dm, ("%.2f - %.2f" % (min(_ok), max(_ok))) if _ok else "NONE"))

sub("8b. ALL LOAD MULTIPLIERS TOGETHER (worst case)")
print("The thermal check is done at the dmult giving the HIGHEST loading;")
print("for the violation check it is enough that a violation occurs at any dmult.")
print("")
print("%-6s %-9s %-10s %-10s %-10s %-10s %s"
      % ("k", "PV MW", "max_trafo%", "max_line%", "min_vmin", "max_n>1.05", "status"))
OK_K = []
for _k in K_GRID:
    _s = DF[(DF["k"] == _k) & (DF["mode"] == "p") & (DF["conv"] == True)]  # noqa: E712
    if len(_s) < len(LOAD_MULTS):
        print("%-6.2f %-9.2f  (DID NOT CONVERGE at some load multipliers - skipped)"
              % (_k, _k * PV_NOM_C0))
        continue
    _tr = float(_s["tr_load"].max())
    _ln = float(_s["ln_load"].max())
    _vn = float(_s["vmin"].min())
    _no = int(_s["n_over"].max())
    if _no == 0:
        _st = "no violation"
    elif _tr > 100.0 or _ln > 100.0:
        _st = "THERMAL OVER"
    elif _vn < V_LO:
        _st = "VOLTAGE COLLAPSE"
    else:
        _st = "USABLE"
        OK_K.append(_k)
    print("%-6.2f %-9.2f %-10.1f %-10.1f %-10.5f %-10d %s"
          % (_k, _k * PV_NOM_C0, _tr, _ln, _vn, _no, _st))
print("")
if OK_K:
    print("USABLE k (thermally sound at every load multiplier): %.2f - %.2f"
          % (min(OK_K), max(OK_K)))
else:
    print("WARNING: no k both produces violations and is thermally defensible")
    print("at every load multiplier. This is a DESIGN CONSTRAINT.")
print("")
print("LIMIT: these tables are a SINGLE SNAPSHOT (peak PV, fixed load).")
print("The final k is calibrated with a TIME SERIES run so that the naive arm gives")
print("the target violation rate. This table only NARROWS THE RANGE.")


# ----------------------------------------------------------------------------
# 9. SAVE
# ----------------------------------------------------------------------------
banner("9. SAVE")
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
OUT = pd.concat([DF.assign(cf="BASE"), DF_CF], ignore_index=True, sort=False)
OUT.to_csv(OUT_CSV, index=False)
print("written: %s  (%d rows)" % (OUT_CSV, len(OUT)))
print("")
print("ANCHOR          : %s" % ("PASSED" if ANCHOR_OK else "FAILED"))
print("H1/H2/H3a/H3b   : %s / %s / %s / %s" % (H1, H2, H3A, H3B))
print("usable k        : %s" % (OK_K if OK_K else "NONE"))
print("")
print("DONE.")
