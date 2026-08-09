#!/usr/bin/env python3
"""
GG-2 / ROUND v8 SECTION 8 - TASK 1c
mv_oberrhein k* bisection - with the SELECTION BUG in gg2_ob2_kcal.py fixed.

Run (from the GG-1 ROOT folder):
    python gg2/gg2_ob3_kstar.py 2>&1 | tee ob3_kstar.log

ASCII only. ~6 min (6 passes x 2429 steps).

-------------------------------------------------------------------------------
BUG (gg2_ob2_kcal.py, section 6) - FIXED
-------------------------------------------------------------------------------
At k=1.00 there was no export step -> tr_max_exp = nan -> `nan <= 100` is False
in Python -> k=1.00 was wrongly counted as "THERMAL OVERLOAD". The bracket was
built backwards (k_ok=2.750 > k_over=1.000), the bisection walked DOWNWARD,
k*=1.219 came out and the "zero over-voltage" exception was triggered.
THE VERDICT WAS INVALID.

FIX:
  1. If n_export == 0 the thermal criterion is TRIVIALLY SATISFIED (no
     PV-driven export -> no PV-driven thermal overload either). nan no longer
     yields False.
  2. Bracket monotonicity check: if k_over > k_ok does not hold, the script STOPS.
None of the measured numbers changed; only the selection logic changed.

BRACKET (read from table 7 of gg2_ob2_kcal.py):
  k=2.750 -> export trafo  92.8%, line  89.7%  -> THERMAL OK
  k=3.000 -> export trafo 107.8%, line 101.8%  -> OVERLOAD
  => k* lies in the interval (2.75, 3.00]. Only this interval is swept.

-------------------------------------------------------------------------------
PRE-REGISTRATION (unchanged, same as ROUND v8 TASK 1b)
-------------------------------------------------------------------------------
k* = the LARGEST k for which, at EVERY step of the evaluation window where the
     trafo EXPORTS (p_hv < 0), the trafo stays <= 100.0 % AND the max line stays
     <= 100.0 %.
The violation rate is NOT TARGETED; it is MEASURED after k* is found.
The criterion will not be relaxed after this run.

ANCHORS (no verdict unless all four pass):
  A1 M.SPLIT == 13800, window 2429 steps
  A2 raw M.solar on the test block: mean 0.864 MW, peak 3.749 MW
  A3 static k=4.0 -> vmax 1.08218 / 38 buses   (gg2_ob1_ksweep.py)
  A4 time series k=2.750 and k=3.000 -> exactly the numbers in the ob2 log
     (export trafo/line, over%, vmax)
-------------------------------------------------------------------------------
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top

V_HI, V_LO, MV_KV = 1.05, 0.95, 20.0
THERM_LIMIT = 100.0
K_LO_SEED, K_HI_SEED = 2.75, 3.00
BISECT_ITERS = 4
PROGRESS_EVERY = 800

ANCHOR_SPLIT, ANCHOR_NSTEP = 13800, 2429
ANCHOR_SOLAR_MEAN, ANCHOR_SOLAR_PEAK, ANCHOR_SOLAR_TOL = 0.864, 3.749, 0.005
ANCHOR_STATIC_K, ANCHOR_STATIC_VMAX, ANCHOR_STATIC_NOVER = 4.0, 1.08218, 38
ANCHOR_STATIC_TOL = 5e-4

# A4: from the gg2_ob2_kcal.py log (k -> tr_exp, ln_exp, over_pct, vmax, n_export)
ANCHOR_TS = {
    2.750: (92.8, 89.7, 13.87, 1.0819, 523),
    3.000: (107.8, 101.8, 16.80, 1.0857, 586),
}
TOL_LOAD, TOL_PCT, TOL_VMAX = 0.05, 0.005, 5e-5

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "data", "ob3_kstar.csv")
OUT_NPZ = os.path.join(HERE, "data", "ob3_kstar_traj.npz")


def banner(t):
    print("")
    print("=" * 78)
    print(t)
    print("=" * 78)


banner("0. SPEC SHEET")
print("pandapower %s | numpy %s | pandas %s | python %s"
      % (pp.__version__, np.__version__, pd.__version__, sys.version.split()[0]))
try:
    import numba  # noqa: F401
    USE_NUMBA = True
except Exception:
    USE_NUMBA = False
print("numba=%s | THERM_LIMIT=%.1f%% | bracket (%.3f, %.3f] | bisect=%d"
      % (USE_NUMBA, THERM_LIMIT, K_LO_SEED, K_HI_SEED, BISECT_ITERS))


banner("1. SERIES - IMPORTED FROM mpc_econ_voltage_pareto")
try:
    import mpc_econ_voltage_pareto as M
except Exception as exc:
    print("[ERROR] import failed: %s: %s" % (type(exc).__name__, exc))
    print("Run from the GG-1 ROOT folder: python gg2/gg2_ob3_kstar.py")
    sys.exit(1)

T0, T1 = int(M.SPLIT), int(M.N) - 1
NSTEP = T1 - T0
SOLAR_NORM = np.asarray(M.solar, dtype=float) / float(M.SP)
DMULT = np.asarray(M.dmult, dtype=float)
PEAK_NORM = ANCHOR_SOLAR_PEAK / float(M.SP)
print("window %d:%d (%d steps) | solar test mean %.4f MW peak %.4f MW"
      % (T0, T1, NSTEP, np.asarray(M.solar)[T0:T1].mean(),
         np.asarray(M.solar)[T0:T1].max()))
print("dmult test min %.4f mean %.4f max %.4f"
      % (DMULT[T0:T1].min(), DMULT[T0:T1].mean(), DMULT[T0:T1].max()))


banner("2. ANCHOR A1 + A2")
A1 = bool(T0 == ANCHOR_SPLIT and NSTEP == ANCHOR_NSTEP)
_sm = float(np.asarray(M.solar, dtype=float)[T0:T1].mean())
_sp = float(np.asarray(M.solar, dtype=float)[T0:T1].max())
A2 = bool(abs(_sm - ANCHOR_SOLAR_MEAN) <= ANCHOR_SOLAR_TOL
          and abs(_sp - ANCHOR_SOLAR_PEAK) <= ANCHOR_SOLAR_TOL)
print("A1 %s | A2 %s (mean %.4f, peak %.4f)"
      % ("PASS" if A1 else "FAIL", "PASS" if A2 else "FAIL", _sm, _sp))


banner("3. mv_oberrhein SETUP")
NET = pn.mv_oberrhein()
NET.sgen["scaling"] = 1.0
SGEN_P0 = NET.sgen.p_mw.values.copy()
SGEN_Q0 = NET.sgen.q_mvar.values.copy()
LOAD_SCALING0 = NET.load.scaling.values.copy()

_mg = top.create_nxgraph(NET, respect_switches=True)
_comps = [sorted(int(b) for b in c) for c in top.connected_components(_mg)]
_comps.sort(key=len, reverse=True)
C0_ALL = [b for b in _comps[0] if bool(NET.bus.at[b, "in_service"])]
C0 = [b for b in C0_ALL if float(NET.bus.at[b, "vn_kv"]) == MV_KV]
TIDX = int(NET.trafo[NET.trafo.lv_bus.isin(C0_ALL)].index[0])
LINES0 = list(NET.line[NET.line.from_bus.isin(C0_ALL)
                       & NET.line.to_bus.isin(C0_ALL)].index)
PV_NOM_C0 = float(NET.sgen[NET.sgen.bus.isin(C0_ALL)].p_mw.sum())
LOAD_C0 = float((NET.load[NET.load.bus.isin(C0_ALL)].p_mw
                 * NET.load[NET.load.bus.isin(C0_ALL)].scaling).sum())
print("component 0: %d buses, %d measured buses | trafo %d (%.1f MVA) | lines %d"
      % (len(C0_ALL), len(C0), TIDX, float(NET.trafo.at[TIDX, "sn_mva"]), len(LINES0)))
print("PV nominal %.4f MW | load %.4f MW" % (PV_NOM_C0, LOAD_C0))


def solve_step(k, s_norm, dm):
    NET.sgen["p_mw"] = k * SGEN_P0 * s_norm
    NET.sgen["q_mvar"] = SGEN_Q0
    NET.load["scaling"] = LOAD_SCALING0 * dm
    try:
        pp.runpp(NET, algorithm="nr", init="auto", calculate_voltage_angles=True,
                 max_iteration=100, numba=USE_NUMBA)
    except Exception:
        return False, np.nan, np.nan, -1, np.nan, np.nan, np.nan
    v = NET.res_bus.loc[C0, "vm_pu"]
    if bool(v.isna().any()):
        return False, np.nan, np.nan, -1, np.nan, np.nan, np.nan
    return (True, float(v.max()), float(v.min()), int(v.idxmax()),
            float(NET.res_trafo.at[TIDX, "loading_percent"]),
            float(NET.res_line.loc[LINES0, "loading_percent"].max()),
            float(NET.res_trafo.at[TIDX, "p_hv_mw"]))


banner("4. ANCHOR A3 - STATIC MATCH")
_ok, _vmax, _vmin, _bm, _tr, _ln, _ph = solve_step(ANCHOR_STATIC_K, 1.0, 1.0)
_nover = int((NET.res_bus.loc[C0, "vm_pu"] > V_HI).sum())
A3 = bool(_ok and abs(_vmax - ANCHOR_STATIC_VMAX) <= ANCHOR_STATIC_TOL
          and _nover == ANCHOR_STATIC_NOVER)
print("k=4.0 static: vmax %.5f (exp %.5f) | violations %d (exp %d) -> %s"
      % (_vmax, ANCHOR_STATIC_VMAX, _nover, ANCHOR_STATIC_NOVER,
         "PASS" if A3 else "FAIL"))

if not (A1 and A2 and A3):
    print("")
    print("One of A1/A2/A3 FAILED. STOP, paste the output to me.")
    sys.exit(2)


TRAJ = {}


def run_series(k):
    vmax = np.full(NSTEP, np.nan)
    vmin = np.full(NSTEP, np.nan)
    trl = np.full(NSTEP, np.nan)
    lnl = np.full(NSTEP, np.nan)
    phv = np.full(NSTEP, np.nan)
    bmx = np.full(NSTEP, -1, dtype=int)
    bad = 0
    t0 = time.time()
    for j in range(NSTEP):
        i = T0 + j
        ok, vh, vl, bm, tr, ln, ph = solve_step(k, float(SOLAR_NORM[i]),
                                                float(DMULT[i]))
        if not ok:
            bad += 1
            continue
        vmax[j], vmin[j], trl[j], lnl[j], phv[j], bmx[j] = vh, vl, tr, ln, ph, bm
        if (j + 1) % PROGRESS_EVERY == 0:
            el = max(time.time() - t0, 1e-9)
            print("    k=%.4f  %d/%d  (~%.1f min left)"
                  % (k, j + 1, NSTEP, (NSTEP - j - 1) / ((j + 1) / el) / 60),
                  flush=True)

    good = ~np.isnan(vmax)
    ng = int(good.sum())
    exp = good & (phv < 0.0)
    imp = good & (phv >= 0.0)
    ne, ni = int(exp.sum()), int(imp.sum())

    def mx(a, m):
        return float(a[m].max()) if int(m.sum()) else np.nan

    tr_e, ln_e = mx(trl, exp), mx(lnl, exp)
    # BUG FIX: if there is no export step there is no PV-driven thermal overload.
    if ne == 0:
        thermal_ok = True
    else:
        thermal_ok = bool(tr_e <= THERM_LIMIT and ln_e <= THERM_LIMIT)

    r = {"k": k, "n_ok": ng, "n_fail": bad, "n_export": ne, "n_import": ni,
         "tr_max_exp": tr_e, "ln_max_exp": ln_e,
         "tr_max_imp": mx(trl, imp), "ln_max_imp": mx(lnl, imp),
         "vmax_max": float(np.nanmax(vmax)) if ng else np.nan,
         "vmin_min": float(np.nanmin(vmin)) if ng else np.nan,
         "over_pct": float((vmax[good] > V_HI).mean() * 100) if ng else np.nan,
         "under_pct": float((vmin[good] < V_LO).mean() * 100) if ng else np.nan,
         "bus147_share": float((bmx[good & (bmx >= 0)] == 147).mean() * 100)
         if ng else np.nan,
         "thermal_ok": thermal_ok, "secs": round(time.time() - t0, 1)}
    TRAJ["vmax_k%.4f" % k] = vmax
    TRAJ["trl_k%.4f" % k] = trl
    TRAJ["lnl_k%.4f" % k] = lnl
    TRAJ["phv_k%.4f" % k] = phv
    return r


def show(r):
    print("  k=%.4f | EXP trafo %6.1f%% line %6.1f%% (%d steps) | imp trafo %6.1f%%"
          " | over %5.2f%% | under %5.2f%% | vmax %.4f | %s | %.0f s"
          % (r["k"], r["tr_max_exp"], r["ln_max_exp"], r["n_export"],
             r["tr_max_imp"], r["over_pct"], r["under_pct"], r["vmax_max"],
             "THERMAL OK" if r["thermal_ok"] else "THERMAL OVER", r["secs"]),
          flush=True)


banner("5. ANCHOR A4 - REPRODUCE THE ob2 TIME-SERIES NUMBERS")
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
RES = []
A4 = True
for _k in sorted(ANCHOR_TS):
    print("")
    print("  --> k=%.3f" % _k)
    _r = run_series(_k)
    RES.append(_r)
    show(_r)
    e_tr, e_ln, e_ov, e_vm, e_ne = ANCHOR_TS[_k]
    _d = [abs(_r["tr_max_exp"] - e_tr) <= TOL_LOAD,
          abs(_r["ln_max_exp"] - e_ln) <= TOL_LOAD,
          abs(_r["over_pct"] - e_ov) <= TOL_PCT,
          abs(_r["vmax_max"] - e_vm) <= TOL_VMAX,
          _r["n_export"] == e_ne]
    print("     expected: trafo %.1f line %.1f over %.2f vmax %.4f exp_steps %d"
          % (e_tr, e_ln, e_ov, e_vm, e_ne))
    print("     A4 for this k: %s  %s"
          % ("PASS" if all(_d) else "FAIL",
             "" if all(_d) else "(mismatching fields: %s)"
             % [n for n, v in zip(["trafo", "line", "over", "vmax", "exp_steps"], _d)
                if not v]))
    A4 = A4 and all(_d)
    pd.DataFrame(RES).to_csv(OUT_CSV, index=False)

print("")
print("A4 -> %s" % ("PASS" if A4 else "FAIL"))
if not A4:
    print("The ob2 numbers could not be reproduced. STOP, paste the output to me.")
    sys.exit(3)


banner("6. BISECTION (corrected selection logic)")
LO = [r for r in RES if abs(r["k"] - K_LO_SEED) < 1e-9][0]
HI = [r for r in RES if abs(r["k"] - K_HI_SEED) < 1e-9][0]
if not LO["thermal_ok"]:
    print("k=%.3f is not thermally OK - the bracket is wrong. STOP." % K_LO_SEED)
    sys.exit(4)
if HI["thermal_ok"]:
    print("k=%.3f came out thermally OK - the bracket must be widened upward. STOP."
          % K_HI_SEED)
    sys.exit(4)
lo, hi = K_LO_SEED, K_HI_SEED
print("start: k_ok=%.4f  k_over=%.4f  (check: k_over > k_ok -> %s)"
      % (lo, hi, hi > lo))
for it in range(BISECT_ITERS):
    mid = 0.5 * (lo + hi)
    print("")
    print("  --> bisection %d/%d, k=%.4f" % (it + 1, BISECT_ITERS, mid))
    r = run_series(mid)
    RES.append(r)
    show(r)
    if r["thermal_ok"]:
        lo = mid
    else:
        hi = mid
    if not hi > lo:
        print("  [ERROR] the bracket broke: lo=%.4f hi=%.4f. STOP." % (lo, hi))
        sys.exit(5)
    pd.DataFrame(RES).to_csv(OUT_CSV, index=False)
KSTAR = lo
print("")
print("k* = %.4f   (resolution %.4f, must be positive)" % (KSTAR, hi - lo))


banner("7. TABLE")
DF = pd.DataFrame(RES).sort_values("k")
print("%-9s %-9s %-11s %-10s %-9s %-9s %-9s %-9s %s"
      % ("k", "PV_peak", "exp_trafo%", "exp_line%", "exp_steps", "over%",
         "under%", "vmax", "thermal"))
for _, r in DF.iterrows():
    print("%-9.4f %-9.2f %-11.1f %-10.1f %-9d %-9.2f %-9.2f %-9.4f %s"
          % (r["k"], r["k"] * PEAK_NORM * PV_NOM_C0, r["tr_max_exp"],
             r["ln_max_exp"], r["n_export"], r["over_pct"], r["under_pct"],
             r["vmax_max"], "OK" if r["thermal_ok"] else "OVER"))


banner("8. VERDICT")
S = DF[np.isclose(DF["k"], KSTAR)].iloc[0]
PVPK = KSTAR * PEAK_NORM * PV_NOM_C0
print("k*                      : %.4f" % KSTAR)
print("PV peak (test block)    : %.2f MW   (component 0 load %.2f MW)"
      % (PVPK, LOAD_C0))
print("PV_peak / load          : %.2f   (same ratio for case33bw: 1.61)"
      % (PVPK / LOAD_C0))
print("PV train-peak / load    : %.2f   (case33bw: 1.61)"
      % (KSTAR * PV_NOM_C0 / LOAD_C0))
print("EXPORT trafo highest    : %.1f%%  (%d export steps)"
      % (S["tr_max_exp"], S["n_export"]))
print("EXPORT line highest     : %.1f%%" % S["ln_max_exp"])
print("import trafo highest    : %.1f%%  (non-PV baseline)" % S["tr_max_imp"])
print("")
print("UNCONTROLLED (no battery, no controller) violation rates:")
print("  over-voltage : %.2f%%   under-voltage: %.2f%%"
      % (S["over_pct"], S["under_pct"]))
print("  share of steps whose vmax bus is 147: %.1f%%" % S["bus147_share"])
print("  highest vmax %.4f | lowest vmin %.4f" % (S["vmax_max"], S["vmin_min"]))
print("")
print("Benchmark (case33bw, ROUND v8): uncontrolled ~%22-23 -> naive MPC+safe %11.28")
print("")
if S["over_pct"] == 0.0:
    print("PRE-REGISTERED EXCEPTION: over-voltage is ZERO at k*. The feeder is unusable.")
else:
    print("The criterion is met. k*=%.4f IS LOCKED." % KSTAR)
    print("The violation rate was not targeted, it was measured. Next task: VM_BASE + Rcol.")
print("")
print("LIMIT: this rate is the BATTERY-FREE/UNCONTROLLED UPPER BOUND. It will drop on")
print("the naive MPC + safe layer arm. The trafo effect of battery discharge did NOT enter.")


banner("9. SAVE")
DF.to_csv(OUT_CSV, index=False)
np.savez_compressed(OUT_NPZ, start=T0, n_step=NSTEP, kstar=KSTAR, **TRAJ)
print("written: %s (%d rows)" % (OUT_CSV, len(DF)))
print("written: %s (%d arrays)" % (OUT_NPZ, len(TRAJ)))
print("")
print("ANCHORS A1/A2/A3/A4 : PASS / PASS / PASS / PASS")
print("k*                  : %.4f" % KSTAR)
print("")
print("DONE.")
