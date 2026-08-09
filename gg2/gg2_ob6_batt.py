#!/usr/bin/env python3
"""
GG-2 / ROUND v8 SECTION 8 - TASK 3
(a) safe_layer_ob.SafeLayerOB ANCHOR  (b) BATTERY SIZING MEASUREMENT

Run (from the GG-1 ROOT folder):
    python gg2/gg2_ob6_batt.py 2>&1 | tee ob6_batt.log

Requires: gg2/safe_layer_ob.py  and  gg2/data/ob_base_vm.npz  (TASK 2 output)
ASCII only. ~5-7 min.

-------------------------------------------------------------------------------
WHY
-------------------------------------------------------------------------------
1) safe_layer.SafeLayer._vac takes vm_pu.min/max over the whole grid.
   mv_oberrhein consists of TWO galvanically separate components and also has a
   110 kV slack bus. Used without a mask, IRRELEVANT buses enter the violation
   count. safe_layer_ob.SafeLayerOB adds a mask; NOTHING ELSE changes. This is
   proved first with an anchor.

2) Battery size. The "carry over the case33bw ratio" decision from ROUND v8
   gives P_MAX = 1.0 x PV_peak = 34.79 MW here. The trafo is 25 MVA. So the
   battery CANNOT run at full power - that is indefensible. The new rule is
   based on measurement.

-------------------------------------------------------------------------------
PRE-REGISTRATION (written BEFORE the run)
-------------------------------------------------------------------------------
MEASUREMENT: at k*=2.859375, for every step with over-voltage, the SMALLEST
       power that brings vmax down into the narrowed band (1.045) by charging at
       bus 147 is found with real AC. A step counts as FIXABLE only if vmin >=
       0.95 also holds at that power - the safe layer requires both bands at
       once anyway (safe_layer.project, the projected_exact branch). Because
       both vmax and vmin decrease monotonically in p, the valid p region is an
       INTERVAL; if vmin breaks at the smallest p it holds at no p -> the step
       is NOT FIXABLE.

RULE (not a result, a physical one):
       P_MAX* = the 95th percentile of the required charging power over the
                FIXABLE steps, rounded UP to 0.5 MW.
       CAP*   = 2.0 hours x P_MAX*   (the energy/power ratio of case33bw is
                PRESERVED: there 6.0 MWh / 3.0 MW = 2.0 hours).

RATIONALE: the battery power is sized against the PHYSICAL problem it has to
solve; the energy/power ratio is kept the same as case33bw on the second feeder
too. The 95th percentile was chosen because the 100th would hang on a single
extreme step. This threshold was fixed BEFORE the run and WILL NOT BE CHANGED
based on the result.

EXPECTED BEHAVIOUR (not a verdict, a record): for case33bw all the remaining
violations came from the battery being FULL (the SOC was binding, not the
instantaneous power). The same structure is expected to hold here; this is NOT
CONFIRMED, it will be measured once the naive arm is run.

ANCHORS (no verdict unless all pass):
  C1  SafeLayerOB(buses=all buses) must give EXACTLY the same result as
      safe_layer.SafeLayer on case33bw: p_safe, status, Vmin, Vmax, runpp count.
  C2  on mv_oberrhein, SafeLayerOB._vac(0) must reproduce the VM_BASE row from
      TASK 2 to within 1e-9.
  C3  at the p* found by the binary search vmax must be <= 1.045, while at
      p*-0.05 MW it must EXCEED 1.045 (i.e. the SMALLEST power really was found).
-------------------------------------------------------------------------------
"""

import os
import sys
import time

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import pandapower as pp
import pandapower.networks as pn

V_HI, V_LO, MARGIN = 1.05, 0.95, 0.005
V_TARGET = V_HI - MARGIN          # 1.045
KSTAR = 2.859375
BB = 147

# The upper corner cannot be taken as FIXED: on this feeder charging more than
# ~25 MW at a single bus makes the power flow DIVERGE (measured). So a valid
# upper corner is searched for with a ladder first, then the binary search runs.
LADDER = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0]
BISECT_N = 14
PCTL = 95.0
ROUND_TO = 0.5
HOURS = 2.0                       # case33bw: 6.0 MWh / 3.0 MW

C1_STRIDE = 121                   # step stride for the case33bw anchor
C2_N = 50                         # number of sample steps for the VM_BASE anchor
C3_N = 25                         # number of sample steps for the minimality check
C3_BACK = 0.05                    # MW

NPZ_IN = os.path.join(_HERE, "data", "ob_base_vm.npz")
OUT_NPZ = os.path.join(_HERE, "data", "ob6_batt.npz")
OUT_CSV = os.path.join(_HERE, "data", "ob6_batt.csv")

RUNPP_KW = dict(algorithm="nr", init="auto", calculate_voltage_angles=True,
                max_iteration=100)


def banner(t):
    print("")
    print("=" * 78)
    print(t)
    print("=" * 78)


banner("0. SPEC SHEET")
print("pandapower %s | numpy %s | python %s"
      % (pp.__version__, np.__version__, sys.version.split()[0]))
try:
    import numba  # noqa: F401
    USE_NUMBA = True
except Exception:
    USE_NUMBA = False
RUNPP_KW["numba"] = USE_NUMBA
print("numba=%s | k*=%.6f | BB=%d | V_TARGET=%.3f" % (USE_NUMBA, KSTAR, BB, V_TARGET))
print("PRE-REGISTERED: P_MAX* = ceil(p%.0f / %.1f)*%.1f MW ; CAP* = %.1f h x P_MAX*"
      % (PCTL, ROUND_TO, ROUND_TO, HOURS))

try:
    from safe_layer import SafeLayer
    from safe_layer_ob import SafeLayerOB
except Exception as exc:
    print("[ERROR] import: %s: %s" % (type(exc).__name__, exc))
    print("safe_layer.py must be in the root folder, safe_layer_ob.py inside gg2/.")
    sys.exit(1)

try:
    import mpc_econ_voltage_pareto as M
except Exception as exc:
    print("[ERROR] mpc import: %s: %s" % (type(exc).__name__, exc))
    print("Run from the GG-1 ROOT folder.")
    sys.exit(1)

T0, T1 = int(M.SPLIT), int(M.N) - 1
NSTEP = T1 - T0
SOLAR_NORM = np.asarray(M.solar, dtype=float) / float(M.SP)
DMULT = np.asarray(M.dmult, dtype=float)
print("window %d:%d (%d steps)" % (T0, T1, NSTEP))


# ---------------------------------------------------------------------------
banner("1. ANCHOR C1 - SafeLayerOB == SafeLayer  (case33bw, mask = all buses)")
SLOB33 = SafeLayerOB(M.net, M.sl.si, M.Rcol, M.S_base,
                     buses=list(M.net.bus.index), runpp_kwargs=None)
_n, _bad, _first = 0, 0, None
_t0 = time.time()
for _i in range(T0, T1, C1_STRIDE):
    for _pa in [-3.0, 0.0, 3.0]:
        for _lo, _hi in [(-3.0, 3.0), (-3.0, 1.0)]:
            M.set_grid(M.solar[_i], M.wind[_i], M.dmult[_i], 0.0)
            _p1, _i1 = M.sl.project(_pa, _lo, _hi)
            M.set_grid(M.solar[_i], M.wind[_i], M.dmult[_i], 0.0)
            _p2, _i2 = SLOB33.project(_pa, _lo, _hi)
            _n += 1
            _same = (abs(_p1 - _p2) < 1e-12
                     and _i1["status"] == _i2["status"]
                     and abs(_i1["Vmin"] - _i2["Vmin"]) < 1e-12
                     and abs(_i1["Vmax"] - _i2["Vmax"]) < 1e-12
                     and _i1["runpp"] == _i2["runpp"])
            if not _same:
                _bad += 1
                if _first is None:
                    _first = (_i, _pa, _lo, _hi, _p1, _p2,
                              _i1["status"], _i2["status"])
C1 = bool(_bad == 0 and _n > 0)
print("cases compared: %d | differences: %d | %.0f s" % (_n, _bad, time.time() - _t0))
if _first:
    print("first difference: %s" % str(_first))
print("C1 -> %s" % ("PASS" if C1 else "FAIL"))


# ---------------------------------------------------------------------------
banner("2. mv_oberrhein SETUP + ANCHOR C2")
if not os.path.exists(NPZ_IN):
    print("[ERROR] not found: %s   (run gg2_ob4_rcol.py first)" % NPZ_IN)
    sys.exit(1)
Z = np.load(NPZ_IN, allow_pickle=True)
VM = Z["vm"]
MV = [int(b) for b in Z["mv_buses"]]
RCOL = Z["Rcol"]
S_BASE = float(Z["S_base"])
_kz = float(Z["kstar"])
print("ob_base_vm.npz: vm %s | mv_buses %d | kstar %.6f | Rcol[BB]=%.6f"
      % (str(VM.shape), len(MV), _kz, RCOL[MV.index(BB)]))
if abs(_kz - KSTAR) > 1e-12:
    print("[ERROR] the k* in the npz (%.8f) is not the same as the k* in this script (%.8f)."
          % (_kz, KSTAR))
    sys.exit(2)

NET = pn.mv_oberrhein()
NET.sgen["scaling"] = 1.0
P0 = NET.sgen.p_mw.values.copy()
Q0 = NET.sgen.q_mvar.values.copy()
LS0 = NET.load.scaling.values.copy()
ST = pp.create_storage(NET, bus=BB, p_mw=0.0, max_e_mwh=1000.0, name="batt")
TIDX = int(NET.trafo[NET.trafo.lv_bus == 319].index[0])
SL = SafeLayerOB(NET, ST, RCOL, S_BASE, buses=MV, runpp_kwargs=RUNPP_KW)


def set_step(i):
    NET.sgen["p_mw"] = KSTAR * P0 * float(SOLAR_NORM[i])
    NET.sgen["q_mvar"] = Q0
    NET.load["scaling"] = LS0 * float(DMULT[i])


_idx_c2 = list(range(T0, T1, max(1, NSTEP // C2_N)))[:C2_N]
_d = 0.0
for _i in _idx_c2:
    set_step(_i)
    _r = SL._vac(0.0)
    if _r is None:
        _d = np.inf
        break
    _row = VM[_i]
    _d = max(_d, abs(_r[0] - float(_row.min())), abs(_r[1] - float(_row.max())))
C2 = bool(_d < 1e-9)
print("C2: %d sample steps, max |diff| = %.3e pu -> %s"
      % (len(_idx_c2), _d, "PASS" if C2 else "FAIL"))

if not (C1 and C2):
    print("")
    print("One of C1/C2 FAILED. The measurement is NOT MADE - paste the output to me.")
    sys.exit(3)


# ---------------------------------------------------------------------------
banner("3. REQUIRED CHARGING POWER MEASUREMENT")
_tsmax = np.nanmax(VM[T0:T1], axis=1)
VIOL = [T0 + int(j) for j in np.where(_tsmax > V_HI)[0]]
print("steps with over-voltage: %d  (367 were expected in TASK 2)" % len(VIOL))


def vmax_at(i, p):
    set_step(i)
    NET.storage.at[ST, "p_mw"] = float(p)
    try:
        pp.runpp(NET, **RUNPP_KW)
    except Exception:
        return None
    v = NET.res_bus.vm_pu.loc[MV]
    if bool(v.isna().any()):
        return None
    return float(v.max()), float(v.min())


def required_power(i):
    """The SMALLEST charging power that brings vmax down to V_TARGET.
    Returns: (p, vmax_p, vmin_p, reason). If p=inf the reason says why none was
    found."""
    r0 = vmax_at(i, 0.0)
    if r0 is None:
        return np.nan, np.nan, np.nan, "base_did_not_converge"
    if r0[0] <= V_TARGET:
        return 0.0, r0[0], r0[1], "not_needed"

    lo, hi = 0.0, None
    diverged_at = None
    for cand in LADDER:
        rc = vmax_at(i, cand)
        if rc is None:
            diverged_at = cand
            break
        if rc[0] <= V_TARGET:
            hi = cand
            break
        lo = cand
    if hi is None:
        why = ("diverged_%.1fMW" % diverged_at) if diverged_at is not None \
            else "ladder_insufficient"
        rl = vmax_at(i, lo) if lo > 0 else r0
        return (np.inf,
                (np.nan if rl is None else rl[0]),
                (np.nan if rl is None else rl[1]), why)

    for _ in range(BISECT_N):
        mid = 0.5 * (lo + hi)
        rm = vmax_at(i, mid)
        if rm is None or rm[0] > V_TARGET:
            lo = mid
        else:
            hi = mid
    rr = vmax_at(i, hi)
    return (hi, (np.nan if rr is None else rr[0]),
            (np.nan if rr is None else rr[1]), "found")


REQ = np.full(len(VIOL), np.nan)
VMX = np.full(len(VIOL), np.nan)
VMN = np.full(len(VIOL), np.nan)
WHY = ["" for _ in VIOL]
_t0 = time.time()
for _j, _i in enumerate(VIOL):
    REQ[_j], VMX[_j], VMN[_j], WHY[_j] = required_power(_i)
    # PRE-REGISTERED: a solution that breaks the vmin band IS NOT VALID (the safe
    # layer would not accept it either). Because vmin decreases monotonically in
    # p, a band broken at the smallest p holds at no p -> the step is NOT FIXABLE.
    if np.isfinite(REQ[_j]) and np.isfinite(VMN[_j]) and VMN[_j] < V_LO:
        REQ[_j] = np.inf
        WHY[_j] = "causes_under_voltage"
    if (_j + 1) % 100 == 0:
        _el = max(time.time() - _t0, 1e-9)
        print("    %d/%d  (~%.1f min left)"
              % (_j + 1, len(VIOL), (len(VIOL) - _j - 1) / ((_j + 1) / _el) / 60),
              flush=True)
print("  finished: %.0f s" % (time.time() - _t0))

FIN = np.isfinite(REQ)
NINF = int((~FIN).sum())
print("")
print("measured steps : %d" % int(FIN.sum()))
print("not found      : %d" % NINF)
if NINF:
    _u, _cnt = np.unique([w for w, f in zip(WHY, FIN) if not f], return_counts=True)
    for _uu, _cc in zip(_u, _cnt):
        print("    %-22s %d" % (_uu, _cc))
_nbase = np.array([int((VM[i] > V_HI).sum()) for i in VIOL])
print("")
print("number of buses above 1.05 at violating steps: median %d, max %d"
      % (int(np.median(_nbase)), int(_nbase.max())))
print("(one battery at one bus; if the violation is widespread one battery is not enough)")
if int(FIN.sum()) == 0:
    print("")
    print("The required power could not be measured at ANY step. No verdict - STOP.")
    sys.exit(4)


# ---------------------------------------------------------------------------
banner("4. ANCHOR C3 - IS THE POWER FOUND REALLY THE MINIMUM")
_sel = [k for k in np.where(FIN & (REQ > C3_BACK))[0]][:C3_N]
_ok3, _n3 = 0, 0
for _k in _sel:
    _i = VIOL[_k]
    _ra = vmax_at(_i, float(REQ[_k]))
    _rb = vmax_at(_i, float(REQ[_k]) - C3_BACK)
    if _ra is None or _rb is None:
        continue
    _n3 += 1
    if _ra[0] <= V_TARGET + 1e-9 and _rb[0] > V_TARGET:
        _ok3 += 1
C3 = bool(_n3 > 0 and _ok3 == _n3)
print("checked: %d | inside the band at p* AND outside at p*-%.2f MW: %d"
      % (_n3, C3_BACK, _ok3))
print("C3 -> %s" % ("PASS" if C3 else "FAIL"))


# ---------------------------------------------------------------------------
banner("5. DISTRIBUTION")
R = REQ[FIN]
for _q in [50, 75, 90, 95, 99, 100]:
    print("  p%-4d : %7.3f MW" % (_q, float(np.percentile(R, _q))))
print("  mean  : %7.3f MW | min %7.3f | max %7.3f"
      % (R.mean(), R.min(), R.max()))
print("")
print("  steps fixable INSTANTANEOUSLY at candidate P_MAX values (over ALL %d violations):"
      % len(VIOL))
for _c in [1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]:
    _cnt = int(np.sum(FIN & (REQ <= _c)))
    print("    P_MAX=%6.2f MW -> %5.1f%%  (%d / %d)"
          % (_c, 100.0 * _cnt / len(VIOL), _cnt, len(VIOL)))
print("")
print("  UPPER BOUND: share of fixable steps = %.1f%% (%d / %d)."
      % (100.0 * FIN.sum() / len(VIOL), int(FIN.sum()), len(VIOL)))
print("  The rest cannot be fixed at a single bus NO MATTER HOW LARGE the battery is.")


# ---------------------------------------------------------------------------
banner("6. VERDICT - BATTERY SIZE")
Q = float(np.percentile(R, PCTL))
P_MAX_STAR = float(np.ceil(Q / ROUND_TO) * ROUND_TO)
CAP_STAR = HOURS * P_MAX_STAR
print("p%.0f required charging power : %.3f MW" % (PCTL, Q))
print("P_MAX* (rounded up)          : %.2f MW" % P_MAX_STAR)
print("CAP*   (%.1f h x P_MAX*)      : %.2f MWh" % (HOURS, CAP_STAR))
print("")
print("comparison with case33bw:")
print("  case33bw : P_MAX 3.00 MW / CAP 6.00 MWh | PV peak 3.00 MW | load 1.86 MW")
print("  oberrhein: P_MAX %.2f MW / CAP %.2f MWh | PV peak %.2f MW | load 20.27 MW"
      % (P_MAX_STAR, CAP_STAR, KSTAR * 12.1657))
print("  P_MAX/PV_peak : case33bw 1.00 -> oberrhein %.3f"
      % (P_MAX_STAR / (KSTAR * 12.1657)))
print("  P_MAX/load    : case33bw 1.61 -> oberrhein %.3f" % (P_MAX_STAR / 20.274))
print("")
print("DECISION TAKEN: the 'carry the case33bw ratio over as is' rule from ROUND v8")
print("has been ABANDONED. That rule would give P_MAX = %.2f MW; since the trafo is"
      % (KSTAR * 12.1657))
print("25 MVA the battery cannot run at full power, which is indefensible. The rule that")
print("replaces it is physical, and the energy/power ratio (%.1f hours) is kept the SAME as case33bw." % HOURS)

_tl = []
for _k in np.where(FIN)[0][:200]:
    _i = VIOL[_k]
    set_step(_i)
    NET.storage.at[ST, "p_mw"] = P_MAX_STAR
    try:
        pp.runpp(NET, **RUNPP_KW)
        _tl.append(float(NET.res_trafo.at[TIDX, "loading_percent"]))
    except Exception:
        pass
if _tl:
    print("")
    print("EXTRA CHECK (not a verdict): over the first %d violating steps, with the battery"
          % len(_tl))
    print("charging at FULL power, trafo loading: mean %.1f%%  max %.1f%%  >100%%: %d steps"
          % (float(np.mean(_tl)), float(np.max(_tl)),
             int(sum(1 for x in _tl if x > 100.0))))


# ---------------------------------------------------------------------------
banner("7. SAVE")
os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
np.savez_compressed(OUT_NPZ, viol_idx=np.array(VIOL, dtype=int), req_mw=REQ,
                    vmax_at_req=VMX, vmin_at_req=VMN, p_max_star=P_MAX_STAR,
                    cap_star=CAP_STAR, kstar=KSTAR, pctl=PCTL,
                    ladder=np.array(LADDER))
pd.DataFrame({"step": VIOL, "req_mw": REQ, "vmax_at_req": VMX,
              "vmin_at_req": VMN, "why": WHY}).to_csv(OUT_CSV, index=False)
print("written: %s" % OUT_NPZ)
print("written: %s (%d rows)" % (OUT_CSV, len(VIOL)))
print("")
print("ANCHORS C1/C2/C3 : %s / %s / %s"
      % tuple("PASS" if x else "FAIL" for x in (C1, C2, C3)))
print("P_MAX* = %.2f MW | CAP* = %.2f MWh" % (P_MAX_STAR, CAP_STAR))
print("")
print("DONE.")
