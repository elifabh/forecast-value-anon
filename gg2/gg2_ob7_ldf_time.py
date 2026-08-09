#!/usr/bin/env python3
"""
GG-2 / ROUND v8 SECTION 8 - TASK 4
(a) VALIDITY of the LinDistFlow horizon model on mv_oberrhein
(b) TIMING measurement of one MPC arm

Run (from the GG-1 ROOT folder):
    python gg2/gg2_ob7_ldf_time.py 2>&1 | tee ob7_ldf_time.log

Requires: gg2/mpc_ob.py, gg2/safe_layer_ob.py, gg2/data/ob_base_vm.npz,
          gg2/data/ob6_batt.npz
ASCII only. ~6-10 min (depends on the very thing we are timing).

-------------------------------------------------------------------------------
WHY
-------------------------------------------------------------------------------
(a) The MPC horizon window is built like this (diag_mpc_forecastvolt.fc_windows_ldf):
        vmb[0] = VM_BASE[i]                       (the real present)
        vmb[t] = VM_BASE[i] + (ldf_v(future) - ldf_v(now))
    That is, it uses the LDF DIFFERENCE, not absolute LinDistFlow. How well that
    difference tracks the real AC difference was NEVER MEASURED for case33bw.
    Running without measuring it on a 108-bus feeder whose PV is spread over 93
    buses would bury an unknown error inside the planner.
    The measurement is FREE: VM_BASE was already computed with battery=0, so we
    have the REAL AC voltage of every step. Not even one power flow is needed.

(b) The number of slack variables grows from 48x33 to 48x108 (3.3x). Before
    launching three arms we need to know how long one step and one arm take.

-------------------------------------------------------------------------------
PRE-REGISTRATION (written BEFORE the run)
-------------------------------------------------------------------------------
The LDF criterion is COMPARATIVE, not absolute. case33bw already produced the
published results; therefore the right question is NOT "is the LDF difference
perfect" but "is it MEASURABLY WORSE on mv_oberrhein than on case33bw".

  For each lag, the RMSE between the real difference  d_AC = VM[i+lag] - VM[i]
  and the model difference                            d_LDF = ldf(i+lag) - ldf(i)
  is computed on both feeders at the SAME steps and the SAME lags.

  VERDICT:
    RMSE_ob <= 2.0 x RMSE_33  ->  we write "the LDF horizon model is NOT
                                  measurably worse on the second feeder than on
                                  the first", and the arms are run.
    RMSE_ob >  2.0 x RMSE_33  ->  the arms are NOT RUN. The model is fixed first
                                  (e.g. a sensitivity column instead of
                                  absolute LDF).
  The 2.0 threshold was fixed BEFORE the run and will not be changed based on
  the result.
  Note: this is not an ABSOLUTE accuracy threshold; it compares the two feeders
  on the same test. The absolute error is reported separately for both feeders.

TIMING: reported as the measured single-step time x 2429 steps x 3 arms.
        No verdict, just a planning number.

ANCHORS:
  L1  mpc_ob.build_rx + ldf_v must produce the same vector as
      diag_mpc_forecastvolt.ldf_v on case33bw, to within 1e-12.
  L2  mpc_ob.solve_window must produce the same p_plan as
      mpc_econ_voltage_pareto.solve_window with the case33bw constants, to
      within 1e-6 (on real windows).
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
import pandapower.topology as top

KSTAR = 2.859375
BB = 147
ROOT_OB = 319
MV_KV = 20.0
LAGS = [1, 6, 12, 24, 47]
N_PAIR = 300            # number of sample steps per lag
RMSE_RATIO_MAX = 2.0
N_TIME = 15             # number of windows for the timing measurement
N_TIME_SL = 15          # number of steps for the safe-layer timing measurement
H = 48

NPZ_VM = os.path.join(_HERE, "data", "ob_base_vm.npz")
NPZ_BT = os.path.join(_HERE, "data", "ob6_batt.npz")
OUT_CSV = os.path.join(_HERE, "data", "ob7_ldf.csv")


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
import cvxpy as cp  # noqa: E402
print("cvxpy %s | numba=%s" % (cp.__version__, USE_NUMBA))
print("PRE-REGISTERED threshold: RMSE_ob / RMSE_33 <= %.1f" % RMSE_RATIO_MAX)

import mpc_ob as MO                      # noqa: E402
import mpc_econ_voltage_pareto as M      # noqa: E402
import diag_mpc_forecastvolt as D        # noqa: E402
from safe_layer_ob import SafeLayerOB    # noqa: E402

T0, T1 = int(M.SPLIT), int(M.N) - 1
NSTEP = T1 - T0
SOLAR_NORM = np.asarray(M.solar, dtype=float) / float(M.SP)
DMULT = np.asarray(M.dmult, dtype=float)
print("window %d:%d (%d steps)" % (T0, T1, NSTEP))


# ---------------------------------------------------------------------------
banner("1. ANCHOR L1 - does build_rx/ldf_v reproduce the original on case33bw")
_net33 = pn.case33bw()
_z33 = _net33.bus.vn_kv.iloc[0] ** 2 / _net33.sn_mva
_slk33 = int(_net33.ext_grid.bus.iloc[0])
_b33 = list(_net33.bus.index)
R33, X33, _ = MO.build_rx(_net33, _slk33, _b33, _z33)
print("R33 %s | max|R33 - D.R| = %.3e | max|X33 - D.X| = %.3e"
      % (str(R33.shape), float(np.abs(R33 - D.R).max()),
         float(np.abs(X33 - D.X).max())))

_Pl33 = np.zeros(len(_b33))
_Ql33 = np.zeros(len(_b33))
for _li, _row in _net33.load.iterrows():
    _Pl33[int(_row.bus)] += float(M.base_p[_li])
    _Ql33[int(_row.bus)] += float(M.base_q[_li])


def ldf33(s, w, dm):
    P = _Pl33 * dm * M.LS
    Q = _Ql33 * dm * M.LS
    P = P.copy()
    P[M.SB] -= s
    P[M.WB] -= w
    return MO.ldf_v(R33, X33, P, Q, M.S_base)


_dmax = 0.0
for _i in range(T0, T1, 137):
    _a = ldf33(M.solar[_i], M.wind[_i], M.dmult[_i])
    _b = D.ldf_v(M.solar[_i], M.wind[_i], M.dmult[_i])
    _dmax = max(_dmax, float(np.abs(_a - _b).max()))
L1 = bool(_dmax < 1e-12)
print("case33bw ldf_v comparison: max|diff| = %.3e -> L1 %s"
      % (_dmax, "PASS" if L1 else "FAIL"))


# ---------------------------------------------------------------------------
banner("2. mv_oberrhein LinDistFlow SETUP")
Z = np.load(NPZ_VM, allow_pickle=True)
VM = Z["vm"]
MV = [int(b) for b in Z["mv_buses"]]
RCOL = Z["Rcol"]
S_BASE = float(Z["S_base"])
if abs(float(Z["kstar"]) - KSTAR) > 1e-12:
    print("[ERROR] npz k* does not match")
    sys.exit(2)
ZB = np.load(NPZ_BT, allow_pickle=True)
P_MAX_OB = float(ZB["p_max_star"])
CAP_OB = float(ZB["cap_star"])
print("VM %s | MV %d | Rcol[BB] %.6f | P_MAX* %.2f MW | CAP* %.2f MWh"
      % (str(VM.shape), len(MV), RCOL[MV.index(BB)], P_MAX_OB, CAP_OB))

NET = pn.mv_oberrhein()
NET.sgen["scaling"] = 1.0
P0 = NET.sgen.p_mw.values.copy()
Q0 = NET.sgen.q_mvar.values.copy()
LS0 = NET.load.scaling.values.copy()
_mg = top.create_nxgraph(NET, respect_switches=True)
_cc = [sorted(int(b) for b in x) for x in top.connected_components(_mg)]
_cc.sort(key=len, reverse=True)
C0 = set(b for b in _cc[0] if bool(NET.bus.at[b, "in_service"]))
OPEN_L = set(NET.switch[(NET.switch.et == "l")
                        & (~NET.switch.closed.astype(bool))].element.astype(int))
ZB_OB = MV_KV ** 2 / S_BASE
ROB, XOB, GOB = MO.build_rx(NET, ROOT_OB, MV, ZB_OB, open_lines=OPEN_L, restrict=C0)
print("R_ob %s | graph %d nodes %d edges | Rcol[BB] check: R[BB,BB]=%.6f (npz %.6f)"
      % (str(ROB.shape), GOB.number_of_nodes(), GOB.number_of_edges(),
         ROB[MV.index(BB), MV.index(BB)], RCOL[MV.index(BB)]))

POS = {b: i for i, b in enumerate(MV)}
PL_OB = np.zeros(len(MV))
QL_OB = np.zeros(len(MV))
for _li, _row in NET.load.iterrows():
    _b = int(_row.bus)
    if _b in POS:
        PL_OB[POS[_b]] += float(_row.p_mw) * float(_row.scaling)
        QL_OB[POS[_b]] += float(_row.q_mvar) * float(_row.scaling)
SG_POS = np.array([POS[int(b)] for b in NET.sgen.bus.values if int(b) in POS])
SG_MASK = np.array([int(b) in POS for b in NET.sgen.bus.values])
SG_P0 = P0[SG_MASK]
print("LDF load axis: P total %.4f MW | Q total %.4f MVAr | sgen %d"
      % (PL_OB.sum(), QL_OB.sum(), len(SG_POS)))


def ldf_ob(s_norm, dm):
    P = PL_OB * dm
    Q = QL_OB * dm
    np.add.at(P, SG_POS, -KSTAR * SG_P0 * s_norm)
    return MO.ldf_v(ROB, XOB, P, Q, S_BASE)


# ---------------------------------------------------------------------------
banner("3. LDF DIFFERENCE ACCURACY - TWO FEEDERS, SAME TEST")
print("real difference  = VM[i+lag] - VM[i]    (AC, battery=0, pre-computed)")
print("model difference = ldf(i+lag) - ldf(i)")
print("")
_idx = list(range(T0, T1 - max(LAGS), max(1, (NSTEP - max(LAGS)) // N_PAIR)))[:N_PAIR]
print("number of sample steps: %d" % len(_idx))

LDF33 = {}
LDFOB = {}
for _i in set(_idx) | set(i + l for i in _idx for l in LAGS):
    LDF33[_i] = ldf33(M.solar[_i], M.wind[_i], M.dmult[_i])
    LDFOB[_i] = ldf_ob(float(SOLAR_NORM[_i]), float(DMULT[_i]))

ROWS = []
print("%-6s | %-26s | %-26s | %s"
      % ("lag", "case33bw RMSE / max (pu)", "oberrhein RMSE / max (pu)", "ratio"))
for _lag in LAGS:
    _e33, _eob = [], []
    for _i in _idx:
        _j = _i + _lag
        d33 = M.VM_BASE[_j] - M.VM_BASE[_i]
        m33 = LDF33[_j] - LDF33[_i]
        _e33.append(np.sqrt(((d33 - m33) ** 2).mean()))
        dob = VM[_j] - VM[_i]
        mob = LDFOB[_j] - LDFOB[_i]
        _eob.append(np.sqrt(((dob - mob) ** 2).mean()))
    r33, rob = float(np.mean(_e33)), float(np.mean(_eob))
    x33, xob = float(np.max(_e33)), float(np.max(_eob))
    ROWS.append({"lag": _lag, "rmse_33": r33, "max_33": x33,
                 "rmse_ob": rob, "max_ob": xob, "oran": rob / r33 if r33 else np.nan})
    print("%-6d | %10.3e / %10.3e | %10.3e / %10.3e | %.2f"
          % (_lag, r33, x33, rob, xob, rob / r33 if r33 else float("nan")))

DFL = pd.DataFrame(ROWS)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
DFL.to_csv(OUT_CSV, index=False)
RATIO = float(DFL["rmse_ob"].mean() / DFL["rmse_33"].mean())
print("")
print("mean over lags: case33bw %.3e | oberrhein %.3e | RATIO %.2f"
      % (DFL["rmse_33"].mean(), DFL["rmse_ob"].mean(), RATIO))
LDF_OK = bool(RATIO <= RMSE_RATIO_MAX)
print("VERDICT: %s (threshold %.1f)"
      % ("PASS - NOT measurably worse on the second feeder"
         if LDF_OK else "FAIL - the arms are NOT RUN, the model will be fixed",
         RMSE_RATIO_MAX))


# ---------------------------------------------------------------------------
banner("4. ANCHOR L2 - mpc_ob.solve_window == mpc_econ_voltage_pareto.solve_window")
_dp, _n2 = 0.0, 0
for _i in range(T0, T0 + 900, 150):
    _Hh = min(H, M.N - _i)
    _vmb, _pr = M.fc_windows(_i, _Hh)
    for _soc in [1.0, 3.0, 5.0]:
        _p1, _s1 = M.solve_window(_vmb, _pr, _soc, 1.0)
        _p2, _s2 = MO.solve_window(_vmb, _pr, _soc, 1.0, M.Rcol, M.S_base,
                                   M.P_MAX, M.CAP, M.ETA, M.dt, M.HI, M.LO,
                                   M.MARGIN, M.SOC_TARGET)
        _dp = max(_dp, abs(float(_p1) - float(_p2)))
        _n2 += 1
L2 = bool(_n2 > 0 and _dp < 1e-6)
print("%d windows | max |p_plan diff| = %.3e -> L2 %s"
      % (_n2, _dp, "PASS" if L2 else "FAIL"))

if not (L1 and L2):
    print("")
    print("One of L1/L2 FAILED. STOP, paste the output to me.")
    sys.exit(3)


# ---------------------------------------------------------------------------
banner("5. TIMING MEASUREMENT")
K = M.solar_fc.shape[1]
SOLAR_FC_NORM = np.asarray(M.solar_fc, dtype=float) / float(M.SP)
DMULT_FC = np.asarray(M.dmult_fc, dtype=float)


def window_ob(i, Hh):
    v_now = ldf_ob(float(SOLAR_NORM[i]), float(DMULT[i]))
    vmb = np.empty((Hh, len(MV)))
    vmb[0] = VM[i]
    for t in range(1, Hh):
        kk = min(t - 1, K - 1)
        v_fut = ldf_ob(float(SOLAR_FC_NORM[i, kk]), float(DMULT_FC[i, kk]))
        vmb[t] = VM[i] + (v_fut - v_now)
    pr = np.empty(Hh)
    pr[0] = M.price[i]
    for t in range(1, Hh):
        kk = min(t - 1, M.price_fc.shape[1] - 1)
        pr[t] = M.price_fc[i, kk]
    return vmb, pr


_tw, _ts, _st = [], [], []
_idxt = list(range(T0, T1 - H, max(1, (NSTEP - H) // N_TIME)))[:N_TIME]
for _i in _idxt:
    _Hh = min(H, M.N - _i)
    _t = time.time()
    _vmb, _pr = window_ob(_i, _Hh)
    _tw.append(time.time() - _t)
    _t = time.time()
    _p, _s = MO.solve_window(_vmb, _pr, CAP_OB / 2, 1.0, RCOL, S_BASE,
                             P_MAX_OB, CAP_OB, M.ETA, M.dt, M.HI, M.LO,
                             M.MARGIN, CAP_OB / 2)
    _ts.append(time.time() - _t)
    _st.append(_s)
print("window construction : mean %.3f s  max %.3f s" % (np.mean(_tw), np.max(_tw)))
print("QP solve (108 buses, H=%d): mean %.3f s  max %.3f s"
      % (H, np.mean(_ts), np.max(_ts)))
print("solver statuses     : %s" % sorted(set(_st)))

_ST = pp.create_storage(NET, bus=BB, p_mw=0.0, max_e_mwh=CAP_OB, name="batt")
_SL = SafeLayerOB(NET, _ST, RCOL, S_BASE, buses=MV,
                  runpp_kwargs=dict(algorithm="nr", init="auto",
                                    calculate_voltage_angles=True,
                                    max_iteration=100, numba=USE_NUMBA))
_tsl, _nrp = [], []
_viol = [T0 + int(j) for j in np.where(np.nanmax(VM[T0:T1], axis=1) > 1.05)[0]]
_pick = ([_viol[j] for j in range(0, len(_viol), max(1, len(_viol) // 8))][:8]
         + _idxt[:7])
for _i in _pick:
    NET.sgen["p_mw"] = KSTAR * P0 * float(SOLAR_NORM[_i])
    NET.sgen["q_mvar"] = Q0
    NET.load["scaling"] = LS0 * float(DMULT[_i])
    _t = time.time()
    _ps, _inf = _SL.project(0.0, -P_MAX_OB, P_MAX_OB)
    _tsl.append(time.time() - _t)
    _nrp.append(_inf["runpp"])
print("safe layer          : mean %.3f s  max %.3f s  (mean %.1f runpp/step)"
      % (np.mean(_tsl), np.max(_tsl), np.mean(_nrp)))

PER = float(np.mean(_tw) + np.mean(_ts) + np.mean(_tsl))
print("")
print("TOTAL PER STEP (estimated) : %.3f s" % PER)
print("ONE ARM   (%d steps)       : %.1f min" % (NSTEP, PER * NSTEP / 60))
print("THREE ARMS (naive/NWP/oracle): %.1f min  (%.2f hours)"
      % (3 * PER * NSTEP / 60, 3 * PER * NSTEP / 3600))
print("")
print("NOTE: the safe layer time was measured on a sample weighted toward")
print("violating steps (8 violating + 7 random). The true mean will be LOWER,")
print("because %.1f%% of the steps have no violation and finish with one runpp."
      % (100.0 * (1 - len(_viol) / NSTEP)))


banner("6. SUMMARY")
print("ANCHOR L1/L2 : %s / %s"
      % ("PASS" if L1 else "FAIL", "PASS" if L2 else "FAIL"))
print("LDF ratio    : %.2f (threshold %.1f) -> %s"
      % (RATIO, RMSE_RATIO_MAX, "PASS" if LDF_OK else "FAIL"))
print("estimated time for three arms: %.1f hours" % (3 * PER * NSTEP / 3600))
print("written: %s" % OUT_CSV)
print("")
print("DONE.")
