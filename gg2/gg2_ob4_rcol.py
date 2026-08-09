#!/usr/bin/env python3
"""
GG-2 / ROUND v8 SECTION 8 - TASK 2
mv_oberrhein: VM_BASE pre-computation + Rcol (LinDistFlow sensitivity column)
VALIDATION.

Run (from the GG-1 ROOT folder):
    python gg2/gg2_ob4_rcol.py 2>&1 | tee ob4_rcol.log

ASCII only. ~10-12 min (16230 steps of VM_BASE + ~350 sensitivity flows).

-------------------------------------------------------------------------------
WHAT IT PRODUCES
-------------------------------------------------------------------------------
gg2/data/ob_base_vm.npz   <- the oberrhein counterpart of `opf_base_vm.npz`
    vm        (16230, 108)  base voltages with battery=0, the 20 kV buses of
                            component 0
    Rcol      (108,)        the chosen LinDistFlow sensitivity column
    S_base    ()            net.sn_mva
    mv_buses  (108,)        COLUMN -> BUS INDEX mapping (it was 0..32 for
                            case33bw; here the bus indices are NOT CONTIGUOUS -
                            use them positionally)
    bb        ()            battery bus (147)
    kstar     ()            2.8594

-------------------------------------------------------------------------------
LOCKED INPUTS
-------------------------------------------------------------------------------
k* = 2.859375  (TASK 1c, gg2_ob3_kstar.py; thermal hosting criterion)

    SPEC-SHEET CORRECTION (a MISTAKE was made on the first run):
    The first version of this script took k* as "2.8594", the value printed in
    the log with `%.4f`. The true value is the EXACT arithmetic of the
    bisection:
        2.75 -> 2.875 (OVER) -> 2.8125 -> 2.84375 -> 2.859375
    The 2.5e-5 difference between 2.8594 and 2.859375 shifts vmax by ~1e-5 pu
    and caused the A5 anchor to fail (diagnosed with gg2_ob5_a5diag.py: storage
    at zero power is NEUTRAL, d=3.57e-13; the problem was the k value).
    RULE: never again anchor on a rounded number.
BB = 147     (TASK 1a/1c; the vmax bus at every violating step)
NO wind, battery = 0 in the VM_BASE computation.
Load: the shipped load.scaling=0.6 ON TOP OF dmult(t). NO LS.
PV(t) = k* * p0 * (M.solar(t) / M.SP).

-------------------------------------------------------------------------------
THE Rcol PROBLEM (the real job of this script)
-------------------------------------------------------------------------------
For case33bw, `diag_mpc_forecastvolt.py` builds the R matrix ONLY from
`net.line`; that grid has no trafo, so it was not a problem. On mv_oberrhein:
  - there is a 110/20 kV supply trafo: vk=11.2%, vkr=0.282%, 25 MVA
    -> at S_base=1 MVA, r=0.000113 pu, x=0.004479 pu
  - the line graph is radial only THROUGH THE SWITCHES (6 open line switches);
    if the switches are ignored the graph gets 2 loops and the shortest-path
    Rcol comes out WRONG
Two candidates:
  A  root = 319 (the 20 kV supply bus), LINES ONLY          [ROUND v8 plan]
  B  root = 318 (the 110 kV slack), lines + trafo resistance r_t
                                                            [a constant added to every bus]
Both are tested against the real dV/dP measured with AC.

PRE-REGISTRATION (written BEFORE the run):
  Sampling: a fixed sample every 97 steps from the test block (25 points, a
  prime stride -> no aliasing with the daily cycle) PLUS the 10 steps with the
  highest base vmax (the points where the safe layer actually acts). The two
  are reported SEPARATELY.
  Battery power: p = -20, -10, -3, -1, +1, +3, +10, +20 MW.
  CHOICE: the variant with the lower RMSE over 108 buses in the |p| <= 3 MW
          range. If the RMSE difference between the two variants is < 1e-5 pu,
          variant A is chosen and "the trafo term is negligible" is written.
  The error over the wide power range (10, 20 MW) IS NOT USED FOR THE VERDICT;
  it is reported SEPARATELY as an input to the battery sizing decision (TASK 3).

ANCHORS:
  A1 M.SPLIT == 13800, window 2429 steps
  A2 raw M.solar on the test block: mean 0.864 MW, peak 3.749 MW
  A3 static k=4.0 -> vmax 1.08218 / 38 buses
  A5 VM_BASE test slice -> the gg2_ob3_kstar.py numbers:
     violation COUNT 367 (an integer, unique), round(vmax,4)=1.0836,
     round(vmin,4)=0.9656. Because the reference has 4 digits, the comparison
     is also made rounded to 4 digits; the exact values are printed to 8 digits
     as well.
  A6 the line graph must be a TREE (taking switches into account),
     108 nodes / 107 edges
-------------------------------------------------------------------------------
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top

V_HI, V_LO, MV_KV = 1.05, 0.95, 20.0
KSTAR = 2.859375
BB = 147
ROOT_A, ROOT_B = 319, 318

SAMPLE_STRIDE = 97
N_TOPV = 10
P_TEST_CORE = [-3.0, -1.0, 1.0, 3.0]
P_TEST_WIDE = [-20.0, -10.0, 10.0, 20.0]
RMSE_TIE = 1e-5

ANCHOR_SPLIT, ANCHOR_NSTEP = 13800, 2429
ANCHOR_SOLAR_MEAN, ANCHOR_SOLAR_PEAK, ANCHOR_SOLAR_TOL = 0.864, 3.749, 0.005
ANCHOR_STATIC_K, ANCHOR_STATIC_VMAX, ANCHOR_STATIC_NOVER = 4.0, 1.08218, 38
ANCHOR_STATIC_TOL = 5e-4
# A5: the ob3 numbers. over_pct was written with %.2f -> the exact count is
# UNIQUELY 367 (over 2429 steps the range 15.105-15.115% contains only 367).
# vmax/vmin were given rounded to 4 digits -> the comparison is also ROUNDED.
ANCHOR_NVIOL = 367
ANCHOR_VMAXMAX_R = 1.0836
ANCHOR_VMINMIN_R = 0.9656

PROGRESS_EVERY = 4000
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_NPZ = os.path.join(HERE, "data", "ob_base_vm.npz")
OUT_CSV = os.path.join(HERE, "data", "ob4_rcol.csv")


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
print("numba=%s | k*=%.4f | BB=%d" % (USE_NUMBA, KSTAR, BB))


banner("1. SERIES")
try:
    import mpc_econ_voltage_pareto as M
except Exception as exc:
    print("[ERROR] import: %s: %s" % (type(exc).__name__, exc))
    print("Run from the GG-1 ROOT folder.")
    sys.exit(1)

NFULL = int(M.N)
T0, T1 = int(M.SPLIT), NFULL - 1
NSTEP = T1 - T0
SOLAR_NORM = np.asarray(M.solar, dtype=float) / float(M.SP)
DMULT = np.asarray(M.dmult, dtype=float)
print("N=%d | window %d:%d (%d steps)" % (NFULL, T0, T1, NSTEP))

A1 = bool(T0 == ANCHOR_SPLIT and NSTEP == ANCHOR_NSTEP)
_sm = float(np.asarray(M.solar, dtype=float)[T0:T1].mean())
_sp = float(np.asarray(M.solar, dtype=float)[T0:T1].max())
A2 = bool(abs(_sm - ANCHOR_SOLAR_MEAN) <= ANCHOR_SOLAR_TOL
          and abs(_sp - ANCHOR_SOLAR_PEAK) <= ANCHOR_SOLAR_TOL)
print("A1 %s | A2 %s (mean %.4f, peak %.4f)"
      % ("PASS" if A1 else "FAIL", "PASS" if A2 else "FAIL", _sm, _sp))


banner("2. GRID + TOPOLOGY")
NET = pn.mv_oberrhein()
NET.sgen["scaling"] = 1.0
SGEN_P0 = NET.sgen.p_mw.values.copy()
SGEN_Q0 = NET.sgen.q_mvar.values.copy()
LOAD_SCALING0 = NET.load.scaling.values.copy()
S_BASE = float(NET.sn_mva)

_mg = top.create_nxgraph(NET, respect_switches=True)
_comps = [sorted(int(b) for b in c) for c in top.connected_components(_mg)]
_comps.sort(key=len, reverse=True)
C0_ALL = [b for b in _comps[0] if bool(NET.bus.at[b, "in_service"])]
MV = [b for b in C0_ALL if float(NET.bus.at[b, "vn_kv"]) == MV_KV]
NMV = len(MV)
POS = {b: i for i, b in enumerate(MV)}
TIDX = int(NET.trafo[NET.trafo.lv_bus.isin(C0_ALL)].index[0])
LINES0 = list(NET.line[NET.line.from_bus.isin(C0_ALL)
                       & NET.line.to_bus.isin(C0_ALL)].index)
Z_MV = MV_KV ** 2 / S_BASE
print("component 0: %d buses | MV measurement set: %d | trafo %d | lines %d"
      % (len(C0_ALL), NMV, TIDX, len(LINES0)))
print("S_base=%.3f MVA | Z_base(20kV)=%.1f ohm | BB=%d position=%d"
      % (S_BASE, Z_MV, BB, POS[BB]))

OPEN_L = set(NET.switch[(NET.switch.et == "l")
                        & (~NET.switch.closed.astype(bool))].element.astype(int))
print("open line switches: %d  (these are REMOVED from the graph - otherwise it has loops)"
      % len(OPEN_L))

G = nx.Graph()
LR = {}
for _i, _ln in NET.line.iterrows():
    if (bool(_ln.in_service) and _i not in OPEN_L
            and int(_ln.from_bus) in C0_ALL and int(_ln.to_bus) in C0_ALL):
        LR[_i] = float(_ln.r_ohm_per_km) * float(_ln.length_km) / Z_MV
        G.add_edge(int(_ln.from_bus), int(_ln.to_bus), lid=_i)
A6 = bool(G.number_of_nodes() == NMV and G.number_of_edges() == NMV - 1
          and nx.is_tree(G))
print("line graph: %d nodes, %d edges, tree=%s -> A6 %s"
      % (G.number_of_nodes(), G.number_of_edges(), nx.is_tree(G),
         "PASS" if A6 else "FAIL"))


banner("3. ANCHOR A3 - STATIC MATCH")
ST = pp.create_storage(NET, bus=BB, p_mw=0.0, max_e_mwh=100.0, name="batt")


def set_grid(k, s_norm, dm, p_batt=0.0):
    NET.sgen["p_mw"] = k * SGEN_P0 * s_norm
    NET.sgen["q_mvar"] = SGEN_Q0
    NET.load["scaling"] = LOAD_SCALING0 * dm
    NET.storage.at[ST, "p_mw"] = float(p_batt)


def run(k, s_norm, dm, p_batt=0.0):
    set_grid(k, s_norm, dm, p_batt)
    try:
        pp.runpp(NET, algorithm="nr", init="auto", calculate_voltage_angles=True,
                 max_iteration=100, numba=USE_NUMBA)
    except Exception:
        return None
    v = NET.res_bus.loc[MV, "vm_pu"].values.astype(float)
    return None if np.isnan(v).any() else v


_v = run(ANCHOR_STATIC_K, 1.0, 1.0, 0.0)
A3 = bool(_v is not None and abs(_v.max() - ANCHOR_STATIC_VMAX) <= ANCHOR_STATIC_TOL
          and int((_v > V_HI).sum()) == ANCHOR_STATIC_NOVER)
print("k=4.0 static: vmax %.5f (exp %.5f) | violations %d (exp %d) -> %s"
      % (_v.max(), ANCHOR_STATIC_VMAX, int((_v > V_HI).sum()),
         ANCHOR_STATIC_NOVER, "PASS" if A3 else "FAIL"))

if not (A1 and A2 and A3 and A6):
    print("")
    print("One of A1/A2/A3/A6 FAILED. STOP, paste the output to me.")
    sys.exit(2)


banner("4. VM_BASE PRE-COMPUTATION (battery = 0, whole series)")
print("%d steps x %d buses. This section takes a long time." % (NFULL, NMV))
VM = np.full((NFULL, NMV), np.nan)
BMAX = np.full(NFULL, -1, dtype=int)
_fail = 0
_t0 = time.time()
for _i in range(NFULL):
    _vv = run(KSTAR, float(SOLAR_NORM[_i]), float(DMULT[_i]), 0.0)
    if _vv is None:
        _fail += 1
        continue
    VM[_i] = _vv
    BMAX[_i] = MV[int(np.argmax(_vv))]
    if (_i + 1) % PROGRESS_EVERY == 0:
        _el = max(time.time() - _t0, 1e-9)
        print("    %d/%d  (~%.1f min left)"
              % (_i + 1, NFULL, (NFULL - _i - 1) / ((_i + 1) / _el) / 60), flush=True)
print("  finished: %.0f s | non-converging steps: %d" % (time.time() - _t0, _fail))

_ts = VM[T0:T1]
_g = ~np.isnan(_ts[:, 0])
_hit = _ts[_g].max(axis=1) > V_HI
_nviol = int(_hit.sum())
_over = float(_hit.mean() * 100)
_vmx = float(np.nanmax(_ts))
_vmn = float(np.nanmin(_ts))
_b147 = float((BMAX[T0:T1][_g][_hit] == BB).mean() * 100) if _hit.sum() else np.nan

banner("5. ANCHOR A5 - REPRODUCING THE ob3 NUMBERS")
A5 = bool(_nviol == ANCHOR_NVIOL
          and round(_vmx, 4) == ANCHOR_VMAXMAX_R
          and round(_vmn, 4) == ANCHOR_VMINMIN_R)
print("violation count : %d       (exp %d)" % (_nviol, ANCHOR_NVIOL))
print("over            : %.4f%%   (the ob3 log said %%15.11)" % _over)
print("vmax  4 digits  : %.4f     (exp %.4f)   EXACT: %.8f"
      % (round(_vmx, 4), ANCHOR_VMAXMAX_R, _vmx))
print("vmin  4 digits  : %.4f     (exp %.4f)   EXACT: %.8f"
      % (round(_vmn, 4), ANCHOR_VMINMIN_R, _vmn))
print("A5 -> %s" % ("PASS" if A5 else "FAIL"))
print("")
print("COMPARISON (gg2_ob5_a5diag.py, WRONG k=2.8594): vmax %.8f -> 4 digits %.4f"
      % (1.08365035, round(1.08365035, 4)))
print("These values will be recorded in the ROUND in FULL.")
print("")
print("Share of VIOLATING steps whose vmax bus is %d: %.2f%%" % (BB, _b147))
print("(the %15.9 in TASK 1c was over ALL steps; this is the CONDITIONAL share)")
if not A5:
    print("")
    print("A5 FAILED. VM_BASE is not trustworthy - STOP, paste the output to me.")
    sys.exit(3)


banner("6. Rcol TWO VARIANTS")
PS = {}
for _b in G.nodes():
    _sp2 = nx.shortest_path(G, ROOT_A, _b)
    PS[_b] = {G[_sp2[t]][_sp2[t + 1]]["lid"] for t in range(len(_sp2) - 1)}
RCOL_A = np.array([sum(LR[l] for l in (PS[j] & PS[BB])) for j in MV], dtype=float)

_tr = NET.trafo.loc[TIDX]
R_T = float(_tr.vkr_percent) / 100.0 * S_BASE / float(_tr.sn_mva)
X_T = np.sqrt(max((float(_tr.vk_percent) / 100.0 * S_BASE / float(_tr.sn_mva)) ** 2
                  - R_T ** 2, 0.0))
RCOL_B = RCOL_A + R_T

print("variant A (root %d, lines only): Rcol[BB]=%.6f  max=%.6f  mean=%.6f"
      % (ROOT_A, RCOL_A[POS[BB]], RCOL_A.max(), RCOL_A.mean()))
print("variant B (root %d, + trafo r) : Rcol[BB]=%.6f  (+%.6f on every bus)"
      % (ROOT_B, RCOL_B[POS[BB]], R_T))
print("trafo pu (S_base=%.1f): r=%.6f  x=%.6f  -> r/Rcol[BB]=%.1f%%"
      % (S_BASE, R_T, X_T, 100 * R_T / RCOL_A[POS[BB]]))
print("NOTE: in LinDistFlow the battery injects P with Q=0 -> to first order")
print("      ONLY R enters the magnitude. x_t does not enter, it is printed for information.")


banner("7. REAL dV/dP MEASUREMENT (AC)")
IDX_STRIDE = list(range(T0, T1, SAMPLE_STRIDE))
_tv = _ts[_g].max(axis=1)
_ord = np.argsort(_tv)[::-1][:N_TOPV]
IDX_TOPV = [T0 + int(np.where(_g)[0][t]) for t in _ord]
print("sample: %d steps (stride %d) + %d steps (highest base vmax)"
      % (len(IDX_STRIDE), SAMPLE_STRIDE, len(IDX_TOPV)))

ROWS = []
for _grp, _idx in [("stride", IDX_STRIDE), ("topv", IDX_TOPV)]:
    for _p in P_TEST_CORE + P_TEST_WIDE:
        _ea, _eb, _ra, _keep = [], [], [], 0
        for _i in _idx:
            _v0 = VM[_i]
            if np.isnan(_v0).any():
                continue
            _v1 = run(KSTAR, float(SOLAR_NORM[_i]), float(DMULT[_i]), _p)
            if _v1 is None:
                continue
            _dv = _v0 - _v1                       # charging (p>0) LOWERS the voltage
            _pa = RCOL_A / (S_BASE * np.maximum(_v0, 1e-3)) * _p
            _pb = RCOL_B / (S_BASE * np.maximum(_v0, 1e-3)) * _p
            _ea.append(np.sqrt(((_dv - _pa) ** 2).mean()))
            _eb.append(np.sqrt(((_dv - _pb) ** 2).mean()))
            if abs(_pa[POS[BB]]) > 1e-12:
                _ra.append(_dv[POS[BB]] / _pa[POS[BB]])
            _keep += 1
        ROWS.append({"grup": _grp, "p_mw": _p, "n": _keep,
                     "rmse_A": float(np.mean(_ea)) if _ea else np.nan,
                     "rmse_B": float(np.mean(_eb)) if _eb else np.nan,
                     "rmse_A_max": float(np.max(_ea)) if _ea else np.nan,
                     "oran_BB_ort": float(np.mean(_ra)) if _ra else np.nan,
                     "oran_BB_min": float(np.min(_ra)) if _ra else np.nan,
                     "oran_BB_max": float(np.max(_ra)) if _ra else np.nan})
        _r = ROWS[-1]
        print("  %-7s p=%+6.1f MW  n=%2d | RMSE A %.3e  B %.3e | "
              "real/LDF@BB mean %.3f (min %.3f max %.3f)"
              % (_r["grup"], _r["p_mw"], _r["n"], _r["rmse_A"], _r["rmse_B"],
                 _r["oran_BB_ort"], _r["oran_BB_min"], _r["oran_BB_max"]),
              flush=True)

DFR = pd.DataFrame(ROWS)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
DFR.to_csv(OUT_CSV, index=False)


banner("8. VERDICT - Rcol VARIANT SELECTION")
CORE = DFR[DFR["p_mw"].abs() <= 3.0]
MA, MB = float(CORE["rmse_A"].mean()), float(CORE["rmse_B"].mean())
print("mean RMSE in the |p| <= 3 MW range:  A %.3e   B %.3e   difference %.3e"
      % (MA, MB, abs(MA - MB)))
if abs(MA - MB) < RMSE_TIE:
    CHOICE, RCOL = "A", RCOL_A
    print("The difference is below %.1e -> VARIANT A chosen." % RMSE_TIE)
    print("To be written: the trafo resistance term is NEGLIGIBLE within the measurement.")
elif MA < MB:
    CHOICE, RCOL = "A", RCOL_A
    print("VARIANT A has the lower RMSE -> A chosen.")
else:
    CHOICE, RCOL = "B", RCOL_B
    print("VARIANT B has the lower RMSE -> B chosen (the trafo resistance matters).")

WIDE = DFR[DFR["p_mw"].abs() >= 10.0]
print("")
print("WIDE POWER RANGE (NOT for the verdict, for the TASK 3 battery sizing):")
for _, r in WIDE.iterrows():
    print("  %-7s p=%+6.1f MW | RMSE A %.3e (worst step %.3e) | ratio %.3f"
          % (r["grup"], r["p_mw"], r["rmse_A"], r["rmse_A_max"], r["oran_BB_ort"]))
print("")
print("This error does NOT affect the SAFE LAYER (which runs real AC), only the")
print("quality of the PLANNER. case33bw had the same structure.")


banner("9. SAVE")
np.savez_compressed(OUT_NPZ, vm=VM, Rcol=RCOL, S_base=S_BASE,
                    mv_buses=np.array(MV, dtype=int), bb=BB, kstar=KSTAR,
                    Rcol_A=RCOL_A, Rcol_B=RCOL_B, r_trafo=R_T, x_trafo=X_T,
                    bmax=BMAX, variant=CHOICE, n_fail=_fail)
print("written: %s" % OUT_NPZ)
print("  vm %s | Rcol %s (variant %s) | S_base %.3f | mv_buses %d"
      % (str(VM.shape), str(RCOL.shape), CHOICE, S_BASE, NMV))
print("written: %s (%d rows)" % (OUT_CSV, len(DFR)))
print("")
print("ANCHORS A1/A2/A3/A5/A6 : %s / %s / %s / %s / %s"
      % tuple("PASS" if x else "FAIL" for x in (A1, A2, A3, A5, A6)))
print("selected Rcol variant: %s" % CHOICE)
print("")
print("WARNING (next task): safe_layer.SafeLayer._vac takes vm_pu.min/max over the")
print("whole grid. On mv_oberrhein component 1 is counted too. Unusable without a MASK.")
print("")
print("DONE.")
