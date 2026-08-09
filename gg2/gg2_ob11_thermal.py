#!/usr/bin/env python3
"""
GG-2 / ROUND v9 - TASK 8
mv_oberrhein: THERMAL AUDIT OF THE ARMS' ACTUAL DISPATCH.

Run (from the GG-1 ROOT folder):
    python gg2/gg2_ob11_thermal.py 2>&1 | tee ob11_thermal.log

Requires: gg2/data/ob8_{naive,nwp,oracle}.npz, gg2/data/ob_base_vm.npz
ASCII only. ~7 min. THE MPC IS NOT RE-RUN.

-------------------------------------------------------------------------------
WHY -- AN UNCLOSED CONTRADICTION
-------------------------------------------------------------------------------
The k* PV capacity was chosen with the criterion "the trafo and the lines must
stay within limits AT EVERY EXPORT STEP" (TASK 1c). But:
  - that criterion was computed WITH THE BATTERY AT ZERO (so it was written in
    the pre-registration),
  - the only thermal check in TASK 3 looked ONLY at violating steps (high solar,
    low import) and found a maximum of 42.8%,
  - the MPC has NO THERMAL CONSTRAINT; the battery may charge up to 8 MW
    following the price signal,
  - the trafo reaches 104.1% at high demand even WITHOUT PV (TASK 1b, k=0).
The TASK 3 pre-registration said "the real trafo loading will be back-checked
after the naive arm". **THAT CHECK WAS NEVER DONE.** This script does it.

If it is not done, the referee objection is: "you sized the PV against the trafo
limit, then broke that limit with your own storage; the hosting criterion is
meaningless."

-------------------------------------------------------------------------------
PRE-REGISTRATION (written BEFORE the run)
-------------------------------------------------------------------------------
For each arm, at every step, an AC power flow is run with the battery at its
ACTUAL power FROM THE TRAJECTORY, and the trafo + maximum line loading are
recorded. The same steps are also run WITH THE BATTERY AT ZERO (arm-independent,
once).

Class distinction:
  PRE-EXISTING  : >100% with battery=0 as well -> comes from modulating a German
                  feeder's load with an Irish demand shape; already declared in
                  TASK 1b.
  STORAGE-ADDED : <=100% with battery=0 but >100% with the battery
                  -> THE BATTERY CREATED THIS. That is the actual question.

SENTENCE TO BE WRITTEN:
  if the number of STORAGE-ADDED steps is ZERO:
    "The storage dispatch adds no thermal violations beyond those already
     present with storage idle."
  if it is NOT ZERO:
    "The storage dispatch pushes the transformer past its rating in N of 2429
     steps (max X%), which the controller does not model. We report this as a
     limitation: the hosting criterion constrains the solar build, not the
     storage dispatch."
  k*, the battery size and the controller WILL NOT BE CHANGED based on the
  result. Whatever the numbers turn out to be, they go into the paper AS THEY
  ARE.

ANCHOR C11: with the battery at the p from the trajectory, the computed Vmax
must reproduce the mx value recorded in the trajectory to within 1e-9 (the same
anchor as C10 in ob10). If it fails, no number is reported.
-------------------------------------------------------------------------------
"""

import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top

KSTAR = 2.859375
BB = 147
MV_KV = 20.0
V_HI, V_LO = 1.05, 0.95
LIMIT = 100.0
ARMS = ["naive", "nwp", "oracle"]
NPZ_VM = os.path.join(_HERE, "data", "ob_base_vm.npz")
OUT_NPZ = os.path.join(_HERE, "data", "ob11_thermal.npz")


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
print("numba=%s | k*=%.6f | thermal limit %.1f%%" % (USE_NUMBA, KSTAR, LIMIT))

import mpc_econ_voltage_pareto as M  # noqa: E402

T0, T1 = int(M.SPLIT), int(M.N) - 1
n = T1 - T0
solar_norm = np.asarray(M.solar, dtype=float) / float(M.SP)
dmult = np.asarray(M.dmult, dtype=float)

Z = np.load(NPZ_VM, allow_pickle=True)
MV = [int(b) for b in Z["mv_buses"]]

NET = pn.mv_oberrhein()
NET.sgen["scaling"] = 1.0
P0 = NET.sgen.p_mw.values.copy()
Q0 = NET.sgen.q_mvar.values.copy()
LS0 = NET.load.scaling.values.copy()
mg = top.create_nxgraph(NET, respect_switches=True)
cc = [sorted(int(b) for b in x) for x in top.connected_components(mg)]
cc.sort(key=len, reverse=True)
C0 = set(b for b in cc[0] if bool(NET.bus.at[b, "in_service"]))
TIDX = int(NET.trafo[NET.trafo.lv_bus.isin(C0)].index[0])
LINES0 = list(NET.line[NET.line.from_bus.isin(C0) & NET.line.to_bus.isin(C0)].index)
ST = pp.create_storage(NET, bus=BB, p_mw=0.0, max_e_mwh=100.0, name="batt")
RPKW = dict(algorithm="nr", init="auto", calculate_voltage_angles=True,
            max_iteration=100, numba=USE_NUMBA)
print("trafo idx %d (%.1f MVA) | lines %d | MV buses %d"
      % (TIDX, float(NET.trafo.at[TIDX, "sn_mva"]), len(LINES0), len(MV)))


def step(i, p_batt):
    NET.sgen["p_mw"] = KSTAR * P0 * float(solar_norm[i])
    NET.sgen["q_mvar"] = Q0
    NET.load["scaling"] = LS0 * float(dmult[i])
    NET.storage.at[ST, "p_mw"] = float(p_batt)
    try:
        pp.runpp(NET, **RPKW)
    except Exception:
        return None
    v = NET.res_bus.vm_pu.loc[MV]
    if bool(v.isna().any()):
        return None
    return (float(NET.res_trafo.at[TIDX, "loading_percent"]),
            float(NET.res_line.loc[LINES0, "loading_percent"].max()),
            float(NET.res_trafo.at[TIDX, "p_hv_mw"]),
            float(v.max()))


banner("1. BASELINE - BATTERY AT ZERO (arm-independent)")
TR0 = np.full(n, np.nan)
LN0 = np.full(n, np.nan)
PH0 = np.full(n, np.nan)
t0 = time.time()
for j in range(n):
    r = step(T0 + j, 0.0)
    if r is None:
        continue
    TR0[j], LN0[j], PH0[j], _ = r
    if (j + 1) % 800 == 0:
        el = max(time.time() - t0, 1e-9)
        print("    %d/%d (~%.1f min left)"
              % (j + 1, n, (n - j - 1) / ((j + 1) / el) / 60), flush=True)
g0 = ~np.isnan(TR0)
base_over_tr = (TR0 > LIMIT) & g0
base_over_ln = (LN0 > LIMIT) & g0
print("  finished: %.0f s" % (time.time() - t0))
print("  trafo : max %.1f%% | >%.0f%% at %d steps (%.2f%%)"
      % (np.nanmax(TR0), LIMIT, int(base_over_tr.sum()),
         100.0 * base_over_tr.sum() / n))
print("      %d of these are in the IMPORT direction (p_hv >= 0)"
      % int((base_over_tr & (PH0 >= 0)).sum()))
print("  lines : max %.1f%% | >%.0f%% at %d steps"
      % (np.nanmax(LN0), LIMIT, int(base_over_ln.sum())))


banner("2. ARMS - WITH THE ACTUAL BATTERY DISPATCH")
RES = {}
ANCH = {}
for arm in ARMS:
    path = os.path.join(_HERE, "data", "ob8_%s.npz" % arm)
    if not os.path.exists(path):
        print("[ERROR] not found: %s" % path)
        sys.exit(1)
    z = np.load(path)
    P, mx = z["p"], z["mx"]
    if int(z["done"]) < n:
        print("[ERROR] %s is incomplete" % path)
        sys.exit(1)
    TR = np.full(n, np.nan)
    LN = np.full(n, np.nan)
    PH = np.full(n, np.nan)
    dmax = 0.0
    t0 = time.time()
    for j in range(n):
        r = step(T0 + j, float(P[j]))
        if r is None:
            continue
        TR[j], LN[j], PH[j], vm = r
        dmax = max(dmax, abs(vm - float(mx[j])))
    ANCH[arm] = dmax
    RES[arm] = (TR, LN, PH)
    print("  %-7s trafo max %6.1f%% | line max %6.1f%% | anchor |dV| %.2e | %.0f s"
          % (arm, np.nanmax(TR), np.nanmax(LN), dmax, time.time() - t0), flush=True)


banner("3. ANCHOR C11")
worst = max(ANCH.values())
C11 = bool(worst < 1e-9)
for arm in ARMS:
    print("  %-7s max |Vmax_recomputed - mx_trajectory| = %.3e" % (arm, ANCH[arm]))
print("  -> C11 %s" % ("PASS" if C11 else "FAIL"))
if not C11:
    print("")
    print("The grid state is NOT the same as the TASK 5 run. No number is reported.")
    sys.exit(2)


banner("4. OVERLOAD DECOMPOSITION")
print("PRE-EXISTING  = >%.0f%% with battery=0 as well   (declared in TASK 1b)" % LIMIT)
print("STORAGE-ADDED = <=%.0f%% with battery=0, >%.0f%% with the battery" % (LIMIT, LIMIT))
print("")
print("%-8s | %-28s | %s" % ("arm", "TRAFO", "LINES"))
print("%-8s | %-9s %-9s %-8s | %-9s %-9s %s"
      % ("", "max%", "existing", "ADDED", "max%", "existing", "ADDED"))
SUM = {}
for arm in ARMS:
    TR, LN, PH = RES[arm]
    g = ~np.isnan(TR)
    tr_over = (TR > LIMIT) & g
    ln_over = (LN > LIMIT) & g
    tr_new = tr_over & ~base_over_tr
    ln_new = ln_over & ~base_over_ln
    SUM[arm] = (int(tr_over.sum()), int(tr_new.sum()),
                int(ln_over.sum()), int(ln_new.sum()),
                float(np.nanmax(TR)), float(np.nanmax(LN)))
    print("%-8s | %-9.1f %-9d %-8d | %-9.1f %-9d %d"
          % (arm, np.nanmax(TR), int((tr_over & base_over_tr).sum()),
             int(tr_new.sum()), np.nanmax(LN),
             int((ln_over & base_over_ln).sum()), int(ln_new.sum())))

print("")
print("What happens at the STORAGE-ADDED steps (naive arm, first 5):")
TRn, LNn, PHn = RES["naive"]
gn = ~np.isnan(TRn)
newn = (TRn > LIMIT) & gn & ~base_over_tr
zn = np.load(os.path.join(_HERE, "data", "ob8_naive.npz"))
shown = 0
for j in np.where(newn)[0]:
    print("   step %5d | p_batt %+6.2f MW | trafo %6.1f%% (battery=0: %5.1f%%)"
          " | solar %.3f dmult %.3f"
          % (T0 + j, float(zn["p"][j]), TRn[j], TR0[j],
             solar_norm[T0 + j], dmult[T0 + j]))
    shown += 1
    if shown >= 5:
        break
if shown == 0:
    print("   NONE - the battery created no new thermal overload at any step.")


banner("5. VERDICT")
tot_new_tr = sum(SUM[a][1] for a in ARMS)
tot_new_ln = sum(SUM[a][3] for a in ARMS)
if tot_new_tr == 0 and tot_new_ln == 0:
    print("THE BATTERY CREATES NO NEW THERMAL OVERLOAD.")
    print("")
    print("Sentence for the paper:")
    print("  The storage dispatch adds no thermal violations beyond those")
    print("  already present with storage idle. The transformer exceeds its")
    print("  rating in the import direction at high demand even with no solar")
    print("  and no storage, because a German feeder's load is modulated by an")
    print("  Irish system-load shape; the hosting criterion is therefore")
    print("  applied to the export direction only.")
else:
    print("THE BATTERY CREATES NEW THERMAL OVERLOAD.")
    for a in ARMS:
        print("  %-7s trafo +%d steps (max %.1f%%) | lines +%d steps (max %.1f%%)"
              % (a, SUM[a][1], SUM[a][4], SUM[a][3], SUM[a][5]))
    print("")
    print("Sentence for the paper (NUMBERS TO BE FILLED IN):")
    print("  The storage dispatch pushes the transformer past its rating in")
    print("  N of %d steps (maximum X%%). The controller does not model thermal" % n)
    print("  limits; we report this as a limitation. The hosting criterion")
    print("  constrains the solar build, not the storage dispatch.")
    print("")
    print("k*, the battery size and the controller WILL NOT BE CHANGED (pre-registration).")

print("")
print("EXTRA INFORMATION - import/export split (naive arm):")
exp = gn & (PHn < 0)
imp = gn & (PHn >= 0)
print("  export steps %d | trafo max %.1f%%"
      % (int(exp.sum()), float(np.nanmax(TRn[exp])) if exp.any() else float("nan")))
print("  import steps %d | trafo max %.1f%%"
      % (int(imp.sum()), float(np.nanmax(TRn[imp])) if imp.any() else float("nan")))

np.savez_compressed(OUT_NPZ, tr_base=TR0, ln_base=LN0, ph_base=PH0,
                    **{("tr_" + a): RES[a][0] for a in ARMS},
                    **{("ln_" + a): RES[a][1] for a in ARMS},
                    **{("ph_" + a): RES[a][2] for a in ARMS})
print("")
print("written: %s" % OUT_NPZ)
print("C11 %s" % ("PASS" if C11 else "FAIL"))
print("")
print("DONE.")
