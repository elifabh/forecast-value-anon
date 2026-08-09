#!/usr/bin/env python3
"""
GG-2 / ROUND v8 SECTION 8 - TASK 7
mv_oberrhein: CURTAILED ENERGY (curtailment) - real AC binary search.

Run (from the GG-1 ROOT folder):
    python gg2/gg2_ob10_curt.py 2>&1 | tee ob10_curt.log

Requires: gg2/data/ob8_{naive,nwp,oracle}.npz, gg2/data/ob_base_vm.npz
ASCII only. ~6-8 min. THE MPC IS NOT RE-RUN.

-------------------------------------------------------------------------------
QUESTION
-------------------------------------------------------------------------------
When the battery cannot fix an over-voltage, the physical remedy is to CURTAIL
solar generation. At every violating step: "what percentage of the solar
generation of component 0 would have had to be curtailed to bring the voltage
down to 1.05?" This is answered EXACTLY with a binary search on the real AC
power flow. The same measurement was made for case33bw (ROUND v8 3.4).

Curtailment is applied ONLY to the 93 sgens of component 0; component 1 is left
untouched (it is a separate grid anyway and does not enter the measurement).

-------------------------------------------------------------------------------
PRE-REGISTRATION (written BEFORE the run)
-------------------------------------------------------------------------------
The difference is reported with the SAME paired daily-block bootstrap as the
violations and the profit (48-step days, the first 50 full days, 10000
replicates, seed 12345, 95% percentile).

  if the naive - NWP interval EXCLUDES zero:
    "Over fifty days on one feeder the forecast prevents X.XX MWh of
     curtailment, 95% CI [a, b]."
  if it INCLUDES zero:
    "The curtailment difference is NOT RESOLVED: X.XX MWh, 95% CI [a, b]."

LIMIT (also written for case33bw): this is the curtailment REQUIRED TO FIX the
violation; the curtailment policy of a real distribution operator was not
modelled. It is an UPPER BOUND estimate.

-------------------------------------------------------------------------------
ANCHOR C10 - THE MOST IMPORTANT ONE
-------------------------------------------------------------------------------
With zero curtailment (c=0), putting the battery at the p value from the
trajectory and running the power flow, the resulting Vmax MUST REPRODUCE the mx
value recorded in the trajectory to within 1e-9. This is the proof that this
script builds the SAME grid state as the TASK 5 run. On case33bw the same
anchor passed with a difference of 0.00e+00.
If it fails, no number is reported.
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
BISECT_N = 16
DAY, NDAY, NBOOT, SEED = 48, 50, 10000, 12345
ANCHOR_TOL = 1e-9

ARMS = ["naive", "nwp", "oracle"]
NPZ_VM = os.path.join(_HERE, "data", "ob_base_vm.npz")
OUT_NPZ = os.path.join(_HERE, "data", "ob10_curt.npz")

# case33bw reference (ROUND v8 3.4)
C33 = {"naive": (126.92, 12.32), "nwp": (119.44, 11.59), "oracle": (112.55, 10.92)}


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
print("numba=%s | k*=%.6f | binary search %d steps" % (USE_NUMBA, KSTAR, BISECT_N))

import mpc_econ_voltage_pareto as M  # noqa: E402

T0, T1 = int(M.SPLIT), int(M.N) - 1
n = T1 - T0
dt = M.dt
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
SG0 = np.array([int(b) in C0 for b in NET.sgen.bus.values])
PV_NOM0 = float(P0[SG0].sum())
ST = pp.create_storage(NET, bus=BB, p_mw=0.0, max_e_mwh=100.0, name="batt")
RPKW = dict(algorithm="nr", init="auto", calculate_voltage_angles=True,
            max_iteration=100, numba=USE_NUMBA)

SOLAR_TOT = float(KSTAR * PV_NOM0 * solar_norm[T0:T1].sum() * dt)
print("component 0: %d sgens, nominal %.4f MW" % (int(SG0.sum()), PV_NOM0))
print("total solar generation of the test block: %.1f MWh" % SOLAR_TOT)


def vmax_at(i, c, p_batt):
    """c = curtailment fraction [0,1], applied to the sgens of component 0."""
    p = KSTAR * P0 * float(solar_norm[i])
    p[SG0] *= (1.0 - c)
    NET.sgen["p_mw"] = p
    NET.sgen["q_mvar"] = Q0
    NET.load["scaling"] = LS0 * float(dmult[i])
    NET.storage.at[ST, "p_mw"] = float(p_batt)
    try:
        pp.runpp(NET, **RPKW)
    except Exception:
        return None
    v = NET.res_bus.vm_pu.loc[MV]
    return None if bool(v.isna().any()) else float(v.max())


def curtail_step(i, p_batt):
    """The SMALLEST curtailment fraction that brings Vmax down to V_HI.
    Returns (c, vmax0)."""
    v0 = vmax_at(i, 0.0, p_batt)
    if v0 is None:
        return np.nan, np.nan
    if v0 <= V_HI:
        return 0.0, v0
    v1 = vmax_at(i, 1.0, p_batt)
    if v1 is None or v1 > V_HI:
        return np.inf, v0
    lo, hi = 0.0, 1.0
    for _ in range(BISECT_N):
        mid = 0.5 * (lo + hi)
        vm = vmax_at(i, mid, p_batt)
        if vm is None or vm > V_HI:
            lo = mid
        else:
            hi = mid
    return hi, v0


banner("1. ARMS - CURTAILMENT MEASUREMENT")
RES = {}
ANCH = {}
for arm in ARMS:
    path = os.path.join(_HERE, "data", "ob8_%s.npz" % arm)
    if not os.path.exists(path):
        print("[ERROR] not found: %s" % path)
        sys.exit(1)
    z = np.load(path)
    mn, mx, P = z["mn"], z["mx"], z["p"]
    if int(z["done"]) < n:
        print("[ERROR] %s is incomplete" % path)
        sys.exit(1)
    viol = np.where((mn < V_LO) | (mx > V_HI))[0]

    curt_mwh = np.zeros(n)
    frac = np.full(n, np.nan)
    dmaxa, nfail = 0.0, 0
    t0 = time.time()
    for j in viol:
        i = T0 + int(j)
        c, v0 = curtail_step(i, float(P[j]))
        if not np.isfinite(v0):
            nfail += 1
            continue
        dmaxa = max(dmaxa, abs(v0 - float(mx[j])))     # ANCHOR C10
        if not np.isfinite(c):
            nfail += 1
            continue
        frac[j] = c
        curt_mwh[j] = c * KSTAR * PV_NOM0 * float(solar_norm[i]) * dt
    ANCH[arm] = dmaxa
    RES[arm] = (curt_mwh, frac, len(viol))
    print("  %-7s violations %4d | curtailment %8.2f MWh (%5.2f%% of solar) | "
          "max curtailment fraction %.3f | anchor |dV| %.2e | %.0f s"
          % (arm, len(viol), curt_mwh.sum(), 100 * curt_mwh.sum() / SOLAR_TOT,
             np.nanmax(frac) if np.isfinite(frac).any() else float("nan"),
             dmaxa, time.time() - t0), flush=True)
    if nfail:
        print("    WARNING: no curtailment found at %d steps (even full curtailment"
              " was not enough, or it did not converge)" % nfail)

banner("2. ANCHOR C10")
worst = max(ANCH.values())
C10 = bool(worst < ANCHOR_TOL)
for arm in ARMS:
    print("  %-7s max |Vmax_recomputed - mx_trajectory| = %.3e" % (arm, ANCH[arm]))
print("  threshold %.0e -> C10 %s" % (ANCHOR_TOL, "PASS" if C10 else "FAIL"))
if not C10:
    print("")
    print("The grid state is NOT the same as the TASK 5 run. No number is reported.")
    sys.exit(2)


banner("3. LEVELS - THE TWO FEEDERS SIDE BY SIDE")
print("%-8s | %-28s | %s" % ("arm", "mv_oberrhein", "case33bw (ROUND v8 3.4)"))
for arm in ARMS:
    c = RES[arm][0].sum()
    print("%-8s | %8.2f MWh  (%5.2f%% solar) | %8.2f MWh  (%5.2f%% solar)"
          % (arm, c, 100 * c / SOLAR_TOT, C33[arm][0], C33[arm][1]))


banner("4. BOOTSTRAP - PAIRED DAILY BLOCK")
NUSE = DAY * NDAY
DRAWS = np.random.default_rng(SEED).integers(0, NDAY, size=(NBOOT, NDAY))
day = {a: RES[a][0][:NUSE].reshape(NDAY, DAY).sum(axis=1) for a in ARMS}
print("(the first %d full days; the arm has %d steps, the remaining %d were dropped)"
      % (NDAY, n, n - NUSE))
print("")

OUT = {}
for lab, a, b in [("naive - NWP", "naive", "nwp"),
                  ("naive - oracle", "naive", "oracle"),
                  ("NWP - oracle", "nwp", "oracle")]:
    obs = float(day[a].mean() * NDAY - day[b].mean() * NDAY)
    bs = (day[a][DRAWS].mean(axis=1) - day[b][DRAWS].mean(axis=1)) * NDAY
    m, sd = float(bs.mean()), float(bs.std(ddof=1))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p0 = float((bs <= 0).mean())
    OUT[lab] = (obs, lo, hi, p0)
    print("  %-16s observed %+8.2f MWh | boot mean %+8.2f  SD %6.2f | "
          "%%95 CI [%+8.2f, %+8.2f] | P(<=0) %.3f"
          % (lab, obs, m, sd, lo, hi, p0))


banner("5. VERDICT")
o, lo, hi, p0 = OUT["naive - NWP"]
if lo > 0:
    print("Over fifty days on one feeder the forecast prevents %.2f MWh of curtailment," % o)
    print("%%95 CI [%.2f, %.2f], P(diff<=0)=%.3f." % (lo, hi, p0))
elif hi < 0:
    print("OPPOSITE DIRECTION: NWP requires MORE curtailment: %.2f MWh, CI [%.2f, %.2f]."
          % (o, lo, hi))
    print("An outcome not anticipated in the pre-registration. No explanation is FABRICATED.")
else:
    print("The curtailment difference is NOT RESOLVED: %.2f MWh, %%95 CI [%.2f, %.2f]."
          % (o, lo, hi))

o3, lo3, hi3, _ = OUT["NWP - oracle"]
print("")
print("NWP - oracle (for case33bw this was NOT RESOLVED on violations but WAS on curtailment):")
print("  violations  : ROUND v8 -> case33bw [-0.13, +1.42] | oberrhein [-1.17, +0.83]")
print("  curtailment : case33bw [+2.33, +12.97] | oberrhein [%.2f, %.2f]" % (lo3, hi3))
if lo3 > 0:
    print("  -> SAME PATTERN: the curtailment metric is more sensitive than the violation count.")
elif hi3 < 0:
    print("  -> OPPOSITE: on oberrhein NWP requires LESS curtailment than oracle.")
else:
    print("  -> THE PATTERN DOES NOT REPEAT: on oberrhein curtailment is unresolved too.")
print("  This is an OBSERVATION, not a headline (same status as ROUND v8 3.4).")

print("")
print("LIMIT: this is the curtailment REQUIRED to fix the violation; the curtailment")
print("policy of a real distribution operator was not modelled. It is an UPPER BOUND.")

np.savez_compressed(OUT_NPZ, solar_tot=SOLAR_TOT, kstar=KSTAR,
                    **{("curt_" + a): RES[a][0] for a in ARMS},
                    **{("frac_" + a): RES[a][1] for a in ARMS})
print("")
print("written: %s" % OUT_NPZ)
print("C10 %s" % ("PASS" if C10 else "FAIL"))
print("")
print("DONE.")
