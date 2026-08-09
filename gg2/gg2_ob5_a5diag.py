#!/usr/bin/env python3
"""
GG-2 / ROUND v8 SECTION 8 - TASK 2 DIAGNOSIS
Why did the A5 anchor in gg2_ob4_rcol.py fail?

Run (from the GG-1 ROOT folder):
    python gg2/gg2_ob5_a5diag.py 2>&1 | tee ob5_a5diag.log

ASCII only. ~3 min (2429 steps x 2 paths).

-------------------------------------------------------------------------------
SYMPTOM
-------------------------------------------------------------------------------
gg2_ob3_kstar.py  (k*=2.8594, test block): over 15.11%  vmax 1.0836  vmin 0.9656
gg2_ob4_rcol.py   (same k*, same block)  : over 15.11%  vmax 1.0837  vmin 0.9656
                                                            ^^^^^^ 1e-4 diff
The over-voltage rate (a DISCRETE count) and vmin match EXACTLY; only vmax
diverges.

TWO CANDIDATES - both are measured in this script, neither is assumed:
  H_A (TOLERANCE) The A5 test in ob4 takes its reference from the 1.0836 that
                  ob3 PRINTED with %.4f. The rounding uncertainty of the
                  reference itself is +/-5e-5, and the test tolerance is also
                  5e-5. So if the true value is around 1.08365 the test is
                  MATHEMATICALLY invalid - it fails even when the two runs are
                  identical.
  H_B (MODEL)     ob4 adds a p=0 `storage` element at bus 147; ob3 does not.
                  If a storage element at zero power is not electrically
                  NEUTRAL, the difference is REAL and VM_BASE is suspect.

PRE-REGISTRATION (BEFORE the run):
  The two paths are run step by step with the same k*, the same series and the
  same runpp settings.
  d = max |vmax_no_storage - vmax_with_storage|  (2429 steps)
    d < 1e-9   -> H_B IS ELIMINATED. Storage at zero power is neutral. The only
                  remaining explanation is H_A: the A5 tolerance was set up
                  incorrectly. Fix: because the reference was rounded to 4
                  digits, the comparison is also made on the ROUNDED value
                  (round(x,4) == 1.0836) and ob4 is re-run.
    d >= 1e-9  -> H_B STANDS. The difference is real; its magnitude is reported
                  and VM_BASE is regenerated using the storage-free path. No
                  explanation is FABRICATED.
  In both cases the violation COUNT (an integer) of the two paths is compared
  as well.
-------------------------------------------------------------------------------
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top

V_HI, V_LO, MV_KV = 1.05, 0.95, 20.0
KSTAR = 2.8594
BB = 147
PROGRESS_EVERY = 800


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
print("numba=%s | k*=%.4f | BB=%d" % (USE_NUMBA, KSTAR, BB))

import mpc_econ_voltage_pareto as M  # noqa: E402

T0, T1 = int(M.SPLIT), int(M.N) - 1
NSTEP = T1 - T0
SOLAR_NORM = np.asarray(M.solar, dtype=float) / float(M.SP)
DMULT = np.asarray(M.dmult, dtype=float)
print("window %d:%d (%d steps)" % (T0, T1, NSTEP))


def build(with_storage):
    net = pn.mv_oberrhein()
    net.sgen["scaling"] = 1.0
    st = None
    if with_storage:
        st = pp.create_storage(net, bus=BB, p_mw=0.0, max_e_mwh=100.0, name="batt")
    return net, st


NET_NO, _ = build(False)
NET_ST, ST = build(True)

_mg = top.create_nxgraph(NET_NO, respect_switches=True)
_c = [sorted(int(b) for b in x) for x in top.connected_components(_mg)]
_c.sort(key=len, reverse=True)
C0_ALL = [b for b in _c[0] if bool(NET_NO.bus.at[b, "in_service"])]
MV = [b for b in C0_ALL if float(NET_NO.bus.at[b, "vn_kv"]) == MV_KV]
print("MV measurement set: %d buses" % len(MV))

P0 = NET_NO.sgen.p_mw.values.copy()
Q0 = NET_NO.sgen.q_mvar.values.copy()
LS0 = NET_NO.load.scaling.values.copy()


def step(net, st, s_norm, dm):
    net.sgen["p_mw"] = KSTAR * P0 * s_norm
    net.sgen["q_mvar"] = Q0
    net.load["scaling"] = LS0 * dm
    if st is not None:
        net.storage.at[st, "p_mw"] = 0.0
    try:
        pp.runpp(net, algorithm="nr", init="auto", calculate_voltage_angles=True,
                 max_iteration=100, numba=USE_NUMBA)
    except Exception:
        return np.nan, np.nan
    v = net.res_bus.loc[MV, "vm_pu"].values.astype(float)
    if np.isnan(v).any():
        return np.nan, np.nan
    return float(v.max()), float(v.min())


banner("1. RUN THE TWO PATHS STEP BY STEP")
VX_NO = np.full(NSTEP, np.nan)
VN_NO = np.full(NSTEP, np.nan)
VX_ST = np.full(NSTEP, np.nan)
VN_ST = np.full(NSTEP, np.nan)
t0 = time.time()
for j in range(NSTEP):
    i = T0 + j
    s, d = float(SOLAR_NORM[i]), float(DMULT[i])
    VX_NO[j], VN_NO[j] = step(NET_NO, None, s, d)
    VX_ST[j], VN_ST[j] = step(NET_ST, ST, s, d)
    if (j + 1) % PROGRESS_EVERY == 0:
        el = max(time.time() - t0, 1e-9)
        print("    %d/%d  (~%.1f min left)"
              % (j + 1, NSTEP, (NSTEP - j - 1) / ((j + 1) / el) / 60), flush=True)
print("  finished: %.0f s" % (time.time() - t0))


banner("2. COMPARISON AT FULL PRECISION")
g = ~np.isnan(VX_NO) & ~np.isnan(VX_ST)
dvx = np.abs(VX_NO[g] - VX_ST[g])
dvn = np.abs(VN_NO[g] - VN_ST[g])
D = float(dvx.max())
print("valid steps             : %d / %d" % (int(g.sum()), NSTEP))
print("max |vmax diff|   d     : %.3e pu" % D)
print("max |vmin diff|         : %.3e pu" % float(dvn.max()))
print("steps with d > 1e-9     : %d" % int((dvx > 1e-9).sum()))
print("")
print("NO storage   : vmax_max %.8f | vmin_min %.8f | %d violating steps (%.4f%%)"
      % (np.nanmax(VX_NO), np.nanmin(VN_NO), int((VX_NO[g] > V_HI).sum()),
         (VX_NO[g] > V_HI).mean() * 100))
print("WITH storage : vmax_max %.8f | vmin_min %.8f | %d violating steps (%.4f%%)"
      % (np.nanmax(VX_ST), np.nanmin(VN_ST), int((VX_ST[g] > V_HI).sum()),
         (VX_ST[g] > V_HI).mean() * 100))
print("")
print("%.4f rounding -> NO storage: %.4f | WITH storage: %.4f"
      % (0.0, round(float(np.nanmax(VX_NO)), 4), round(float(np.nanmax(VX_ST)), 4)))
print("what the ob3 log says   : 1.0836")
print("what the ob4 log says   : 1.0837")


banner("3. VERDICT")
if D < 1e-9:
    print("H_B ELIMINATED: storage at zero power is electrically NEUTRAL (d=%.3e < 1e-9)." % D)
    print("The two paths give the SAME solution.")
    print("")
    print("That leaves H_A: the A5 tolerance was set up incorrectly. The reference (1.0836,")
    print("printed with %.4f) is itself uncertain by +/-5e-5, and the test tolerance is")
    print("also 5e-5. This test can fail even when the two runs are identical - my mistake.")
    print("")
    print("FIX: the A5 comparison will be made on the ROUNDED value")
    print("(round(vmax,4) == 1.0836), or the tolerance will be raised to 1.5e-4.")
    print("gg2_ob4_rcol.py will be RE-RUN with this fix.")
    print("")
    print("NOTE: the true vmax_max = %.8f. This value will be recorded in the ROUND"
          % np.nanmax(VX_NO))
    print("in full, and no anchor will ever again be set on a 4-digit number.")
else:
    print("H_B STANDS: storage at zero power is NOT NEUTRAL. d=%.3e pu" % D)
    print("The difference is REAL. It exceeds 1e-9 at %d steps." % int((dvx > 1e-9).sum()))
    print("NO EXPLANATION IS FABRICATED. VM_BASE will be regenerated storage-free;")
    print("the cause will be investigated separately. Paste the output to me as is.")

print("")
print("DONE.")
