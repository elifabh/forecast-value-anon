#!/usr/bin/env python3
"""
GG-2 / ROUND v8 SECTION 8 - TASK 5
mv_oberrhein: THREE ARMS (naive / NWP / oracle) - FAIR-INFORMATION MPC.

Run (from the GG-1 ROOT folder):
    python gg2/gg2_ob8_arms.py 2>&1 | tee ob8_arms.log
Single arm:
    python gg2/gg2_ob8_arms.py --arm nwp 2>&1 | tee ob8_nwp.log
After an interruption:
    python gg2/gg2_ob8_arms.py --resume 2>&1 | tee -a ob8_arms.log

Requires: gg2/mpc_ob.py, gg2/safe_layer_ob.py,
          gg2/data/ob_base_vm.npz, gg2/data/ob6_batt.npz,
          gg2/data/nwp_solar_fc_capnorm.npz
ASCII only. ~21 min/arm, three arms ~65 min (measured in TASK 4).

-------------------------------------------------------------------------------
CONTROLLER SPEC SHEET (every number is recorded with these fields)
-------------------------------------------------------------------------------
  feeder           : mv_oberrhein, component 0, 108 x 20 kV buses
  k*               : 2.859375        (TASK 1c, thermal hosting criterion)
  battery          : bus 147, P_MAX 8.00 MW / CAP 16.00 MWh, RTE 0.90
                     (TASK 3, p95 of the required charging power; 2-hour ratio
                     as in case33bw)
  wind             : NONE (decision K1 - this grid has no wind plant)
  voltage window   : VM_BASE[i] + LDF difference (FROM THE FORECAST)   [fair]
  LDF              : root 319, lines only, open switches removed
                     (TASK 2; difference accuracy is 1.54x that of case33bw,
                     threshold 2.0)
  safe layer       : SafeLayerOB, mask = the 20 kV buses of component 0
                     (TASK 3 C1: with the mask off, IDENTICAL to the original)
  w_econ           : 1.0             (same as case33bw)
  H                : 48
  MARGIN           : 0.005
  slice            : M.SPLIT : M.N-1  (2429 steps, SAME as case33bw)

The ONLY VARIABLE is the solar forecast. The dmult and price forecasts are
SHARED BY ALL THREE ARMS.

-------------------------------------------------------------------------------
PRE-REGISTRATION (written BEFORE the run)
-------------------------------------------------------------------------------
LEVELS ALONE ARE NOT A VERDICT. This script only produces the trajectories.
The difference will be reported in a SEPARATE script using a PAIRED daily-block
bootstrap (48-step days, 10000 replicates, seed 12345, 95% percentile
interval) - the SAME method used for case33bw.

Sentences to be written:
  if the interval EXCLUDES zero :
    "The effect is observed on the second feeder as well: violations drop from
     X.XX% to Y.YY%, a difference of Z.ZZ points, 95% CI [a, b]."
  if the interval INCLUDES zero :
    "On the second feeder the difference goes the same way but is NOT RESOLVED:
     Z.ZZ points, 95% CI [a, b]. It CANNOT be claimed that the effect also
     exists on the second feeder."
In either case k*, the battery size, the controller settings and the
measurement window WILL NOT BE CHANGED BASED ON THE RESULT.

The same interval is given for the profit difference; "profit increased" is
NEVER written (it was not written for case33bw either).

ANCHORS (at the START of each arm, before the run begins):
  A7  k*, P_MAX* and CAP* in the npz files must match those in this script.
  A8  at the first step, with battery p=0, the Vmax seen by the safe layer must
      equal the maximum of VM_BASE[T0] to within 1e-9 (the run depends on
      VM_BASE).
  A9  the shape of the solar forecast matrix must equal that of M.solar_fc.

-------------------------------------------------------------------------------
PLACEBO ARM (--arm placebo)  -- PRE-REGISTRATION, written BEFORE the run
-------------------------------------------------------------------------------
The NWP matrix is shifted 7 days backwards: same statistical shape, but its
CONTENT belongs to the wrong day. In case33bw this arm came out MEASURABLY
WORSE than the real NWP (+1.42 points, 95% CI [+0.33, +2.67]).

  if the placebo - NWP interval EXCLUDES zero:
    "On the second feeder too the effect comes from the CONTENT of the
     forecast: shifting the matrix by seven days worsens the difference by
     X.XX points, 95% CI [a, b]."
  if it INCLUDES zero:
    "On the second feeder the placebo does not separate from the real forecast;
     the content attribution COULD NOT BE PROVEN for this feeder."
No setting will be changed based on the result.

-------------------------------------------------------------------------------
VIOLATION DECOMPOSITION (NOT a tautology)
-------------------------------------------------------------------------------
"X% of the violations occur at saturation" is a TAUTOLOGY: when the safe layer
saturates it sets p_safe = p_hi, so the charging headroom is zero BY
DEFINITION (ROUND v8 F3).
Instead, each violation step is compared against the "required charging power"
measured in TASK 3 (gg2/data/ob6_batt.npz):
   req = inf        -> SPATIAL: no charging power at the storage bus brings the
                       maximum bus voltage inside the band the controller
                       targets while keeping the minimum above its limit. The
                       forecast CANNOT CHANGE this.
   req > p_hi       -> SOC/POWER: the charging capacity available at that
                       moment was insufficient. The forecast CAN AFFECT this
                       (it manages SOC in advance).
   req <= p_hi      -> UNEXPECTED: the safe layer should have been able to fix
                       it. If any occur they are reported, and no explanation
                       is FABRICATED.
This decomposition shows how large the violation mass is that the forecast
could affect. On case33bw the equivalent split was not measured independently
of the arms, so it is reported for this feeder only.
-------------------------------------------------------------------------------
"""

import argparse
import os
import sys
import time
from pathlib import Path

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
ROOT_OB = 319
MV_KV = 20.0
W_ECON = 1.0
H = 48
CKPT_EVERY = 300
SHIFT_DAYS = 7          # placebo shift (same as case33bw p7)

NPZ_VM = os.path.join(_HERE, "data", "ob_base_vm.npz")
NPZ_BT = os.path.join(_HERE, "data", "ob6_batt.npz")
NPZ_NWP = os.path.join(_HERE, "data", "nwp_solar_fc_capnorm.npz")
OUT_DIR = os.path.join(_HERE, "data")


def banner(t):
    print("")
    print("=" * 78)
    print(t)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["naive", "nwp", "oracle", "placebo"],
                    default=None,
                    help="run a single arm (default: naive/nwp/oracle in "
                         "order; 'placebo' must be requested SEPARATELY)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--nwp", default=NPZ_NWP)
    a = ap.parse_args()

    banner("0. SPEC SHEET")
    print("pandapower %s | numpy %s | python %s"
          % (pp.__version__, np.__version__, sys.version.split()[0]))
    try:
        import numba  # noqa: F401
        use_numba = True
    except Exception:
        use_numba = False
    import cvxpy as cp
    print("cvxpy %s | numba=%s" % (cp.__version__, use_numba))

    import mpc_ob as MO
    import mpc_econ_voltage_pareto as M
    from safe_layer_ob import SafeLayerOB

    T0, T1 = int(M.SPLIT), int(M.N) - 1
    n = T1 - T0
    solar_norm = np.asarray(M.solar, dtype=float) / float(M.SP)
    dmult = np.asarray(M.dmult, dtype=float)
    dmult_fc = np.asarray(M.dmult_fc, dtype=float)
    K = M.solar_fc.shape[1]

    Z = np.load(NPZ_VM, allow_pickle=True)
    VM = Z["vm"]
    MV = [int(b) for b in Z["mv_buses"]]
    RCOL = Z["Rcol"]
    S_BASE = float(Z["S_base"])
    ZB = np.load(NPZ_BT, allow_pickle=True)
    P_MAX = float(ZB["p_max_star"])
    CAP = float(ZB["cap_star"])
    REQ_BY_STEP = dict(zip([int(x) for x in ZB["viol_idx"]],
                           [float(x) for x in ZB["req_mw"]]))
    ETA, dt = M.ETA, M.dt
    SOC_T = CAP / 2.0

    print("")
    print("feeder mv_oberrhein | k*=%.6f | %d MV buses | battery bus %d"
          % (KSTAR, len(MV), BB))
    print("P_MAX %.2f MW | CAP %.2f MWh | ETA %.5f | dt %.2f | w_econ %.1f | H %d"
          % (P_MAX, CAP, ETA, dt, W_ECON, H))
    print("slice %d:%d (%d steps) | LO %.2f HI %.2f MARGIN %.3f"
          % (T0, T1, n, M.LO, M.HI, M.MARGIN))

    A7 = bool(abs(float(Z["kstar"]) - KSTAR) < 1e-12
              and abs(float(ZB["kstar"]) - KSTAR) < 1e-12)
    print("A7 npz k* consistency -> %s" % ("PASS" if A7 else "FAIL"))
    if not A7:
        sys.exit(2)

    # ---- grid + LDF + safe layer ----
    NET = pn.mv_oberrhein()
    NET.sgen["scaling"] = 1.0
    P0 = NET.sgen.p_mw.values.copy()
    Q0 = NET.sgen.q_mvar.values.copy()
    LS0 = NET.load.scaling.values.copy()
    mg = top.create_nxgraph(NET, respect_switches=True)
    cc = [sorted(int(b) for b in x) for x in top.connected_components(mg)]
    cc.sort(key=len, reverse=True)
    C0 = set(b for b in cc[0] if bool(NET.bus.at[b, "in_service"]))
    OPEN_L = set(NET.switch[(NET.switch.et == "l")
                            & (~NET.switch.closed.astype(bool))].element.astype(int))
    ROB, XOB, _ = MO.build_rx(NET, ROOT_OB, MV, MV_KV ** 2 / S_BASE,
                              open_lines=OPEN_L, restrict=C0)
    POS = {b: i for i, b in enumerate(MV)}
    PL = np.zeros(len(MV))
    QL = np.zeros(len(MV))
    for li, row in NET.load.iterrows():
        b = int(row.bus)
        if b in POS:
            PL[POS[b]] += float(row.p_mw) * float(row.scaling)
            QL[POS[b]] += float(row.q_mvar) * float(row.scaling)
    SG_POS = np.array([POS[int(b)] for b in NET.sgen.bus.values if int(b) in POS])
    SG_MASK = np.array([int(b) in POS for b in NET.sgen.bus.values])
    SG_P0 = P0[SG_MASK]

    ST = pp.create_storage(NET, bus=BB, p_mw=0.0, max_e_mwh=CAP, name="batt")
    RPKW = dict(algorithm="nr", init="auto", calculate_voltage_angles=True,
                max_iteration=100, numba=use_numba)
    SL = SafeLayerOB(NET, ST, RCOL, S_BASE, buses=MV, runpp_kwargs=RPKW)

    def ldf(s_norm, dm):
        P = PL * dm
        Q = QL * dm
        np.add.at(P, SG_POS, -KSTAR * SG_P0 * s_norm)
        return MO.ldf_v(ROB, XOB, P, Q, S_BASE)

    def set_grid(i, p_batt):
        NET.sgen["p_mw"] = KSTAR * P0 * float(solar_norm[i])
        NET.sgen["q_mvar"] = Q0
        NET.load["scaling"] = LS0 * float(dmult[i])
        NET.storage.at[ST, "p_mw"] = float(p_batt)

    def window(i, Hh, sfc):
        v_now = ldf(float(solar_norm[i]), float(dmult[i]))
        vmb = np.empty((Hh, len(MV)))
        vmb[0] = VM[i]
        for t in range(1, Hh):
            kk = min(t - 1, sfc.shape[1] - 1)
            vmb[t] = VM[i] + (ldf(float(sfc[i, kk]), float(dmult_fc[i, kk])) - v_now)
        pr = np.empty(Hh)
        pr[0] = M.price[i]
        for t in range(1, Hh):
            kk = min(t - 1, M.price_fc.shape[1] - 1)
            pr[t] = M.price_fc[i, kk]
        return vmb, pr

    # ---- A8: does the safe layer depend on VM_BASE ----
    set_grid(T0, 0.0)
    r0 = SL._vac(0.0)
    d8 = abs(r0[1] - float(VM[T0].max())) if r0 else np.inf
    A8 = bool(d8 < 1e-9)
    print("A8 safe layer <-> VM_BASE: |diff| = %.3e -> %s"
          % (d8, "PASS" if A8 else "FAIL"))
    if not A8:
        sys.exit(3)

    # ---- solar forecast matrices (normalized from the case33bw MW scale) ----
    def get_fc(arm):
        if arm == "naive":
            return np.asarray(M.solar_fc, dtype=float) / float(M.SP), "daily-naive"
        if arm == "oracle":
            idx = np.minimum(np.arange(M.N)[:, None] + np.arange(1, K + 1)[None, :],
                             M.N - 1)
            return np.asarray(M.solar, dtype=float)[idx] / float(M.SP), "solar-ORACLE"
        p = Path(a.nwp)
        if not p.exists():
            print("[ERROR] not found: %s" % p)
            sys.exit(4)
        z = np.load(p)
        sf = np.asarray(z["solar"], dtype=float) / float(M.SP)
        if arm == "nwp":
            return sf, ("NWP cap-norm (%s, alpha=%.3f)"
                        % (z["model"], float(z["a_mean"])))
        # PLACEBO: the same matrix is read SHIFT days earlier. Same statistical
        # shape, but its CONTENT belongs to the wrong day. In case33bw the p7
        # arm came out measurably WORSE than the real NWP (+1.42 [0.33, 2.67]).
        sh = SHIFT_DAYS * 48
        idx = np.maximum(np.arange(sf.shape[0]) - sh, 0)
        return sf[idx], "PLACEBO: NWP matrix shifted back by %d days" % SHIFT_DAYS

    arms = [a.arm] if a.arm else ["naive", "nwp", "oracle"]
    summary = []
    SPLIT = {}

    for arm in arms:
        sfc, tag = get_fc(arm)
        A9 = bool(sfc.shape == M.solar_fc.shape)
        banner("ARM: %s  (%s)" % (arm.upper(), tag))
        print("A9 forecast matrix shape %s (exp %s) -> %s"
              % (str(sfc.shape), str(M.solar_fc.shape), "PASS" if A9 else "FAIL"))
        if not A9:
            sys.exit(5)

        out = Path(os.path.join(OUT_DIR, "ob8_%s.npz" % arm))
        mn = np.zeros(n)
        mx = np.zeros(n)
        P = np.zeros(n)
        SOC = np.zeros(n)
        PHI = np.zeros(n)
        soc = CAP / 2.0
        j0, nsat = 0, 0

        if out.exists():
            if not a.resume:
                print("output already exists: %s  -> this arm SKIPPED "
                      "(continue with --resume)" % out)
                c = np.load(out)
                if int(c["done"]) >= n:
                    mn, mx, P, SOC = c["mn"], c["mx"], c["p"], c["soc"]
                    v = ((mn < M.LO) | (mx > M.HI)).mean() * 100
                    kar = float(-(M.price[T0:T1] * P).sum() * dt)
                    summary.append((arm, v, kar, tag))
                continue
            c = np.load(out)
            done = int(c["done"])
            if done >= n:
                print("checkpoint already complete -> summary only")
                mn, mx, P, SOC = c["mn"], c["mx"], c["p"], c["soc"]
                if "p_hi" in c:
                    PHI = c["p_hi"]
                j0 = n
            else:
                mn[:done], mx[:done] = c["mn"][:done], c["mx"][:done]
                P[:done], SOC[:done] = c["p"][:done], c["soc"][:done]
                if "p_hi" in c:
                    PHI[:done] = c["p_hi"][:done]
                soc = float(SOC[done - 1])
                j0 = done
                print("RESUMING: %d/%d, soc=%.3f" % (done, n, soc))

        t0 = time.time()
        for j in range(j0, n):
            i = T0 + j
            Hh = min(H, M.N - i)
            vmb, pr = window(i, Hh, sfc)
            p_plan, _ = MO.solve_window(vmb, pr, soc, W_ECON, RCOL, S_BASE,
                                        P_MAX, CAP, ETA, dt, M.HI, M.LO,
                                        M.MARGIN, SOC_T)
            p_plan = float(np.clip(p_plan, -P_MAX, P_MAX))
            p_lo = max(-P_MAX, -soc * ETA / dt)
            p_hi = min(P_MAX, (CAP - soc) / (ETA * dt))
            p_plan = float(np.clip(p_plan, p_lo, p_hi))
            set_grid(i, 0.0)
            p_safe, sinfo = SL.project(p_plan, p_lo, p_hi)
            if sinfo["status"] == "saturated":
                nsat += 1
            mn[j], mx[j], P[j] = sinfo["Vmin"], sinfo["Vmax"], p_safe
            PHI[j] = p_hi
            soc = min(max(soc + (ETA * p_safe * dt if p_safe >= 0
                                 else p_safe * dt / ETA), 0.0), CAP)
            SOC[j] = soc
            if (j + 1) % CKPT_EVERY == 0:
                el = max(time.time() - t0, 1e-9)
                rate = (j + 1 - j0) / el
                print("  %d/%d  (%.0f steps/min, ~%.1f min left)"
                      % (j + 1, n, rate * 60, (n - j - 1) / rate / 60), flush=True)
                np.savez(out, mn=mn, mx=mx, p=P, soc=SOC, p_hi=PHI,
                         start=T0, done=j + 1, kstar=KSTAR, p_max=P_MAX,
                         cap=CAP, arm=arm)
        np.savez(out, mn=mn, mx=mx, p=P, soc=SOC, p_hi=PHI, start=T0, done=n,
                 kstar=KSTAR, p_max=P_MAX, cap=CAP, arm=arm)

        viol = (mn < M.LO) | (mx > M.HI)
        v = float(viol.mean() * 100)
        kar = float(-(M.price[T0:T1] * P).sum() * dt)
        lowv = int((mn < M.LO).sum())
        print("")
        print("  violations %.2f%%  (%d steps; low voltage %d) | profit %.1f EUR"
              % (v, int(viol.sum()), lowv, kar))
        n_sp, n_soc, n_unx = 0, 0, 0
        for jj in np.where(viol)[0]:
            rq = REQ_BY_STEP.get(int(T0 + jj), None)
            if rq is None or not np.isfinite(rq):
                n_sp += 1
            elif rq > PHI[jj] + 1e-9:
                n_soc += 1
            else:
                n_unx += 1
        nv = max(int(viol.sum()), 1)
        print("  saturated %d" % nsat)
        print("  violation decomposition (not a tautology, vs TASK 3 measurement):")
        print("    SPATIAL   (unfixable at any battery size) : %4d  (%.1f%%)"
              % (n_sp, 100.0 * n_sp / nv))
        print("    SOC/POWER (capacity insufficient at t)    : %4d  (%.1f%%)"
              % (n_soc, 100.0 * n_soc / nv))
        print("    UNEXPECTED                                : %4d  (%.1f%%)"
              % (n_unx, 100.0 * n_unx / nv))
        print("    available charging power p_hi at violation steps: "
              "mean %.2f / P_MAX %.2f MW"
              % (float(PHI[viol].mean()) if nv else float("nan"), P_MAX))
        print("  SOC mean %.2f MWh (%.0f%% full) | written: %s"
              % (SOC.mean(), 100 * SOC.mean() / CAP, out))
        summary.append((arm, v, kar, tag))
        SPLIT[arm] = (n_sp, n_soc, n_unx)

    banner("SUMMARY - mv_oberrhein")
    print("uncontrolled (no battery, TASK 2): 15.11%")
    print("")
    print("%-8s %-9s %-11s %-9s %-9s %s"
          % ("arm", "viol%", "profit EUR", "spatial", "SOC/pow", "solar forecast"))
    for arm, v, kar, tag in summary:
        sp = SPLIT.get(arm, (0, 0, 0))
        print("%-8s %-9.2f %-11.1f %-9d %-9d %s" % (arm, v, kar, sp[0], sp[1], tag))
    if len(summary) >= 2:
        d = dict((s[0], s[1]) for s in summary)
        if "naive" in d and "nwp" in d:
            print("")
            print("naive - NWP = %+.2f points  (LEVEL DIFFERENCE - NOT A VERDICT)"
                  % (d["naive"] - d["nwp"]))
            print("The verdict is given only by the paired daily-block bootstrap.")
    print("")
    print("DONE.")


if __name__ == "__main__":
    main()
