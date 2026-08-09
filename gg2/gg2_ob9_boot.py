#!/usr/bin/env python3
"""
GG-2 / ROUND v8 SECTION 8 - TASK 6
PAIRED DAILY-BLOCK BOOTSTRAP - the mv_oberrhein arms.

Run (from the GG-1 ROOT folder):
    python gg2/gg2_ob9_boot.py 2>&1 | tee ob9_boot.log

Requires: gg2/data/ob8_{naive,nwp,oracle}.npz
          and, for the ANCHOR, the case33bw trajectories
          (default gg2/nwp_fair_traj_*.npz)
ASCII only. ~1 min.

-------------------------------------------------------------------------------
METHOD (the SAME one used for case33bw)
-------------------------------------------------------------------------------
  day          = 48 steps
  used         = the first 50 full days (2400 steps; an arm has 2429 steps,
                 the remaining 29 are dropped)
  resampling   : 50 days, WITH REPLACEMENT, 10000 replicates, seed 12345
  PAIRING      : the same day draws are applied to ALL arms (a single `draws`
                 array)
  reporting    : mean, SD, 95% PERCENTILE interval, skewness, P(diff <= 0)

A percentile interval is reported, NOT a symmetric error bar; for case33bw the
distribution was right-skewed (ROUND v8 F2/3.1).

-------------------------------------------------------------------------------
PRE-REGISTRATION (written BEFORE the run; same as TASK 5, unchanged)
-------------------------------------------------------------------------------
  if the interval EXCLUDES zero:
    "The effect is observed on the second feeder as well: violations drop from
     X.XX% to Y.YY%, a difference of Z.ZZ points, 95% CI [a, b]."
  if the interval INCLUDES zero:
    "On the second feeder the difference goes the same way but is NOT RESOLVED:
     Z.ZZ points, 95% CI [a, b]. It CANNOT be claimed that the effect also
     exists on the second feeder."
  An interval is given for profit; "profit increased" is NEVER written.
  k*, the battery, the controller and the window WILL NOT BE CHANGED based on
  the result.

-------------------------------------------------------------------------------
ANCHOR B1 - THE MOST IMPORTANT ONE
-------------------------------------------------------------------------------
The same bootstrap code is FIRST applied to the case33bw trajectories and must
reproduce the headline figure from ROUND v8 3.1:
      naive - NWP = +1.54 points
The point estimate (over 50 days, INVOLVES NO RNG) must hold to within 0.01
points. Because the SD and the interval endpoints depend on the RNG variant,
they are checked with a LOOSE tolerance (SD 0.72 +/- 0.08; upper/lower endpoint
within 0.20) and the exact values are printed.
If B1 fails, NO VERDICT IS GIVEN for mv_oberrhein.
-------------------------------------------------------------------------------
"""

import glob
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

DAY = 48
NDAY = 50
NBOOT = 10000
SEED = 12345

ANCH_DIFF = 1.54
ANCH_SD = 0.72
ANCH_LO, ANCH_HI = 0.29, 3.12
TOL_DIFF, TOL_SD, TOL_CI = 0.01, 0.08, 0.20

OB = {"naive": os.path.join(_HERE, "data", "ob8_naive.npz"),
      "nwp": os.path.join(_HERE, "data", "ob8_nwp.npz"),
      "oracle": os.path.join(_HERE, "data", "ob8_oracle.npz")}
C33 = {"naive": os.path.join(_HERE, "nwp_fair_traj_naive.npz"),
       "nwp": os.path.join(_HERE, "nwp_fair_traj_nwp.npz"),
       "oracle": os.path.join(_HERE, "nwp_fair_traj_oracle.npz")}


def banner(t):
    print("")
    print("=" * 78)
    print(t)
    print("=" * 78)


banner("0. SPEC SHEET")
print("numpy %s | python %s" % (np.__version__, sys.version.split()[0]))
print("day %d steps | %d days | %d replicates | seed %d | RNG numpy default_rng"
      % (DAY, NDAY, NBOOT, SEED))

import mpc_econ_voltage_pareto as M  # noqa: E402

T0 = int(M.SPLIT)
LO, HI, dt = M.LO, M.HI, M.dt
NUSE = DAY * NDAY
PRICE = np.asarray(M.price, dtype=float)[T0:T0 + NUSE]


def load(path):
    z = np.load(path)
    mn, mx, p = z["mn"], z["mx"], z["p"]
    if len(mn) < NUSE:
        return None
    v = ((mn < LO) | (mx > HI)).astype(float)[:NUSE]
    day_viol = v.reshape(NDAY, DAY).mean(axis=1) * 100.0
    day_prof = (-(PRICE * p[:NUSE]) * dt).reshape(NDAY, DAY).sum(axis=1)
    return day_viol, day_prof


DRAWS = np.random.default_rng(SEED).integers(0, NDAY, size=(NBOOT, NDAY))


def compare(a, b, label, unit):
    """a - b, paired (the same draws)."""
    obs = float(a.mean() - b.mean())
    bs = a[DRAWS].mean(axis=1) - b[DRAWS].mean(axis=1)
    m, sd = float(bs.mean()), float(bs.std(ddof=1))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    sk = float(((bs - m) ** 3).mean() / (sd ** 3)) if sd > 0 else float("nan")
    p0 = float((bs <= 0).mean())
    print("  %-16s observed %+9.2f %-4s | boot mean %+9.2f  SD %7.2f | "
          "%%95 CI [%+8.2f, %+8.2f] | skew %+5.2f | P(<=0) %.3f"
          % (label, obs, unit, m, sd, lo, hi, sk, p0))
    return dict(label=label, obs=obs, mean=m, sd=sd, lo=float(lo), hi=float(hi),
                skew=sk, p0=p0)


# ---------------------------------------------------------------------------
banner("1. ANCHOR B1 - REPRODUCE THE case33bw HEADLINE")
missing = [k for k, v in C33.items() if not os.path.exists(v)]
if missing:
    print("[ERROR] case33bw trajectories not found: %s" % missing)
    print(".npz files under gg2/:")
    for f in sorted(glob.glob(os.path.join(_HERE, "*.npz"))):
        print("   %s" % os.path.basename(f))
    print("")
    print("Send me the correct file names; --  STOP.")
    sys.exit(1)

d33 = {}
for k, v in C33.items():
    r = load(v)
    if r is None:
        print("[ERROR] %s is too short (at least %d steps required)" % (v, NUSE))
        sys.exit(2)
    d33[k] = r
    print("  %-7s level (50 days): violations %.2f%% | profit %.1f EUR"
          % (k, r[0].mean(), r[1].sum()))

print("")
res33 = compare(d33["naive"][0], d33["nwp"][0], "naive - NWP", "pts")
B1 = bool(abs(res33["obs"] - ANCH_DIFF) <= TOL_DIFF
          and abs(res33["sd"] - ANCH_SD) <= TOL_SD
          and abs(res33["lo"] - ANCH_LO) <= TOL_CI
          and abs(res33["hi"] - ANCH_HI) <= TOL_CI)
print("")
print("  ROUND v8 3.1 : +%.2f | SD %.2f | CI [%.2f, %.2f]"
      % (ANCH_DIFF, ANCH_SD, ANCH_LO, ANCH_HI))
print("  B1 -> %s" % ("PASS" if B1 else "FAIL"))
if not B1:
    print("")
    print("The bootstrap code could not reproduce the case33bw headline.")
    print("NO VERDICT IS GIVEN for mv_oberrhein. Paste the output to me.")
    sys.exit(3)

_ = compare(d33["nwp"][0], d33["oracle"][0], "NWP - oracle", "pts")
_ = compare(d33["naive"][1], d33["nwp"][1], "naive - NWP profit", "EUR")


# ---------------------------------------------------------------------------
banner("2. mv_oberrhein - LEVELS")
dob = {}
for k, v in OB.items():
    if not os.path.exists(v):
        print("[ERROR] not found: %s  (run gg2_ob8_arms.py first)" % v)
        sys.exit(4)
    r = load(v)
    if r is None:
        print("[ERROR] %s is incomplete" % v)
        sys.exit(4)
    dob[k] = r
    print("  %-7s 50 days: violations %.2f%% | profit %.1f EUR   (2429 steps"
          " reported separately)" % (k, r[0].mean(), r[1].sum()))

banner("3. mv_oberrhein - VIOLATION DIFFERENCES")
r1 = compare(dob["naive"][0], dob["nwp"][0], "naive - NWP", "pts")
r2 = compare(dob["naive"][0], dob["oracle"][0], "naive - oracle", "pts")
r3 = compare(dob["nwp"][0], dob["oracle"][0], "NWP - oracle", "pts")

banner("4. mv_oberrhein - PROFIT DIFFERENCES")
k1 = compare(dob["naive"][1], dob["nwp"][1], "naive - NWP", "EUR")
k2 = compare(dob["naive"][1], dob["oracle"][1], "naive - oracle", "EUR")
k3 = compare(dob["nwp"][1], dob["oracle"][1], "NWP - oracle", "EUR")


# ---------------------------------------------------------------------------
banner("4b. PLACEBO (if present)")
PB = os.path.join(_HERE, "data", "ob8_placebo.npz")
rp = None
if os.path.exists(PB):
    r = load(PB)
    if r is None:
        print("  ob8_placebo.npz is incomplete - skipped")
    else:
        dob["placebo"] = r
        print("  placebo 50 days: violations %.2f%% | profit %.1f EUR" % (r[0].mean(), r[1].sum()))
        rp = compare(dob["placebo"][0], dob["nwp"][0], "placebo - NWP", "pts")
        _ = compare(dob["placebo"][0], dob["naive"][0], "placebo - naive", "pts")
        print("")
        print("  case33bw p7 reference: +1.42 points, %95 CI [+0.33, +2.67]")
else:
    print("  ob8_placebo.npz MISSING - the placebo arm has not been run.")
    print("  (python gg2/gg2_ob8_arms.py --arm placebo)")


banner("5. VERDICT")
print("PRE-REGISTERED SENTENCE (violations, naive - NWP):")
if r1["lo"] > 0:
    print("  The effect is observed on the second feeder too: violations fall"
          " from %.2f%% to %.2f%%,"
          % (dob["naive"][0].mean(), dob["nwp"][0].mean()))
    print("  a difference of %.2f points, %%95 CI [%.2f, %.2f], P(diff<=0)=%.3f."
          % (r1["obs"], r1["lo"], r1["hi"], r1["p0"]))
elif r1["hi"] < 0:
    print("  OPPOSITE DIRECTION: NWP is WORSE than naive, %.2f points,"
          " %%95 CI [%.2f, %.2f]."
          % (r1["obs"], r1["lo"], r1["hi"]))
    print("  This outcome was NOT ANTICIPATED in the pre-registration."
          " No explanation is FABRICATED.")
else:
    print("  On the second feeder the difference is in the same direction but"
          " is NOT RESOLVED: %.2f points,"
          % r1["obs"])
    print("  %%95 CI [%.2f, %.2f]. It CANNOT be claimed that the effect also"
          % (r1["lo"], r1["hi"]))
    print("  exists on the second feeder.")

print("")
print("PROFIT (naive - NWP):")
if k1["lo"] <= 0 <= k1["hi"]:
    print("  The profit difference was NOT RESOLVED: %.0f EUR, %%95 CI [%.0f, %.0f]."
          % (k1["obs"], k1["lo"], k1["hi"]))
    print("  The data supports neither a gain nor a loss.")
else:
    print("  The profit difference EXCLUDES zero: %.0f EUR, %%95 CI [%.0f, %.0f]."
          % (k1["obs"], k1["lo"], k1["hi"]))
    print("  For case33bw this interval included zero; DIFFERENT behaviour, to be")
    print("  written up separately. The phrase 'profit increased' is still NOT USED.")

print("")
print("GAP TO PERFECT FORESIGHT (NWP - oracle):")
if r3["lo"] <= 0 <= r3["hi"]:
    print("  NOT RESOLVED: %.2f points, %%95 CI [%.2f, %.2f]. The day-ahead"
          % (r3["obs"], r3["lo"], r3["hi"]))
    print("  forecast is NOT measurably worse than perfect solar foresight.")
else:
    print("  RESOLVED: %.2f points, %%95 CI [%.2f, %.2f]."
          % (r3["obs"], r3["lo"], r3["hi"]))

if rp is not None:
    print("")
    print("PLACEBO (content attribution):")
    if rp["lo"] > 0:
        print("  On the second feeder too the effect comes from the CONTENT of")
        print("  the forecast: shifting the matrix by seven days worsens the"
              " difference by %.2f points, %%95 CI" % rp["obs"])
        print("  [%.2f, %.2f]." % (rp["lo"], rp["hi"]))
    else:
        print("  The placebo does NOT SEPARATE from the real forecast: %.2f points,"
              " %%95 CI" % rp["obs"])
        print("  [%.2f, %.2f]. The content attribution COULD NOT BE PROVEN for"
              " this feeder." % (rp["lo"], rp["hi"]))

print("")
print("THE TWO FEEDERS SIDE BY SIDE (violations, naive - NWP):")
print("  case33bw     : %+.2f points, CI [%.2f, %.2f]"
      % (res33["obs"], res33["lo"], res33["hi"]))
print("  mv_oberrhein : %+.2f points, CI [%.2f, %.2f]"
      % (r1["obs"], r1["lo"], r1["hi"]))
print("")
print("LIMIT: the two feeders use the SAME Irish series and the SAME test block.")
print("NOT an independent replication; a different grid on the same days.")
print("")
print("DONE.")
