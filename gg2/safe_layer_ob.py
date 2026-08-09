# safe_layer_ob.py -- BUS-MASKED version of SafeLayer (for mv_oberrhein).
#
# WHY: safe_layer.SafeLayer._vac takes res_bus.vm_pu.min()/max() over the whole
# grid. That is fine for case33bw (a single component, 33 buses). In
# mv_oberrhein the grid consists of TWO galvanically separate components; the
# voltages of component 1 (70 buses) would also enter the count. The 110 kV
# slack bus would enter as well.
#
# THE ONLY THING THAT CHANGES: which buses _vac looks at + the runpp call
# arguments. The project/_root/_scan logic is EXACTLY THE SAME as safe_layer.py.
#
# If buses=None is given, the behaviour is identical to safe_layer.SafeLayer
# (the anchor test is performed through this).
#
# Convention (unchanged): p>0 = CHARGE (consumption, LOWERS the voltage),
#                         p<0 = DISCHARGE (RAISES the voltage)

import numpy as np
import pandapower as pp

V_LO, V_HI, MARGIN = 0.95, 1.05, 0.005
TOL = 5e-4   # MW


class SafeLayerOB:
    """Storage conv.: p>0=charge (V drops). Projection verified against a real
    AC solve. buses: indices of the buses where the voltage is MEASURED.
    If None, the whole grid."""

    def __init__(self, net, storage_idx, Rcol, S_base, buses=None,
                 runpp_kwargs=None):
        self.net = net
        self.si = storage_idx
        self.Rcol = Rcol
        self.S_base = S_base
        self.buses = None if buses is None else list(buses)
        self.runpp_kwargs = dict(runpp_kwargs or {})
        self.last_runpp = 0

    def _vac(self, p):
        self.last_runpp += 1
        self.net.storage.at[self.si, "p_mw"] = float(p)
        try:
            pp.runpp(self.net, **self.runpp_kwargs)
        except Exception:
            return None
        v = (self.net.res_bus.vm_pu if self.buses is None
             else self.net.res_bus.vm_pu.loc[self.buses])
        if bool(v.isna().any()):
            return None
        return float(v.min()), float(v.max())

    def _root(self, x0, f0, x1, f1, fval, tol=TOL, itmax=10):
        """Regula falsi: f decreasing, root inside the bracket f(x0)>0>=f(x1)."""
        for _ in range(itmax):
            if abs(x1 - x0) < tol:
                break
            xm = x1 - f1 * (x1 - x0) / (f1 - f0) if f1 != f0 else 0.5 * (x0 + x1)
            xm = min(max(xm, min(x0, x1) + 1e-9), max(x0, x1) - 1e-9)
            fm = fval(xm)
            if fm is None:
                fm = np.inf
            if fm > 0:
                x0, f0 = xm, fm
            else:
                x1, f1 = xm, fm
        return x1

    def project(self, p_agent, p_lo, p_hi):
        self.last_runpp = 0
        Vhi_m, Vlo_m = V_HI - MARGIN, V_LO + MARGIN
        pa = float(np.clip(p_agent, p_lo, p_hi))

        r = self._vac(pa)
        if r and r[0] >= V_LO and r[1] <= V_HI:
            return pa, dict(status="agent_ok", Vmin=r[0], Vmax=r[1],
                            runpp=self.last_runpp)

        over = (r is None) or (r[1] > V_HI)
        under = (r is not None) and (r[0] < V_LO)

        if over and not under:
            rh = self._vac(p_hi)
            if rh is None or rh[1] > Vhi_m:
                if rh is not None and rh[0] >= V_LO:
                    res = round(max(0, rh[1] - V_HI), 4)
                    st = "saturated" if res > 0 else "projected_margin"
                    return p_hi, dict(status=st, Vmin=rh[0], Vmax=rh[1],
                                      residual_pu=res, runpp=self.last_runpp)
                return self._scan(p_lo, p_hi)
            f_pa = (np.inf if r is None else r[1] - Vhi_m)
            a = self._root(pa, f_pa, p_hi, rh[1] - Vhi_m,
                           lambda p: (lambda q: None if q is None else q[1] - Vhi_m)
                           (self._vac(p)))
            p_safe = float(np.clip(pa, a, p_hi))
            rr = self._vac(p_safe)
            if rr and rr[0] >= V_LO and rr[1] <= V_HI:
                return p_safe, dict(status="projected_exact", Vmin=rr[0],
                                    Vmax=rr[1], bound=round(a, 3),
                                    runpp=self.last_runpp)
            return self._scan(p_lo, p_hi)

        if under and not over:
            rl = self._vac(p_lo)
            if rl is None or rl[0] < Vlo_m:
                if rl is not None and rl[1] <= V_HI:
                    res = round(max(0, V_LO - rl[0]), 4)
                    st = "saturated" if res > 0 else "projected_margin"
                    return p_lo, dict(status=st, Vmin=rl[0], Vmax=rl[1],
                                      residual_pu=res, runpp=self.last_runpp)
                return self._scan(p_lo, p_hi)
            f_pa = (-np.inf if r is None else r[0] - Vlo_m)
            b = self._root(pa, -f_pa, p_lo, -(rl[0] - Vlo_m),
                           lambda p: (lambda q: np.inf if q is None else Vlo_m - q[0])
                           (self._vac(p)))
            p_safe = float(np.clip(pa, p_lo, b))
            rr = self._vac(p_safe)
            if rr and rr[0] >= V_LO and rr[1] <= V_HI:
                return p_safe, dict(status="projected_exact", Vmin=rr[0],
                                    Vmax=rr[1], bound=round(b, 3),
                                    runpp=self.last_runpp)
            return self._scan(p_lo, p_hi)

        return self._scan(p_lo, p_hi)

    def _scan(self, p_lo, p_hi):
        best = None
        for p in np.linspace(p_lo, p_hi, 13):
            r = self._vac(p)
            if r is None:
                continue
            viol = max(0, r[1] - V_HI) + max(0, V_LO - r[0])
            if best is None or viol < best[1]:
                best = (p, viol, r)
        if best is None:
            return 0.0, dict(status="saturated", Vmin=np.nan, Vmax=np.nan,
                             residual_pu=np.nan, runpp=self.last_runpp)
        st = "saturated" if best[1] > 0 else "projected_margin"
        return best[0], dict(status=st, Vmin=best[2][0], Vmax=best[2][1],
                             residual_pu=round(best[1], 4), runpp=self.last_runpp)