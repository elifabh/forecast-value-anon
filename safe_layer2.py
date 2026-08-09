# safe_layer2.py -- IKI-BATARYA gercek-AC dogrulamali guvenlik katmani (bus17=A, bus14=B).
# Konvansiyon: p>0=sarj (V duser). Iki bataryanin onerdigi (pA,pB) gercek sebekede ihlal
# yapiyorsa, ikisini ORANTILI olcekleyerek/projekteyerek en kucuk-mudahaleli guvenli noktayi bulur.
# Tek-batarya safe_layer.py ile ayni felsefe; fark: 2D aksiyon -> orantili olcek + AC tarama.
import numpy as np
import pandapower as pp
from pandapower.auxiliary import LoadflowNotConverged

V_LO, V_HI, MARGIN = 0.95, 1.05, 0.005

class SafeLayer2:
    def __init__(self, net, idxA, idxB):
        self.net=net; self.iA=idxA; self.iB=idxB; self.last_runpp=0
    def _vac(self,pA,pB):
        self.last_runpp+=1
        self.net.storage.at[self.iA,"p_mw"]=float(pA)
        self.net.storage.at[self.iB,"p_mw"]=float(pB)
        try:
            pp.runpp(self.net)
            return self.net.res_bus.vm_pu.min(), self.net.res_bus.vm_pu.max()
        except LoadflowNotConverged:
            return None
    def _ok(self,r): return (r is not None) and (r[0]>=V_LO) and (r[1]<=V_HI)

    def project(self, pA, pB, loA, hiA, loB, hiB):
        self.last_runpp=0
        pA=float(np.clip(pA,loA,hiA)); pB=float(np.clip(pB,loB,hiB))
        r=self._vac(pA,pB)
        if self._ok(r):
            return pA,pB,dict(status="agent_ok",Vmin=r[0],Vmax=r[1],runpp=self.last_runpp)

        over=(r is None) or (r[1]>V_HI); under=(r is not None) and (r[0]<V_LO)
        # overvoltage -> daha cok SARJ gerek (p arttir, +yon); under -> daha cok DESARJ (p azalt)
        if over and not under: tA,tB=hiA,hiB
        elif under and not over: tA,tB=loA,loB
        else: tA,tB=hiA,hiB   # karisik: once over hedefi

        # orantili projeksiyon: pa..target arasi en kucuk a, ihlali kaldiran
        best=None
        for a in np.linspace(0,1,13):
            qA=pA+a*(tA-pA); qB=pB+a*(tB-pB)
            rr=self._vac(qA,qB)
            if rr is None: continue
            viol=max(0,rr[1]-V_HI)+max(0,V_LO-rr[0])
            if best is None or viol<best[2]:
                best=(qA,qB,viol,rr)
            if viol<=0: break
        qA,qB,viol,rr=best
        st = "agent_ok" if viol<=0 and a==0 else ("projected_exact" if viol<=0 else "saturated")
        return qA,qB,dict(status=st,Vmin=rr[0],Vmax=rr[1],
                          residual_pu=round(viol,4),runpp=self.last_runpp)