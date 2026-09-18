#!/usr/bin/env python3
"""Independent symbolic ADM check of the implemented R0=M=1 trumpet.

Spherical components, rather than the Cartesian conformal implementation.
Reference: Dennison & Baumgarte, arXiv:1403.5484, equations 15--20.
Only geometry/constraints are checked: standard 1+log stationarity is NOT claimed.
Requires SymPy; prints exact simplified residuals.
"""
import json
import sympy as s
r=s.symbols('r',positive=True)
R=r+1
psi=s.sqrt(R/r)
phi=s.log(psi)
grr=(R/r)**2
gtt=R**2
alpha=r/R
beta=r/R**2
Krr=-1/r**2
Ktt=s.Integer(1)
K=s.simplify(Krr/grr+2*Ktt/gtt)
kr,kt=Krr/grr,Ktt/gtt
Rrr=-4*s.diff(phi,r,2)-4*s.diff(phi,r)/r
Rtt=-2*r*r*s.diff(phi,r,2)-6*r*s.diff(phi,r)-4*r*r*s.diff(phi,r)**2
Ricci=s.simplify(Rrr/grr+2*Rtt/gtt)
Drr=s.diff(alpha,r,2)-2*s.diff(phi,r)*s.diff(alpha,r)
Dtt=(r+2*r*r*s.diff(phi,r))*s.diff(alpha,r)
checks={
 'Hamiltonian':Ricci+K*K-kr*kr-2*kt*kt,
 'radial_momentum':s.diff(kr,r)+2/R*(kr-kt)-s.diff(K,r),
 'dt_gamma_rr':-2*alpha*Krr+beta*s.diff(grr,r)+2*grr*s.diff(beta,r),
 'dt_gamma_theta_theta':-2*alpha*Ktt+beta*s.diff(gtt,r),
 'dt_K_rr':-Drr+alpha*(Rrr+K*Krr-2*Krr*Krr/grr)+beta*s.diff(Krr,r)+2*Krr*s.diff(beta,r),
 'dt_K_theta_theta':-Dtt+alpha*(Rtt+K*Ktt-2*Ktt*Ktt/gtt)+beta*s.diff(Ktt,r),
}
result={key:str(s.simplify(value)) for key,value in checks.items()}
assert all(value=='0' for value in result.values()),result
print(json.dumps(result,indent=2))
