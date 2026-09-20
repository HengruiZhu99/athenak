"""Symbolic continuum check; no AthenaK evolution or discrete stability claim."""
from pathlib import Path
import sys,json
sys.path.insert(0,'/Users/hz0693/research/TDE/athenak-review/review/vacuum-preservation-20260918/python-deps')
import sympy as s
alpha,L2,C2,G,v=s.symbols('alpha L2 C2 G v', nonzero=True)
S2=s.Rational(4,3)*G
a=alpha*S2/(L2-S2)
b=alpha*S2/(2*(C2-S2))
checks={
 'W_Khat_wave_coefficient':s.factor(a*(L2-S2)-alpha*S2),
 'W_Theta_wave_coefficient':s.factor(b*(C2-S2)-alpha*S2/2),
 'Gamma_Khat_time_coefficient':s.factor(-4*a/(3*v)+4*alpha*v/(3*(L2-S2))).subs(v**2,S2),
 'Gamma_Khat_normal_coefficient':s.factor(-s.Rational(4,3)*(a+alpha)+4*alpha*L2/(3*(L2-S2))),
 'Gamma_Theta_time_coefficient':s.factor(-4*b/(3*v)+2*alpha*v/(3*(C2-S2))).subs(v**2,S2),
 'Gamma_Theta_normal_coefficient':s.factor(-s.Rational(4,3)*b-s.Rational(2,3)*alpha+2*alpha*C2/(3*(C2-S2))),
}
checks={k:s.factor(x.subs(G,s.Rational(3,4)*v**2)) for k,x in checks.items()}
assert all(x==0 for x in checks.values()),checks
out={'scope':'Frozen continuum principal identities; excludes coincident speeds, variable coefficients, and stability inference','checks':{k:str(x) for k,x in checks.items()},'a':str(a),'b':str(b),'eta_Gamma_n':'4*eta/(3*vSL)*div(beta)','eta_Gamma_A':'eta/vST*(Dn beta_A-DA beta_n)'}
p=Path(__file__).with_name('gauge-completion-check.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
