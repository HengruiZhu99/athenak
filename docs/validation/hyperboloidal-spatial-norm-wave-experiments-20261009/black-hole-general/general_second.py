"""Independent exact general-parameter outer ADM/gauge Taylor projection."""
import json
from pathlib import Path
import sympy as s
O=s.symbols('Omega',real=True)
S,a,M,rho=s.symbols('S a M rho',positive=True)
nu,etaR=s.symbols('nu eta_regular',nonnegative=True)
dA,dB=s.symbols('delta_a2 delta_b2',real=True)
eta=rho*S/a**2;C=S/a*(1-1/rho)
def tr(v,n=2):return s.series(v,O,0,n).removeO().expand()
r=S-a*O-a*a*O*O/(2*S)
op=-r/(a*S);L=S/a-O;b=r/a
m=tr(M*O/(2*r),3);psi=1+m
N=tr((1-m)/psi,3);F=tr((1-m)/psi**3,3);chi=tr(psi**-4,3)
alpha=tr(N*L+dA*O*O,3);beta=tr(-b*F+dB*O*O,3)
ahat=L;bhat=-b
ap=tr(s.diff(alpha,O)*op);bp=tr(s.diff(beta,O)*op);cp=tr(s.diff(chi,O)*op)
ahatp=-op;bhatp=-1/a
P=tr(-(3-2*m+m*m)/(a*(1-m)*psi**3),3)
wn=tr(-beta*op/alpha,3)
Qtrace=tr((P-3*wn)/O)
Nraw=tr(chi*op*op-wn*wn,3)
assert s.simplify(Nraw.coeff(O,0))==0 and s.simplify(Nraw.coeff(O,1))==0
assert s.simplify(Qtrace.coeff(O,0)-(-3/S+M/(a*S)))==0
Ar=tr(-2*M/(3*a*r)*(2-m)/(psi**2*(1-m*m)))
geom=tr(op*op*(beta*cp+2*alpha*chi*(Ar+Qtrace/3)-2*chi*bp))
dadiv=tr((alpha-ahat)/O);dbdiv=tr((beta-bhat)/O)
pd=tr((P+3/a)/O);chdiv=tr((chi-1)/O)
adot=tr(beta*ap-bhat*ahatp-nu*alpha*tr(s.log(alpha/ahat))
         -alpha*alpha*pd-(alpha+ahat)*dadiv/a-(alpha*dbdiv+bhat*dadiv)*op)
bdot=tr(beta*bp-bhat*bhatp-etaR*O*dbdiv
         +alpha*alpha*chi*(cp/(2*chi)-ap/alpha+ahatp/ahat)-eta*(dbdiv+C*chdiv))
ndot=tr(geom+2*wn/alpha*(op*bdot+wn*adot))
for v in [adot,bdot,geom,ndot]:assert s.simplify(v.coeff(O,0))==0
n1=s.factor(ndot.coeff(O,1))
base=s.factor(n1.subs({dA:0,dB:0}));ac=s.factor(s.diff(n1,dA));bc=s.factor(s.diff(n1,dB))
assert ac==0
correction=s.factor(-base/bc)
assert s.simplify(n1.subs(dB,correction))==0
special={S:1,a:s.Rational(1,2),M:s.Rational(1,2),rho:s.Rational(3,2),nu:s.Rational(3,2),etaR:1}
assert s.simplify(correction.subs(special)-s.Rational(29,40))==0
ell=s.symbols('length_scale',positive=True)
scaling={S:ell*S,a:ell*a,M:ell*M,nu:nu/ell,etaR:etaR/ell}
assert s.simplify(correction.subs(scaling,simultaneous=True)-correction)==0
assert s.simplify(base.subs(scaling,simultaneous=True)-base/ell**3)==0
assert s.simplify(bc.subs(scaling,simultaneous=True)-bc/ell**3)==0
assert s.simplify(base.subs(rho,4)-2*M*(M-a*a*(2*etaR-nu))/(S*a**4))==0
out={k:str(v) for k,v in [('Omega_coefficient',n1),('original_coefficient',base),('delta_a2_coefficient',ac),('delta_b2_coefficient',bc),('correction',correction),('Nraw_Omega2',s.factor(Nraw.coeff(O,2))),('Qtrace_scri',s.factor(Qtrace.coeff(O,0)))]}
Path(__file__).with_name('general_second.json').write_text(json.dumps(out,indent=2)+'\n')
for k,v in out.items():print(k,'=',v)
print('PASS general exact first/second-null gate, beta correction reduces to 29/40')
print('PASS coefficient is dimensionless and dt N_raw Omega coefficient scales as length^(-3); rho4 obstruction formula exact')
