"""Linear spherical full-state constraint map and incoming boundary identities.

No evolution code; verifies independent warped-product scalar curvature and
covariant radial momentum against the maps used for a future CPBC derivation.
"""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'vacuum-preservation-20260918/python-deps'))
import sympy as s
r=s.symbols('r',positive=True);sigma,omegaT,omegaQ=s.symbols('sigma omega_Theta omega_Q',real=True)
R=1+r;a=r/R;ch=a*a;K=1/R**2;b=r/R**2;c=a*a;v=c-b
u,h,k,th,A,q=[s.Function(z)(r) for z in ('u','h','k','Theta','aTF','q')]
# h=delta gtilde_rr, tangential perturbation=-h/2.
# A is the tracefree radial projection of delta Atilde; delta Atilde_rr=A−2Kh/3.
DR=ch*(s.diff(h,r,2)+5*s.diff(h,r)/r+3*h/r**2)+2*s.diff(u,r,2)+4*s.diff(u,r)/r-2*h*s.diff(ch,r,2)-2*s.diff(ch,r)*s.diff(h,r)-4*h*s.diff(ch,r)/r-5*s.diff(ch,r)*s.diff(u,r)/ch+s.Rational(5,2)*h*s.diff(ch,r)**2/ch+s.Rational(5,2)*u*s.diff(ch,r)**2/ch**2
# Independent physical metric ds²=X dr²+rho² dOmega².
X=1/ch;rho=R;dX=h/ch-u/ch**2;drho=-R*h/4-R*u/(2*ch)
rp=s.diff(rho,r);rpp=s.diff(rho,r,2);Xp=s.diff(X,r)
F=1-rp**2/X-2*rho*rpp/X+rho*rp*Xp/X**2
DF=-2*rp*s.diff(drho,r)/X+rp**2*dX/X**2-2*(drho*rpp+rho*s.diff(drho,r,2))/X+2*rho*rpp*dX/X**2+(drho*rp+rho*s.diff(drho,r))*Xp/X**2+rho*rp*s.diff(dX,r)/X**2-2*rho*rp*Xp*dX/X**3
DR_warp=2*DF/rho**2-4*F*drho/rho**3
assert s.simplify(DR-DR_warp)==0
assert s.simplify(2*F/rho**2-2*K*K)==0
j=A+s.Rational(2,3)*K*h+(k+2*th)/3
H=DR+4*K*j
f=1/r-s.diff(ch,r)/(2*ch)
df=-s.diff(h,r)/4-s.diff(u/ch,r)/2
momentum_direct=s.diff(j,r)-s.diff(k+2*th,r)+f*(3*j-k-2*th)-4*K*df
M=s.diff(A,r)-s.Rational(2,3)*s.diff(k,r)-s.Rational(4,3)*s.diff(th,r)+3*A/R+s.Rational(2,3)*K*h/R+s.Rational(5,3)*K*s.diff(h,r)+2*K*s.diff(u/ch,r)
assert s.simplify(momentum_direct-M)==0
Gamma=q+s.diff(h,r)+3*h/r
C1=a*th+ch*Gamma/2+s.diff(u,r)
C2=4*k/(3*a)+2*th/(3*a)-2*A/a-Gamma+s.diff(h,r)
FT=a*H/2+c*s.diff(th,r)+a*ch*(s.diff(q,r)+2*q/r)/2+(v*omegaT-2*sigma)*th
FQ=2*a*M+2*a*s.diff(th,r)+c*s.diff(q,r)+(v*omegaQ-2*sigma)*q
LT=s.simplify(FT-a*s.diff(C1,r));LQ=s.simplify(FQ+c*s.diff(C2,r))
variables=[u,h,k,th,A,q,s.diff(u,r),s.diff(h,r)]

def coefficients(expr):
 out={str(z):str(s.factor(s.diff(expr,z))) for z in variables}
 assert s.simplify(expr-sum(s.diff(expr,z)*z for z in variables))==0
 return {k:v for k,v in out.items() if v!='0'}
# No second-normal derivative remains in either lower-order lift source.
out={'exact_symbolic_checks':{'Hamiltonian_spatial_curvature_vs_warped_metric':True,'background_Hamiltonian':True,'radial_momentum_covariant_divergence':True,'constraint_incoming_normal_derivative_identities':True},'definitions':{'C1':'alpha Theta + chi Gamma/2 + u_prime','C2':'4k/(3alpha)+2Theta/(3alpha)-2aTF/alpha-Gamma+h_prime','F_Theta':'alpha*C1_prime+L_Theta','F_Q':'-chi*C2_prime+L_Q','outgoing_coordinate_speed':str(v),'falloff_weights':'coordinate p1: omegaT=omegaQ=1/r; physical amplitude: omegaT=1/(r+1), omegaQ=1/r'},'L_Theta_coefficients':coefficients(LT),'L_Q_coefficients':coefficients(LQ)}
p=Path(__file__).parent/'boundary-lift-identities.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
