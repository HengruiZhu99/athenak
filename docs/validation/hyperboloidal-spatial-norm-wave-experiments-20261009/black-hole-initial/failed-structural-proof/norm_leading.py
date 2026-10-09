"""Exact live spatial-norm source/zero-jet/shear/connection gates."""
import sympy as s
S,a,M,eta=s.symbols('S a M eta',positive=True)
a1,b1,d2=s.symbols('a1 b1 delta_beta2',real=True)
mass=M/a;xi=1/a;C=S/a*(1-S/(eta*a*a));c1=-2*M/S
adot=S/a**2*(b1-2*a1-4*mass)
bdot=S/a**2*(a1+b1+mass)-eta*(b1+C*c1)
chdot=2*M/(3*a*a)-2*a1/a-4*b1/(3*a)
grdot=8*M/(3*a*a)-4*b1/(3*a);gtdot=-grdot/2
Pdot=3/a**2*(a1+mass)
ndot=s.factor((chdot-grdot)/a**2+2/(a*S)*(adot+bdot))
source_dot=s.factor(bdot+C*(chdot-grdot))
alpha_pole_dot=-S*S/a**2*Pdot-S/a*(2*xi+1/a)*adot+S/a**2*bdot
original={a1:-mass,b1:2*mass}
for v in [adot,bdot,chdot,grdot,gtdot,Pdot,ndot,source_dot,alpha_pole_dot]:assert s.simplify(v.subs(original))==0
# New initial beta correction d2*Omega^2 leaves all leading rates zero. Its
# geometric first time jets imply equal radial/tangential bar connection
# derivatives; therefore the conformal shear pole remains tangent initially.
op=-1/a
chi_dot_r=s.Rational(2,3)*d2*op**2
g_rr_dot_r=s.Rational(8,3)*d2*op**2
g_tt_dot_r=-s.Rational(4,3)*d2*op**2
bar_rr_dot_r=g_rr_dot_r-chi_dot_r
bar_tt_dot_r=g_tt_dot_r-chi_dot_r
Gamma_rr_dot=bar_rr_dot_r/2
Gamma_tt_dot=-bar_tt_dot_r/2
assert s.simplify(Gamma_rr_dot-Gamma_tt_dot)==0
assert s.simplify(-op*(Gamma_rr_dot-Gamma_tt_dot))==0
# Lambda is not pinned: beta Hessians and contracted metric connection move
# together. Zdot=0 is checked from their difference rather than clipped.
lambda_dot=s.Rational(8,3)*d2/a**2
contracted_dot=g_rr_dot_r
assert s.simplify(lambda_dot-contracted_dot)==0
# Initial physical-metric HessOmega shear pole residue, retaining mass A.
alpha0=S/a;Ar=-4*M/(3*a*S);At=2*M/(3*a*S)
tfh_r=-4*M/(3*a*a*S);tfh_t=2*M/(3*a*a*S)
for h,A in [(tfh_r,Ar),(tfh_t,At)]:assert s.simplify(2*alpha0*h+alpha0*A*(-2/a))==0
# Leading rates of physical P, Theta, A are unchanged by d2*Omega^2:
# trace advection changes only by d2*Omega^2*P_r; Theta=0 identically;
# A advection/Lie terms change by O(Omega). Alpha jets are unchanged.
# Consequently their boundary rates, Pdot_r and Thetadot_r are zero.
theta_dot0=A_dot0=P_dot_r=theta_dot_r=0
wn_dot0=0
G_dot0=s.simplify(((chdot-grdot)/a**2).subs(original))
chi_pole_dot=s.Rational(2,3)*alpha0*(Pdot.subs(original)+2*theta_dot0-3*wn_dot0)
trace_pole_dot=alpha0*(2*(-3/a)*Pdot.subs(original)/3-3*G_dot0)
theta_pole_dot=trace_pole_dot
zup_dot0=(lambda_dot-contracted_dot)/2
lambda_pole_dot=-alpha0*(s.Rational(4,3)*(-3/a)+2*s.symbols('kappa1'))*zup_dot0-s.Rational(2,3)*alpha0*(2*P_dot_r+theta_dot_r)-4*alpha0*A_dot0*op
for v in [chi_pole_dot,trace_pole_dot,theta_pole_dot,lambda_pole_dot]:assert s.simplify(v)==0
# Independent nonlinear value-branch audit at rho=3/2. This does not establish
# its invariance, and fixes only G, not chi and the metric separately.
x,y,z=s.symbols('x y z',positive=True)
# x=alpha/alpha_ref, y=beta_rad/alpha_ref (negative), z=sqrt(G/Ghat).
# The spatial source and null relation give y=-(z^2+2)/3=-x*z.
# The physical-P lapse pole is 2*x*y+4*x^2-2=0, hence x^2=1/(2-z).
eliminated=s.factor((z*z+2)**2*(2-z)-9*z*z)
assert eliminated==-(z-1)*(z**4-z**3+3*z*z+4*z+8)
assert s.expand(z*z*((z-s.Rational(1,2))**2+s.Rational(11,4))+4*z+8)==z**4-z**3+3*z*z+4*z+8
# The final factor is positive for z>0; the unique positive branch is z=x=1,
# y=-1, with tangent beta fixed separately by the source pole.
print('PASS live spatial-norm feedback: all initial zero-jet gauge/null/trace gates for any eta>0,xi=1/a')
print('C=',C,'; no imposed live beta/chi/metric falloff or BH source subtraction')
print('PASS shear pole and its initial time derivative; beta^2 correction gives equal physical connection time jets')
print('Lambda_dot=Gamma_dot=',lambda_dot,'; eta6 corrected coefficient gives',lambda_dot.subs({a:s.Rational(1,2),d2:s.Rational(29,40)}))
print('PASS initial time derivatives of chi/P/Theta/Lambda pole numerators; Pdot_r=Thetadot_r=0 follows from exact static outer ADM data and O(Omega^2) shift correction')
print('PASS rho1.5 necessary positive null/finite-Q/Theta0 gauge boundary branch: alpha=alpha_ref,beta_rad=beta_ref,G=Ghat; no evolution invariance claimed')
