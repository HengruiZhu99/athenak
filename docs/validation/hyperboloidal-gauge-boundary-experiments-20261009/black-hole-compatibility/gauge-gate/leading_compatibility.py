"""Independent finite-pole boundary-value gates for fixed vacuum BH ADM jets."""
import sympy as s
S,a,M,xi,eta,alpha=s.symbols('S a M xi eta alpha',positive=True)
a1,b1=s.symbols('a1 b1',real=True)
A=S/a;B=-S/a;op=-1/a;mass=M/a
adot=S/a**2*(b1-2*xi*a*a1-4*mass)
bdot=S/a**2*(a1+b1+mass)-eta*b1
chdot=2*M/(3*a*a)-2*a1/a-4*b1/(3*a)
grdot=8*M/(3*a*a)-4*b1/(3*a)
gtdot=-grdot/2
ndot=s.factor(op*op*(chdot-grdot)+2/(a*S)*(adot+bdot))
n1=2/(a*S)*(a1+b1-mass)
# Derive Pdot from all regular/pole pieces of the geometric trace equation.
# Omega*normA and Omega*lapalpha vanish; retain cross, Hessian and advection.
# Theta=Z=0 is this initial vacuum case, not an arbitrary live-field falloff.
ap=(1-a1)/a;chip=2*M/(a*S);Pp=-4*M/(a*a*S)
lapOmega=-3/(a*S)-chip*op/2
trace_pole_div=A*(6/(a*S)-2*M/(a*a*S))
Pdot=s.factor(3*ap*op+A*lapOmega+B*Pp+trace_pole_div)
theta_dot=s.simplify(2*A*lapOmega+trace_pole_div)
assert s.simplify(Pdot-3/a**2*(a1+mass))==0
assert theta_dot==0
assert s.simplify(Pdot+3/(2*a)*(chdot-grdot))==0
trace_dt=s.factor(Pdot-3*(adot+bdot)/S)
assert s.simplify(trace_dt+3*a*s.Rational(1,2)*ndot)==0
alpha_pole_dt=s.factor(-A*A*Pdot-A*(2*xi+1/a)*adot+A/a*bdot)
# Fixed beta, finite trace Q and Theta=0 imply P=-3*B*op/alpha.
# Positive coefficients in the factored lapse pole then pin alpha too.
P=-3*B*op/alpha;Pref=-3/a
alpha_pole=-alpha**2*(P-Pref)-xi*(alpha+A)*(alpha-A)-B*op*(alpha-A)
assert s.simplify(alpha_pole+(alpha-A)*((3*alpha+A)/a+xi*(alpha+A)))==0
c,d=s.symbols('c d',real=True)
gate=s.factor(ndot.subs(a1,mass-b1))
beta_pin_pair={b1:2*mass/c,eta:c*S/a**2,xi:d/a}
paired=s.factor(gate.subs(beta_pin_pair,simultaneous=True))
assert s.simplify(paired-2/a**3*mass*(4*(1+d)/c-6-2*d))==0
required_c=2*(1+d)/(3+d)
assert s.simplify(paired.subs(c,required_c))==0
num={S:1,a:s.Rational(1,2),M:s.Rational(1,2),xi:s.Rational(3,2),eta:10}
for name,pair in [('original',{a1:-1,b1:2}),('null_only',{a1:-s.Rational(9,2),b1:s.Rational(11,2)}),('beta_only',{a1:s.Rational(1,5),b1:s.Rational(4,5)})]:
 print(name,{label:s.simplify(expr.subs(num).subs(pair)) for label,expr in [('alphaDot',adot),('betaDot',bdot),('chiDot',chdot),('g_rrDot',grdot),('g_ttDot',gtdot),('Pdot',Pdot),('Ndot',ndot),('dtSalpha',alpha_pole_dt)]})
assert s.simplify(ndot.subs(num).subs({a1:s.Rational(1,5),b1:s.Rational(4,5)}))==-s.Rational(376,5)
solution={a1:-M/a,b1:2*M/a,xi:1/a,eta:S/a**2}
for expr in [n1,adot,bdot,ndot,alpha_pole_dt,Pdot,chdot,grdot,gtdot]:assert s.simplify(expr.subs(solution,simultaneous=True))==0
print('PASS full leading finite-pole gate: original jets, xi=1/a, eta=S/a^2')
print('paired beta/null condition alone: eta*a^2/S =',required_c,'with d=xi*a; for d>=0, c in [2/3,2)')
print('eta10 overdetermined for fixed mass ADM data retaining N_raw=O(Omega^2)')
print('alpha pole = -(alpha-alphaRef)*[(3alpha+alphaRef)/a+xi(alpha+alphaRef)] on beta-pin/finite-trace vacuum manifold')
