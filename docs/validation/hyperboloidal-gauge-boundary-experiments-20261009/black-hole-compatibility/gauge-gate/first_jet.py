"""Independent scri first-jet gate; fixed ADM mass data, free gauge jets."""
import sympy as s
S,a,M,xi,eta,sigma=s.symbols('S a M xi eta sigma',positive=True)
a1,b1=s.symbols('a1 b1',real=True)
alpha0=S/a;beta0=-S/a;op=-1/a
c1=-2*M/S;p1=4*M/(a*S);Ar=-4*M/(3*a*S)
dbeta=-1/a+b1*op;dalpha=1/a+a1*op;dchi=c1*op
divbeta=dbeta+2*beta0/S
wnref=op;dwn_div=(a1+b1)/S
kdiv=-3/S+p1-3*dwn_div
chi_dot=beta0*dchi-s.Rational(2,3)*divbeta+s.Rational(2,3)*alpha0*kdiv
metric_dot=-2*alpha0*Ar-s.Rational(2,3)*divbeta+2*dbeta
geometry_null_dot=op*op*(chi_dot-metric_dot)
Ra=beta0*(dalpha-1/a)
Sa_div=-alpha0**2*p1-xi*2*alpha0*a1-(alpha0*b1+beta0*a1)*op
alpha_dot=s.factor(Ra+Sa_div)
beta_dot=s.factor(beta0*(dbeta+1/a)+alpha0**2*(dchi/2-(dalpha-1/a)/alpha0))
null_dot=s.factor(geometry_null_dot+2*wnref/alpha0*(op*beta_dot+wnref*alpha_dot))
expected=2/a**3*(2*b1-2*xi*a*a1-4*M/a)
assert s.simplify(null_dot-expected)==0
eta_dot=s.factor(null_dot-2*eta*b1/(a*S))
assert s.simplify(eta_dot-2/a**3*((2-eta*a*a/S)*b1-2*xi*a*a1-4*M/a))==0
n1=s.factor(c1/a**2-2*wnref*dwn_div)
assert s.simplify(n1-2/(a*S)*(a1+b1-M/a))==0
# Independent rederived preferred-source terms, with the actual Minkowski
# outer source held fixed. The regular correction at scri is -3*S/a^2.
f0p_div=3*a/S**2+3*dwn_div/alpha0+((alpha0*b1+beta0*a1)*op+2*xi*alpha0*a1)/alpha0**3
preferred_beta_delta=-3*S/a**2-alpha0**2*beta0*f0p_div
preferred_dot=s.factor(null_dot+2*wnref*op/alpha0*preferred_beta_delta)
assert s.simplify(preferred_dot-4*S/a**2*n1)==0
feedback_dot=s.factor(preferred_dot+2*wnref*op/alpha0*(sigma*alpha0**2/op*n1))
assert s.simplify(feedback_dot-(4-2*sigma)*S/a**2*n1)==0
solve=s.solve([a1+b1-M/a,eta_dot],[a1,b1])
assert s.simplify(solve[a1]+(2+eta*a*a/S)/(2+2*xi*a-eta*a*a/S)*M/a)==0
assert s.simplify(solve[b1]-(4+2*xi*a)/(2+2*xi*a-eta*a*a/S)*M/a)==0
geometric={a1:-M/a,b1:2*M/a}
assert s.simplify(null_dot.subs(geometric)-4*xi*M/a**3)==0
assert s.simplify(eta_dot.subs(geometric)-(4*xi*M/a**3-4*eta*M/(a*a*S)))==0
assert s.simplify(preferred_dot.subs(geometric))==0
assert s.simplify(feedback_dot.subs(geometric))==0
numeric={S:1,a:s.Rational(1,2),M:s.Rational(1,2),xi:s.Rational(3,2),eta:10}
print('PASS independent fixed-ADM scri first-jet derivation')
for label,expr in [('geometry null contribution',geometry_null_dot),('alpha dot',alpha_dot),('beta dot',beta_dot),('baseline Ndot',null_dot),('eta Ndot',eta_dot),('preferred Ndot',preferred_dot),('feedback Ndot',feedback_dot),('N_raw first coefficient',n1)]:
 print(label, '=', expr, '; original geometric BH jets =',s.simplify(expr.subs(geometric)))
print('numerical original Ndot baseline/eta10/preferred:',[s.simplify(x.subs(geometric).subs(numeric)) for x in [null_dot,eta_dot,preferred_dot]])
print('compatible free gauge first jets:',solve)
print('eta10 numerical compatible jets:',{k:s.simplify(v.subs(numeric)) for k,v in solve.items()})
print('trace Q=(P-3*wn)/Omega scri limit:',s.simplify(kdiv))
assert s.simplify(kdiv.subs(b1,M/a-a1)-(-3/S+M/(a*S)))==0
print('a1+b1=M/a keeps trace Q limit -3/S+M/(a*S), separately from N_raw null jets')
