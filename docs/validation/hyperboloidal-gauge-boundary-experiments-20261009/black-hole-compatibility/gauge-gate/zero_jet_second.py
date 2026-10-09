"""Exact rational second-jet gate at S=1,a=M=1/2,xi=2,eta=4."""
import sympy as s
O=s.symbols('Omega',real=True);da2,db2=s.symbols('delta_a2 delta_b2',real=True)
def tr(v,n=2):return s.series(v,O,0,n).removeO().expand()
r=tr(s.sqrt(1-O),3);op=-2*r;L=2-O;b=2*r
m=tr(O/(4*r),3);psi=1+m
N=tr((1-m)/psi,3);F=tr((1-m)/psi**3,3);chi=tr(psi**-4,3)
ahat=L;bhat=-b;alpha=tr(N*L+da2*O*O,3)
beta=tr(-b*F+db2*O*O,3)
ap=tr(s.diff(alpha,O)*op);bp=tr(s.diff(beta,O)*op);cp=tr(s.diff(chi,O)*op)
ahatp=-op;bhatp=-2
P=tr(-2*(3-2*m+m*m)/((1-m)*psi**3),3)
wn=tr(-beta*op/alpha,3);what=tr(-bhat*op/ahat,3)
Qtrace=tr((P-3*wn)/O)
Nraw=tr(chi*op*op-wn*wn,3)
assert Nraw.coeff(O,0)==0 and Nraw.coeff(O,1)==0
assert Qtrace.coeff(O,0)==-2
Ar=tr(-s.Rational(2,3)/(r)*(2-m)/(psi**2*(1-m*m)))
geom=tr(op*op*(beta*cp+2*alpha*chi*(Ar+Qtrace/3)-2*chi*bp))
dadiv=tr((alpha-ahat)/O);dbdiv=tr((beta-bhat)/O);pd=tr((P+6)/O)
adot=tr(beta*ap-bhat*ahatp-s.Rational(3,2)*alpha*tr(s.log(alpha/ahat))
        -alpha*alpha*pd-2*(alpha+ahat)*dadiv-(alpha*dbdiv+bhat*dadiv)*op)
bdot=tr(beta*bp-bhat*bhatp-O*dbdiv+alpha*alpha*chi*(cp/(2*chi)-ap/alpha+ahatp/ahat)-4*dbdiv)
ndot=tr(geom+2*wn/alpha*(op*bdot+wn*adot))
assert ndot.coeff(O,0)==0
assert s.simplify(ndot.coeff(O,1)-(-36+48*db2))==0
assert ndot.subs({da2:0,db2:s.Rational(3,4)})==0
print('PASS exact zero-jet pair second gate; original first jets, initial N_raw=O(Omega^2), Qtrace_scri=-2')
print('dt N_raw =',ndot,'+ O(Omega^2)')
print('delta_a2=0,delta_b2=3/4 cancels this second gate only; no dynamic manifold proof')
