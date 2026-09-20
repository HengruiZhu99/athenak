"""Constraint-satisfying mass/spatial-gauge variations, without imposing BCs."""
import json
import radial_operator as ro
from radial_operator import s,r,U,ROOT,alpha as a,chi as ch,K,beta as b

u,h,k,th,A,G,ell,shift=U
q=G-s.diff(h,r)-3*h/r
m=s.diff(A,r)-2*s.diff(k,r)/3-4*s.diff(th,r)/3+3*A/(1+r)+2*K*h/(3*(1+r))+s.Rational(5,3)*K*s.diff(h,r)+2*K*s.diff(u/ch,r)
dr=ch*(s.diff(h,r,2)+5*s.diff(h,r)/r+3*h/r**2)+2*s.diff(u,r,2)+4*s.diff(u,r)/r-2*h*s.diff(ch,r,2)-2*s.diff(ch,r)*s.diff(h,r)-4*h*s.diff(ch,r)/r-5*s.diff(ch,r)*s.diff(u,r)/ch+s.Rational(5,2)*h*s.diff(ch,r)**2/ch+s.Rational(5,2)*u*s.diff(ch,r)**2/ch**2
ham=dr+4*K*(A+2*K*h/3+(k+2*th)/3)
c1=a*th+ch*G/2+s.diff(u,r)
c2=4*k/(3*a)+2*th/(3*a)-2*A/a-G+s.diff(h,r)
mass=s.symbols('mass',positive=True)
mass_family=[(r/(r+mass))**2,0,mass/(r+mass)**2,0,
             -4*mass/(3*(r+mass)**2),0,r/(r+mass),mass*r/(r+mass)**2]
dm=[s.diff(x,mass).subs(mass,1) for x in map(s.sympify,mass_family)]
xi=s.Function('xi')(r)
hg=4*(s.diff(xi,r)-xi/r)/3
gauge=[xi*s.diff(ch,r)-2*ch*(s.diff(xi,r)+2*xi/r)/3,hg,xi*s.diff(K,r),0,
       -4*xi*s.diff(K,r)/3-2*K*hg/3,s.diff(hg,r)+3*hg/r,
       xi*s.diff(a,r),xi*s.diff(b,r)-b*s.diff(xi,r)]
def evaluate(expr,profile):
    return s.factor(expr.subs(dict(zip(U,profile)),simultaneous=True).doit())
rows=[]
for name,profile in [('mass_variation_along_R0_equals_M',dm),('radial_spatial_coordinate_variation',gauge)]:
    vals={key:evaluate(expr,profile) for key,expr in zip(('Theta','Q','H','M','C1','C2'),(th,q,ham,m,c1,c2))}
    assert all(vals[key]==0 for key in ('Theta','Q','H','M'))
    assert vals['C1']!=0 and vals['C2']!=0
    rows.append({'name':name,'constraints_exact_zero':True,'state':[str(x) for x in profile],
                 'C1':str(vals['C1']),'C2':str(vals['C2'])})
out={'checks':rows,'interpretation':'Volume constraint preservation follows from exact intertwining. These physical directions need not satisfy the separately chosen incoming gauge/boundary data or remain stationary under adapted gauge. Nonzero C1/C2 are not physical constraint violations.'}
(ROOT/'physical-directions.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
