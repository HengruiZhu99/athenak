"""Verify induced physical constraints and an independent Cartesian RHS path."""
import sys,json
import numpy as np
import radial_operator as ro
from radial_operator import s,r,U,alpha as a,chi as ch,K,beta as b,kap,ROOT
sys.path.insert(0,str(ROOT.parents[1]/'discrete-symbol'))
from point_operator import point_rhs

u,h,k,th,A,G,ell,shift=U
q=G-s.diff(h,r)-3*h/r
M=s.diff(A,r)-2*s.diff(k,r)/3-4*s.diff(th,r)/3+3*A/(1+r)+2*K*h/(3*(1+r))+s.Rational(5,3)*K*s.diff(h,r)+2*K*s.diff(u/ch,r)
DR=ch*(s.diff(h,r,2)+5*s.diff(h,r)/r+3*h/r**2)+2*s.diff(u,r,2)+4*s.diff(u,r)/r-2*h*s.diff(ch,r,2)-2*s.diff(ch,r)*s.diff(h,r)-4*h*s.diff(ch,r)/r-5*s.diff(ch,r)*s.diff(u,r)/ch+s.Rational(5,2)*h*s.diff(ch,r)**2/ch+s.Rational(5,2)*u*s.diff(ch,r)**2/ch**2
H=DR+4*K*(A+2*K*h/3+(k+2*th)/3)
constraints=[th,q,H,M]
A2,A1,A0=ro.symbolic_coefficients()
rhs=list(A2*s.Matrix([s.diff(z,r,2) for z in U])+A1*s.Matrix([s.diff(z,r) for z in U])+A0*s.Matrix(U))
sig=kap*a
target=[b*s.diff(th,r)+a*H/2+a*ch*(s.diff(q,r)+2*q/r)/2-2*sig*th,
        b*s.diff(q,r)+2*a*(M+s.diff(th,r))-2*sig*q,
        b*s.diff(H,r)-2*a*ch*s.diff(M,r)+(-4*a*ch/r+a*s.diff(ch,r)-4*ch*s.diff(a,r))*M+2*a*K*H+4*a*ch*K*s.diff(q,r)-4*sig*K*th,
        b*s.diff(M,r)+(s.diff(b,r)+a*K)*M-a*s.diff(H,r)/2-s.diff(a,r)*H-a*s.diff(ch,r)*s.diff(q,r)-(2*ch*s.diff(a,r)+a*s.diff(ch,r))*q/r+2*sig*s.diff(th,r)+2*s.diff(sig,r)*th]
residuals=[]
for name,C,T in zip(('Theta','Q','H','M'),constraints,target):
    dt=sum(s.diff(C,s.diff(U[j],r,d))*s.diff(rhs[j],r,d) for j in range(8) for d in range(3))
    error=s.factor(dt-T)
    residuals.append({'constraint':name,'exact_zero':error==0,'difference':str(error)})
    print(name,error,flush=True)

# Independent NumPy point_operator transcription, with analytic spherical
# jets at each chosen radius. The symbolic implementation does not call it.
v,d,dd=ro.state_jets();jets=[s.diff(x,r,degree) for degree in range(3) for x in U]
symbols=s.symbols('p0:24');mapping=dict(zip(jets,symbols))
def compile_dual_array(arr):
    shape=arr.shape
    bg=s.lambdify(r,[z.b for z in arr.flat],modules='numpy',cse=True)
    pert=s.lambdify((r,*symbols),[z.p.xreplace(mapping) for z in arr.flat],modules='numpy',cse=True)
    return lambda rr,p: np.array(bg(rr),float).reshape(shape)+1j*1e-20*np.array(pert(rr,*p),float).reshape(shape)
vf,df,ddf=[compile_dual_array(z) for z in (v,d,dd)]
rng=np.random.default_rng(85310);checks=[]
for rr in (.2,.37,.75,1.,2.,4.,8.):
    c2,c1,c0=ro.coefficients(rr)
    for n in range(5):
        p=rng.normal(size=24);V=vf(rr,p);D=df(rr,p);DD=ddf(rr,p)
        adv=np.einsum('a,az->z',V[19:22],D)
        fout=point_rhs(V,D,DD,adv).imag/1e-20
        direct=fout[[0,1,7,17,8,14,18,19]]
        direct[4]=(2*fout[8]-fout[11]-fout[13])/3
        direct[6]=float(b.subs(r,rr))*p[8+6]-2*float(a.subs(r,rr))*p[2]
        direct[7]=float(b.subs(r,rr))*p[8+7]+2*p[5]-2*p[7]
        expected=c0@p[:8]+c1@p[8:16]+c2@p[16:]
        checks.append({'r':rr,'absolute_error':float(np.max(abs(direct-expected))),
                       'relative_error':float(np.linalg.norm(direct-expected)/np.linalg.norm(expected))})
out={'state_order':list(ro.names),'scope':'Volume only; exact continuum radial jets, current Z4c original damping, adapted G2 gauge. No boundary closure or spectral claim.',
     'constraint_intertwining':residuals,'independent_cartesian_point_rhs':checks,
     'max_point_absolute_error':max(z['absolute_error'] for z in checks),
     'max_point_relative_error':max(z['relative_error'] for z in checks)}
out['passed']=all(z['exact_zero'] for z in residuals) and out['max_point_relative_error']<1e-12
(ROOT/'validation.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='independent_cartesian_point_rhs'},indent=2))
if not out['passed']:sys.exit(1)
