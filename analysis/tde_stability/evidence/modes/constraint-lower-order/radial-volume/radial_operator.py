"""Exact linear spherical current-Z4c volume operator on R0=M=1 trumpet.

No AthenaK source, jobs, boundary closure, or spatial discretization is changed.
U=(delta_chi,h,delta_Khat,Theta,A_TF,delta_Gamma_rad,delta_alpha,delta_beta_rad).
Returns A2,A1,A0 in U_t=A2 U''+A1 U'+A0 U. Coefficients are derived with
first-order dual arithmetic from Cartesian tensor equations on the x axis.
"""
from pathlib import Path
import sys, json, itertools
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[2]/'vacuum-preservation-20260918/python-deps'))
import sympy as s

r,kap=s.symbols('r kappa', positive=True)
names=('u','h','k','Theta','A','Gamma','ell','shift')
U=[s.Function(n)(r) for n in names]
R=1+r; alpha=r/R; chi=alpha**2; K=1/R**2; beta=r/R**2
pairs=((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))

class Dual:
    def __init__(self,b=0,p=0): self.b=s.sympify(b); self.p=s.sympify(p)
    @staticmethod
    def coerce(x): return x if isinstance(x,Dual) else Dual(x)
    def __add__(self,o):
        o=self.coerce(o); return Dual(self.b+o.b,self.p+o.p)
    __radd__=__add__
    def __neg__(self):return Dual(-self.b,-self.p)
    def __sub__(self,o):return self+-self.coerce(o)
    def __rsub__(self,o):return self.coerce(o)+-self
    def __mul__(self,o):
        o=self.coerce(o); return Dual(self.b*o.b,self.b*o.p+self.p*o.b)
    __rmul__=__mul__
    def __truediv__(self,o):
        o=self.coerce(o); return Dual(self.b/o.b,self.p/o.b-self.b*o.p/o.b**2)
    def __rtruediv__(self,o):return self.coerce(o)/self
    def __pow__(self,n):return Dual(self.b**n,n*self.b**(n-1)*self.p)
    def diff(self,n=1):return Dual(s.diff(self.b,r,n),s.diff(self.p,r,n))

def zeros(shape):
    a=np.empty(shape,object)
    for idx in np.ndindex(shape):a[idx]=Dual()
    return a

def scalar_jet(f):
    d=zeros((3,));dd=zeros((3,3));d[0]=f.diff();dd[0,0]=f.diff(2)
    dd[1,1]=dd[2,2]=f.diff()/r
    return f,d,dd

def vector_jet(f):
    v=zeros((3,));d=zeros((3,3));dd=zeros((3,3,3));v[0]=f
    d[0,0]=f.diff();d[1,1]=d[2,2]=f/r;dd[0,0,0]=f.diff(2)
    for a in (1,2):
        dd[a,a,0]=dd[0,a,a]=dd[a,0,a]=f.diff()/r-f/r**2
    return v,d,dd

def tensor_jet(t,w):
    v=zeros((3,3));d=zeros((3,3,3));dd=zeros((3,3,3,3))
    for a in range(3):v[a,a]=t;d[0,a,a]=t.diff();dd[0,0,a,a]=t.diff(2)
    v[0,0]+=w;d[0,0,0]+=w.diff();dd[0,0,0,0]+=w.diff(2)
    for a in (1,2):
        d[a,0,a]=d[a,a,0]=w/r
        for b in range(3):dd[a,a,b,b]=t.diff()/r
        dd[a,a,0,0]+=w.diff()/r-2*w/r**2
        dd[a,a,a,a]+=2*w/r**2
        for aa,bb in ((0,a),(a,0)):
            dd[aa,bb,0,a]=dd[aa,bb,a,0]=w.diff()/r-w/r**2
    dd[1,2,1,2]=dd[1,2,2,1]=dd[2,1,1,2]=dd[2,1,2,1]=w/r**2
    return v,d,dd

def state_jets():
    u,h,k,th,aa,gg,ell,shift=U
    v=zeros((25,));d=zeros((3,25));dd=zeros((3,3,25))
    for idx,f in [(0,Dual(chi,u)),(7,Dual(K,k)),(17,Dual(0,th)),(18,Dual(alpha,ell))]:
        v[idx],d[:,idx],dd[:,:,idx]=scalar_jet(f)
    for offset,t,w in [(1,Dual(1,-h/2),Dual(0,3*h/2)),
                       (8,Dual(2*K/3,-aa/2-2*K*h/3),Dual(-2*K,3*aa/2))]:
        vv,dd1,dd2=tensor_jet(t,w)
        for z,(a,b) in enumerate(pairs):v[offset+z]=vv[a,b];d[:,offset+z]=dd1[:,a,b];dd[:,:,offset+z]=dd2[:,:,a,b]
    for offset,f in [(14,Dual(0,gg)),(19,Dual(beta,shift))]:
        v[offset:offset+3],d[:,offset:offset+3],dd[:,:,offset:offset+3]=vector_jet(f)
    return v,d,dd

def derive_rhs():
    v,d,dd=state_jets()
    g=zeros((3,3));gi=zeros((3,3));aa=zeros((3,3));dg=zeros((3,3,3));ddg=zeros((3,3,3,3))
    for z,(i,j) in enumerate(pairs):
        g[i,j]=g[j,i]=v[1+z];aa[i,j]=aa[j,i]=v[8+z]
        dg[:,i,j]=dg[:,j,i]=d[:,1+z];ddg[:,:,i,j]=ddg[:,:,j,i]=dd[:,:,1+z]
    # On the chosen radial ray the conformal metric is diagonal. This is its
    # exact first variation, not a physical-metric inverse approximation.
    for i in range(3):gi[i,i]=1/g[i,i]
    low=zeros((3,3,3));up=zeros((3,3,3));Cg=zeros((3,))
    for c,i,j in itertools.product(range(3),repeat=3):
        low[c,i,j]=(dg[i,j,c]+dg[j,i,c]-dg[c,i,j])/2
        up[c,i,j]=sum(gi[c,z]*low[z,i,j] for z in range(3))
    for c in range(3):Cg[c]=sum(gi[i,j]*up[c,i,j] for i,j in itertools.product(range(3),repeat=2))
    phi=[-d[a,0]/(4*v[0]) for a in range(3)]
    phidd=zeros((3,3));Ric=zeros((3,3));hess=zeros((3,3))
    for a,b in itertools.product(range(3),repeat=2):
        phidd[a,b]=-dd[a,b,0]/(4*v[0])+4*phi[a]*phi[b]-sum(up[c,a,b]*phi[c] for c in range(3))
    for a,b in itertools.product(range(3),repeat=2):
        ric=Dual()
        for c in range(3):
            ric+=(g[c,a]*d[b,14+c]+g[c,b]*d[a,14+c]+Cg[c]*(low[a,b,c]+low[b,a,c]))/2
            for e in range(3):
                ric-=gi[c,e]*ddg[c,e,a,b]/2
                for f in range(3):ric+=gi[c,e]*(up[f,c,a]*low[b,f,e]+up[f,c,b]*low[a,f,e]+up[f,a,e]*low[f,c,b])
        Ric[a,b]=ric+4*phi[a]*phi[b]-2*phidd[a,b]-2*g[a,b]*sum(gi[c,e]*(phidd[c,e]+2*phi[c]*phi[e]) for c,e in itertools.product(range(3),repeat=2))
        hess[a,b]=dd[a,b,18]-2*(phi[a]*d[b,18]+phi[b]*d[a,18])-sum(up[c,a,b]*d[c,18] for c in range(3))+2*g[a,b]*sum(gi[c,e]*phi[c]*d[e,18] for c,e in itertools.product(range(3),repeat=2))
    Aup=zeros((3,3));AA=zeros((3,3))
    for a,b in itertools.product(range(3),repeat=2):
        Aup[a,b]=sum(gi[a,c]*aa[c,e]*gi[e,b] for c,e in itertools.product(range(3),repeat=2))
        AA[a,b]=sum(aa[a,c]*gi[c,e]*aa[e,b] for c,e in itertools.product(range(3),repeat=2))
    alphaF=v[18];chiF=v[0];KF=v[7]+2*v[17];theta=v[17]
    div=sum(d[a,19+a] for a in range(3))
    adv=[sum(v[19+a]*d[a,z] for a in range(3)) for z in range(25)]
    scalar=chiF*sum(gi[a,b]*Ric[a,b] for a,b in itertools.product(range(3),repeat=2))
    lap=chiF*sum(gi[a,b]*hess[a,b] for a,b in itertools.product(range(3),repeat=2))
    A2=sum(Aup[a,b]*aa[a,b] for a,b in itertools.product(range(3),repeat=2))
    fout=zeros((25,))
    fout[0]=adv[0]-2*chiF*div/3+2*chiF*alphaF*KF/3
    fout[7]=-lap+alphaF*(A2+KF*KF/3)+adv[7]+kap*alphaF*theta
    fout[17]=adv[17]+alphaF*((scalar+2*KF*KF/3-A2)/2-2*kap*theta)
    for a in range(3):
        DA=sum(-3*Aup[a,b]*d[b,0]/(2*chiF)-gi[a,b]*(2*d[b,7]+d[b,17])/3 for b in range(3))
        DA+=sum(up[a,b,c]*Aup[b,c] for b,c in itertools.product(range(3),repeat=2))
        LG=adv[14+a]+2*Cg[a]*div/3-sum(Cg[b]*d[b,19+a] for b in range(3))
        LG+=sum(gi[a,b]*dd[b,c,19+c]/3+gi[b,c]*dd[b,c,19+a] for b,c in itertools.product(range(3),repeat=2))
        fout[14+a]=2*alphaF*DA+LG-2*kap*alphaF*(v[14+a]-Cg[a])-2*sum(Aup[a,b]*d[b,18] for b in range(3))
    for z,(a,b) in enumerate(pairs):
        Lg=adv[1+z]-2*g[a,b]*div/3+sum(d[a,19+c]*g[b,c]+d[b,19+c]*g[a,c] for c in range(3))
        LA=adv[8+z]-2*aa[a,b]*div/3+sum(d[a,19+c]*aa[b,c]+d[b,19+c]*aa[a,c] for c in range(3))
        fout[1+z]=-2*alphaF*aa[a,b]+Lg
        fout[8+z]=chiF*(-hess[a,b]+alphaF*Ric[a,b])-g[a,b]*(-lap+alphaF*scalar)/3+alphaF*(KF*aa[a,b]-2*AA[a,b])+LA
    # Adapted residual gauge uses only fixed-background advection. It omits
    # delta beta * grad(alpha_bg) and standard lapse-product perturbations.
    fout[18]=Dual(0,beta*s.diff(U[6],r)-2*alpha*U[2])
    fout[19]=Dual(0,beta*s.diff(U[7],r)+2*U[5]-2*U[7])
    bg=[s.factor(f.b) for f in fout]
    assert all(x==0 for x in bg),[(i,x) for i,x in enumerate(bg) if x!=0]
    ans=[fout[i].p for i in [0,1,7,17]]
    ans += [(2*fout[8].p-fout[11].p-fout[13].p)/3,fout[14].p,fout[18].p,fout[19].p]
    ans=[s.factor(x) for x in ans]
    assert s.factor(fout[8].p+2*K*fout[1].p/3-ans[4])==0
    return ans

def symbolic_coefficients(force=False):
    cache=ROOT/'coefficients.json'
    if cache.exists() and not force:
        raw=json.loads(cache.read_text())
        return [s.Matrix([[s.sympify(v,locals={'r':r,'kappa':kap}) for v in row] for row in raw[key]]) for key in ('A2','A1','A0')]
    rhs=derive_rhs(); mats=[]
    for degree in (2,1,0):
        mat=s.zeros(8)
        for i in range(8):
            for j in range(8):mat[i,j]=s.factor(s.diff(rhs[i],s.diff(U[j],r,degree)))
        mats.append(mat)
    for i in range(8):
        assert s.factor(rhs[i]-sum(mats[d][i,j]*s.diff(U[j],r,2-d) for d in range(3) for j in range(8)))==0
    raw={key:[[str(x) for x in row] for row in m.tolist()] for key,m in zip(('A2','A1','A0'),mats)}
    raw['order']=list(names);raw['scope']='Exact linear spherical current Z4c, adapted G2 gauge, sigma=alpha*kappa, eta2; no boundary/discretization.'
    cache.write_text(json.dumps(raw,indent=2)+'\n')
    return mats

def coefficients(rad,rate=.1):
    """Return A2,A1,A0 with shape (...,8,8), for positive scalar/vector radii."""
    rad=np.asarray(rad);mats=symbolic_coefficients();out=[]
    for mat in mats:
        vals=s.lambdify((r,kap),list(mat),modules='numpy',cse=True)(rad,rate)
        out.append(np.stack([np.broadcast_to(x,rad.shape) for x in vals],axis=-1).reshape(rad.shape+(8,8)))
    return tuple(out)

if __name__=='__main__':
    mats=symbolic_coefficients(force='--force' in sys.argv)
    print(json.dumps({'order':names,'nonzero_counts':[sum(x!=0 for x in m) for m in mats], 'background_rhs_exact_zero':True,'algebraic_A_tangent_exact':True}))
