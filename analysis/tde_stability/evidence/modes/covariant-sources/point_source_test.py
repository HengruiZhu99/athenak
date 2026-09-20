"""Compare the actual C++ helper with physical-metric Christoffel derivatives.

The reference constructs D_i Z_j directly, retaining arbitrary derivatives of Q
until subtraction of the Ricci C_ij term. It does not reuse the expanded E formula.
"""
from pathlib import Path
import io, json, subprocess
import numpy as np
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parent
rng=np.random.default_rng(73471)
def sym(a): return .5*(a+a.T)
def point():
    raw=sym(rng.normal(size=(3,3))*.2)
    g=expm(raw-np.eye(3)*np.trace(raw)/3)
    gi=np.linalg.inv(g)
    dg=[]
    for _ in range(3):
        raw=sym(rng.normal(size=(3,3)))
        dg.append(raw-g*np.sum(gi*raw)/3)
    dg=np.array(dg)
    A=sym(rng.normal(size=(3,3)))
    A-=g*np.sum(gi*A)/3
    return dict(alpha=10**rng.uniform(-2,0),chi=10**rng.uniform(-3,0),
        K=rng.normal(),Theta=rng.normal()*1e-3,g=g,gi=gi,A=A,
        Q=rng.normal(size=3)*1e-3,da=rng.normal(size=3),
        dc=rng.normal(size=3),dg=dg,db=rng.normal(size=(3,3)))
def reference(p):
    g,gi,chi=p['g'],p['gi'],p['chi']
    physical=g/chi
    dpi=p['dg']/chi-g[None,:,:]*p['dc'][:,None,None]/chi**2
    pi=chi*gi
    C=np.einsum('kl,ilj->kij',pi,dpi)*.5
    C+=np.einsum('kl,jli->kij',pi,dpi)*.5
    C-=np.einsum('kl,lij->kij',pi,dpi)*.5
    dQ=rng.normal(size=(3,3))*.01
    Z=.5*g@p['Q']
    dZ=.5*(np.einsum('ijk,k->ij',p['dg'],p['Q'])+dQ@g.T)
    DZ=dZ-np.einsum('kij,k->ij',C,Z)
    absorbed=.5*(g@dQ.T+dQ@g.T)
    E=DZ+DZ.T-absorbed
    trace=np.sum(pi*E)
    Zup=pi@Z
    zdalpha=Zup@p['da']
    alpha,K,th=p['alpha'],p['K'],p['Theta']
    khat=2*zdalpha
    theta=.5*alpha*trace-alpha*K*th-zdalpha
    gamma=-2*th*gi@p['da']-(2/3)*alpha*K*p['Q']
    gamma+=((2/3)*np.trace(p['db'])*np.eye(3)-p['db'].T)@p['Q']
    arhs=alpha*chi*(E-physical*trace/3)-2*alpha*th*p['A']
    return np.r_[khat,theta,gamma,arhs.ravel()]
points=[point() for _ in range(64)]
for p in points[:8]:
    q=p.copy();q.update(Q=np.zeros(3),Theta=0.)
    points.append(q)
rows=[]
for p in points:
    rows.append(np.r_[p['alpha'],p['chi'],p['K'],p['Theta'],
        p['g'].ravel(),p['gi'].ravel(),p['A'].ravel(),p['Q'],p['da'],p['dc'],
        p['dg'].ravel(),p['db'].ravel()])
text=str(len(points))+'\n'+'\n'.join(' '.join(format(v,'.17g') for v in row) for row in rows)+'\n'
proc=subprocess.run([str(ROOT/'point_source_driver')],input=text,text=True,
                    capture_output=True,check=True)
actual=np.loadtxt(io.StringIO(proc.stdout))
expected=np.array([reference(p) for p in points])
error=abs(actual-expected)
trace_A=[float(np.sum(p['gi']*a[5:].reshape(3,3))) for p,a in zip(points,actual)]
report=dict(points=len(points),reference='Physical Christoffel construction of D_i Z_j',
    max_absolute_error=float(error.max()),
    relative_l2=float(np.linalg.norm(error)/np.linalg.norm(expected)),
    constraint_surface_max=float(abs(actual[64:]).max()),
    max_trace_A_rhs=float(np.max(abs(np.array(trace_A)))),
    finite=bool(np.isfinite(actual).all()))
assert report['max_absolute_error']<1e-13
assert report['constraint_surface_max']==0
assert report['max_trace_A_rhs']<1e-13
assert report['finite']
(ROOT/'point-source-results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
