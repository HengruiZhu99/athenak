"""Independent embedding/analytic Lambda and vacuum constraint checks."""
import json, subprocess, tempfile
from pathlib import Path
import numpy as np
from embedding_probe import BoostedEmbedding
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
temporary=tempfile.TemporaryDirectory(prefix='boost-embedding-')
probe=str(Path(temporary.name)/'probe')
subprocess.run(['c++','-std=c++17','-O2','-I'+str(ROOT),str(ROOT/'tst/hyperbolic_shift_3d/test_native_boosted_wormhole.cpp'),'-o',probe],check=True)
b=BoostedEmbedding(); z=BoostedEmbedding(0)
points=np.array([[r*a,r*c,r*d] for r in [.025,.07,.16,.22,.4,.55,.7,.85,.97] for a,c,d in [(1,0,0),(-.6,.8,0),(.3,.4,np.sqrt(.75))]])
def native(points,v=.1):
 p=subprocess.run([probe,str(v)],input=''.join(' '.join(map(str,x))+'\n' for x in points),text=True,capture_output=True,check=True)
 return np.array([list(map(float,s.split())) for s in p.stdout.splitlines()])
f=native(points); errors=[]; ls=[];zero=[]
for x,q in zip(points,f):
 g=b.geometry(x); sym=lambda m:np.array([m[i,j] for i in range(3) for j in range(i,3)])
 errors.append(max(abs(q[0]-g['chi']),np.max(abs(q[1:7]-sym(g['g']-np.eye(3)))),np.max(abs(q[7:13]-sym(g['A']))),abs(q[13]-3-g['K'])))
 h=2e-5;L=np.zeros(3)
 for j in range(3):
  d=np.eye(3)[j]*h
  gi=[np.linalg.inv(b.geometry(x+k*d)['g']) for k in [-2,-1,1,2]]
  L-= (gi[0][:,j]-8*gi[1][:,j]+8*gi[2][:,j]-gi[3][:,j])/(12*h)
 ls.append(float(np.max(abs(q[15:18]-L))))
for x,q in zip(points,native(points,0)):
 g=z.geometry(x);zero.append(max(np.max(abs(q[1:7])),np.max(abs(q[15:18])),abs(q[0]-g['chi'])))
# Ricci from independent centered metric derivatives and momentum divergence.
def constraints(x,h):
 I=np.eye(3);g0=b.geometry(x);g=g0['metric'];K=g0['curvature'];inv=np.linalg.inv(g)
 dg=np.empty((3,3,3));ddg=np.empty((3,3,3,3));dK=np.empty((3,3,3))
 for a in range(3):
  samples=[b.geometry(x+k*h*I[a]) for k in [-2,-1,1,2]]
  dg[a]=(samples[0]['metric']-8*samples[1]['metric']+8*samples[2]['metric']-samples[3]['metric'])/(12*h)
  dK[a]=(samples[0]['curvature']-8*samples[1]['curvature']+8*samples[2]['curvature']-samples[3]['curvature'])/(12*h)
  ddg[a,a]=(-samples[0]['metric']+16*samples[1]['metric']-30*g+16*samples[2]['metric']-samples[3]['metric'])/(12*h*h)
  for c in range(a):
   val=np.zeros((3,3));coeff=[1,-8,8,-1]
   for i,k in enumerate([-2,-1,1,2]):
    for j,l in enumerate([-2,-1,1,2]):val+=coeff[i]*coeff[j]*b.geometry(x+h*(k*I[a]+l*I[c]))['metric']
   ddg[a,c]=ddg[c,a]=val/(144*h*h)
 dinv=np.array([-inv@d@inv for d in dg]);G=np.zeros((3,3,3));dG=np.zeros((3,3,3,3))
 for k in range(3):
  for i in range(3):
   for j in range(3):
    for l in range(3):
     bracket=dg[i,l,j]+dg[j,l,i]-dg[l,i,j]
     G[k,i,j]+=.5*inv[k,l]*bracket
     for a in range(3):dG[a,k,i,j]+=.5*(dinv[a,k,l]*bracket+inv[k,l]*(ddg[a,i,l,j]+ddg[a,j,l,i]-ddg[a,l,i,j]))
 Ric=np.zeros((3,3))
 for i in range(3):
  for j in range(3):
   for k in range(3):
    Ric[i,j]+=dG[k,k,i,j]-dG[j,k,i,k]
    for l in range(3):Ric[i,j]+=G[k,k,l]*G[l,i,j]-G[k,j,l]*G[l,i,k]
 trace=np.trace(inv@K);mix=inv@K
 H=np.sum(inv*Ric)+trace*trace-np.trace(mix@mix)
 dmix=np.array([dinv[a]@K+inv@dK[a] for a in range(3)]);mom=np.zeros(3)
 for i in range(3):
  mom[i]=-np.trace(dmix[i])
  for j in range(3):
   mom[i]+=dmix[j,j,i]
   for k in range(3):mom[i]+=G[j,j,k]*mix[k,i]-G[k,j,i]*mix[j,k]
 return float(H),float(np.sqrt(mom@inv@mom))
baseline_error=float(np.max(abs(native(points,0)-native(points,'base'))))
assert baseline_error<2e-10,baseline_error
res=[]
for x in [[.12,.08,.03],[.21,.03,.07],[.38,.2,.11],[.6,.2,.1]]:
 vals=[constraints(np.array(x),h) for h in [.002,.001,.0005]];res.append(dict(x=x,steps=[.002,.001,.0005],H_M=vals))
out=dict(zero_boost_all_fields_vs_baseline=baseline_error,native_embedding_max_error=max(errors),analytic_Lambda_vs_independent_FD_max=max(ls),zero_boost_metric_Lambda_error=max(zero),vacuum_constraints=res)
assert max(errors)<2e-10 and max(ls)<2e-5 and max(zero)<2e-10,out
(OUT/'embedding-verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
