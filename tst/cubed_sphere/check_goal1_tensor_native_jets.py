#!/usr/bin/env python3
"""Independent Cartesian physical pullback/ADM derivatives vs factored native jets."""
from pathlib import Path
import argparse,json,ast,hashlib,math,time
from functools import lru_cache
import mpmath as m
from goal1_global_clock_oracle import Clock
from goal1_tensor_native_oracle import profile
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);a=p.parse_args();root=Path(__file__).resolve().parents[2];m.mp.dps=80;start=time.time();cache=json.loads((a.evidence.parent/'native-global-clock-v1/reused-height-cache.json').read_text());old=root/'tst/cubed_sphere/check_goal1_tensor_oracle.py';tree=ast.parse(old.read_text());ns={'mp':m,'math':math,'profile':profile};exec(compile(ast.Module(body=[v for v in tree.body if isinstance(v,ast.FunctionDef)and v.name in ['radial','tensor']],type_ignores=[]),str(old),'exec'),ns);physical_tensor=ns['tensor'];pairs=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
class Direct:
 def __init__(self,r,E):self.clock=Clock(r,cache);self.E=E
 def reference(self,x):return self._reference(tuple(x),m.mp.prec)
 @lru_cache(maxsize=14000)
 def _reference(self,x,precision):
  r=m.sqrt(sum(v*v for v in x));n=m.matrix([v/r for v in x]);o,L,A,b=self.clock.reference(r);G=(L/A)**2;gamma=m.eye(3)+(G-1)*n*n.T;beta=-b*A/L*n;chi=G**(-m.mpf(1)/3)
  return gamma,A,beta,chi,o
 def base(self,x,t):return self._base(tuple(x),t,m.mp.prec)
 @lru_cache(maxsize=24000)
 def _base(self,x,t,precision):
  r=m.sqrt(sum(v*v for v in x));n=m.matrix([v/r for v in x]);gamma,A,beta,chi,o=self.reference(x);oo,L,aa,b=self.clock.reference(r);X=[v/o for v in x];T=t+self.clock.uref(r)+r/o;J=(m.eye(3)+(L/o-1)*n*n.T)/o;dGamma=o*o*J.T*physical_tensor(T,X,self.E)*J;inverse=gamma**-1;dBeta=-inverse*dGamma*beta;dAlpha=-(beta.T*dGamma*beta)[0]/(2*A);dChi=-chi*sum((inverse*dGamma)[i,i]for i in range(3))/3;dMetric=chi*dGamma+dChi*gamma
  return dGamma,dBeta,dAlpha,dChi,dMetric
 def matrix_derivative(self,fn,x,d):
  return m.matrix([[m.diff(lambda *y,i=i,j=j:fn(y)[i,j],x,tuple(int(k==d)for k in range(3)))for j in range(3)]for i in range(3)])
 def vector_derivative(self,fn,x,d):return m.matrix([m.diff(lambda *y,i=i:fn(y)[i],x,tuple(int(k==d)for k in range(3)))for i in range(3)])
 def inverse_delta(self,x,t):
  gamma,A,beta,chi,o=self.reference(x);inverse=(gamma**-1)/chi;return -inverse*self.base(x,t)[4]*inverse
 def values(self,x,t):return self._values(tuple(x),t,m.mp.prec)
 @lru_cache(maxsize=24000)
 def _values(self,x,t,precision):
  gamma,alpha,beta,chi,o=self.reference(x);inverse=gamma**-1;dg,db,da,dc,dmetric=self.base(x,t);Gd=[self.matrix_derivative(lambda y:self.reference(y)[0],x,d)for d in range(3)];Bd=[self.vector_derivative(lambda y:self.reference(y)[2],x,d)for d in range(3)];Dgd=[self.matrix_derivative(lambda y:self.base(y,t)[0],x,d)for d in range(3)];Dbd=[self.vector_derivative(lambda y:self.base(y,t)[1],x,d)for d in range(3)];Dgt=m.matrix([[m.diff(lambda tau,i=i,j=j:self.base(x,tau)[0][i,j],t)for j in range(3)]for i in range(3)]);K=m.matrix(3);dK=m.matrix(3)
  for i in range(3):
   for j in range(3):
    L0=sum(beta[k]*Gd[k][i,j]+gamma[k,j]*Bd[i][k]+gamma[i,k]*Bd[j][k]for k in range(3));K[i,j]=L0/(2*alpha)
    L1=sum(beta[k]*Dgd[k][i,j]+dg[k,j]*Bd[i][k]+dg[i,k]*Bd[j][k]+db[k]*Gd[k][i,j]+gamma[k,j]*Dbd[i][k]+gamma[i,k]*Dbd[j][k]for k in range(3));dK[i,j]=(-Dgt[i,j]+L1)/(2*alpha)-da*K[i,j]/alpha
  trK=sum((inverse*K)[i,i]for i in range(3));dtrK=sum((inverse*dK-inverse*dg*inverse*K)[i,i]for i in range(3));op=m.matrix([m.diff(lambda *y:self.reference(y)[4],x,tuple(int(k==d)for k in range(3)))for d in range(3)]);w=-(beta.T*op)[0]/alpha;dw=-(db.T*op)[0]/alpha-w*da/alpha;dP=o*dtrK+3*dw;dA=dc*(K-gamma*trK/3)+chi*(dK-dg*trK/3-gamma*dtrK/3);lam=m.matrix([0,0,0])
  for i in range(3):
   for j in range(3):lam[i]-=m.diff(lambda *y,i=i,j=j:self.inverse_delta(y,t)[i,j],x,tuple(int(k==j)for k in range(3)))
  return [dc]+[dmetric[i,j]for i,j in pairs]+[dP]+[dA[i,j]for i,j in pairs]+list(lam)+[m.mpf(0),da]+list(db)
records=[]
for sector,E in [('plus',m.diag([1,-1,0])),('cross',m.matrix([[0,1,0],[1,0,0],[0,0,0]]))]:
 for rv,tv in [('.3','0'),('.7','.3'),('.95','.4')]:
  fixture=json.loads((a.evidence/f'fixture-{sector}-r{rv}-t{tv}.json').read_text());x=list(map(m.mpf,fixture['xyz']));r=m.mpf(rv);t=m.mpf(tv);direct=Direct(r,E);v=direct.values(x,t);first=[[m.diff(lambda *y,i=i:direct.values(y,t)[i],x,tuple(int(k==d)for k in range(3)))for d in range(3)]for i in range(22)];second={}
  for i in [*range(0,8),17,18,19,20,21]:second[i]={(d,e):m.diff(lambda *y,i=i:direct.values(y,t)[i],x,tuple(int(k==d)+int(k==e)for k in range(3)))for d,e in pairs}
  flat=[]
  def scalar(i):return [v[i],*first[i],*[second.get(i,{}).get((d,e),m.mpf(0))for d,e in pairs]]
  for i in [0,18,7,17]:flat+=scalar(i)
  for i in range(3):flat+=scalar(19+i)+scalar(14+i)
  for i in range(6):flat+=scalar(1+i)+[v[8+i],*first[8+i]]
  target=list(map(m.mpf,fixture['native184']));time_target=[m.diff(lambda tau,i=i:direct.values(x,tau)[i],t)for i in range(22)];errors=[abs(aa-bb)/max(1,abs(aa),abs(bb))for aa,bb in zip(flat,target)];terr=[abs(aa-m.mpf(bb))/max(1,abs(aa),abs(m.mpf(bb)))for aa,bb in zip(time_target,fixture['native22_time'])];row={'sector':sector,'radius':rv,'time':tv,'native184_scaled_error':float(max(errors)),'native22_time_scaled_error':float(max(terr)),'seconds':time.time()-start};records.append(row);(a.evidence/'independent-jets-partial.json').write_text(json.dumps({'scope':'Direct physical TT Cartesian pullback/ADM derivatives, separate from radial tensor coefficients; no native RHS imported','records':records},indent=2)+'\n');print(json.dumps(row),flush=True)
report={'scope':'Independent Cartesian TT pullback, linear ADM curvature/trace and inverse-metric divergence versus factored radial native target jets/time','precision_digits':80,'cases':len(records),'records':records,'max_native184_scaled_error':max(x['native184_scaled_error']for x in records),'max_native22_time_scaled_error':max(x['native22_time_scaled_error']for x in records),'passed':max(max(x['native184_scaled_error'],x['native22_time_scaled_error'])for x in records)<1e-50,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'physical_TT_source_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'no_new_evolved_fields':True};(a.evidence/'independent-jets-review.json').write_text(json.dumps(report,indent=2)+'\n');assert report['passed']
