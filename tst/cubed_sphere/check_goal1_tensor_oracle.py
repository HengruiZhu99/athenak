#!/usr/bin/env python3
"""Independent physical Cartesian TT Hertz-potential identity pilot.

This establishes an oracle construction, not native Z4c propagation. No
physical plane wave is extended to scri. A smooth compact retarded profile
is needed for the complete-domain benchmark; Gaussian is a local identity test.
"""
import argparse, hashlib, json, math
from pathlib import Path
import mpmath as mp
mp.mp.dps=75
p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
protocol={'precision_decimal_digits':75,'identity_scaled_budget':1e-50,
    'sectors':['plus','cross'],'points':[[.3,.4,.5],[.7,-.2,.4]],'times':[0.,.2],
    'scope':'Independent inertial linear vacuum TT and physical harmonic gauge identities; not native transformed RHS/evolution',
    'negative_control':'Drop the isotropic trace-cancelling term'}
(a.output/'tensor-oracle-protocol-v1.json').write_text(json.dumps(protocol,indent=2)+'\n')
def hermite(n,z):
    if n==0:return mp.mpf(1)
    old=mp.mpf(1);value=2*z
    for j in range(1,n):old,value=value,2*z*value-2*j*old
    return value
def profile(u,j):
    width=mp.mpf('.4');z=(u+mp.mpf('.5'))/width
    return (-1)**j*hermite(j,z)*mp.exp(-z*z)/width**j
def radial(R,t,n,m):
    return (-1)**n*sum(mp.binomial(n,k)*math.factorial(n-k)*profile(t-R,m+k)/R**(1+n-k) for k in range(n+1))
def tensor(t,X,E,omit=False):
    R=mp.sqrt(sum(x*x for x in X));n=mp.matrix([x/R for x in X]);en=E*n;enn=(n.T*en)[0]
    phir=radial(R,t,1,0);phirr=radial(R,t,2,0);phirrr=radial(R,t,3,0);phirrrr=radial(R,t,4,0)
    A=phirrrr-6*phirrr/R+15*phirr/R**2-15*phir/R**3
    B=phirrr/R-3*phirr/R**2+3*phir/R**3;C=phirr/R**2-phir/R**3
    C2=radial(R,t,1,2)/R;A2=radial(R,t,2,2)-C2
    return mp.matrix([[E[i,j]*(radial(R,t,0,4)-2*C2+C)
        +(-A2+B)*(en[i]*n[j]+en[j]*n[i])
        +(0 if omit else mp.mpf('.5')*(A2+B)*(1 if i==j else 0)*enn)
        +mp.mpf('.5')*A*n[i]*n[j]*enn for j in range(3)] for i in range(3)])
records=[]
for sector,E in [('plus',mp.diag([1,-1,0])),('cross',mp.matrix([[0,1,0],[1,0,0],[0,0,0]]))]:
 for point in protocol['points']:
  X=list(map(lambda x:mp.mpf(str(x)),point))
  for tv in protocol['times']:
   t=mp.mpf(str(tv));H=tensor(t,X,E);scale=max(1,max(abs(x) for x in H));trace=sum(H[i,i] for i in range(3));div=[];wave=[]
   for i in range(3):
    d=0
    for j in range(3):
     order=[0,0,0];order[j]=1
     d+=mp.diff(lambda *x:tensor(t,x,E)[i,j],X,tuple(order))
    div.append(d)
    for j in range(3):
     lap=0
     for k in range(3):
      order=[0,0,0];order[k]=2
      lap+=mp.diff(lambda *x:tensor(t,x,E)[i,j],X,tuple(order))
     wave.append(mp.diff(lambda tau:tensor(tau,X,E)[i,j],t,2)-lap)
   # Independent Cartesian differential-projector action, not radial reduction.
   def phi(tau,x):
    R=mp.sqrt(sum(v*v for v in x));return profile(tau-R,0)/R
   projector_error=0
   for i in range(3):
    for j in range(3):
     expected=E[i,j]*mp.diff(lambda tau:phi(tau,X),t,4)
     for k in range(3):
      order=[0,0,0];order[j]+=1;order[k]+=1
      expected-=E[i,k]*mp.diff(lambda *x:radial(mp.sqrt(sum(v*v for v in x)),t,0,2),X,tuple(order))
      order=[0,0,0];order[i]+=1;order[k]+=1
      expected-=E[j,k]*mp.diff(lambda *x:radial(mp.sqrt(sum(v*v for v in x)),t,0,2),X,tuple(order))
      for l in range(3):
       if E[k,l]==0:continue
       order=[0,0,0];order[k]+=1;order[l]+=1
       if i==j:expected+=mp.mpf('.5')*E[k,l]*mp.diff(lambda *x:radial(mp.sqrt(sum(v*v for v in x)),t,0,2),X,tuple(order))
       order[i]+=1;order[j]+=1
       expected+=mp.mpf('.5')*E[k,l]*mp.diff(lambda *x:phi(t,x),X,tuple(order))
     projector_error=max(projector_error,abs(H[i,j]-expected)/scale)
   negative=tensor(t,X,E,True);negative=abs(sum(negative[i,i] for i in range(3)))/scale
   values={'trace':abs(trace)/scale,'divergence':max(map(abs,div))/scale,'wave':max(map(abs,wave))/scale,'projector':projector_error}
   records.append({'sector':sector,'point':point,'time':tv,'errors':{k:float(v) for k,v in values.items()},
       'negative_trace_error':float(negative),'passed':max(values.values())<=mp.mpf('1e-50')})
report={'passed':all(r['passed'] for r in records) and max(r['negative_trace_error'] for r in records)>mp.mpf('.01'),
    'records':records,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'native_oracle_admitted':False,'exact_scri_transformed_jets_admitted':False,
    'scope':protocol['scope']}
(a.output/'tensor-oracle-review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':report['passed'],'maximum_identity_error':max(v for r in records for v in r['errors'].values()),'negative_trace_error':max(r['negative_trace_error'] for r in records)},indent=2))
