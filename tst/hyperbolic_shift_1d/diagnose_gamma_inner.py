"""Metric-rate decomposition and restricted inner linearization diagnostics.

Optional parity variants change derivative stencils only for this diagnostic.
They are not adopted evolution equations, and no bulk field is projected.
"""
import argparse,hashlib,inspect,json
from pathlib import Path
import numpy as np
from puncture import PunctureEvolution
from numerics import RadialOperators,puncture_dissipation
from elliptic import connection

class DiagnosticParity:
 def __init__(self,base,fields):
  self.base=base;self.fields=fields
  self.reg=RadialOperators(base.r,width=7,subtract_local=True)
 def __getattr__(self,key):return getattr(self.base,key)
 def derivatives(self,values,parity):
  d,dd=self.base.derivatives(values,parity)
  if values.ndim==2 and values.shape[1]==8:
   for field,sign in self.fields:
    v=values[:,field].copy();v[0]=0.
    d[:,field],dd[:,field]=self.reg.derivatives(v,sign)
  return d,dd
 def advection_derivative(self,values,parity,beta):
  d=self.base.advection_derivative(values,parity,beta)
  if values.ndim==2 and values.shape[1]==8:
   for field,sign in self.fields:
    v=values[:,field,None].copy();v[0]=0.
    d[:,field]=self.reg.advection_derivative(v,np.array([sign]),beta)[:,0]
  return d

p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--nodes',type=int,default=12);p.add_argument('--spectrum',action='store_true');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
m=json.loads((a.case/'manifest.json').read_text());z=np.load(a.case/'profiles.npz')
kwargs={k:v for k,v in m['arguments'].items() if k in inspect.signature(PunctureEvolution).parameters and k not in ['n','library']}
def evolution():
 e=PunctureEvolution(m['arguments']['n'],a.library,**kwargs)
 assert np.array_equal(e.r,z['r'])
 return e
e=evolution();rows=[];arrays={}
selection=sorted(set([0,len(z['q'])//2,len(z['q'])-1]))
for index in selection:
 q=z['q'][index];t=float(z['t'][index]);rate=e.rhs(q,diagnostics=True);v=e.last;r=e.r;g=v['g'];alpha=v['alpha'];beta=v['beta'];d=v['physical_first'];Arr=v['physical'][:,2]
 chart=q[:,:8].copy();chart[1:,6]=np.log(q[1:,6]/e.chart.refs['alpha'][1:,0]);adv=e.op.advection_derivative(chart,np.ones(8),beta)
 ko=puncture_dissipation(q,r,e.h,e.ko,.8,measure=e.jac,order=8)/e.jac[:,None]
 terms=np.c_[beta*adv[:,1],-2*alpha*Arr,4*g/3*(d[:,7]-1-np.divide(beta,r,out=np.zeros_like(beta),where=r>0)),ko[:,1]]
 error=float(max(abs(np.sum(terms[1:-1],axis=1)-rate[1:-1,1])));assert error<1e-9
 V=connection(r[1:],g[1:],d[1:,1]);key=f't{index}'
 arrays[key+'_profile']=np.c_[r[1:],g[1:],alpha[1:],beta[1:],q[1:,5],V,v['diag'][1:,0],v['diag'][1:,3],rate[1:,1],terms[1:]]
 rows.append(dict(t=t,key=key,metric_rate_decomposition_error=error,first_positive_node=arrays[key+'_profile'][0].tolist()))
spectra=[]
if a.spectrum:
 indices=np.arange(10,(a.nodes+1)*10)
 base=z['last'].copy()
 for label,fields in [('current',[]),('lambda-odd',[(5,-1)]),('g-even',[(1,1)]),('g-even-lambda-odd',[(1,1),(5,-1)])]:
  ev=evolution()
  if fields:ev.op=DiagnosticParity(ev.op,fields)
  scales=np.maximum(abs(base),1.)
  scales[:,6]=np.maximum(abs(base[:,6]),1e-5)
  sr=scales.ravel()[indices]
  for eps in [1e-5,2e-6]:
   columns=[]
   for j in indices:
    q=base.copy();q.flat[j]+=eps*scales.flat[j];plus=ev.rhs(q).ravel()[indices]
    q=base.copy();q.flat[j]-=eps*scales.flat[j];minus=ev.rhs(q).ravel()[indices]
    columns.append((plus-minus)/(2*eps*sr))
   J=np.asarray(columns).T;values,vectors=np.linalg.eig(J);k=int(np.argmax(values.real));mode=vectors[:,k].reshape(a.nodes,10)
   record=dict(variant=label,epsilon=eps,spectral_abscissa=float(values[k].real),imaginary_part=float(values[k].imag),peak_radius=float(ev.r[1+np.argmax(np.linalg.norm(mode,axis=1))]),field_norms=np.linalg.norm(mode,axis=0).tolist())
   spectra.append(record);print(json.dumps(record),flush=True)
   arrays[f'{label}-{eps}_eigenvalues']=values;arrays[f'{label}-{eps}_mode']=mode
result=dict(case=str(a.case),manifest_sha256=hashlib.sha256((a.case/'manifest.json').read_bytes()).hexdigest(),profile_columns=['r','g','alpha','beta','Lambda','V','H','Zr','gdot','transport','curvature','shear','KO'],profiles=rows,spectra=spectra,
 scope='Exact decomposition of the implemented metric rate away from endpoints. Restricted Jacobians perturb only inner positive nodes and hold the exterior fixed; artificial-interface and nonnormal effects prevent a full-domain stability claim. Parity alternatives are diagnostic-only and retain all positive-node fields independently.')
(a.output/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(a.output/'diagnosis.npz',**arrays)
print(json.dumps(dict(metric_rate_errors=[x['metric_rate_decomposition_error'] for x in rows],spectra=len(spectra))))
