#!/usr/bin/env python3
"""Independent80-digit derivatives of native holonomic momentum and inverse."""
import hashlib,json
from pathlib import Path
import mpmath as mp
mp.mp.dps=80
root=Path(__file__).resolve().parents[2];out=root/'docs/validation/goal1-minkowski-20261018/typed-native-ad-v1';assert not(out/'review.json').exists();rows=[]
for run in json.loads((out/'runs.json').read_text()):
 p=out/(run['label']+'.jsonl');assert hashlib.sha256(p.read_bytes()).hexdigest()==run['stdout_sha256'];rows.extend(json.loads(line)for line in p.read_text().splitlines()if line.startswith('{'))
def tensor(v):return mp.matrix([[v[0],v[1],v[2]],[v[1],v[3],v[4]],[v[2],v[4],v[5]]])
def chol(g):
 l00=mp.sqrt(g[0,0]);l10=g[1,0]/l00;l20=g[2,0]/l00;l11=mp.sqrt(g[1,1]-l10*l10);l21=(g[2,1]-l10*l20)/l11;l22=mp.sqrt(g[2,2]-l20*l20-l21*l21);return mp.matrix([[l00,0,0],[l10,l11,0],[l20,l21,l22]])
def velocity(g,V):
 L=chol(g);a=V[0,0]/(2*L[0,0]);da=a/L[0,0];c=(V[1,0]-L[1,0]*a)/L[0,0];d=(V[2,0]-L[2,0]*a)/L[0,0];b=(V[1,1]-2*L[1,0]*c)/(2*L[1,1]);db=b/L[1,1];e=(V[2,1]-c*L[2,0]-L[1,0]*d-L[2,1]*b)/L[1,1];return[da,db,c,d,e]
def config(u,r):
 g=tensor(u[2:8]);ig=g**-1;a=u[1];chi=u[0];D=sum(u[28+3*i+i]-r[28+3*i+i]for i in range(3));strain=mp.matrix(3,3)
 for i in range(3):
  for j in range(3):strain[i,j]=-mp.mpf(2)/3*g[i,j]*D+sum(g[k,i]*(u[28+3*j+k]-r[28+3*j+k])+g[k,j]*(u[28+3*i+k]-r[28+3*i+k])for k in range(3))
 return g,ig,a,chi,D,strain
def momentum(u,r,o):
 g,ig,a,chi,D,strain=config(u,r);V=-2*a*(tensor(u[13:19])-tensor(r[13:19]))+strain;p0=-mp.mpf(2)/3*chi*D+mp.mpf(2)/3*a*chi*((u[11]-r[11])+2*(u[12]-r[12]))/o;v=velocity(g,V);result=[p0]+v+[-a*a*(u[11]-r[11])/o]
 for i in range(3):result.append(a*a*chi*(u[19+i]-r[19+i])+sum(ig[i,j]*(a*a*(u[22+j]-r[22+j])/2-a*chi*(u[25+j]-r[25+j]))for j in range(3)))
 return result
def inverse(u,r,o,pi):
 g,ig,a,chi,D,strain=config(u,r);dp=-o*pi[6]/(a*a);P=r[11]+dp;theta=r[12]+(3*o*(pi[0]+mp.mpf(2)/3*chi*D)/(2*a*chi)-dp)/2;sv=velocity(g,2*a*tensor(r[13:19])+strain);v=[(s-p)/(2*a)for s,p in zip(sv,pi[1:6])];L=chol(g);Ld=mp.matrix([[L[0,0]*v[0],0,0],[v[2],L[1,1]*v[1],0],[v[3],v[4],-L[2,2]*(v[0]+v[1])]]);A=Ld*L.T+L*Ld.T;lam=[]
 for i in range(3):lam.append(r[19+i]+(pi[7+i]-sum(ig[i,j]*(a*a*(u[22+j]-r[22+j])/2-a*chi*(u[25+j]-r[25+j]))for j in range(3)))/(a*a*chi))
 return[P,theta]+[A[i,j]for i,j in[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]]+lam
def poly(j,x):return mp.mpf(j[0])+sum(mp.mpf(j[1+i])*x[i]for i in range(3))+sum(mp.mpf(j[4+3*i+k])*x[i]*x[k]/2 for i in range(3)for k in range(3))
def derivatives(f):
 z=(mp.mpf(0),)*3;orders=[(0,0,0)]+[tuple(int(i==a)for i in range(3))for a in range(3)]+[tuple(int(i==a)+int(i==b)for i in range(3))for a in range(3)for b in range(3)];return[mp.diff(lambda*x:f(x),z,o)for o in orders]
oracle=[]
for row in rows:
 if'input_u'not in row:continue
 inputs={k:row[k]for k in['input_u','input_ref','input_Omega','target_Pi']}
 def evaluate(x,reverse):
  u=[poly(v,x)for v in inputs['input_u']];r=[poly(v,x)for v in inputs['input_ref']];o=poly(inputs['input_Omega'],x)
  return inverse(u,r,o,[poly(v,x)for v in inputs['target_Pi']])if reverse else momentum(u,r,o)
 errors=[]
 for reverse,key in[(False,'forward_Pi'),(True,'inverse_p')]:
  maximum=0
  for i,cpp in enumerate(row[key]):
   truth=derivatives(lambda x:evaluate(x,reverse)[i]);maximum=max(maximum,max(float(abs(mp.mpf(v)-w)/max(1,abs(w)))for v,w in zip(cpp,truth)))
  errors.append(maximum)
 oracle.append({'kind':row['kind'],'case':row['case'],'momentum_all_jet_scaled':errors[0],'inverse_all_jet_scaled':errors[1],'passed':max(errors)<=1e-10});print('oracle',row['kind'],row['case'],max(errors),flush=True)
points=[r for r in rows if r['kind']!='native-core-AD'];core=[r for r in rows if r['kind']=='native-core-AD'];keys=['preserved_Real_value_scaled','AD_inverse_momentum_jet_roundtrip_scaled','AD_tracefree_A_jet_scaled','AD_configuration_change_scaled'];maxima={k:max(r[k]for r in points)for k in keys};corekeys=['native22_value_scaled','native22_FD_first_scaled','native22_FD_second_scaled','FD_refinement_uncertainty_scaled'];coremax={k:max(r[k]for r in core)for k in corekeys}
report={'scope':__doc__,'independent_precision_digits':80,'points':len(points),'independent_oracle_points':len(oracle),'map_values_and_AD_invariants_pass':max(maxima.values())<=1e-10 and maxima['AD_configuration_change_scaled']==0,'independent_map_derivatives_pass':all(x['passed']for x in oracle),'map_maxima':maxima,'oracle':oracle,'native_core_points':len(core),'native_core_maxima':coremax,'native_core_AD_agreement_pass':coremax['native22_value_scaled']<=1e-10 and max(coremax[k]for k in corekeys[1:])<=1e-6,'native_core_FD_scope':'Independent finite-difference adapter check, not final derivative/source accuracy oracle. Steps .003/.0015, fivepoint, uncertainty retained.','physical_reference_higher_jets_admitted':False,'native_reduced_P_source_implemented':False,'native_trajectories_run':0,'exact_scri_implemented':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(out/'review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print({k:v for k,v in report.items()if k!='oracle'})
