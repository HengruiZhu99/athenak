#!/usr/bin/env python3
"""Full actual frozen native3D wave reduction; energy witness, no evolution."""
import argparse,ast,hashlib,json,math
from pathlib import Path
import numpy as np
np.seterr(all='raise')
p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();out=a.output
protocol=json.loads((out/'protocol-v1.json').read_text());base=Path('docs/validation/goal1-minkowski-20261018')
helper=Path('tst/cubed_sphere/check_constraint_incoming_lift.py')
for node in ast.parse(helper.read_text()).body:
 if isinstance(node,ast.FunctionDef) and node.name in ['mm','symbols']:exec(compile(ast.Module(body=[node],type_ignores=[]),str(helper),'exec'),globals())
gramhelper=Path('tst/cubed_sphere/check_goal1_outer_port.py')
for node in ast.parse(gramhelper.read_text()).body:
 if isinstance(node,ast.FunctionDef) and node.name=='gram':exec(compile(ast.Module(body=[node],type_ignores=[]),str(gramhelper),'exec'),globals())
all_symbols=list(symbols(base/'oblique-symbol.txt'))[:30]
requests=[list(map(float,s.split())) for s in (base/'oblique-symbol.samples').read_text().splitlines()][:30]
ops=[json.loads(s) for s in (base/'canonical-ko-v2/tapered-radial-matrices.jsonl').read_text().splitlines() if s.startswith('{')]
wave=[];witness=[]
for state in range(5):
 cases=all_symbols[state*6:state*6+6];M=[s['matrix'] for s in cases];inverse=cases[0]['inverse'];G=gram(cases[0]['basis'][:11,:10],inverse)
 B=M[0][:10,10:];Binv=np.linalg.inv(B);beta=np.array([-(s['lo']+s['hi'])/2 for s in cases[:3]])
 A=[m[:10,:10] for m in M[:3]];Ae=[A[i]-beta[i]*np.eye(10) for i in range(3)];D=[m[10:,10:] for m in M[:3]]
 alpha=requests[state*6][7+18];chi=requests[state*6][7];metric=alpha*alpha*chi*inverse
 C=np.zeros((3,3,10,10))
 for i in range(3):C[i,i]=M[i][10:,:10]
 design=[];residual=[]
 for s,m in zip(cases[3:],M[3:]):
  n=s['normal'];design.append([2*n[0]*n[1],2*n[0]*n[2],2*n[1]*n[2]])
  residual.append(m[10:,:10]-sum(n[i]*n[i]*C[i,i] for i in range(3)))
 cross=np.linalg.solve(np.array(design),np.array(residual).reshape(3,100)).reshape(3,10,10)
 for pair,val in zip([(0,1),(0,2),(1,2)],cross):C[pair[0],pair[1]]=C[pair[1],pair[0]]=val
 errors=[]
 for i in range(3):
  errors.append(float(np.linalg.norm(mm(Ae[i],B)+mm(B,D[i])-beta[i]*B)/max(1,np.linalg.norm(B))))
  for j in range(3):
   target=metric[i,j]*np.eye(10);actual=(mm(Ae[i],Ae[j])+mm(Ae[j],Ae[i]))/2+mm(B,C[i,j])
   errors.append(float(np.linalg.norm(actual-target)/max(1,np.linalg.norm(target))))
 Bchange=max(float(np.linalg.norm(m[:10,10:]-B)) for m in M)
 wave.append({'state':state,'native_mixed_wave_errors':errors,'native_B_covector_change':Bchange,'oblique_design_condition':float(np.linalg.cond(design)),
  'positive_spatial_metric_minimum':float(np.linalg.eigvalsh(metric)[0]),'passed':bool(max(errors+[Bchange])<=1e-10)})
 for normal in range(3):
  for tangent in range(3):
   if normal==tangent:continue
   for col in range(10):
    v=np.eye(10)[col];t=Ae[tangent][:,col];zero=bool(np.linalg.norm(t)<1e-12)
    for op in ops:
     N=op['nodes'];H=np.array(op['H']);Q=np.array(op['Q']);D1=np.array(op['D']);x=(np.arange(N)+.5)/N
     f=.001*(np.sin(5*math.pi*x)+.2*np.cos(3*math.pi*x));df=np.einsum('ij,j->i',D1,f)
     # q=y*f*v at three symmetric tangential nodes; centered y derivative is exact.
     pi_normal=-f[:,None]*t[None,:]/2;pi_full=-pi_normal
     v2=mm(mm(Q,pi_normal),Binv.T);v3=mm(mm(Q,pi_full),Binv.T)
     rate2=float(2*np.einsum('n,ni,ij,nj->',H,pi_full,G,mm(v2,B.T)))
     rate3=float(2*np.einsum('n,ni,ij,nj->',H,pi_full,G,mm(v3,B.T)))
     target=float(np.einsum('ni,ij,nm,mj->',pi_full,G,H[:,None]*Q+Q.T*H[None,:],pi_full))
     defect=max(abs(rate3-target),abs(rate2+target))/max(1,abs(target),abs(rate2),abs(rate3))
     E=float(np.einsum('n,ni,ij,nj->',H,pi_full,G,pi_full)+v.dot(G.dot(v))*(metric[normal,normal]*np.dot(H,df*df)/6+metric[tangent,tangent]*np.dot(H,f*f)+np.dot(H,f*f)/6))
     # Invert the original q row at y=-.5,0,.5 and verify compatible full Pi.
     compatibility=0.
     for y in [-.5,0,.5]:
      qn=y*df[:,None]*v;qt=f[:,None]*v
      pp=mm(pi_normal-mm(qn,Ae[normal].T),Binv.T)
      actual=mm(qn,Ae[normal].T)+mm(qt,Ae[tangent].T)+mm(pp,B.T)
      compatibility=max(compatibility,float(np.max(abs(actual-pi_full))))
     witness.append({'state':state,'normal':normal,'tangent':tangent,'q_component':col,'nodes':N,'stretch':op['stretch'],'algebraically_zero_tangential_coupling':zero,
      'v2_full_gradient_energy_rate':rate2,'v3_full_gradient_energy_rate':rate3,'negative_line_form':target,'energy_identity_error':defect,
      'compatible_gradient_lift_error':compatibility,'full_positive_energy':E,'v2_logarithmic_rate':rate2/E,'v3_logarithmic_rate':rate3/E,
      'passed_v3':bool(defect<=1e-10 and compatibility<=1e-10 and rate3<=1e-10 and (zero or (rate2>0 and rate3<0)))})
nonzero=[r for r in witness if not r['algebraically_zero_tangential_coupling']]
negative=max(r['v2_logarithmic_rate'] for r in nonzero if r['nodes']==24)>1e-12
report={'scope':__doc__,'full_native_3D_wave_reduction_pass':all(r['passed'] for r in wave),'v2_full_gradient_contraction_pass':False if negative else None,
 'v2_positive_energy_witness_detected':negative,'v3_algebra_dissipation_pass':all(r['passed_v3'] for r in witness),
 'states':len(wave),'wave_checks':wave,'witnesses':witness,'nonzero_witnesses':len(nonzero),'algebraic_zero_records':len(witness)-len(nonzero),
 'maximum_v2_positive_logarithmic_rate':max(r['v2_logarithmic_rate'] for r in nonzero),'native_v3_evolved':False,'full_method_adopted':False,
 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'input_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [base/'oblique-symbol.txt',base/'oblique-symbol.samples',base/'canonical-ko-v2/tapered-radial-matrices.jsonl']}}
(out/'review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print({k:v for k,v in report.items() if k not in ['witnesses','wave_checks','input_sha256']})
print('failed wave',[(r['state'],max(r['native_mixed_wave_errors'])) for r in wave if not r['passed']]);print('failed v3',sum(not r['passed_v3'] for r in witness))
