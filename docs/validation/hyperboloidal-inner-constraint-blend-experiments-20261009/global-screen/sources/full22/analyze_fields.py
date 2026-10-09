"""Field and second-order-unit norms using actual native ghosted field derivatives."""
from pathlib import Path
import argparse,json,subprocess,time,hashlib
import numpy as np
w=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('gauge',choices=['production','spatialnorm']);p.add_argument('--stop',type=float,default=2.);a=p.parse_args();g=a.gauge;src=w/f'{g}-projected-krylov-m50-80-h0.1-t{a.stop}.npz';data=np.load(src);values=data['values'];times=data['times'];names=data['names'].tolist();meta=json.loads((w/f'{g}-cache0.0001-metadata.json').read_text());N=meta['points'];coords=np.array(meta['xyz_omega_volume_ginv_chi']);h=meta['spacing'];weights=h**3*coords[:,4];G=np.zeros((N,3,3));ix=[0,0,0,1,1,2];iy=[0,1,2,1,2,2]
for k,(i,j) in enumerate(zip(ix,iy)):G[:,i,j]=G[:,j,i]=coords[:,5]*coords[:,6+k]
cfg=[0,1,2,3,4,5,6,18,19,20,21];mom=[7,8,9,10,11,12,13,14,15,16,17];groups={'chi':[0],'metric':[1,2,3,4,5,6],'P':[7],'A':[8,9,10,11,12,13],'Lambda':[14,15,16],'Theta':[17],'alpha':[18],'beta':[19,20,21]};err=(w/f'{g}-fields-native.stderr').open('w');proc=subprocess.Popen([str(w/'diagnostic-fields')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err);m=json.loads(proc.stdout.readline());assert m['points']==N
result={'gauge':g,'semantics':'prescribed bulk C1 blend projected continuous global tangent; not the full covariant system; canonical short-pilot comparison recorded separately','field_derivative_executable_sha256':hashlib.sha256((w/'diagnostic-fields').read_bytes()).hexdigest(),'field_derivative_source_sha256':hashlib.sha256((w/'diagnostic_fields.cpp').read_bytes()).hexdigest(),'configuration_fields':['chi','g','alpha','beta'],'momentum_fields':['P','A','Lambda','Theta'],'second_order_units_norm':'sqrt integral sqrtgamma [sum_configuration (value^2/S^2+gammaInv gradient^2)+sum_momenta value^2], S=1. Native stored upper tensor components counted once. This is component scaling, not invariant or proved energy.','h_balanced_norm':'sqrt integral sqrtgamma [sum_configuration value^2/h^2+sum_momenta value^2]; grid-dependent equivalent norm used only to diagnose units','histories':[]};t0=time.monotonic()
for col,name in enumerate(names):
 hist=[]
 for it,t in enumerate(times):
  v=np.asarray(values[it,:,col],dtype=np.float64);proc.stdin.write(v.tobytes());proc.stdin.flush();count=N*22*4*8;b=bytearray()
  while len(b)<count:
   q=proc.stdout.read(count-len(b))
   if not q:raise RuntimeError('native field diagnostic stopped')
   b.extend(q)
  jet=np.frombuffer(b,dtype=np.float64).reshape(N,22,4);val2=jet[:,:,0]**2;grad2=np.einsum('pfi,pij,pfj->pf',jet[:,:,1:],G,jet[:,:,1:]);q=np.sum(val2[:,cfg],axis=1);d=np.sum(grad2[:,cfg],axis=1);m=np.sum(val2[:,mom],axis=1);row={'time':float(t),'configuration_H1_momentum_L2':float(np.sqrt(np.sum(weights*(q+d+m)))),'h_balanced_L2':float(np.sqrt(np.sum(weights*(q/h**2+m)))),'groups':{key:{'value_l2':float(np.sqrt(np.sum(weights[:,None]*val2[:,ids]))),'gradient_l2':float(np.sqrt(np.sum(weights[:,None]*grad2[:,ids])))} for key,ids in groups.items()}};hist.append(row)
 for row in hist:
  row['configuration_H1_momentum_L2_amplification']=row['configuration_H1_momentum_L2']/hist[0]['configuration_H1_momentum_L2'];row['h_balanced_L2_amplification']=row['h_balanced_L2']/hist[0]['h_balanced_L2']
 result['histories'].append({'name':name,'history':hist});print(g,name,'final',hist[-1],'max_second_order',max((r['configuration_H1_momentum_L2_amplification'],r['time']) for r in hist),flush=True)
proc.stdin.close();proc.wait();err.close();result['seconds']=time.monotonic()-t0;result['server_exit']=proc.returncode;(w/f'{g}-projected-krylov-t{a.stop}-field-analysis.json').write_text(json.dumps(result,indent=2)+'\n');print('done',g,result['seconds'],flush=True)
