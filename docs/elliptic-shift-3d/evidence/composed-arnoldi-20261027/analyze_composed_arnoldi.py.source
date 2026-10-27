#!/usr/bin/env python3
"""Independent small-matrix products and full spatial witness diagnostics."""
import argparse,hashlib,json,math,os,subprocess,time,warnings
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);a=p.parse_args();out=a.evidence.resolve();m={};sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();recorded=[]
def mv(A,x):return np.array([math.fsum(float(a)*float(b) for a,b in zip(row,x)) for row in A])
def product(A,B):return np.array([[math.fsum(float(a)*float(b) for a,b in zip(row,col)) for col in B.T] for row in A])
def norm(x):return math.sqrt(math.fsum(float(a)*float(a) for a in np.asarray(x).ravel()))
def spatial(path):
 d=np.genfromtxt(path,delimiter=',',names=True);r=np.sqrt(d['x']**2+d['y']**2+d['z']**2);block=d['block'].astype(int);cart=block<8;layer=np.where(cart,-1,(block-8)//6);ratio=r/np.maximum.reduce([abs(d['x']),abs(d['y']),abs(d['z'])]);radii=np.array([.45,.55,.60,.65,.70,.75,1.]);lower=np.where(layer==0,.25*ratio,radii[np.maximum(layer-1,0)]);upper=radii[np.maximum(layer,0)];w=np.where(cart,(.25/6)**3,4/6**3*r*r*(upper-lower)/ratio**3);w/=np.sum(w)
 v=np.array([d['v'+str(c)] for c in range(3)]).T;z=np.array([d['Pv'+str(c)] for c in range(3)]).T;av=np.array([d['APv'+str(c)] for c in range(3)]).T
 dot=lambda x,y:math.fsum(float(wi)*math.fsum(float(a)*float(b) for a,b in zip(xi,yi)) for wi,xi,yi in zip(w,x,y))
 result={'all_input_norm':math.sqrt(dot(v,v)),'all_APv_norm':math.sqrt(dot(av,av)),'all_Pv_norm':math.sqrt(dot(z,z)),'full_Rayleigh_v_APv':dot(v,av),'physical_Rayleigh_z_Az':dot(z,av)/dot(z,z),'v_Pv':dot(v,z),'layers':{}}
 for l in range(-1,7):
  select=layer==l;result['layers'][str(l)]={'cells':int(sum(select)),'measure':float(sum(w[select])),'input_norm_contribution':math.sqrt(float(sum(w[select]*np.sum(v[select]**2,axis=1)))),'physical_correction_norm_contribution':math.sqrt(float(sum(w[select]*np.sum(z[select]**2,axis=1)))),'APv_norm_contribution':math.sqrt(float(sum(w[select]*np.sum(av[select]**2,axis=1))))}
 for key,data in [('input',v),('physical_correction',z),('output',av)]:
  point=int(np.argmax(np.max(abs(data),axis=1)));result[key+'_pointmax']={'point':int(d['point'][point]),'block':int(block[point]),'layer':int(layer[point]),'xyz':[float(d[a][point]) for a in ('x','y','z')],'radius':float(r[point]),'logical_ijk':[int(d[a][point]) for a in ('i','j','k')],'maximum':float(np.max(abs(data[point])))}
 return result
with warnings.catch_warnings(record=True) as caught:
 warnings.simplefilter('always')
 for which in ('expanded','composed'):
  data=json.loads((out/(which+'.H.json')).read_text());H=np.array(data['hessenberg']);S=H[:-1];_,s,Vt=np.linalg.svd(H,full_matrices=False);values,Y=np.linalg.eigh((S+S.T)/2);comm=product(S.T,S)-product(S,S.T);coeff=Vt[-1];actual=json.loads((out/(which+'.witness.json')).read_text());predicted=norm(mv(H,coeff));sp=spatial(out/(which+'.witness.csv'));assert abs(sp['all_APv_norm']-actual['independent_full_APv_norm'])<1e-12
  result={'smallest_restricted_full_action_sv':float(s[-1]),'fresh_APv_norm':actual['independent_full_APv_norm'],'scalar_fsum_predicted_norm':predicted,'prediction_absolute_error':abs(predicted-actual['independent_full_APv_norm']),'projected_commutator_fsum':norm(comm)/norm(S)**2,'projected_numerical_range_real_min':float(values[0]),'projected_numerical_range_real_max':float(values[-1]),'singular_witness_spatial':sp,'curves':[]}
  e=np.zeros(H.shape[0]);e[0]=1
  for j in (16,32,48,64,80):
   y=np.linalg.lstsq(H[:j+1,:j],e[:j+1],rcond=None)[0];result['curves'].append({'columns':j,'relative_residual_fsum':norm(e[:j+1]-mv(H[:j+1,:j],y))})
  if which=='composed':
   prefix=out/'composed-negative';target=out/'composed-negative.Q.bin'
   if not target.exists():os.link(out/'composed.Q.bin',target)
   np.savetxt(out/'composed-negative.coefficients.dat',Y[:,0],fmt='%.17g');cmd=[str(out/'probe'),'composed','witness',str(prefix),str(out/'composed-negative.coefficients.dat')];start=time.monotonic()
   with (out/'composed-negative.stdout').open('w') as stdout,(out/'composed-negative.stderr').open('w') as stderr:code=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=180,env=dict(os.environ,OMP_NUM_THREADS='1')).returncode
   assert code==0;negative=spatial(out/'composed-negative.witness.csv');assert abs(negative['full_Rayleigh_v_APv']-values[0])<1e-10;result['fresh_negative_numerical_range_witness']={'command':cmd,'seconds':time.monotonic()-start,'exit_code':code,'projected_rayleigh':float(values[0]),'independent_full_rayleigh':negative['full_Rayleigh_v_APv'],'spatial':negative}
  m[which]=result
 recorded=[str(w.message) for w in caught]
m['small_matrix_warnings']=recorded;m['independent_products']='Scalar math.fsum; LAPACK SVD/eigh/lstsq; no BLAS matrix products';m['warning_context']='Initial runner emitted NumPy matmul floating-status warnings with finite results. This independent analysis avoids those products and verifies the same values and fresh full operator witnesses.';m['source_sha256']=sha(Path(__file__));(out/'independent-analysis.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({k:{a:b for a,b in v.items() if a not in ['singular_witness_spatial','fresh_negative_numerical_range_witness']} for k,v in m.items() if isinstance(v,dict)},indent=2))
