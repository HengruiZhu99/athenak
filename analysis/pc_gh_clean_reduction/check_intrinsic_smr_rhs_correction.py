#!/usr/bin/env python3
"""Counterfactual same-state RHS response to a synchronized ghost correction."""
import argparse,json,math,subprocess,sys,hashlib,copy
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
from intrinsic_stage import read_dump

def jets(data):
 h=data['header'];assert h.get('payload','evolution_state')=='evolution_state';u=data['state'];ks,ke,js,je,iss,ie=h['active_kji'];sl=[slice(ks,ke+1),slice(js,je+1),slice(iss,ie+1)];active=u[(slice(None),slice(None),*sl)];du=[];ko=np.zeros_like(active);order=h['order'];radius=order//2+1
 assert h['ghosts_valid'] and h['rank']==0
 for d in range(3):
  v=np.zeros_like(active)
  if u.shape[4-d]>1:
   inv=np.array([1/b['spacing'][d] for b in h['blocks']])[:,None,None,None,None]
   def shifted(offset):
    t=list(sl);axis=2-d;t[axis]=slice(sl[axis].start+offset,sl[axis].stop+offset);return u[(slice(None),slice(None),*t)]
   for j,c in enumerate({2:[.5],4:[2/3,-1/12],6:[.75,-.15,1/60]}[order],1):v+=c*(shifted(j)-shifted(-j))*inv
   for j in range(-radius,radius+1):ko-=h['dissipation']*(-1.)**j*math.comb(2*radius,radius+j)*shifted(j)*inv/(2**(2*radius))
  du.append(v)
 flat=lambda x:np.moveaxis(x,1,-1).reshape(-1,50)
 values=flat(active);rate=h['reduction_rate']*(values[:,0]*values[:,1] if h['reduction_profile']=='lapse_scaled' else np.ones(len(values)))
 rows=np.concatenate([values,*[flat(v) for v in du],rate[:,None],np.full((len(values),1),2.),np.ones((len(values),1))],axis=1)
 return rows,flat(ko),flat(data['rhs']) if 'rhs' in data else None

p=argparse.ArgumentParser();p.add_argument('--dumps',type=Path,required=True);p.add_argument('--kernel',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);entries=[]
# eta2/kappa1 are fixed in the input of this controlled fixture.
assert 'shift_eta = 2' in (a.dumps/'used.athinput').read_text() and 'kappa = 1' in (a.dumps/'used.athinput').read_text()
for cycle,stage in [(0,1),(1,3)]:
 for operation in ['pre-rk','pre-coherent','post-coherent']:
  files=list(a.dumps.glob(f'intrinsic-stage-r0-*-c{cycle}-s{stage}-{operation}.dat'));assert len(files)==1
  d=read_dump(files[0]);rows,ko,actual=jets(d);entries.append((cycle,stage,operation,rows,ko,actual))
for cycle,stage in [(0,1),(1,3)]:
 pre=read_dump(next(a.dumps.glob(f'intrinsic-stage-r0-*-c{cycle}-s{stage}-pre-coherent.dat')))
 post=read_dump(next(a.dumps.glob(f'intrinsic-stage-r0-*-c{cycle}-s{stage}-post-coherent.dat')))
 for family,start,stop in [('p',20,23),('l',23,26),('S',26,41),('B',41,50)]:
  d=copy.deepcopy(pre);d['state'][:,start:stop]=post['state'][:,start:stop]
  rows,ko,actual=jets(d);entries.append((cycle,stage,'family-'+family,rows,ko,actual))
allrows=np.concatenate([e[3] for e in entries]);np.savetxt(a.output/'input.txt',allrows,header=str(len(allrows)),comments='',fmt='%.17g')
command=[str(a.kernel.resolve()),str((a.output/'input.txt').resolve()),str((a.output/'output.txt').resolve()),str((a.output/'configuration.txt').resolve())]
with (a.output/'run.log').open('w') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=120)
assert r.returncode==0
out=np.loadtxt(a.output/'output.txt')[:,:50];pos=0;rhs={};validation=[]
for cycle,stage,operation,rows,ko,actual in entries:
 value=out[pos:pos+len(rows)]+ko;pos+=len(rows);rhs[cycle,stage,operation]=value
 if actual is not None:
  err=float(np.max(abs(value-actual)/(1+abs(actual))));assert err<2e-12;validation.append(err)
records=[]
for cycle,stage in [(0,1),(1,3)]:
 before=rhs[cycle,stage,'pre-coherent'];after=rhs[cycle,stage,'post-coherent'];delta=after-before;assert np.max(abs(delta[:,:10]))==0
 families={f:rhs[cycle,stage,'family-'+f]-before for f in ['p','l','S','B']}
 closure=float(np.max(abs(sum(families.values())-delta)));assert closure<2e-12
 row=int(np.argmax(abs(delta[:,19])));sample=read_dump(next(a.dumps.glob(f'intrinsic-stage-r0-*-c{cycle}-s{stage}-pre-coherent.dat')));h=sample['header'];ks,ke,js,je,iss,ie=h['active_kji'];m,k,j,i=np.unravel_index(row,(len(h['blocks']),ke-ks+1,je-js+1,ie-iss+1));block=h['blocks'][m]
 location=dict(gid=block['gid'],active_kji=[int(k),int(j),int(i)],xyz=[block['origin'][d]+([i,j,k][d]+.5)*block['spacing'][d] for d in range(3)],C_correction=float(delta[row,19]),family_C={f:float(v[row,19]) for f,v in families.items()})
 records.append(dict(C_maximum_location=location,family_sum_closure=closure,family_component_maximum={f:np.max(abs(v),axis=0).tolist() for f,v in families.items()},cycle=cycle,stage=stage,component_maximum=np.max(abs(delta),axis=0).tolist(),signed_at_max=delta[np.argmax(abs(delta),axis=0),np.arange(50)].tolist(),first10_unchanged=True))
 np.savez_compressed(a.output/f'rhs-c{cycle}-s{stage}.npz',before=before,after=after,correction=delta,**{f+'_correction':v for f,v in families.items()})
result=dict(status='PASS_COUNTERFACTUAL_ONLY',kernel_sha256=hashlib.sha256(a.kernel.read_bytes()).hexdigest(),actual_preRK_maximum_normalized_errors=validation,records=records,scope='actual PointRHS plus independently assembled FD/KO matches recorded RHS; evaluate both synchronized ghost states without changing active fields; no trajectory attribution or subsidiary-defect claim')
(a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(validation=validation,maximum_corrections=[max(v['component_maximum']) for v in records]),indent=2))
