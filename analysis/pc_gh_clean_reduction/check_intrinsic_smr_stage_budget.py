#!/usr/bin/env python3
"""Actual synchronized SMR reconstruction budgets, with old-binary neutrality."""
import argparse,hashlib,json,subprocess,sys,time,shlex
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
from intrinsic_stage import read_dump,assemble_ranks
from intrinsic_restart import read_restart
from analyze_intrinsic_interface_budget import block_budget

def quantities(data):
 h=data['header'];assert h.get('payload','evolution_state')=='evolution_state';ks,ke,js,je,iss,ie=h['active_kji'];assert ks==0 and iss==js and h['ghosts_valid']
 result=[];weights=[]
 for m,b in enumerate(h['blocks']):
  u=data['state'][m];q,_=block_budget(u,u,np.zeros((3,*u.shape)),b['spacing'],iss,(1,je-js+1,ie-iss+1),h['order']);result.append(np.concatenate([q[k] for k in ['reduction','curl','qcurl']]).reshape(78,-1));weights.extend([b['spacing'][0]*b['spacing'][1]]*((je-js+1)*(ie-iss+1)))
 return np.concatenate(result,axis=1),np.array(weights)
def stats(v,w):
 return dict(component_RMS=np.sqrt(np.sum(v*v*w,axis=1)/np.sum(w)).tolist(),component_maximum=np.max(abs(v),axis=1).tolist(),signed_at_max=v[np.arange(len(v)),np.argmax(abs(v),axis=1)].tolist(),cell_at_max=np.argmax(abs(v),axis=1).tolist())

p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--launcher',default='');p.add_argument('--ranks',type=int,default=1);a=p.parse_args();a.output.mkdir(exist_ok=True);runs=[];records=[]
for mode in ['none','residual_shifted']:
 folders=[]
 for enabled,exe in [(False,a.reference),(True,a.binary)]:
  d=a.output/(mode+('-dump' if enabled else '-reference'));d.mkdir();folders.append(d)
  text=(a.fixtures/f'fd6-2d-{mode}'/'used.athinput').read_text().replace('nlim = 0','nlim = 2').replace('research_dt_ceiling = 0.001','research_dt_ceiling = 0.000125').replace('<pc_gh>',f'<pc_gh>\nintrinsic_stage_dump = {str(enabled).lower()}');(d/'used.athinput').write_text(text);command=shlex.split(a.launcher)+[str(exe.resolve()),'-i','used.athinput'];start=time.monotonic()
  with (d/'run.log').open('w') as log:r=subprocess.run(command,cwd=d,stdout=log,stderr=subprocess.STDOUT,timeout=180)
  assert r.returncode==0 and f'Number of parallel ranks = {a.ranks}' in (d/'run.log').read_text()
  runs.append(dict(directory=str(d),command=command,returncode=r.returncode,seconds=time.monotonic()-start,binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),input_sha256=hashlib.sha256((d/'used.athinput').read_bytes()).hexdigest()))
 arrays=[read_restart(sorted(d.glob('rst/*.rst'))[-1],allow_refinement=True)['state'] for d in folders];assert np.array_equal(*arrays)
 parts={};files=sorted(folders[1].glob('intrinsic-stage-*.dat'));assert len(files)==(48 if mode=='residual_shifted' else 30)*a.ranks
 for f in files:
  d=read_dump(f);h=d['header'];parts.setdefault((h['cycle'],h['stage'],h['operation']),[]).append(d)
 snapshots={k:assemble_ranks(v,a.ranks) for k,v in parts.items()};stages=[]
 for cycle in [0,1]:
  for stage in [1,2,3]:
   pre=snapshots[cycle,stage,'pre-coherent'];post=snapshots[cycle,stage,'post-coherent'];rk=snapshots[cycle,stage,'post-rk'];exchange=snapshots[cycle,stage,'post-exchange']
   assert not rk['header']['ghosts_valid'] and pre['header']['ghosts_valid'] and post['header']['ghosts_valid']
   assert np.array_equal(pre['state'][:,:20],post['state'][:,:20]);assert np.array_equal(pre['active'],post['active']);assert np.array_equal(rk['active'],pre['active']);assert np.array_equal(post['state'],exchange['state'])
   before,w=quantities(pre);after,w2=quantities(post);assert np.array_equal(w,w2);delta=after-before;assert np.max(abs(delta[:30]))==0
   if mode=='residual_shifted':
    source=snapshots[cycle,stage,'source-residual'];received=snapshots[cycle,stage,'received-residual']
    assert source['header']['payload']==received['header']['payload']=='transfer_residual'
    change=(received['state']-source['state'])[:,20:]
    identity=float(np.max(abs((post['state']-pre['state'])[:,20:]-change)))
    assert identity<2e-12
   else:identity=0.
   if mode=='none':assert np.array_equal(pre['state'],post['state']) and np.max(abs(delta))==0
   state_delta=post['state']-pre['state'];ordinary_delta=pre['state']-rk['state']
   np.savez_compressed(folders[1]/f'budget-c{cycle}-s{stage}.npz',before=before,after=after,correction=delta,cell_area=w,coherent_state_delta=state_delta,ordinary_stored_state_delta=ordinary_delta)
   stages.append(dict(residual_identity_error=identity,cycle=cycle,stage=stage,primary_change=0,active_change=0,reduction_change=0,auxiliary_ghost_change=float(np.max(abs(state_delta))),ordinary_stored_change=float(np.max(abs(ordinary_delta))),correction=stats(delta,w),before=stats(before,w),after=stats(after,w),scope='ordinary ghost delta is raw only; no derivatives on stale post-RK state'))
 # Reject a colliding dump in a separate directory, leaving main evidence untouched.
 collision=a.output/(mode+'-collision');collision.mkdir();(collision/'used.athinput').write_text((folders[1]/'used.athinput').read_text());first=next(f for f in files if '-r0-' in f.name and '-c0-s1-pre-rk.' in f.name);sentinel=collision/first.name;sentinel.write_bytes(first.read_bytes());original=hashlib.sha256(sentinel.read_bytes()).hexdigest()
 with (collision/'run.log').open('w') as log:r=subprocess.run(command,cwd=collision,stdout=log,stderr=subprocess.STDOUT,timeout=180)
 assert r.returncode!=0 and 'cannot exclusively create stage dump' in (collision/'run.log').read_text();assert hashlib.sha256(sentinel.read_bytes()).hexdigest()==original
 records.append(dict(mode=mode,ranks=a.ranks,status='PASS_BUDGET_ONLY',bitwise_neutral=True,collision_rejected=True,stages=stages));(a.output/'results.json').write_text(json.dumps(records,indent=2)+'\n');(a.output/'runs.json').write_text(json.dumps(runs,indent=2)+'\n');print(mode,'PASS',flush=True)
