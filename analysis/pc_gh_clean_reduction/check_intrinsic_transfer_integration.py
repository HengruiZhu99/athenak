#!/usr/bin/env python3
"""Bounded static-periodic transfer integration checks; no accuracy claim."""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
from intrinsic_restart import read_restart
p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--template',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--ranks',type=int,default=1);p.add_argument('--mpiexec',default='/opt/homebrew/bin/mpiexec');p.add_argument('--block-n',type=int,default=8);p.add_argument('--orders',type=int,nargs='+',default=[2,4,6]);p.add_argument('--dimensions',type=int,nargs='+',default=[2,3]);a=p.parse_args()
a.output.mkdir(exist_ok=True)
results=[]
for order in a.orders:
 for dim in a.dimensions:
  pair=[]
  for mode in ('none','residual_shifted'):
   run=a.output/f'fd{order}-{dim}d-{mode}';run.mkdir()
   text=a.template.read_text().replace('spatial_order = 6',f'spatial_order = {order}').replace('nlim = 20','nlim = 0').replace('intrinsic_diagnostics = true','intrinsic_diagnostics = false').replace('<pc_gh>',f'<pc_gh>\ncoherent_transfer = {mode}')
   if dim==3:
    mesh,tail=text.split('<meshblock>');text=mesh.replace('nx3 = 1','nx3 = 16')+'<meshblock>'+tail.replace('nx3 = 1','nx3 = 8')
   mesh,tail=text.split('<meshblock>')
   for axis in range(1,dim+1):
    mesh=mesh.replace(f'nx{axis} = 16',f'nx{axis} = {2*a.block_n}')
    tail=tail.replace(f'nx{axis} = 8',f'nx{axis} = {a.block_n}')
   text=mesh+'<meshblock>'+tail
   text+='\n<mesh_refinement>\nrefinement = static\n<refined_region1>\nx1min = 0.125\nx1max = 0.375\nx2min = 0.1625\nx2max = 0.4875\nx3min = 0.2125\nx3max = 0.6375\nlevel = 1\n'
   used=run/'used.athinput';used.write_text(text);start=time.monotonic()
   with (run/'run.log').open('w') as log:
    command=[str(a.binary.resolve()),'-i',str(used.resolve())]
    if a.ranks>1: command=[a.mpiexec,'-n',str(a.ranks)]+command
    rc=subprocess.run(command,cwd=run,stdout=log,stderr=subprocess.STDOUT,timeout=120).returncode
   assert rc==0,(run,rc)
   assert f'Number of parallel ranks = {a.ranks}' in (run/'run.log').read_text()
   files=sorted(run.glob('rst/*.rst'));assert files
   data=read_restart(files[0],allow_refinement=True);pair.append(data)
   results.append(dict(order=order,dimension=dim,mode=mode,ranks=a.ranks,command=command,returncode=rc,seconds=time.monotonic()-start,leaves=len(data['locations']),levels=np.unique(data['locations'][:,3]).tolist(),input_sha256=hashlib.sha256(used.read_bytes()).hexdigest()))
  x,y=pair;assert np.array_equal(x['locations'],y['locations'])
  assert len(np.unique(x['locations'][:,3]))==2
  u,v=x['state'],y['state'];ng,nx,ny,nz=x['mb'][:4]
  primary=float(np.max(abs(u[:,:20]-v[:,:20])))
  active=(slice(None),slice(None),slice(ng,ng+nz) if nz>1 else slice(0,1),slice(ng,ng+ny),slice(ng,ng+nx))
  fixed=float(np.max(abs(u[active]-v[active])))
  change=float(np.max(abs(u[:,20:]-v[:,20:])))
  assert primary==0 and fixed==0
  results[-1].update(primary_change=primary,active_change=fixed,auxiliary_ghost_change=change,status='PASS_INVARIANCE_ONLY')
  (a.output/'results.json').write_text(json.dumps(dict(binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),runs=results),indent=2)+'\n')
  print(order,dim,primary,fixed,change,flush=True)
