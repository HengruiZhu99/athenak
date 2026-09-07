#!/usr/bin/env python3
"""Trace the scalar derivative/transfer commutator from measured private buffers."""
import argparse,json,sys
from fractions import Fraction
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
from intrinsic_stage import read_dump

def matrix(size):
 result=np.zeros((size,size))
 for position in range(size):
  start=max(0,min(position-3,size-7));x=position-start
  for node in range(7):
   others=[b for b in range(7) if b!=node];den=np.prod([node-b for b in others]);value=Fraction(0)
   for omitted in others:
    value+=Fraction(int(np.prod([x-b for b in others if b!=omitted])),int(den))
   result[position,start+node]=float(value)
 return result

def target(data):
 u=data['state'][:,0]-1.;blocks=data['header']['blocks'];result=[]
 for axis in range(3):
  if u.shape[3-axis]==1:v=np.zeros_like(u)
  else:
   v=np.moveaxis(u,3-axis,-1);v=np.einsum('ij,...j->...i',matrix(v.shape[-1]),v);v=np.moveaxis(v,-1,3-axis);v/=np.array([b['spacing'][axis] for b in blocks])[:,None,None,None]
  result.append(v)
 return np.stack(result,axis=1)

p=argparse.ArgumentParser();p.add_argument('--dumps',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);records=[]
for cycle in [0,1]:
 for stage in [1,2,3]:
  def load(operation):
   files=list(a.dumps.glob(f'intrinsic-stage-r0-*-c{cycle}-s{stage}-{operation}.dat'));assert len(files)==1;return read_dump(files[0])
  pre=load('pre-coherent');post=load('post-coherent');source=load('source-residual');received=load('received-residual');probe=load('transported-state-probe')
  assert source['header']['payload']==received['header']['payload']=='transfer_residual';assert probe['header']['payload']=='transported_state_probe'
  g=pre['state'][:,20:23];after=post['state'][:,20:23];es=source['state'][:,20:23];er=received['state'][:,20:23];tg=probe['state'][:,20:23]
  r=target(pre);source_check=float(np.max(abs((g-es)-r)));assert source_check<2e-12
  # T is linear on fixed-topology buffers; T(R)=T(G)-T(E).
  transported_target=tg-er;commutator=r-transported_target;repeat=tg-g;correction=after-g
  h=pre['header'];ks,ke,js,je,iss,ie=h['active_kji'];ghost=np.ones(g.shape[2:],bool);ghost[ks:ke+1,js:je+1,iss:ie+1]=False
  closure=float(np.max(abs((commutator+repeat-correction)[:,:,ghost])));assert closure<2e-12
  # Rounding of the independent derivative check limits this closure, not evolution.
  m=next(i for i,b in enumerate(h['blocks']) if b['gid']==3);samples=[]
  for direction in [0,1]:
   for offset in [1,2,3]:
    k,j,i=0,je,ie
    if direction==0:i+=offset
    else:j+=offset
    samples.append(dict(direction=direction,stored_kji=[k,j,i],G=float(g[m,direction,k,j,i]),source_E=float(es[m,direction,k,j,i]),received_E=float(er[m,direction,k,j,i]),target=float(r[m,direction,k,j,i]),transferred_target=float(transported_target[m,direction,k,j,i]),commutator=float(commutator[m,direction,k,j,i]),repeated_G_transfer=float(repeat[m,direction,k,j,i]),correction=float(correction[m,direction,k,j,i])))
  np.savez_compressed(a.output/f'p-c{cycle}-s{stage}.npz',source_E=es,received_E=er,target=r,transferred_target=transported_target,commutator=commutator,repeated_G_transfer=repeat,correction=correction,ghost_mask=ghost)
  records.append(dict(cycle=cycle,stage=stage,source_target_max_error=source_check,ghost_identity_error=closure,maximum_ghost_correction=float(np.max(abs(correction[:,:,ghost]))),maximum_repeated_G_transfer=float(np.max(abs(repeat[:,:,ghost]))),maximum_commutator=float(np.max(abs(commutator[:,:,ghost]))),corner_normal_samples=samples))
(a.output/'results.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='corner_normal_samples'} for r in records],indent=2))
