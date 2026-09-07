#!/usr/bin/env python3
"""Signed initial-transfer budgets against analytic sinusoidal jets on SMR leaves."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
from intrinsic_restart import read_restart

PAIRS=[(0,1),(0,2),(1,2)]
def auxiliary(u):
 return np.array([np.concatenate([u[20+d:21+d],u[23+d:24+d],u[26+5*d:31+5*d],u[41+3*d:44+3*d]]) for d in range(3)])
def potentials(u):return np.concatenate([u[:1],(u[0]*u[1])[None],u[2:10]])
def qfield(u):
 # Q= (dT) T^T + T (dT)^T, with independent intrinsic S.
 t=np.zeros((3,3,*u.shape[1:]),dtype=u.dtype)
 t[0,0]=np.exp(u[2]);t[1,1]=np.exp(u[3]);t[2,2]=np.exp(-u[2]-u[3]);t[1,0]=u[4];t[2,0]=u[5];t[2,1]=u[6]
 q=[]
 for d in range(3):
  s=u[26+5*d:31+5*d];v=np.zeros_like(t);v[0,0]=t[0,0]*s[0];v[1,1]=t[1,1]*s[1];v[2,2]=-t[2,2]*(s[0]+s[1]);v[1,0]=s[2];v[2,0]=s[3];v[2,1]=s[4]
  q.append(np.einsum('ir...,jr...->ij...',v,t)+np.einsum('ir...,jr...->ij...',t,v))
 return np.array(q)
def block_budget(u,exact,jet,h,ng,shape,order):
 nz,ny,nx=shape;sl=(slice(ng,ng+nz) if nz>1 else slice(0,1),slice(ng,ng+ny) if ny>1 else slice(0,1),slice(ng,ng+nx))
 def active(f):return f[(...,*sl)]
 def D(f,d):
  out=np.zeros_like(active(f))
  if shape[2-d]==1:return out
  for j,c in enumerate({2:[.5],4:[2/3,-1/12],6:[.75,-.15,1/60]}[order],1):
   plus=list(sl);minus=list(sl);axis=2-d;plus[axis]=slice(sl[axis].start+j,sl[axis].stop+j);minus[axis]=slice(sl[axis].start-j,sl[axis].stop-j)
   out+=c*(f[(...,*plus)]-f[(...,*minus)])/h[d]
  return out
 g=auxiliary(u);q=qfield(u);exg=auxiliary(exact)
 dp=np.array([np.concatenate([j[:1],(j[0]*exact[1]+exact[0]*j[1])[None],j[2:10]]) for j in jet])
 dg=np.array([auxiliary(j) for j in jet])
 # Complex-step differentiates the independent algebraic Q expression at analytic jets.
 dq=np.array([qfield(exact.astype(complex)+1e-30j*j).imag/1e-30 for j in jet])
 quantities=dict(reduction=active(g)-np.array([D(potentials(u),d) for d in range(3)]),curl=np.array([D(g[j],i)-D(g[i],j) for i,j in PAIRS]),qcurl=np.array([D(q[j],i)-D(q[i],j) for i,j in PAIRS]))
 oracle=dict(reduction=active(exg-dp),curl=active(np.array([dg[i,j]-dg[j,i] for i,j in PAIRS])),qcurl=active(np.array([dq[i,j]-dq[j,i] for i,j in PAIRS])))
 for d in (quantities,oracle):
  d['qcurl']=np.array([d['qcurl'][:,i,j] for i,j in [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]]).swapaxes(0,1)
  for key in d:d[key]=d[key].reshape(-1,nz,ny,nx)
 return quantities,oracle

def analyze(folder,order,dim,out):
 data=[read_restart(sorted((folder/f'fd{order}-{dim}d-{mode}').glob('rst/*.rst'))[0],allow_refinement=True) for mode in ('none','residual_shifted')]
 a,b=data;assert a['cycle']==b['cycle']==0 and np.array_equal(a['locations'],b['locations'])
 ng,nx,ny,nz=a['mb'][:4];assert all(np.array_equal(a[k],b[k]) for k in ('mb','domain'))
 fields={};volumes=[];locs=[];boundary_counts=[];levels=[]
 for m,loc in enumerate(a['locations']):
  lev=loc[3]-a['root_level'];h=a['domain'][6:9]/2.**lev
  # Inactive coordinates do not enter phase or differential operators.
  grid=[a['domain'][d]+(loc[d]*[nx,ny,nz][d]+np.arange(a['state'].shape[4-d])-(ng if [nx,ny,nz][d]>1 else 0)+.5)*h[d] for d in range(3)]
  z,y,x=np.meshgrid(grid[2],grid[1],grid[0],indexing='ij');length=a['domain'][3:6]-a['domain'][:3];wave=2*np.pi/length;wave[dim:]=0
  phase=wave[0]*x+wave[1]*y+wave[2]*z;v=np.arange(50)[:,None,None,None];amp=float(a['header']['problem'].get('amplitude','.001'))
  exact=(v<2)+amp*np.sin(phase+.17*v)/(1+.03*v);jet=np.array([amp*w*np.cos(phase+.17*v)/(1+.03*v) for w in wave])
  active=(slice(ng,ng+nz) if nz>1 else slice(0,1),slice(ng,ng+ny) if ny>1 else slice(0,1),slice(ng,ng+nx))
  assert np.max(abs(a['state'][m][(...,*active)]-exact[(...,*active)]))<5e-16, 'not the analytic initial state'
  qa,oracle=block_budget(a['state'][m],exact,jet,h,ng,(nz,ny,nx),order);qb,oracleb=block_budget(b['state'][m],exact,jet,h,ng,(nz,ny,nx),order)
  for key in qa:
   for label,val in [('before',qa[key]),('after',qb[key]),('error_before',qa[key]-oracle[key]),('error_after',qb[key]-oracle[key]),('correction',qb[key]-qa[key])]:fields.setdefault(key+'_'+label,[]).append(val.reshape(val.shape[0],-1))
  kk,jj,ii=np.indices((nz,ny,nx));radius=order//2
  boundary=((ii<radius)|(ii>=nx-radius)).astype(int)
  if dim>1:boundary+=((jj<radius)|(jj>=ny-radius))
  if dim>2:boundary+=((kk<radius)|(kk>=nz-radius))
  boundary_counts.extend(boundary.ravel());levels.extend([int(lev)]*(nx*ny*nz))
  dv=float(np.prod(h[:dim]));volumes.extend([dv]*(nx*ny*nz));locs.extend([[m,*loc]]*(nx*ny*nz))
 fields={k:np.concatenate(v,axis=1) for k,v in fields.items()};weights=np.array(volumes);volume=float(weights.sum());assert abs(volume-np.prod((a['domain'][3:6]-a['domain'][:3])[:dim]))<1e-12
 result={'volume':volume,'cells':len(weights),'leaf_levels':a['locations'].tolist(),'metrics':{}}
 for key,v in fields.items():
  norm=np.sqrt(np.sum(v*v*weights,axis=1)/volume);indices=np.argmax(abs(v),axis=1)
  result['metrics'][key]=dict(component_RMS=norm.tolist(),group_RMS=float(np.sqrt(np.sum(norm*norm))),component_maximum=np.max(abs(v),axis=1).tolist(),signed_at_max=v[np.arange(len(v)),indices].tolist(),flattened_cell_at_max=indices.tolist())
 result['region_metrics']={}
 boundary_counts=np.array(boundary_counts);levels=np.array(levels)
 for lev in np.unique(levels):
  for count in range(dim+1):
   mask=(levels==lev)&(boundary_counts==count)
   if not mask.any():continue
   region={}
   for key,v in fields.items():
    region[key]=dict(group_RMS=float(np.sqrt(np.sum(v[:,mask]**2*weights[mask])/weights[mask].sum())),maximum=float(np.max(abs(v[:,mask]))))
   result['region_metrics'][f'level{lev}-near{count}-block-faces']=dict(volume=float(weights[mask].sum()),cells=int(mask.sum()),metrics=region)
 assert np.max(abs(fields['reduction_correction']))==0
 assert np.max(abs(fields['curl_correction'][:,boundary_counts==0]))==0
 np.savez_compressed(out,**fields,cell_volume=weights,boundary_count=boundary_counts,physical_level=levels,leaf_locations=a['locations'],cell_leaf_location=np.array(locs),active_shape=np.array([nz,ny,nx]))
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--folders',type=Path,nargs=3,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--order',type=int,default=6);p.add_argument('--dimension',type=int,default=2);a=p.parse_args();a.output.mkdir(exist_ok=False)
 records=[analyze(f,a.order,a.dimension,a.output/f'signed-n{n}.npz') for f,n in zip(a.folders,[8,16,32])]
 convergence={}
 for key in ['reduction','curl','qcurl']:
  for mode in ['before','after']:
   errors=[r['metrics'][key+'_error_'+mode]['group_RMS'] for r in records];rates=np.log2(np.array(errors[:-1])/errors[1:]);convergence[key+'_'+mode]=dict(errors=errors,rates=rates.tolist(),status='PASS_EXPLORATORY' if np.all(rates>=.5) else 'FAIL')
 result=dict(scope='initial analytic transfer only; no evolution or full gate qualification',records=records,convergence=convergence)
 (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(convergence,indent=2))
