#!/usr/bin/env python3
"""Compare matching SMR leaves with point interpolation and signed convergence."""
import argparse,json,sys,csv
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
from intrinsic_restart import read_restart

NPOINTS=6

def interpolation(n):
 w=np.zeros((n,2*n))
 for i in range(n):
  x=2*i+.5;start=max(0,min(2*i-(NPOINTS//2-1),2*n-NPOINTS));nodes=np.arange(start,start+NPOINTS)
  for j,k in enumerate(nodes):
   others=nodes[nodes!=k];w[i,k]=np.prod((x-others)/(k-others))
 return w

def down(u):
 assert u.shape[-1]==u.shape[-2] and u.shape[-1]%2==0
 w=interpolation(u.shape[-1]//2)
 if NPOINTS>6:
  anchor=u[:,:,:1,:1]
  return anchor+np.einsum('ay,mvyx,bx->mvab',w,u-anchor,w,optimize=False)
 return np.einsum('ay,mvyx,bx->mvab',w,u,w,optimize=False)

def active(d):
 ng,nx,ny,nz=d['mb'][:4];assert nz==1 and nx==ny
 return d['state'][:,:,0,ng:ng+ny,ng:ng+nx]

def weights(d):
 n=d['mb'][1];scale=2.**(d['locations'][:,3]-d['root_level']);area=np.prod(d['domain'][6:8])/scale**2
 assert abs(np.sum(area)*n*n-1.3)<2e-14
 return area[:,None,None,None]
def norm(u,w):return np.sqrt(np.sum(u*u*w,axis=(0,2,3))/1.3)
def ratios(d1,d2,w):
 n1,n2=norm(d1,w),norm(d2,w);inner=np.sum(d1*d2*w,axis=(0,2,3))/1.3
 return np.log2(n1/n2),inner/(n1*n2),n1,n2

def interpolation_check():
 errors=[]
 for n in [8,16,32]:
  x=(np.arange(2*n)+.5)/(2*n);coarse=(np.arange(n)+.5)/n
  u=np.sin(2*np.pi*(x[:,None]+x[None,:]))[None,None];expected=np.sin(2*np.pi*(coarse[:,None]+coarse[None,:]))
  errors.append(float(np.sqrt(np.mean((down(u)[0,0]-expected)**2))))
 rates=np.log2(np.array(errors[:-1])/errors[1:]);assert np.all(rates>=5.5),(errors,rates)
 return dict(errors=errors,rates=rates.tolist())

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--points',type=int,choices=[6,10],default=6);a=p.parse_args();NPOINTS=a.points;a.output.mkdir(exist_ok=False);check=interpolation_check();(a.output/'interpolation.json').write_text(json.dumps(check,indent=2)+'\n')
 if a.runs is None:print(check);sys.exit(0)
 results={};histories={}
 for mode in ['none','residual_shifted']:
  data=[]
  for name in [f'n8-{mode}',f'n16-{mode}',f'n32-{mode}',f'n32-{mode}-halfdt']:
   folder=a.runs/name;d=read_restart(sorted(folder.glob('rst/*.rst'))[-1],allow_refinement=True);assert abs(d['time']-.02)<1e-14;assert d['cycle'] in [160,320];data.append(d)
   history=[]
   for f in folder.glob('intrinsic-diagnostics-*.csv'):
    with f.open() as h:rows=list(csv.DictReader(h))
    assert len(rows)==89
    history.append(dict(time=float(rows[0]['time']),cycle=int(rows[0]['cycle']),components=rows))
   history.sort(key=lambda x:x['time']);assert len(history)==5 and np.allclose([h['time'] for h in history],[0,.005,.010,.015,.020],rtol=0,atol=1e-14)
   histories[name]=history
  assert all(np.array_equal(d['locations'],data[0]['locations']) for d in data)
  u=[active(d) for d in data];d1=u[0]-down(u[1]);fine_difference=u[1]-down(u[2]);d2=down(fine_difference);temporal=u[2]-u[3]
  order,alignment,n1,n2=ratios(d1,d2,weights(data[0]));fine_norm=norm(fine_difference,weights(data[1]));time_norm=norm(temporal,weights(data[2]));groups={}
  for name,s in [('primary',slice(0,20)),('all50',slice(0,50))]:
   a1=np.sqrt(np.sum(n1[s]**2));a2=np.sqrt(np.sum(n2[s]**2));cos=float(np.sum(alignment[s]*n1[s]*n2[s])/(a1*a2));rate=float(np.log2(a1/a2));ratio=float(np.sqrt(np.sum(time_norm[s]**2)/np.sum(fine_norm[s]**2)))
   groups[name]=dict(order=rate,alignment=cos,temporal_contamination=ratio,status='PASS_EXPLORATORY' if rate>=3 and cos>=.99 and ratio<=.05 else 'FAIL')
  results[mode]=dict(groups=groups,component_order=order.tolist(),component_alignment=alignment.tolist(),component_temporal_contamination=(time_norm/fine_norm).tolist())
  np.savez_compressed(a.output/(mode+'-signed.npz'),coarse_difference=d1,fine_difference_on_coarse=d2,fine_difference=fine_difference,temporal_difference=temporal,locations=data[0]['locations'],coarse_cell_area=weights(data[0]))
 (a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n');(a.output/'histories.json').write_text(json.dumps(histories,indent=2)+'\n');print(json.dumps({k:v['groups'] for k,v in results.items()},indent=2))
