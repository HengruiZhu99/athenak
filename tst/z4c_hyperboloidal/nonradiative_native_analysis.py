#!/usr/bin/env python3
"""Point-matched full T_ab from projected native rates and accepted time records."""
import argparse,json,sys,time,hashlib,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'vis/python'))
from scri_collect import collect
from scri_reader import read,reconstruct_sampled_jets
from scri_geometry import endpoint_jets,spacetime_derivatives,ricci
from scri_frame import generator_geometry
from scri_news import stencil,analyze

def metric_rate(jets,x,rate):
 chi=jets[:,0,0];gt=np.empty((len(x),3,3));dgt=np.empty_like(gt)
 for k,(i,j) in enumerate(((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))):
  gt[:,i,j]=gt[:,j,i]=jets[:,k+1,0]+float(i==j)
  dgt[:,i,j]=dgt[:,j,i]=rate[:,k+1]
 gamma=gt/chi[:,None,None];dgam=dgt/chi[:,None,None]-gamma*(rate[:,0]/chi)[:,None,None]
 beta=jets[:,22:25,0]-x;db=rate[:,8:11];alpha=jets[:,18,0]+(1+np.sum(x*x,axis=-1))/2
 out=np.zeros((len(x),4,4));out[:,1:,1:]=dgam
 out[:,0,1:]=np.einsum('qij,qj->qi',dgam,beta)+np.einsum('qij,qj->qi',gamma,db)
 out[:,1:,0]=out[:,0,1:]
 out[:,0,0]=-2*alpha*rate[:,7]+np.einsum('qi,qij,qj->q',beta,dgam,beta)+2*np.einsum('qi,qij,qj->q',db,gamma,beta)
 return out

def run(root,output,lmax=6):
 start=time.monotonic();cache=output/'cache';records=collect(root,cache,capacity=2048,record_pattern='**/*.o3.*.r00000000.scri')
 assert len(records)>=10
 grid=np.load(cache/'grid.npz');x=grid['x'];times=np.array([r['time'] for r in records]);archive=np.load(cache/'metric.npy',mmap_mode='r')
 epochs={};T_rhs=[];T_fd=[];lapse=[];rates=[];penalties=[];lifts=[];poles=[];reconstruction=0
 for q,record in enumerate(records):
  path=Path(record['source']);parts=[]
  for rank in range(record['ranks']):
   file=path.with_name(path.name.replace('r00000000.scri',f'r{rank:08d}.scri'));s=read(file);ep=file.parent/s['epoch']
   if ep not in epochs:epochs[ep]=read(ep)
   e=epochs[ep]
   assert int(s['record'][0])==record['cycle'] and float(s['time'][0])==record['time']
   if q in (0,len(records)-1):
    reconstruction=max(reconstruction,float(np.max(abs(reconstruct_sampled_jets(s,e)-s['sampled_residual_jets']),initial=0)))
   parts.append((s,e))
  ids=np.concatenate([e['surface_ids'][:,[0,2,3]] for s,e in parts]);order=np.lexsort(ids.T[::-1])
  jets=np.concatenate([endpoint_jets(s,e) for s,e in parts])[order]
  dr=np.concatenate([s['projected_metric_rates'] for s,e in parts])[order]
  lapse.append(np.concatenate([s['lapse_identity_terms'] for s,e in parts])[order])
  penalties.append(np.concatenate([s['boundary_penalty_rates'] for s,e in parts])[order])
  spacing=np.concatenate([e['rows'][:,2] for s,e in parts])[order]
  strength={float(re.search(r'^\s*scri_penalty\s*=\s*(\S+)',e['parameters'],re.M).group(1)) for s,e in parts}
  assert len(strength)==1 and next(iter(strength))>0
  strength=next(iter(strength));speed=lapse[-1][:,5]
  # Invert the recorded penalty formula without a new evolution evaluation.
  defects=np.column_stack((jets[:,0,0]*(1-1/speed**2),jets[:,13,0],jets[:,14,0]))
  geometric=penalties[-1][:,[0,13,14]]/(4*strength)+2*defects/spacing[:,None]
  lapse_pole=(-jets[:,13,0]+1-speed)/spacing
  poles.append(np.column_stack((geometric,lapse_pole)))
  lift=np.concatenate([s['normal_lift_jets'] for s,e in parts])[order]
  lifts.append([float(np.max(abs(lift[...,section]),initial=0)) for section in (slice(0,1),slice(1,4),slice(4,13))])
  rhs=metric_rate(jets,x,dr);window,center=stencil(times,q)
  g,dg,ddg=spacetime_derivatives(archive[window],times[window],center)
  rates.append(float(np.max(abs(dg[:,0]-rhs))))
  tensors=[]
  for use_rhs in (True,False):
   derivatives=dg.copy()
   if use_rhs:derivatives[:,0]=rhs
   _,conn,inv=ricci(g,derivatives,ddg)
   _,src,nt,H=generator_geometry(x,inv,conn)
   tensors.append(H-g*(src*nt)[:,None,None])
  T_rhs.append(tensors[0]);T_fd.append(tensors[1])
 output.mkdir(parents=True,exist_ok=True)
 np.savez(output/'pointwise.npz',times=times,x=x,T_rhs=T_rhs,T_fd=T_fd,lapse_terms=lapse,penalty_rates=penalties,pole_over_h=poles,pole_fields=['chi','P','Theta','alpha'],metric_rate_difference=rates)
 series={'time':times.tolist(),'T_rhs':np.max(abs(np.array(T_rhs)),axis=(1,2,3)).tolist(),
  'T_fd':np.max(abs(np.array(T_fd)),axis=(1,2,3)).tolist(),
  'T_difference':np.max(abs(np.array(T_rhs)-np.array(T_fd)),axis=(1,2,3)).tolist(),
  'lapse_min':np.min(lapse,axis=1).tolist(),'lapse_max':np.max(lapse,axis=1).tolist(),
  'penalty_max':np.max(abs(np.array(penalties)),axis=1).tolist(),'pole_over_h_max':np.max(abs(np.array(poles)),axis=1).tolist(),'lift_jet_max':lifts,
  'metric_rate_difference':rates}
 (output/'series.json').write_text(json.dumps(series,indent=2)+'\n')
 interior=slice(3,-3)
 summary={'records':len(records),'end_M':float(3*times[-1]),'derivative_reconstruction_error':reconstruction,
  'T_rhs_max':float(np.max(abs(np.array(T_rhs)[interior]))),'T_fd_max':float(np.max(abs(np.array(T_fd)[interior]))),
  'T_path_difference':float(np.max(abs(np.array(T_rhs)[interior]-np.array(T_fd)[interior]))),
  'lapse_identity_error':float(np.max(abs(np.array(lapse)[:,:,10]))),
  'clock_max':float(np.max(abs(np.array(lapse)[:,:,0]))),'metric_rate_difference_max':max(rates[3:-3])}
 summary['news']=analyze(cache,output/'news',lmax=lmax,mass=1/3)
 summary['seconds']=time.monotonic()-start
 (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='news'}),flush=True)
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);p.add_argument('--lmax',type=int,default=6);a=p.parse_args();run(a.root,a.output,a.lmax)
