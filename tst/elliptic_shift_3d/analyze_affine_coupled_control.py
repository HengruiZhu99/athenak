#!/usr/bin/env python3
"""Read-only same-grid affine pilot comparison against selected independent1D."""
import argparse,hashlib,json,os,tempfile
from pathlib import Path
import numpy as np
from compare_coupled_family import FIELDS,interpolate,norms,topology
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=Path('docs/elliptic-shift-3d/evidence');out=a.output;out.mkdir(parents=True,exist_ok=False);inputs={}
def tracked(p):inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def csv(p):return np.genfromtxt(tracked(p if p.exists() else p.with_name(p.name+'.gz')),delimiter=',',names=True)
ref=np.load(tracked(r/'selected-taper-pilot-20261027/radial-n2049.npz'));coarse=np.load(tracked(r/'selected-taper-pilot-20261027/radial-n1025.npz'))
params=json.loads(tracked(r/'selected-taper-pilot-20261027/reference-parameters.json').read_text());assert params['reference']=='Minkowski' and params['time']==.0002 and params['telegraph_width']==.24
base=r/'related-scri-family-20261027/n6';report=dict(reference_parameters=params,scientific_acceptance=False,scope='Same grid full3D classicalRK4 t=.0002; all cells and64directions retained; no radialized evolution',cases={})
coords=None;profiles={}
for name,folder in [('baseline',base),('affine_candidate',a.candidate)]:
 d=csv(folder/'coupled.csv');s=csv(folder/'coupled.csv.spheres.csv');xyz=np.column_stack([d[x] for x in 'xyz']);assert np.all(np.isfinite(d.view(float))) and np.all(d['weight']>0)
 if coords is None:coords=xyz
 else:assert np.array_equal(xyz,coords)
 w=d['weight'];masks,_=topology(d,6,[.45,.55,.60,.65,.70,.75,1]);truth={f:interpolate(ref['r'],ref[f],d['r']) for f in FIELDS};error={f:d[f]-truth[f] for f in FIELDS}
 beta=np.column_stack([d['delta_beta_'+x]-d[x] for x in 'xyz']);normal=xyz/d['r'][:,None];be=beta-truth['beta_r'][:,None]*normal;tangent=np.linalg.norm(beta-np.sum(beta*normal,axis=1)[:,None]*normal,axis=1)
 row=dict(cells=len(d),field_errors={f:norms(error[f],w) for f in FIELDS},beta_vector=norms(np.linalg.norm(be,axis=1),w),beta_tangent=norms(tangent,w),reference_1025_minus_2049={f:norms(interpolate(coarse['r'],coarse[f],d['r'])-truth[f],w) for f in FIELDS},regions={},angular={})
 for n,mask in masks.items():
  row['regions'][n]=dict(cells=int(mask.sum()),field_errors={f:norms(error[f][mask],w[mask]) for f in FIELDS},H=norms(d['H'][mask],w[mask]),M=norms(d['M_norm'][mask],w[mask])) if mask.any() else dict(cells=0)
 rs=np.unique(s['r'])
 for f in FIELDS:
  mean=[];scatter=[]
  for radius in rs:
   mask=s['r']==radius;assert mask.sum()==64 and len(np.unique(s['direction'][mask]))==64;mean.append(float(s[f][mask].mean()));scatter.append(float(s[f][mask].std()))
  row['angular'][f]=dict(radii=rs.tolist(),mean=mean,scatter=scatter,reference=interpolate(ref['r'],ref[f],rs).tolist())
 np.savez_compressed(out/(name+'-all-cell.npz'),xyz=xyz,weight=w,fields=np.array(FIELDS),values=np.column_stack([d[f]for f in FIELDS]),errors=np.column_stack([error[f]for f in FIELDS]),beta_vector_error=be,beta_tangent=tangent,H=d['H'],M=d['M_norm'])
 np.savez_compressed(out/(name+'-raw-angular.npz'),radius=s['r'],direction=s['direction'],fields=np.array(FIELDS),values=np.column_stack([s[f]for f in FIELDS]))
 report['cases'][name]=row
with tempfile.TemporaryDirectory(dir=out)as cache:
 os.environ['MPLCONFIGDIR']=cache;os.environ['XDG_CACHE_HOME']=cache
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
 for n,row in report['cases'].items():
  v=row['angular']['beta_r'];radius=np.array(v['radii']);mean=np.array(v['mean']);truth=np.array(v['reference']);scatter=np.array(v['scatter']);axes[0].plot(radius,mean-truth,'.-',label=n);axes[1].semilogy(radius,scatter,'.-',label=n)
 axes[0].set(ylabel='Radial shift mean minus matched1D',xlabel='compactified radius');axes[1].set(ylabel='Raw angular scatter',xlabel='compactified radius')
 for ax in axes:ax.legend();ax.grid(alpha=.2)
 fig.suptitle('Affine candidate coupled gate FAIL; identical grid, no excluded cells');fig.savefig(out/'radial-shift-control.png',dpi=160);plt.close(fig)
report['input_sha256']=inputs;report['analysis_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();(out/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:dict(beta_radial_rms=v['field_errors']['beta_r']['penrose_volume_rms'],beta_vector_rms=v['beta_vector']['penrose_volume_rms'],beta_tangent_rms=v['beta_tangent']['penrose_volume_rms']) for k,v in report['cases'].items()}))
