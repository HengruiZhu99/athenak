#!/usr/bin/env python3
"""Read-only native Driver snapshots versus matched selected spherical reference.
All active cells enter norms; angular groups contain actual nodes, not interpolated
64-ray samples. This comparison cannot establish late-time collapse or convergence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
from compare_coupled_family import FIELDS, interpolate, norms, topology


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--onecore',type=Path,default=Path('docs/elliptic-shift-3d/evidence/native-wormhole-controls-r3-20261027'));p.add_argument('--reference',type=Path,default=Path('docs/elliptic-shift-3d/evidence/selected-taper-pilot-20261027'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 out=a.output;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();hashes={}
 def track(p):hashes[str(p)]=sha(p);return p
 ref=np.load(track(a.reference/'radial-n2049.npz'));coarse=np.load(track(a.reference/'radial-n1025.npz'));parameters=json.loads(track(a.reference/'reference-parameters.json').read_text())
 assert parameters['telegraph_width']==.24 and parameters['time']==.0002 and parameters['reference']=='Minkowski' and parameters['mass']==1/3
 summary={'scope':'Actual native Driver two-cycle one-core/eight-core controls at t=.0002 versus independently evolved N2049 spherical profiles. All scientific FAIL; no spatial-convergence or collapse-safety claim.','reference_parameters':parameters,'temporal_control_limit':'Native actual Driver Ketcheson 2S RK4 versus stored radial classical RK4 at dt1e-4; independent timestep refinement remains required.','weight_definition':'CSV map_volume*sqrt(det(g_tilde))/chi^(3/2), Penrose spatial proper-volume quadrature. Every active cell retained; no Omega factor, floor, clipping or radial exclusion.','region_note':'All is the acceptance norm over every cell. Other regions overlap; no region replaces it. Inferred Cartesian spacing tier is a diagnostic, not a native AMR level.','angular_method':'All actual nodal Cartesian components projected radially for diagnostics only. Geometrically coincident radii grouped by round(r,14), retaining raw values/directions and per-group counts. No interpolated rays, symmetrized evolution or radial-value projection.','cases':{}}
 plots={}
 for key,base in [('one_core',a.onecore),('eight_core',a.native)]:
  m=json.loads(track(base/'manifest.json').read_text());trace=m['native_trace'];assert trace[-1]['time']==.0002
  q=np.genfromtxt(track(base/'full/native_wormhole.00002.csv'),delimiter=',',names=True)
  assert len(q)==trace[-1]['cells'] and all(np.all(np.isfinite(q[f])) for f in q.dtype.names)
  xyz=np.column_stack([q[f] for f in ['x','y','z']]);r=np.linalg.norm(xyz,axis=1);assert np.all(r>0) and np.all(r<1)
  n=xyz/r[:,None];w=q['weight'];assert np.all(w>0) and np.all(q['alpha']>0) and np.all(q['chi']>0)
  def contraction(start,identity=False):
   c=np.column_stack([q[f'state_{i}'] for i in range(start,start+6)])
   return (1 if identity else 0)+c[:,0]*n[:,0]**2+c[:,3]*n[:,1]**2+c[:,5]*n[:,2]**2+2*(c[:,1]*n[:,0]*n[:,1]+c[:,2]*n[:,0]*n[:,2]+c[:,4]*n[:,1]*n[:,2])
  beta=np.column_stack([q['beta_'+f] for f in ['x','y','z']]);physical={'alpha':q['alpha'],'chi':q['chi'],'g_rr':contraction(1,True),'A_rr':contraction(7),'Khat':q['state_13']-3,'Theta':q['Theta'],'Lambda_r':sum(q[f'state_{15+i}']*n[:,i] for i in range(3)),'beta_r':np.sum(beta*n,axis=1)}
  derived=np.empty(len(q),dtype=list(q.dtype.descr)+[('r','f8'),('M_norm','f8')]);
  for f in q.dtype.names:derived[f]=q[f]
  derived['r']=r;derived['M_norm']=q['M'];masks,mesh=topology(derived,6,[.45,.55,.60,.65,.70,.75,1])
  truth={f:interpolate(ref['r'],ref[f],r) for f in FIELDS};error={f:physical[f]-truth[f] for f in FIELDS};refdiff={f:interpolate(coarse['r'],coarse[f],r)-truth[f] for f in FIELDS}
  beta_error=beta-truth['beta_r'][:,None]*n;beta_tangent=np.linalg.norm(beta-physical['beta_r'][:,None]*n,axis=1)
  row={'cells':len(q),'minimum_radius':float(r.min()),'Cartesian_h':sorted(set(mesh['spacing'][np.isfinite(mesh['spacing'])].round(15).tolist())),'native_core_blocks':m.get('native_core_blocks',1),'physical_errors':{f:norms(error[f],w) for f in FIELDS},'reference_N1025_minus_N2049':{f:norms(refdiff[f],w) for f in FIELDS},'regions':{},'angular':{},'original_vector_residual':{'physical_volume_rms_stage_max':trace[-1]['shift_original_rms_max'],'point_stage_max':trace[-1]['shift_point_max'],'source':'actual Driver independently checked original vector equation, not potential/preconditioner residual; derivative-incomplete CSV does not permit independent regional reconstruction'},'gates':{name:{'value':trace[-1][name],'unchanged_target':target,'passes':trace[-1][name]<=target} for name,target in [('scri_tensor_pole',1e-7),('scri_ghost_clock',1e-13),('scri_ghost_null',1e-12),('scri_ghost_tangent',1e-12)]},'scientific_acceptance':False}
  for name,mask in masks.items():
   if not mask.any():row['regions'][name]={'cells':0,'unavailable':True};continue
   ww=w[mask];row['regions'][name]={'cells':int(mask.sum()),'Penrose_volume':float(ww.sum()),'physical_errors':{f:norms(error[f][mask],ww) for f in FIELDS},'diagnostics':{f:norms(q[f][mask],ww) for f in ['H','M','Theta','Z']},'beta_vector_error':norms(np.linalg.norm(beta_error[mask],axis=1),ww),'beta_tangent':norms(beta_tangent[mask],ww)}
  for f in ['H','M','Theta','Z']:
   assert abs(row['regions']['all']['diagnostics'][f]['penrose_volume_rms']-trace[-1][f+'_rms'])<5e-12*max(1,abs(trace[-1][f+'_rms']))
  groups=np.round(r,14);radii=np.unique(groups)
  for f in FIELDS:
   means=[];scatter=[];minima=[];maxima=[];counts=[];rms=[]
   exact=interpolate(ref['r'],ref[f],radii)
   for rr,expected in zip(radii,exact):
    mask=groups==rr;v=physical[f][mask];assert np.max(abs(r[mask]-rr))<1e-14
    means.append(float(v.mean()));scatter.append(float(v.std()));minima.append(float(v.min()));maxima.append(float(v.max()));counts.append(len(v));rms.append(float(np.sqrt(np.mean((v-expected)**2))))
   row['angular'][f]={'radii':radii.tolist(),'counts':counts,'reference':exact.tolist(),'mean':means,'scatter':scatter,'min':minima,'max':maxima,'angular_rms_error':rms,'mean_error':(np.array(means)-exact).tolist(),'maximum_scatter':max(scatter)}
  np.savez_compressed(out/(key+'-all-cells-and-raw-angular.npz'),points=xyz,radius=r,angular_group_radius=groups,unit_directions=n,weights=w,fields=np.array(FIELDS),physical=np.column_stack([physical[f] for f in FIELDS]),reference=np.column_stack([truth[f] for f in FIELDS]),errors=np.column_stack([error[f] for f in FIELDS]),H=q['H'],M=q['M'],Theta=q['Theta'],Z=q['Z'],beta=beta,beta_vector_error=beta_error,beta_tangent=beta_tangent)
  summary['cases'][key]=row;plots[key]=(r,physical)
 with tempfile.TemporaryDirectory(prefix='plot-cache-',dir=out) as cache:
  os.environ['MPLCONFIGDIR']=cache;os.environ['XDG_CACHE_HOME']=cache
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(4,2,figsize=(11,12),layout='constrained')
  for ax,f in zip(axes.flat,FIELDS):
   ax.plot(ref['r'],ref[f],color='k',label='Independent selected spherical N2049')
   for key,color in [('one_core','tab:blue'),('eight_core','tab:orange')]:
    v=summary['cases'][key]['angular'][f];rr=np.array(v['radii']);means=np.array(v['mean']);sd=np.array(v['scatter']);ax.plot(rr,means,'.',ms=3,color=color,label=key.replace('_',' '));ax.fill_between(rr,means-sd,means+sd,color=color,alpha=.15)
   ax.grid(alpha=.2);ax.set(xlabel='compact radius',ylabel=f)
  axes[0,0].legend(fontsize=7);fig.suptitle('Actual native Driver at t=.0002, selected taper .24\nActual-node angular mean +/- scatter; both scientific FAIL')
  fig.savefig(out/'native-radial-profiles.png',dpi=145);plt.close(fig)
  fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');names=['all','core_r_lt_01','cartesian','cartesian_shell_stencil_neighbors','radial_shell_stencil_neighbors','scri_r_ge_09'];labels=['all','r<.1','Cartesian','cube-shell','shell radial','r>=.9']
  for key,color in [('one_core','tab:blue'),('eight_core','tab:orange')]:
   v=summary['cases'][key];x=np.arange(len(FIELDS));axes[0,0].semilogy(x,[v['physical_errors'][f]['penrose_volume_rms'] for f in FIELDS],'.-',color=color,label=key);axes[0,1].semilogy(x,[v['angular'][f]['maximum_scatter'] for f in FIELDS],'.-',color=color)
   for ax,f in zip(axes[1],['H','M']):ax.semilogy(np.arange(len(names)),[v['regions'][n]['diagnostics'][f]['penrose_volume_rms'] for n in names],'.-',color=color)
  axes[0,0].set_title('All-cell volume RMS against spherical');axes[0,0].legend();axes[0,1].set_title('Maximum actual-node angular scatter')
  for ax in axes[0]:ax.set_xticks(np.arange(len(FIELDS)),FIELDS,rotation=35,ha='right');ax.grid(alpha=.2)
  for ax,f in zip(axes[1],['H','M']):ax.set_xticks(np.arange(len(names)),labels,rotation=30,ha='right');ax.set_title(f+' RMS; overlapping diagnostics');ax.grid(alpha=.2)
  fig.suptitle('Native static core resolution control; all cells retained\nShell mesh unchanged; no uniform-resolution convergence order')
  fig.savefig(out/'native-errors-and-regions.png',dpi=145);plt.close(fig)
 summary['input_hashes']=hashes;summary['inputs_unchanged']=all(sha(Path(p))==v for p,v in hashes.items());assert summary['inputs_unchanged'];summary['script_sha256']=sha(Path(__file__));summary['command']=sys.argv;summary['seconds']=time.monotonic()-start
 (out/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:{'H_rms':v['regions']['all']['diagnostics']['H']['penrose_volume_rms'],'beta_r_error':v['physical_errors']['beta_r']['penrose_volume_rms'],'alpha_error':v['physical_errors']['alpha']['penrose_volume_rms'],'minimum_radius':v['minimum_radius']} for k,v in summary['cases'].items()}))

if __name__=='__main__':main()
