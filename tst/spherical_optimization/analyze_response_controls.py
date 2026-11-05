#!/usr/bin/env python3
"""Same-grid final pulse controls with explicit parameter and amplitude differences."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from kernel import FIELDS

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--reference',nargs=2,type=Path,required=True);ap.add_argument('--comparison',nargs=3,action='append',required=True,metavar=('LABEL','BASELINE','PERTURBED'));ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);inputs={}
 def read(p):
  inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text()) if p.suffix=='.json' else np.load(p)
 def pair(folders):
  bc,pc=[read(p/'manifest.json')['config'] for p in folders]
  assert bc['perturbation']=='none' and pc['perturbation']!='none'
  assert all(bc[k]==pc[k] for k in bc if k not in ('output','perturbation','perturb_amplitude'))
  assert all(read(p/'result.json')['status']=='completed_unqualified' for p in folders)
  b,p=[read(x/'last-accepted.npz') for x in folders];assert np.array_equal(b['r'],p['r']) and b['t_M']==p['t_M'];return pc,b['r'],p['q']-b['q'],float(b['t_M'])
 ref,r,delta,t=pair(a.reference);rows=[];arrays=dict(r=r,reference=delta,t_M=t)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');axes[0].semilogy(r,np.maximum(np.max(abs(delta),axis=1),1e-16),label='Reference')
 for index,(label,b,p) in enumerate(a.comparison):
  cfg,rr,response,tt=pair([Path(b),Path(p)]);assert np.array_equal(r,rr) and tt==t
  changes={k:[ref.get(k),cfg.get(k)] for k in set(ref)|set(cfg) if k!='output' and ref.get(k)!=cfg.get(k)}
  assert set(changes)<=set(('fixed_dt','outer_lapse_damping','perturb_amplitude')),changes
  ratio=cfg['perturb_amplitude']/ref['perturb_amplitude'];assert ratio>0;scaled=response/ratio;difference=scaled-delta;arrays[f'comparison{index}']=scaled
  regions=[]
  for region,mask in [('all',r>0),('exterior',r>.8)]:
   d=scaled[mask];i,j=np.unravel_index(abs(d).argmax(),d.shape);regions.append(dict(region=region,max_response=float(abs(d).max()),max_response_difference=float(abs(difference[mask]).max()),relative_L2_difference=float(np.linalg.norm(difference[mask])/max(np.linalg.norm(delta[mask]),1e-300)),peak_r=float(r[mask][i]),peak_field=FIELDS[j]))
  rows.append(dict(label=label,parameter_changes=changes,amplitude_ratio=ratio,regions=regions));axes[0].semilogy(r,np.maximum(np.max(abs(scaled),axis=1),1e-16),label=label);axes[1].semilogy(r,np.maximum(np.max(abs(difference),axis=1),1e-16),label=label)
 for ax in axes:ax.set(xlim=(.8,1),xlabel='Compact radius');ax.grid(alpha=.2);ax.legend(fontsize=8)
 axes[0].set(ylabel='Maximum amplitude-normalized paired response',title=f'Final response at {t:g}M');axes[1].set(ylabel='Difference from reference',title='Explicit controls')
 fig.savefig(a.output/'response-controls.png',dpi=160);plt.close(fig);np.savez_compressed(a.output/'responses.npz',**arrays)
 out=dict(reference=[str(p) for p in a.reference],rows=rows,inputs=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Same-grid paired differences, normalized to reference pulse amplitude. All cross-case parameter changes are listed. Raw component norms are not energy norms; finite-duration agreement or amplitude linearity is not all-time stability.')
 (a.output/'comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
