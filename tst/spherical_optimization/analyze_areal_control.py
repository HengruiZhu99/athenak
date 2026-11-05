#!/usr/bin/env python3
"""Temporal contamination and unsmoothed localization of the areal control."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from physical_constraints import evaluate

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,required=True);ap.add_argument('--halfstep',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);inputs={};rows=[];profiles=[]
 def load(path):
  inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();return json.loads(path.read_text()) if path.suffix=='.json' else np.load(path)
 for folder in [a.runs/f'areal-n{n}-repeat1' for n in (128,256,512)]+[a.halfstep]:
  manifest=load(folder/'manifest.json');cfg=manifest['config'];initial=load(folder/'initial.npz');last=load(sorted(folder.glob('profile-*.npz'))[-1]);reference=load(folder/'reference-jets.npz')
  r=initial['r'];om=initial['omega'];outer=r[:-1]>initial['configuration'][3]/6;R=3*r[:-1][outer]/(om[:-1,0][outer]*np.sqrt(initial['q'][:-1,0][outer]));sample=np.geomspace(25,100,301)
  def constraints(q,d,dd):
   v=evaluate(r,q,d,dd,om);v[:,:2]/=9;v[:,2:]/=3;return v[outer]
  physical=constraints(last['q'],last['d'],last['dd']);initial_phys=constraints(initial['q'],reference['d'],reference['dd']);values=PchipInterpolator(R,physical)(sample);mask=(R>=25)&(R<=100)
  row=dict(case=folder.name,n=cfg['n'],physical_rms=np.sqrt(np.mean(values**2,axis=0)).tolist(),initial_physical_rms=np.sqrt(np.mean(PchipInterpolator(R,initial_phys)(sample)**2,axis=0)).tolist(),peak_H_reference_R_M=float(R[mask][abs(physical[mask,0]).argmax()]),peak_abs_H=float(abs(physical[mask,0]).max()))
  rows.append(row);profiles.append((folder.name,R[mask],physical[mask,0]))
  if cfg['n']==256:
   if folder==a.halfstep:half=last;hc=cfg;hv=values
   else:full=last;fc=cfg;fv=values
 assert np.array_equal(half['r'],full['r']) and float(half['t_M'])==float(full['t_M'])==10
 for key in fc:
  if key not in ('fixed_dt','output'):assert fc[key]==hc[key],key
 assert fc['fixed_dt']==2*hc['fixed_dt']
 mask=(full['r']>.65)&(full['r']<.995);delta=half['q']-full['q'];time=dict(max_field_difference=float(abs(delta[mask]).max()),per_field_max_difference=abs(delta[mask]).max(axis=0).tolist(),constraint_difference_rms=np.sqrt(np.mean((hv-fv)**2,axis=0)).tolist(),constraint_relative_L2_difference=float(np.linalg.norm(hv-fv)/np.linalg.norm(hv)))
 out=dict(rows=rows,halfstep_comparison=time,inputs=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Two timesteps bound the observed temporal change, not an order estimate. No spatial-convergence claim for areal slicing; unsmoothed physical constraints on common reference coordinates, no constraint reset.')
 (a.output/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
 for name,R,H in profiles:ax.plot(R,H,'o-',ms=2,label=name)
 ax.set(xlabel='Reference R/M',ylabel='Dimensionless physical H',title='Areal slicing at 10M: unsmoothed samples');ax.grid(alpha=.25);ax.legend(fontsize=8);fig.savefig(a.output/'areal-constraint.png',dpi=160);plt.close(fig)
 print(json.dumps(dict(rows=rows,halfstep_comparison=time)))
if __name__=='__main__':main()
