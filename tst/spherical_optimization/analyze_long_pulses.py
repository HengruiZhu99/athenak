#!/usr/bin/env python3
"""Saved paired 20M responses at two timesteps; no stability inference from survival."""
import argparse,json
from pathlib import Path
import numpy as np
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[];last={}
 for c in [.5,.25]:
  baseline=a.evidence/f'chi-r2-none-20M-c{c:g}-v16';perturbed=a.evidence/f'chi-r2-gauge-20M-c{c:g}-v16'
  cfg=json.loads((baseline/'manifest.json').read_text())['config'];pcfg=json.loads((perturbed/'manifest.json').read_text())['config'];assert cfg['fixed_dt']==pcfg['fixed_dt']
  for file in sorted(baseline.glob('profile-*.npz')):
   b=np.load(file);p=np.load(perturbed/file.name);assert b['t_M']==p['t_M'] and np.array_equal(b['r'],p['r'])
   dq=p['q']-b['q'];mask=b['r']>.8;values=dq[mask]
   rows.append(dict(C_char_target=c,t_M=float(b['t_M']),exterior_max=float(np.max(abs(values))),exterior_RMS=float(np.sqrt(np.mean(values**2))),lapse_shift_max=float(np.max(abs(values[:,8:10])))))
  last[c]=dq
 diff=last[.5]-last[.25];mask=b['r']>.8
 out=dict(rows=rows,final_response_timestep_max_difference=float(np.max(abs(diff[mask]))),final_response_timestep_relative_L2_difference=float(np.linalg.norm(diff[mask])/np.linalg.norm(last[.25][mask])),qualification='Same fixed-grid paired smooth gauge pulse; raw exterior r>.8 total-field norms, dominated by D/Lambda, not an energy norm. Timestep refinement separates delayed response growth from RK error; neither completion nor a finite amplification proves stability.')
 (a.output/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
 for c in [.5,.25]:
  v=[x for x in rows if x['C_char_target']==c]
  for ax,key in zip(axes,['exterior_max','lapse_shift_max']):ax.semilogy([x['t_M'] for x in v],[x[key] for x in v],'o-',label=f'C={c:g}')
 for ax,label in zip(axes,['Maximum exterior field response','Maximum lapse/shift response']):ax.set(xlabel='t/M',ylabel=label);ax.grid(alpha=.25);ax.legend()
 fig.savefig(a.output/'growth.png',dpi=150);plt.close(fig);print(json.dumps(out))
if __name__=='__main__':main()
