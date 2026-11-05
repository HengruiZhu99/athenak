#!/usr/bin/env python3
"""Compare saved full-grid paired pulse responses for explicitly named policies."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pair',nargs=3,action='append',required=True,metavar=('LABEL','BASELINE','PERTURBED'));ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[];inputs={}
 def read(path):
  inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();return json.loads(path.read_text()) if path.suffix=='.json' else np.load(path)
 for label,baseline,perturbed in a.pair:
  b,p=Path(baseline),Path(perturbed);bc=read(b/'manifest.json')['config'];pc=read(p/'manifest.json')['config']
  for key in bc:
   if key not in ('output','perturbation'):assert bc[key]==pc[key],key
  assert bc['perturbation']=='none' and pc['perturbation']!='none'
  assert read(b/'result.json')['status']==read(p/'result.json')['status']=='completed_unqualified'
  for path in sorted(b.glob('profile-*.npz')):
   bs=read(path);ps=read(p/path.name);assert bs['t_M']==ps['t_M'] and np.array_equal(bs['r'],ps['r']);delta=(ps['q']-bs['q'])[bs['r']>.8]
   rows.append(dict(label=label,t_M=float(bs['t_M']),max_field_response=float(abs(delta).max()),field_response_RMS=float(np.sqrt(np.mean(delta**2))),max_lapse_shift_response=float(abs(delta[:,8:10]).max())))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
 for label,_,_ in a.pair:
  v=[x for x in rows if x['label']==label]
  for ax,key in zip(axes,['max_field_response','max_lapse_shift_response']):ax.semilogy([x['t_M'] for x in v],[x[key] for x in v],'o-',label=label)
 for ax,title in zip(axes,['Raw exterior field response','Lapse/shift response']):ax.set(xlabel='t/M',ylabel='Maximum paired difference',title=title);ax.grid(alpha=.25);ax.legend()
 fig.savefig(a.output/'response.png',dpi=160);plt.close(fig)
 out=dict(rows=rows,inputs=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Paired differences on r>.8, not energy norms. Configurations match within each pair; cross-policy timesteps can differ and are disclosed in input manifests. Finite response or completion is not a stability proof.')
 (a.output/'comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([x for x in rows if x['t_M']>19.99]))
if __name__=='__main__':main()
