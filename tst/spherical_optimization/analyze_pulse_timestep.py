#!/usr/bin/env python3
"""Compare paired final responses at two timesteps on exactly the same grid."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--coarse',nargs=2,type=Path,required=True);ap.add_argument('--fine',nargs=2,type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);inputs={};pairs=[];configs=[]
 def read(p):
  inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text()) if p.suffix=='.json' else np.load(p)
 for folders in (a.coarse,a.fine):
  bc,pc=[read(p/'manifest.json')['config'] for p in folders]
  assert bc.get('compatible_lapse',0.)==pc.get('compatible_lapse',0.)
  assert all(bc[k]==pc[k] for k in bc if k not in ('output','perturbation'));assert bc['perturbation']=='none' and pc['perturbation']!='none'
  assert all(read(p/'result.json')['status']=='completed_unqualified' for p in folders)
  b,p=[read(x/'last-accepted.npz') for x in folders];assert np.array_equal(b['r'],p['r']) and b['t_M']==p['t_M'];pairs.append((b['r'],p['q']-b['q'],float(b['t_M'])));configs.append(bc)
 assert all(configs[0][k]==configs[1][k] for k in configs[0] if k not in ('output','fixed_dt'))
 assert configs[1]['fixed_dt']==configs[0]['fixed_dt']/2
 assert np.array_equal(pairs[0][0],pairs[1][0]) and pairs[0][2]==pairs[1][2]
 r=pairs[0][0];coarse=pairs[0][1];fine=pairs[1][1];rows=[];fields=['chi','gr','gt','Ar','At','Khat','Theta','Lambda','alpha','beta','B','D']
 for region,mask in [('all',r>0),('exterior',r>.8)]:
  c,f=coarse[mask],fine[mask];j=np.unravel_index(abs(f).argmax(),f.shape)
  rows.append(dict(region=region,coarse_max=float(abs(c).max()),fine_max=float(abs(f).max()),relative_L2_difference=float(np.linalg.norm(c-f)/np.linalg.norm(f)),max_response_difference=float(abs(c-f).max()),fine_peak_r=float(r[mask][j[0]]),fine_peak_field=fields[j[1]]))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
 for ax,indices,label in [(axes[0],list(range(12)),'All fields'),(axes[1],[8,9],'Lapse/shift')]:
  ax.semilogy(r,np.max(abs(coarse[:,indices]),axis=1),label='C .4');ax.semilogy(r,np.max(abs(fine[:,indices]),axis=1),'--',label='C .2');ax.set(xlim=(.7,1),xlabel='Compact radius',ylabel='Maximum paired response',title=label);ax.grid(alpha=.2);ax.legend()
 fig.savefig(a.output/'response-timestep.png',dpi=160);plt.close(fig)
 np.savez_compressed(a.output/'responses.npz',r=r,coarse=coarse,fine=fine,t_M=pairs[0][2])
 out=dict(rows=rows,inputs=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Same-grid final paired differences; two steps exclude leading RK sensitivity but do not measure temporal order or prove spatial stability. Raw component norm, not an energy.');(a.output/'comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
