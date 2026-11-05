#!/usr/bin/env python3
"""Reproduce discrete-symbol plots from immutable numerical diagnostics."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--local',type=Path,required=True);ap.add_argument('--global-audit',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 local=json.loads((a.local/'audit.json').read_text());curves=np.load(a.local/'symbols.npz');glob=json.loads((a.global_audit/'audit.json').read_text());operators=np.load(a.global_audit/'operators.npz');rows=[]
 fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
 for column,n in enumerate((64,128)):
  row=min([x for x in local['rows'] if x['n']==n],key=lambda x:abs(x['r']-.8 if n==64 else x['r']-.82));r=np.load(Path(row['spectrum'])/'linearization.npz')['r'];i=int(np.argmin(abs(r-row['r'])));tag=f'n{n}-i{i}';theta=curves[tag+'-theta'];band=theta>.8*np.pi
  for name in row['models']:
   v=curves[tag+'-'+name];axes[0,column].plot(theta/np.pi,v[:,0],label=name.replace('_',' '));j=np.where(band)[0][v[band,0].argmax()];rows.append(dict(n=n,r=row['r'],model=name,high_phase_max_real=float(v[j,0]),imaginary=float(v[j,1]),phase_over_pi=float(theta[j]/np.pi)))
  ax=axes[0,column];ax.set(xlabel='Grid phase / pi',ylabel='Maximum real eigenvalue',title=f"Local frozen symbol: N{n}, r={row['r']:.4f}");ax.axhline(0,color='k',lw=.5);ax.grid(alpha=.2);ax.legend(fontsize=8)
  ax=axes[1,column]
  for name in ('original','all','lapse','shift','geometry'):
   v=operators[f'n{n}-{name}-eigenvalues'];mask=(v.real>-.5)&(abs(v.imag)<70);ax.scatter(v[mask].real,v[mask].imag,s=18,alpha=.7,label=name)
  ax.set(xlabel='Real eigenvalue',ylabel='Imaginary eigenvalue',title=f'Full frozen operator: N{n}');ax.axvline(0,color='k',lw=.5);ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.savefig(a.output/'discrete-mechanism.png',dpi=170);plt.close(fig)
 inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.local/'audit.json',a.local/'symbols.npz',a.global_audit/'audit.json',a.global_audit/'operators.npz']}
 (a.output/'summary.json').write_text(json.dumps(dict(rows=rows,inputs=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
