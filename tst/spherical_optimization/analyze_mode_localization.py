#!/usr/bin/env python3
"""Locate saved leading frozen-operator modes; norms are diagnostic, not energy."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--spectra',nargs='+',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[]
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
 for folder in a.spectra:
  audit=json.loads((folder/'audit.json').read_text());z=np.load(folder/'linearization.npz');mode=abs(z['leading_mode']);i,j=np.unravel_index(mode.argmax(),mode.shape);scaled=mode/z['scales'];norm=np.sqrt(np.sum(scaled**2,axis=1));r=z['r']
  row=dict(spectrum=str(folder),n=audit['n'],eigenvalue=audit['leading_eigenvalues'][0],physical_component_peak_r=float(r[i]),physical_component_peak_independent_slot=int(j),scaled_mode_peak_r=float(r[norm.argmax()]),raw_exterior_fraction=float(np.sum(mode[r>.8]**2)/np.sum(mode**2)),scaled_exterior_fraction=float(np.sum(scaled[r>.8]**2)/np.sum(scaled**2)),source_sha256=hashlib.sha256((folder/'linearization.npz').read_bytes()).hexdigest());rows.append(row)
  ax.semilogy(r,np.maximum(norm/norm.max(),1e-12),label=f"N{audit['n']}, Re={row['eigenvalue'][0]:.4g}, Im={row['eigenvalue'][1]:.4g}")
 cfg=audit['run_configuration'];initial=np.load(Path(cfg['output'])/'initial.npz');window=initial['configuration'][7:9];ax.axvspan(*window,color='grey',alpha=.15,label='inner/outer gauge blend')
 ax.set(xlabel='Compact radius',ylabel='Saved-scale mode magnitude / maximum',ylim=(1e-9,2),title='Frozen 10M modes: interior transition localization');ax.grid(alpha=.25);ax.legend(fontsize=8);fig.savefig(a.output/'modes.png',dpi=160);plt.close(fig)
 out=dict(rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Right eigenmodes of frozen time-dependent discrete operators; raw/scaled fractions are disclosed Euclidean norms, not physical energies. Resolution-dependent eigenvalues are evidence to investigate, not by themselves a continuum instability proof.')
 (a.output/'localization.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
