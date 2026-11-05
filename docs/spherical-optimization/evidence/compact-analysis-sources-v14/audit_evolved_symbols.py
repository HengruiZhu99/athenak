#!/usr/bin/env python3
"""Full 26-field radial/transverse/oblique point symbols on saved evolved data."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from kernel import Kernel
from audit import symbol_audit

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--stride',type=int,default=4);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 cfg=json.loads((a.run/'manifest.json').read_text())['config'];k=Kernel.from_config(a.library,cfg);rows=[]
 files=[a.run/'initial.npz',*sorted(a.run.glob('profile-*.npz'))]
 for p in files:
  state=np.load(p);r=state['r'];q=state['q'];index=np.unique(np.r_[np.arange(0,len(r)-1,a.stride),len(r)-2]);values=symbol_audit(k,r[index],q[index]);np.savetxt(a.output/(p.stem+'-symbols.csv'),values,delimiter=',',header='r,direction,max_abs_speed,max_imag,raw_eigenvector_condition,eigenspace_condition,defective_clusters,residual',comments='')
  rows.append(dict(file=p.name,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),time_M=float(state['t_M']) if 't_M' in state else 0.,sampled_radii=r[index].tolist(),max_speed=float(max(values[:,2])),max_imag=float(max(values[:,3])),max_eigenspace_condition=float(max(values[:,5])),defective_clusters=int(sum(values[:,6])),max_residual=float(max(values[:,7]))))
 (a.output/'audit.json').write_text(json.dumps(dict(run=str(a.run),library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),config=cfg,states=rows,qualification='Pointwise interior 26-variable algebraic-tangent symbols include transverse sectors. Sampling is not endpoint uniformity, nonspherical evolution or 3D stability proof.'),indent=2)+'\n');print(json.dumps(rows));k.close()
if __name__=='__main__':main()
