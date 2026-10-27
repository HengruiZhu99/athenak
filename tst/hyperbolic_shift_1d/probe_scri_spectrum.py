"""Restricted outer-collar linearization; not a full-domain stability proof."""
import argparse,json,time
from pathlib import Path
import numpy as np
from puncture import PunctureEvolution
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--nodes',type=int,default=16);a=p.parse_args()
z=np.load(a.case/'profiles.npz');m=json.loads((a.case/'manifest.json').read_text());n=len(z['r']);e=PunctureEvolution(n,a.library,restoring=m['arguments']['restoring'],inner_shift=m['arguments'].get('inner_shift',0.),lapse_floor=m['arguments'].get('lapse_floor',1e-10));rows=[]
indices=np.arange((n-a.nodes)*10,n*10);indices=indices[~np.isin(indices,[(n-1)*10+6,(n-1)*10+7,(n-1)*10+9])]
for i in [0,len(z['q'])//2,len(z['q'])-1]:
 base=z['q'][i];columns=[];eps=2e-6;start=time.monotonic()
 for j in indices:
  q=base.copy();q.flat[j]+=eps;plus=e.rhs(q).ravel()[indices]
  q=base.copy();q.flat[j]-=eps;minus=e.rhs(q).ravel()[indices];columns.append((plus-minus)/(2*eps))
 J=np.asarray(columns).T;values=np.linalg.eigvals(J);k=np.argmax(values.real)
 rows.append(dict(t=float(z['t'][i]),nodes=a.nodes,r_min=float(e.r[-a.nodes]),spectral_abscissa=float(values[k].real),imaginary_part=float(values[k].imag),max_abs=float(max(abs(values))),wall_seconds=time.monotonic()-start))
result=dict(source_case=str(a.case),scope='Only outer collar perturbed; interior held fixed. Artificial inner interface and nonnormal effects prevent a full-domain stability claim.',rows=rows)
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
