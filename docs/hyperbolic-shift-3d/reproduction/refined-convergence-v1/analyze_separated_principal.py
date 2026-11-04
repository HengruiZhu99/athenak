"""Bounded initial full-symbol comparison; not a global hyperbolicity proof."""
from pathlib import Path
import gzip,json,hashlib,shutil
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=next(p for p in HERE.parents if (p/'CMakeLists.txt').exists());RAW=ROOT/'output/refined-convergence/principal-probe';OUT=HERE/'separated-shift-control';OUT.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True);result={}
for ax,name in zip(axes,['central6_stationary_matched_plateau','central20_stationary_smr025_radial9']):
 path=RAW/(name+'-separated.jsonl');rows=[json.loads(x) for x in path.read_text().splitlines()];base=[q for q in rows if q['override']==0];new=[q for q in rows if q['override']==1];assert len(base)==len(new)==361
 gapmin=min(q['gap'] for q in new);assert gapmin>0
 imag=0.;leak=0.;outer=0.;speed_error=0.
 for b,q in zip(base,new):
  P=np.array(q['P']);g=np.array(q['inverse_metric']).reshape(3,3);c=np.array([g[0,0],2*g[0,1],2*g[0,2],g[1,1],2*g[1,2],g[2,2]])
  C=np.zeros((2,28));C[0,1:7]=c;C[1,7:13]=c;Q=np.linalg.svd(C,full_matrices=True)[2][2:].T
  A=np.einsum("ia,ij,jb->ab",Q,P,Q);leak=max(leak,float(np.max(abs(np.einsum("ai,ij,jb->ab",C,P,Q)))));ev=np.linalg.eigvals(A);imag=max(imag,float(max(abs(ev.imag))))
  for sign in (-1,1):
   for key in ('lapse_speed2','shift_speed2'):
    expected=q['beta_x']+sign*np.sqrt(q[key]);speed_error=max(speed_error,float(min(abs(ev-expected))))
  if q['inner_weight']==0:outer=max(outer,float(np.max(abs(P-np.array(b['P'])))))
 assert imag<1e-10 and leak<1e-10 and outer==0 and speed_error<1e-8
 result[name]=dict(samples=len(new),min_all_direction_squared_speed_gap=gapmin,max_eigen_imaginary=imag,max_trace_tangent_leak=leak,max_expected_gauge_eigenvalue_distance=speed_error,outer_symbol_max_change=outer,raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 for items,label in [(base,'constant k²'),(new,'separated inner driver')]:ax.plot([q['r'] for q in items],[q['gap'] for q in items],label=label)
 ax.axhline(0,color='k',lw=.6);ax.set(xlabel='compact r',ylabel='shift² − lapse² speed',title='Moderate center' if 'central6' in name else 'Enlarged center');ax.legend();ax.grid(alpha=.25)
 with gzip.open(OUT/(name+'-matrices.jsonl.gz'),'wb') as f:f.write(path.read_bytes())
result['qualification']='Radial frozen-coefficient full production symbol on stationary initial states with algebraic trace constraints projected; positive lapse/longitudinal-shift gap, real sampled spectrum and expected gauge speeds. Other repeated modes, arbitrary backgrounds, time evolution and nonlinear well-posedness are not established. No long-time accuracy claim.'
(OUT/'principal-verification.json').write_text(json.dumps(result,indent=2)+'\n');shutil.copy2(RAW/'build-reproduced-separated-1.json',OUT/'principal-build.json');fig.savefig(OUT/'speed-gaps.png',dpi=160);plt.close(fig);print(json.dumps(result,indent=2))
