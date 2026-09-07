#!/usr/bin/env python3
"""Compare initial analytic interpolation differences to evolved differences."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import analyze_intrinsic_smr_evolution as mod
from intrinsic_restart import read_restart
p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();results={}
for mode in ['none','residual_shifted']:
 ds=[read_restart(sorted((a.runs/f'n{n}-{mode}').glob('rst/*.rst'))[0],allow_refinement=True) for n in [8,16,32]];u=[mod.active(d) for d in ds];stats={}
 for points,name in [(6,'analysis'),(10,'analysis-tenpoint')]:
  mod.NPOINTS=points;initial=mod.norm(u[0]-mod.down(u[1]),mod.weights(ds[0]));final=np.load(a.runs/name/(mode+'-signed.npz'))['coarse_difference'];fn=mod.norm(final,mod.weights(ds[0]));stats[str(points)]={}
  for group,s in [('primary',slice(0,20)),('all50',slice(None))]:
   numerator=float(np.sqrt(np.sum(initial[s]**2)));denominator=float(np.sqrt(np.sum(fn[s]**2)));stats[str(points)][group]=dict(initial_interpolation_RMS=numerator,final_difference_RMS=denominator,initial_to_final_ratio=numerator/denominator)
 results[mode]=stats
with a.output.open('x') as f:json.dump(results,f,indent=2);f.write('\n')
