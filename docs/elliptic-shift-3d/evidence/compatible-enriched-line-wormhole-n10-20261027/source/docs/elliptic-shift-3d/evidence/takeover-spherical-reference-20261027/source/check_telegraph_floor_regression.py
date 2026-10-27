#!/usr/bin/env python3
"""Offline regression at the recorded lapse-underflow failure checkpoint."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from telegraph import TelegraphEngine,options
from puncture_chart import PunctureChart
from wormhole_isotropic import IsotropicWormhole
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--fast-library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();original=sha(a.checkpoint)
with np.load(a.checkpoint,allow_pickle=False) as z:q=z['q'].copy();B=z['B'].copy();rays=z['rays'].copy();comp=z['compensation'].copy();t=float(z['t']);m=json.loads(str(z['metadata']))
v=m['arguments'];opts=options(v['n'],a.library,v['nu'],v['cfl'])
for k in ['width','ko_order','quadrature_width','puncture_ko','puncture_ko_radius','grid_stretch','grid_map','mapped_ko']:setattr(opts,k,v[k])
ev=PunctureChart(opts);ev.args.mass=v['mass'];ev.initial_wormhole(IsotropicWormhole(v['mass'],ev.args.initial_quadrature_order));dt=m['time_lattice']['dt'];records=[]
for floor,steps in [(0.,1),(1e-10,round(1/dt))]:
 e=TelegraphEngine(ev,a.fast_library,v['kappa'],v['tau'],v['inner'],v['outer'],v['transition'],len(rays),v['ray_start'],v['inner_profile'],v['ray_interpolation'],v['ray_exit'],floor);e.compensation(comp)
 cq=q.copy();cb=B.copy();cr=rays.copy();done,error=e.advance(cq,cb,cr,steps,dt,t)
 rec=dict(lapse_floor=floor,requested_steps=steps,completed_steps=done,error=error,time=t+done*dt,diagnostics=e.diagnostics())
 if floor==0:assert done==0 and error=='g=1 gauge or positivity violation'
 else:
  assert done==steps and error is None,error
  row,b,diag=ev.report(cq,t+done*dt);outside=ev.r>=row['apparent_horizon_r'];w=ev.r**2*ev.grid_jacobian
  rec.update(relative_AH_mass_error=abs(row['apparent_horizon_mass']-v['mass'])/v['mass'],exterior_RMS=dict(zip(['H','M','Theta','Z'],map(float,np.sqrt(np.sum(w[outside,None]*diag[outside,:4]**2,axis=0)/sum(w[outside]))))))
  np.savez_compressed(a.output/'repaired-state.npz',q=cq,B=cb,rays=cr,r=ev.r,t=t+done*dt,compensation=e.compensation())
 records.append(rec);e.close()
assert sha(a.checkpoint)==original
result=dict(parent_checkpoint=str(a.checkpoint.resolve()),parent_checkpoint_sha256=original,new_library_sha256=sha(a.fast_library),geometry_library_sha256=sha(a.library),records=records,scope='Independent offline copies; native floor=0 reproduces the original rejection, while floor=1e-10 is checked for one additional time unit. No claim of long-time stability.')
(a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
