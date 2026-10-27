#!/usr/bin/env python3
"""Isolated method-change probe from a saved profile; not an exact restart."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from telegraph import TelegraphEngine,options
from puncture_chart import PunctureChart
from wormhole_isotropic import IsotropicWormhole
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--profile',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--fast-library',type=Path,required=True);p.add_argument('--mode',choices=['log','total'],required=True);p.add_argument('--duration',type=float,default=2.);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
m=json.loads((a.case/'manifest.json').read_text());v=m['arguments'];opts=options(v['n'],a.library,v['nu'],v['cfl'])
for k in ['width','ko_order','quadrature_width','puncture_ko','puncture_ko_radius','grid_stretch','grid_map','mapped_ko']:setattr(opts,k,v[k])
ev=PunctureChart(opts);ev.args.mass=v['mass'];ev.initial_wormhole(IsotropicWormhole(v['mass'],ev.args.initial_quadrature_order))
with np.load(a.profile,allow_pickle=False) as z:q=z['q'].copy();B=z['B'].copy();rays=z['rays'].copy();t=float(z['t'])
e=TelegraphEngine(ev,a.fast_library,v['kappa'],v['tau'],v['inner'],v['outer'],v['transition'],len(rays),v['ray_start'],v['inner_profile'],v['ray_interpolation'],v['ray_exit'],v['lapse_floor'],a.mode)
dt=m['time_lattice']['dt'];steps=round(a.duration/dt);done_total=0;rows=[];error=None;start=time.perf_counter()
while done_total<steps:
 count=min(1024,steps-done_total);done,error=e.advance(q,B,rays,count,dt,t+done_total*dt);done_total+=done
 if error:break
 row,b,diag=ev.report(q,t+done_total*dt);mask=ev.r>=row['apparent_horizon_r'];w=ev.r**2*ev.grid_jacobian;rows.append(dict(t=row['t'],relative_AH_error=abs(row['apparent_horizon_mass']-v['mass'])/v['mass'],exterior_RMS=list(map(float,np.sqrt(np.sum(w[mask,None]*diag[mask,:4]**2,axis=0)/sum(w[mask]))))))
result=dict(mode=a.mode,initial_time=t,time=t+done_total*dt,completed_steps=done_total,requested_steps=steps,error=error,runtime_seconds=time.perf_counter()-start,diagnostics=e.diagnostics(),rows=rows,profile_sha256=hashlib.sha256(a.profile.read_bytes()).hexdigest(),library_sha256=hashlib.sha256(a.fast_library.read_bytes()).hexdigest(),scope='Isolated profile-based method probe with zeroed compensation. Not an exact restart; original run is untouched.');e.close();(a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(a.output/'state.npz',q=q,B=B,rays=rays,t=result['time']);print(json.dumps({k:result[k] for k in ['mode','time','error','runtime_seconds']}),flush=True)
