"""Compare native constraints with independent finite-difference boundary jets.

No state or trajectory is altered. This detects whether the independent
boundary reductions could hide error in the reported exterior constraint norm.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--supplement',type=Path);a=p.parse_args()
s=json.loads((a.campaign/'campaign.json').read_text());assert s['active_child'] is None
sys.path.insert(0,str(a.campaign.resolve()/'source/tst/nonadvective_shift_1d'))
from candidate import NonadvectiveEvolution
rows=[]
events=s['finished'].copy()
if a.supplement:
 supplement=json.loads(a.supplement.read_text());assert supplement['status']=='completed';events+=supplement['finished']
for event in events:
 case=a.campaign/event['name'];m=json.loads((case/'manifest.json').read_text());cfg=m['configuration']
 e=NonadvectiveEvolution(library=a.campaign.resolve()/'source/libspherical_z4c.dylib',**cfg)
 with np.load(case/'profiles.npz') as z:
  frames=list(zip(z['t'],z['q']))
  if not any(abs(float(t)-float(z['last_time']))<1e-13 for t,_ in frames):frames.append((float(z['last_time']),z['last'].copy()))
  for time,q in frames:
   x=e.geometry(q,True);fd=q.copy();fd[-1,10:13]-=x['reduction'];y=e.geometry(fd,True)
   difference=np.max(abs(x['diag']-y['diag']),axis=0)
   if cfg['puncture']:
    from horizon import apparent_horizons
    ah=apparent_horizons(e.r[1:-1],x['chi'][1:-1]*np.sqrt(x['g'][1:-1]),x['diag'][1:-1,11]);radius=ah.get('apparent_horizon_r',np.inf)
   else:radius=0.
   weights=e.r**2*(e.jac if cfg['puncture'] else 1);weights[[0,-1]]*=.5;ext=e.r>radius
   norm=lambda diag:{field:float(np.sqrt(np.sum(weights[ext]*diag[ext,i]**2)/sum(weights[ext]))) for i,field in enumerate(['H','M','Theta','Z'])}
   rows.append(dict(case=event['name'],t=float(time),fd_exterior_norms=norm(y['diag']),used_jet_exterior_norms=norm(x['diag']),maximum_constraint_diagnostic_difference=difference[:4].tolist(),endpoint_reduction_defect=x['reduction'].tolist()))
result=dict(rows=rows,max_constraint_difference=np.max([r['maximum_constraint_diagnostic_difference'] for r in rows],axis=0).tolist(),scope='All saved profiles and last accepted states; compare evolution-used derivative jets against independently reconstructed finite-difference jets in the native constraint diagnostic. No evolution, masking, or Lambda replacement.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
