#!/usr/bin/env python3
"""Interim two-timestep evidence; the required dt/4 control remains pending."""
from pathlib import Path
import sys,json,hashlib
root=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(root/'tst/elliptic_shift_1d'))
from analyze_telegraph_time import load,controlled_inputs,differences
base=load(root/'output/elliptic-shift-1d/telegraph-scri-pchip-convergence-v1/n2049')
half=load(root/'output/elliptic-shift-1d/telegraph-scri-pchip-time-v1/dt-half')
controlled_inputs(base,half,{'cfl'})
assert base['manifest']['time_lattice']['dt']==2*half['manifest']['time_lattice']['dt']
rows=[dict(t=t,**differences(base['data'][t],half['data'][t])) for t in sorted(base['data'])]
result=dict(records=[{k:v for k,v in x.items() if k!='data'} for x in [base,half]],history=rows,
 maxima={key:max(row[key] for row in rows) for key in rows[0] if key!='t'},
 required_triplet_complete=False,goal_status='incomplete',
 scope='Completed 2049-point dt and dt/2 trajectories through10, exact common times and numerical source/library bindings verified. Differences alone do not establish temporal order or the required temporal-to-spatial error ratio. The dt/4 and finer spatial runs remain pending.')
p=Path(__file__).with_name('review.json')
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['maxima'],indent=2))
