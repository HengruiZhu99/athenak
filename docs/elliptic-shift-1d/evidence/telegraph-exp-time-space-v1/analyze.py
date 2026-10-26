#!/usr/bin/env python3
"""Provisional dt/2-to-spatial differences; dt/4 remains required."""
from pathlib import Path
import sys,json,hashlib
root=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(root/'tst/elliptic_shift_1d'))
from analyze_telegraph_time import load,controlled_inputs,differences
base=load(root/'output/elliptic-shift-1d/telegraph-exp-extension-v2/n2049')
fine=load(root/'output/elliptic-shift-1d/telegraph-exp-fine-v1/n4097')
controlled_inputs(base,fine,{'n'})
rows=[dict(t=t,**differences(base['data'][t],fine['data'][t])) for t in sorted(base['data'])]
p=root/'docs/elliptic-shift-1d/evidence/telegraph-exp-half-time-v1/review.json'
time=json.loads(p.read_text())['maxima'];metrics={}
for key in time:
 space=max(row[key] for row in rows)
 fraction=time[key]/space if space else (0 if time[key]==0 else None)
 metrics[key]=dict(half_step_difference=time[key],spatial_difference=space,ratio=fraction)
result=dict(metrics=metrics,half_step_review=str(p),half_step_review_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
 spatial_ancestry=[base['ancestry'],fine['ancestry']],required_triplet_complete=False,goal_status='incomplete',
 scope='Complete exact t=0..10 trajectories. Raw dt-to-dt/2 differences at N2049 compared with N2049-to-N4097 fixed-CFL spatial differences. These are diagnostic ratios, not certified error bounds. The independent dt/4 run remains required.')
p=Path(__file__).with_name('review.json')
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(metrics['ray_logJ_max']))
