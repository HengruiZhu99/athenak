#!/usr/bin/env python3
"""Require same-device restart exactness and active-state ghost independence."""
import argparse,json
from pathlib import Path
import numpy as np
from compare_closure_field_norms import read
p=argparse.ArgumentParser();p.add_argument('parent',type=Path);p.add_argument('followup',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
summary=json.loads((a.followup/'summary.json').read_text())
for r in summary['cases'].values():assert r['exit_code']==0 and r['reached_step_target'] and r['reference_unchanged'] and r['all_rows_finite']
r=read(a.followup/'resumed-eight');f=read(a.followup/'fresh-eight');base=read(a.parent/'pulse-interior');ghost=read(a.followup/'ghost-degree5')
assert r['time']==f['time'] and r['cycle']==f['cycle']==8
assert (a.followup/'resumed-eight/probe_checkpoint.bin').read_bytes()==(a.followup/'fresh-eight/probe_checkpoint.bin').read_bytes()
assert base['cycle']==ghost['cycle']==4 and base['time']==ghost['time']
assert all(x==y for i,(x,y) in enumerate(zip(base['signature'],ghost['signature'])) if i!=12)
errors=[]
for x,y in zip(base['packs'],ghost['packs']):
 assert x.shape==y.shape
 errors.append(float(np.max(abs(x[:,:,4:-4,4:-4,4:-4]-y[:,:,4:-4,4:-4,4:-4]))))
assert errors==[0.,0.],errors
out=dict(same_device_restart_bitwise_exact=True,active_ghost_independence_max_errors=errors,binary_sha256=summary['binary_sha256'],scope='Short execution admission only; no convergence or long-time stability claim')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(out)
