#!/usr/bin/env python3
"""Independent full-domain component norms from uniform periodic intrinsic restarts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.dont_write_bytecode=True
from intrinsic_restart import read_restart,global_state
from intrinsic_diagnostics import diagnostics,norms
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',type=Path,required=True,help='directory of synchronized restart files')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
records=[]
for path in sorted(a.input.glob('*.rst')):
    data=read_restart(path);u=global_state(data)
    order=int(data['header']['pc_gh']['spatial_order'])
    fields=diagnostics(u,data['domain'][6:9],order)
    assert all(np.isfinite(v).all() for v in fields.values())
    volume=np.prod(data['domain'][3:6]-data['domain'][:3])
    record=dict(file=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        time=data['time'],cycle=data['cycle'],blocks=len(data['locations']),spatial_order=order,
        spacing=data['domain'][6:9].tolist(),norms=norms(fields,volume))
    np.savez(a.output/(path.stem+'.npz'),**fields)
    records.append(record)
assert records
(a.output/'results.json').write_text(json.dumps(dict(scope='independent offline global-periodic primary H/M and all reductions/curls; not physical evolution qualification',mask='full active domain, no excision',records=records),indent=2)+'\n')
