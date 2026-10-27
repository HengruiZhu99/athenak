#!/usr/bin/env python3
"""Frozen spherical initial field sampled on concentric shells, not solved."""
import argparse,json
from pathlib import Path
import numpy as np
from audit_endpoint_layouts import run
from concentric_sbp import ConcentricGrid

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise')
rows=[]
for n in [10,12]:
    result=run(layouts=[(0.,.4,.55,.7,1.)],factory=ConcentricGrid,n=n)
    for row in result['rows']:
        row['physical_radial_knots']=row.pop('radial_partition_s');rows.append(row)
result.update(scope=__doc__,rows=rows)
(a.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
