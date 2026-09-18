#!/usr/bin/env python3
"""Reduce audited stages to global active-cell maxima, preserving operation/time.

The input audit's regional sets overlap. Select maxima, never add their counts.
The recorded time is the step time; RK stage is a separate required coordinate.
"""
import argparse
import csv
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('files',type=Path,nargs='+')
p.add_argument('--field',action='append',default=[])
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
selected={}
for path in a.files:
    for row in csv.DictReader(path.open()):
        if row['region']=='ghost' or (a.field and row['field'] not in a.field):continue
        key=(int(row['cycle']),int(row['stage']),row['operation'],row['field'])
        if key not in selected or float(row['max_abs'])>float(selected[key]['max_abs']):
            selected[key]=row
out=[]
for row in selected.values():
    r={k:int(row[k]) for k in ['cycle','stage','rank','gid','level','i','j','k']}
    r.update({k:float(row[k]) for k in ['time','x','y','z','max_abs']})
    r.update({k:row[k] for k in ['operation','field','region']})
    r['nonzero_peak']=r['max_abs']!=0
    r['coordinate_radius_M']=(r['x']**2+r['y']**2+r['z']**2)**.5
    out.append(r)
a.output.write_text(json.dumps(out,indent=2)+'\n')
