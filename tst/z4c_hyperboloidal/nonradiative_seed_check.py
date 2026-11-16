#!/usr/bin/env python3
"""Verify exact physical constraint-data preservation, including persisted fields."""
import argparse,json,re
from pathlib import Path
import numpy as np
from restart_roundtrip import parse,payload
p=argparse.ArgumentParser();p.add_argument('control',type=Path);p.add_argument('gauge',type=Path);p.add_argument('output',type=Path);p.add_argument('--halfdt',type=Path);a=p.parse_args()
c,g=parse(a.control),parse(a.gauge)
for key in ('blocks','record_blocks','cycle','time','dt','atlas','topology','identity','stride'):assert c[key]==g[key],key
assert c['cycle']==0 and c['time']==0
x,y=payload(c),payload(g);changed=[]
for f in range(39):
 if not np.array_equal(x[:,:,f].view('u8'),y[:,:,f].view('u8')):changed.append(f)
assert changed==[18],changed
r=dict(changed_fields=changed,physical_constraint_data_bitwise_equal=True,gauge_auxiliaries_bitwise_equal=True,
 lapse_peak=float(np.max(abs(y[:,:,18]-x[:,:,18]))),scalar_values=int(x.size),
 constraint_argument='EvolvedConstraints depends on chi, determinant-one metric, A, P, Theta and Lambda; all are bitwise unchanged. This proves equal initial discrete H/M/Z, not zero discretization error.')
if a.halfdt:
 h=parse(a.halfdt)
 for key in ('blocks','record_blocks','cycle','time','atlas','topology','stride'):assert g[key]==h[key],key
 pattern=rb'p\.requested_dt=([^;]+);'
 full_requested=float.fromhex(re.search(pattern,g['identity']).group(1).decode())
 half_requested=float.fromhex(re.search(pattern,h['identity']).group(1).decode())
 assert full_requested==2*half_requested
 assert re.sub(pattern,b'p.requested_dt=CONTROL;',g['identity'])==re.sub(pattern,b'p.requested_dt=CONTROL;',h['identity'])
 assert g['dt']==2*h['dt']
 assert np.array_equal(y.view('u8'),payload(h).view('u8'))
 r['halfdt_all_initial_fields_bitwise_equal']=True
 r['halfdt_identity_difference']={'p.requested_dt':[full_requested,half_requested]}
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
