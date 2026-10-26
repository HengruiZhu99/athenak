#!/usr/bin/env python3
"""Resolve closure component differences against both closures' own time controls."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('closure',type=Path);p.add_argument('alternative_time',type=Path);p.add_argument('baseline_time',type=Path);p.add_argument('output',type=Path);a=p.parse_args();c,h,q=[json.loads(p.read_text()) for p in [a.closure,a.alternative_time,a.baseline_time]];assert c['time']==h['time']==q['time'];checks=[]
for region,data in c['regions'].items():
 for x,y,z in zip(data['components'],h['regions'][region]['components'],q['regions'][region]['components']):
  assert x['component']==y['component']==z['component']
  for f in ['l2','linf']:checks.append(dict(region=region,component=x['component'],norm=f,closure_difference=x[f],alternative_time_difference=y[f],baseline_time_difference=z[f],resolved_at_10_percent=max(y[f],z[f])<=.1*x[f]))
r=dict(checks=checks,resolved_count=sum(x['resolved_at_10_percent'] for x in checks),unresolved_count=sum(not x['resolved_at_10_percent'] for x in checks),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.closure,a.alternative_time,a.baseline_time]},scope='Matched middle-grid component coordinate-volume differences; both own half-CFL controls include restart milestones. Not invariant constraint norms or cross-grid pointwise convergence. No diagnostic floor or region exclusion.')
a.output.write_text(json.dumps(r,indent=2)+'\n');print(r['resolved_count'],r['unresolved_count'])
