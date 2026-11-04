#!/usr/bin/env python3
"""Summarize optional profiling without summing non-simultaneous label peaks."""
import argparse,pathlib,json,statistics
p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--output',required=True);a=p.parse_args();root=pathlib.Path(a.directory);groups={}
for f in root.glob('job1-components/*.allocations.json'):
 name=f.name.split('.pid')[0];x=json.loads(f.read_text());groups.setdefault(name,[]).append(x)
summary={}
for name,xs in sorted(groups.items()):
 records=[]
 for x in xs:
  invalid=[k for k,v in x['labels'].items() if v['live']>10**12 or v['peak']>10**12]
  assert all(v['live']<10**12 and v['peak']<10**12 for v in x['spaces'].values())
  ranked=sorted(x['kernels'].items(),key=lambda kv:kv[1]['seconds'],reverse=True)
  records.append({'rank':x['rank'],'spaces':x['spaces'],'excluded_invalid_live_labels':invalid,'top_kernel_intervals':dict(ranked[:12]),'radial_prepare':{k:v for k,v in x['kernels'].items() if 'radial derivatives' in k}})
 normal=[]
 pattern='overhead-p4-r*-m0.receipt.json' if name=='profile-m0' else 'transfer-p4-r*-m0.receipt.json' if name=='profile-boost-m0' else 'transfer-p4-r*-m1a.receipt.json' if name=='profile-boost-m1a' else None
 if pattern:
  normal=[json.loads(f.read_text())['seconds'] for f in (root/'gpu-job1').glob(pattern)]
 receipt=json.loads((root/'gpu-job1'/(name+'.receipt.json')).read_text())
 summary[name]={'ranks':records,'process_seconds':receipt['seconds'],'unprofiled_reference_seconds':normal,'intrusive_overhead_ratio_to_reference_median':receipt['seconds']/statistics.median(normal) if normal else None,'caveat':'Single separate profiling run; setup dominates; not an independent kernel speedup estimate. Per-space peaks cover observed Kokkos allocations only. Invalid unsigned live-label counters excluded; raw evidence unchanged.'}
pathlib.Path(a.output).write_text(json.dumps(summary,indent=2)+'\n')
