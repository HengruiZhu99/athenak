#!/usr/bin/env python3
"""Evaluate the frozen varying-residual actual-mesh transfer ladder."""
import argparse
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('root',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
rows=[]
for dim in [2,3]:
 for topology in ['uniform','smr']:
  for repeat in range(3):
   residual=[];curl=[];curl_rms=[];ghost_rms=[];before=[];components=[]
   for n in [8,16,32]:
    path=args.root/f'n{n}-{topology}'/f'fd6-{dim}d'
    records=[json.loads(line) for f in path.glob('transfer-mesh-rank*.jsonl') for line in f.read_text().splitlines()]
    records=[r for r in records if r['repeat']==repeat]
    assert records and all(r['fixed_change']==0 for r in records)
    residual.append(max(r['after_reference_residual_error'] for r in records))
    before.append(max(r['before_reference_residual_error'] for r in records))
    ghost_rms.append(np.sqrt(sum(r['ghost_error_square_after'] for r in records)/sum(r['ghost_component_volume'] for r in records)))
    tensor_key='curl_tensor_error_after_max' if 'curl_tensor_error_after_max' in records[0] else 'curl_error_after_max'
    curl.append(np.max([r[tensor_key] for r in records],axis=0))
    if 'ghost_component_error_after_max' in records[0]:
     components.append(np.max([r['ghost_component_error_after_max'] for r in records],axis=0))
    curl_rms.append(np.sqrt(np.sum([r['curl_error_after_square'] for r in records],axis=0)/sum(r['active_volume'] for r in records)))
   residual=np.array(residual);curl=np.array(curl)
   rates=np.log2(residual[:-1]/residual[1:]);curl_rates=np.log2(curl[:-1]/curl[1:])
   minimum=5.5 if topology=='uniform' else (0.8 if dim==2 else 3.5)
   curl_ok=(curl_rates>=minimum) | (np.maximum(curl[:-1],curl[1:])<=2e-12)
   residual_ok=bool(np.max(residual)<=2e-12) if topology=='uniform' else bool(np.min(rates)>=(1.7 if dim==2 else 4.5))
   row=dict(dimension=dim,topology=topology,repeat=repeat,residual_max=residual.tolist(),ordinary_residual_max=before,residual_rates=rates.tolist(),ghost_rms=np.asarray(ghost_rms).tolist(),curl_max=curl.tolist(),curl_rms=np.asarray(curl_rms).tolist(),curl_rates=curl_rates.tolist(),curl_min_rate=minimum,residual_status='PASS' if residual_ok else 'FAIL',curl_status='PASS' if curl_ok.all() else 'FAIL',failed_curl_family_pairs=np.argwhere(~curl_ok).tolist(),scope='family curl maxima and aggregate ghost errors; not full amplification or evolution')
   if components:
    components=np.array(components)
    component_rates=np.log2(components[:-1]/components[1:])
    component_ok=(components<=2e-12) if topology=='uniform' else ((component_rates>=(1.7 if dim==2 else 4.5)) | (np.maximum(components[:-1],components[1:])<=2e-12))
    row['curl_tensor_max']=row.pop('curl_max')
    row['curl_tensor_rates']=row.pop('curl_rates')
    row['failed_curl_tensor_pairs']=row.pop('failed_curl_family_pairs')
    row.update(residual_component_max=components.tolist(),residual_component_rates=component_rates.tolist(),failed_residual_component_pairs=np.argwhere(~component_ok).tolist(),curl_components='family*3 + (01,02,12)')
    residual_ok=residual_ok and component_ok.all()
    row['residual_status']='PASS' if residual_ok else 'FAIL'
    row['scope']='all 33 residual and all 33 curl components; not full amplification or evolution'
   row['status']='PASS' if residual_ok and curl_ok.all() else 'FAIL'
   rows.append(row)
args.output.write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:
 print(r['dimension'],r['topology'],r['repeat'],r['status'],'residual rates',r['residual_rates'],'failed curl entries',r.get('failed_curl_tensor_pairs',r.get('failed_curl_family_pairs')))
