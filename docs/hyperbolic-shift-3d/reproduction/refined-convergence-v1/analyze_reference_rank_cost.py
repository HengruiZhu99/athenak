"""Rank-level cost/traffic evidence; timers overlap and are not strong-scaling data."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent;I=D/'stationary-reference-v2-segment01/output';O=D/'reference-rank-cost';O.mkdir(exist_ok=True)
rows=[];inputs={}
for p in I.glob('state.rank*.generations.jsonl'):
 a=json.loads(p.read_text().splitlines()[-1]);assert a['event']=='final' and a['step']==12753;a['rank']=int(p.name.split('rank')[1].split('.')[0]);rows.append(a);inputs[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
rows.sort(key=lambda a:a['rank']);assert [a['rank'] for a in rows]==list(range(48))
sends=sum(a['sparse_send_bytes_per_call'] for a in rows);receives=sum(a['sparse_receive_bytes_per_call'] for a in rows);assert sends==receives
summary=lambda v:dict(min=min(v),median=float(np.median(v)),max=max(v))
r=dict(source='admitted first-segment generation records',inputs=inputs,active_cells=summary([a['active'] for a in rows]),halo_rows=summary([a['halo_rows'] for a in rows]),rhs_seconds=summary([a['nested_seconds']['rhs'] for a in rows]),halo_seconds=summary([a['nested_seconds']['halo'] for a in rows]),overlapping_error_sync_seconds=summary([a['nested_seconds']['error_sync_overlapping'] for a in rows]),aggregate_nominal_send_bytes_per_call=sends,aggregate_nominal_receive_bytes_per_call=receives,aggregate_nominal_send_bytes=sum(a['sparse_send_bytes_per_call']*a['sparse_calls'] for a in rows),qualification='Observed counts and host timings from one4-node run. Sparse byte count is application payload accounting, not measured network utilization/bandwidth. Nested phase timers and synchronization overlap; do not sum them or equate synchronization time to avoidable latency. Equal active-cell counts do not imply equal work or communication. Existing measured/repartition APIs are candidates for a later controlled performance test; no production partition or science changed.')
(O/'analysis.json').write_text(json.dumps(r,indent=2)+'\n')
with (O/'ranks.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['rank','active','overlap','halo_rows','nominal_send_bytes_per_call','rhs_seconds','halo_seconds','error_sync_overlapping_seconds','work'])
 for a in rows:w.writerow([a['rank'],a['active'],a['overlap'],a['halo_rows'],a['sparse_send_bytes_per_call'],a['nested_seconds']['rhs'],a['nested_seconds']['halo'],a['nested_seconds']['error_sync_overlapping'],json.dumps(a['work'],sort_keys=True)])
fig,axes=plt.subplots(3,1,figsize=(11,8),layout='constrained');rank=[a['rank'] for a in rows]
axes[0].plot(rank,[a['active'] for a in rows],'.-',label='owned active cells');axes[0].plot(rank,[a['halo_rows'] for a in rows],'.-',label='halo rows');axes[0].set(ylabel='Count');axes[0].legend()
for name in ('rhs','halo','error_sync_overlapping'):axes[1].plot(rank,[a['nested_seconds'][name] for a in rows],'.-',label=name)
axes[1].set(ylabel='Nested host seconds');axes[1].legend(fontsize=8)
axes[2].plot(rank,[a['sparse_send_bytes_per_call']/1e6 for a in rows],'.-',label='send');axes[2].plot(rank,[a['sparse_receive_bytes_per_call']/1e6 for a in rows],'.-',label='receive');axes[2].set(xlabel='MPI rank',ylabel='Nominal MB per sparse call');axes[2].legend()
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Reference-v2 first segment: equal cell counts, unequal local work/communication\nNested timing curves overlap; not an exclusive cost or scaling comparison');fig.savefig(O/'ranks.png',dpi=160);fig.savefig(O/'ranks.pdf');plt.close(fig)
print(json.dumps({k:v for k,v in r.items() if k!='inputs'},indent=2))
