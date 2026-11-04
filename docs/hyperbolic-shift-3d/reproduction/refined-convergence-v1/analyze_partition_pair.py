"""Preserve the admitted GPU pair and compare actual stepping costs."""
from pathlib import Path
import hashlib
import json
import shutil
import gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D=Path(__file__).resolve().parent
R=next(p for p in D.parents if (p/'CMakeLists.txt').exists())
P=R/'output/refined-convergence/stationary-reference-v2-aurora/campaigns/stationary20-radial18-ownership-pair-v1'
O=D/'gpu-partition-comparison';O.mkdir(exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
plan=json.loads((P/'plan.json').read_text())
terminal=json.loads((P/'terminal-verification.json').read_text())
release=json.loads((P/'allocation-released.json').read_text())
worker=json.loads((P/'output/worker.json').read_text())
assert terminal['worker_status']=='completed' and terminal['scheduler_exit']==0
assert release['terminal_verification_sha256']==sha(P/'terminal-verification.json')
assert worker['plan_sha256']==sha(P/'plan.json')
for name,v in worker['artifacts'].items():
    f=P/'output'/name;assert f.stat().st_size==v['bytes'] and sha(f)==v['sha256'],name
verification=dict(all_artifacts_locally_verified=True,count=len(worker['artifacts']),
                  plan_sha256=sha(P/'plan.json'),artifacts=worker['artifacts'])
(O/'local-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
for f in P.iterdir():
    if f.is_file():shutil.copy2(f,O/f.name)
for f in (P/'output').rglob('*'):
    if not f.is_file() or f.name.endswith('.checkpoint'):continue
    dest=O/'output'/f.relative_to(P/'output');dest.parent.mkdir(exist_ok=True,parents=True)
    if f.name.endswith('.samples.jsonl'):
        with f.open('rb') as src,gzip.open(str(dest)+'.gz','wb') as dst:shutil.copyfileobj(src,dst)
    else:shutil.copy2(f,dest)
summary=dict(job=terminal['job_id'],checkpoint_byte_identical=terminal['parity_passed'],
             measured_steps=plan['measured_steps'],cases={})
for name in ('equal','weighted'):
    F=P/'output'/name
    perf=[json.loads((F/f'state.rank{r}.performance.json').read_text()) for r in range(48)]
    generations=[json.loads((F/f'state.rank{r}.generations.jsonl').read_text().splitlines()[-1]) for r in range(48)]
    assert all(p['status']=='completed' and p['exclusive_phases']['step']['calls']==plan['measured_steps'] for p in perf)
    phases={}
    for key in perf[0]['exclusive_phases']:
        values=np.array([p['exclusive_phases'][key]['seconds'] for p in perf])
        phases[key]=dict(min=float(values.min()),median=float(np.median(values)),max=float(values.max()))
    nested={}
    for key in ('rhs','halo','closure','exchange','error_sync_overlapping'):
        values=np.array([g['nested_seconds'][key] for g in generations])
        nested[key]=dict(median=float(np.median(values)),max=float(values.max()))
    summary['cases'][name]=dict(exclusive_phase_seconds=phases,nested_seconds=nested,
        step_seconds_by_rank=[p['exclusive_phases']['step']['seconds'] for p in perf],
        active_by_rank=[g['active'] for g in generations],
        halo_rows_by_rank=[g['halo_rows'] for g in generations],
        nominal_send_bytes_per_sparse_call=sum(g['sparse_send_bytes_per_call'] for g in generations),
        max_host_RSS_bytes=max(p['peak_rss_bytes'] for p in perf),
        checkpoint_sha256=sha(F/'state.checkpoint'))
a=summary['cases']['equal']['exclusive_phase_seconds']['step']
b=summary['cases']['weighted']['exclusive_phase_seconds']['step']
summary['step_speedup_equal_over_weighted']={key:a[key]/b[key] for key in ('median','max')}
summary['adopted_for_physics']=False
summary['qualification']='Single ordered pair on the same four-node allocation, same admitted restart and256 steps. Host exclusive stepping timers are separate from setup/diagnostics/checkpoint I/O. Nested timers overlap: do not add them. Nominal sparse bytes are application payload, not measured network bandwidth. This is not strong scaling or a universal speedup; warm ordering and short duration remain limitations. Do not adopt a differing checkpoint without explicit physical comparison.'
(O/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for name in ('equal','weighted'):
    s=summary['cases'][name]
    axes[0].plot(range(48),np.array(s['step_seconds_by_rank'])/256,'.-',label=name)
    axes[1].plot(range(48),s['halo_rows_by_rank'],'.-',label=name)
axes[0].set(xlabel='MPI rank',ylabel='Host stepping seconds per step')
axes[0].set_ylim(0,1.15*max(a['max'],b['max'])/256)
axes[0].set_title(f"Weighted median cost change: {100*(b['median']/a['median']-1):+.2f}%")
axes[1].set(xlabel='MPI rank',ylabel='Halo rows')
for ax in axes:ax.grid(alpha=.2);ax.legend()
fig.suptitle('Matched four-node ownership pair; same physical state, no scaling claim')
fig.savefig(O/'stepping-and-halos.png',dpi=170);fig.savefig(O/'stepping-and-halos.pdf');plt.close(fig)
print(json.dumps(summary,indent=2))
