"""Verify preservation and compare the two fresh failed driver controls."""
from pathlib import Path
import json, hashlib, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'CMakeLists.txt').exists())
RAW=ROOT/'output/refined-convergence/frame-separation-aurora/campaigns'
OUT=HERE/'stationary20-constant-failure';OUT.mkdir(exist_ok=True)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
c=RAW/'stationary20-constant-n16-segment01';s=RAW/'stationary20-separated-n16-segment01'
w=json.loads((c/'output/worker.json').read_text());verified={}
for name,expected in w['artifacts'].items():
 p=c/'output'/name
 actual={'bytes':p.stat().st_size,'sha256':digest(p)}
 assert actual==expected,(name,actual,expected)
 verified[name]=actual
assert json.loads((c/'allocation-released.json').read_text())
(OUT/'local-verification.json').write_text(json.dumps(dict(all_artifacts_verified=True,count=len(verified),artifacts=verified),indent=2)+'\n')
for name in ['plan.json','submission.json','terminal-verification.json','terminal-qstat.txt','allocation-released.json']:
 shutil.copy2(c/name,OUT/name)
for name in ['worker.json','native.log','state.history.jsonl','state.failure.json']:
 shutil.copy2(c/'output'/name,OUT/name)
a=np.memmap(s/'output/state.failed-stage.bin',dtype='<f8',mode='r').reshape(-1,31)
b=np.memmap(c/'output/state.failed-stage.bin',dtype='<f8',mode='r').reshape(-1,31)
assert a.shape==b.shape
profiles=np.load(ROOT/'output/refined-convergence/failed-field-profiles/profiles.npy')
assert profiles.ndim==2 and profiles.shape[1]==20
q=profiles[profiles[:,0]==1];ids=q[:,1].astype(int);radius=q[:,11]
regions={}
for lo,hi in [(0,1),(.6,.78),(.78,.875),(.875,1)]:
 ix=ids[(radius>=lo)&(radius<hi)]
 x=a[ix];y=b[ix];d=abs(x-y)
 regions[f'{lo:g}-{hi:g}']=dict(cells=len(ix),max_absolute=float(d.max()),max_scaled=float((d/(1+abs(x))).max()),chi_max_absolute=float(d[:,0].max()))
perf={};hist={}
for label,path in [('constant',c),('separated',s)]:
 hist[label]=[json.loads(line) for line in (path/'output/state.history.jsonl').read_text().splitlines()]
 p=[json.loads(f.read_text()) for f in sorted((path/'output').glob('state.rank*.performance.json'))]
 perf[label]={key:dict(median_seconds=float(np.median([x['exclusive_phases'][key]['seconds'] for x in p])),min_seconds=float(min(x['exclusive_phases'][key]['seconds'] for x in p)),max_seconds=float(max(x['exclusive_phases'][key]['seconds'] for x in p))) for key in p[0]['exclusive_phases']}
fig,axs=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for label,rows in hist.items():
 t=[r['t_over_M'] for r in rows]
 for ax,key in [(axs[0,0],'H'),(axs[0,1],'M')]:
  ax.semilogy(t,[r['fixed'][2][key] for r in rows],marker='o',label=label)
 for ax,key in [(axs[1,0],'pole_max'),(axs[1,1],'horizon_trial_expansion_rms')]:
  ax.semilogy(t,[r[key] for r in rows],marker='o',label=label)
for ax,y in zip(axs.flat,['H norm, fixed r > .6','M norm, fixed r > .6','Cumulative pole maximum','Horizon expansion RMS']):
 ax.set(xlabel='t/M',ylabel=y);ax.grid(alpha=.2)
axs[0,0].legend();fig.suptitle('Matched inner-driver controls both fail at 0.463875 M\nFour recorded diagnostics; failed captures are not admitted endpoints')
fig.savefig(OUT/'matched-history.png',dpi=170);fig.savefig(OUT/'matched-history.pdf');plt.close(fig)
result=dict(artifacts_verified=len(verified),failures={label:json.loads((path/'output/state.failure.json').read_text()) for label,path in [('constant',c),('separated',s)]},nominal_shell_field_differences=regions,exclusive_performance=perf,qualification='Same source, mesh, initial geometry, outer gauge and dt; inner driver policy differs. Failed active fields only, no failed surface saved, no endpoint admission. Exclusive phases may be compared individually; nested timers must not be summed. No controlled scaling or speedup claim.')
(OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(artifacts_verified=len(verified),regions=regions,performance=perf),indent=2))
