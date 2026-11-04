"""Read-only comparison of two N16 radial layouts; no evolution order claim."""
from pathlib import Path
import json,hashlib,csv,gzip,shutil,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CMakeLists.txt').exists());DOC=Path(__file__).resolve().parent
OUT=DOC/'reference-radial18-initial';OUT.mkdir(exist_ok=True)
RAW=ROOT/'output/refined-convergence/reference-radial18-initial';BASE=ROOT/'output/refined-convergence/initial-native-rhs'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
command=json.loads((RAW/'commands.json').read_text());assert command['rhs_returncode']==0 and command['checkpoint_unchanged']
assert sha(RAW/'state.checkpoint')==command['checkpoint_sha256']
x=json.loads((RAW/'rhs.json').read_text());base=json.loads((BASE/'n16.json').read_text());history=json.loads((RAW/'state.history.jsonl').read_text().splitlines()[0])
assert x['initial_reconstructed'] and x['steps_taken']==0
checks={k:abs(x[k]-v)/(1+abs(v)) for k,v in dict(full_H=history['H'],full_M=history['M'],cut03_H=history['fixed'][1]['H'],cut03_M=history['fixed'][1]['M']).items()}
assert max(checks.values())<1e-11
components={'chi':[(0,1)],'metric':[(f,2 if f in (2,3,5) else 1) for f in range(1,7)],'A':[(f,2 if f in (8,9,11) else 1) for f in range(7,13)],'K':[(13,1)],'Lambda':[(f,1) for f in range(15,18)],'alpha':[(18,1)],'D':[(f,1) for f in range(25,28)]}
def rate(data,fields):
 bands=[b for b in data['regions'] if b['lo']>=.9 and b['hi']<=.95];assert len(bands)==2
 return math.sqrt(sum(b['volume']*sum(w*b['modes'][0]['rms'][f]**2 for f,w in fields) for b in bands)/sum(b['volume'] for b in bands))
rates={name:dict(radial12=rate(base,fields),radial18=rate(x,fields),reduction_factor=rate(base,fields)/rate(x,fields)) for name,fields in components.items()}
norms={k:dict(radial12=base[k],radial18=x[k],reduction_factor=base[k]/x[k]) for k in ('cut03_H','cut03_M')}
files=[RAW/n for n in ('commands.json','checkpoint-verification.json','rhs.json','rhs.log','initial.log','state.history.jsonl','state.report.json')]
files+=list(RAW.glob('rhs.rank*.csv'))
for p in files:
 if p.suffix=='.csv':
  with gzip.open(OUT/(p.name+'.gz'),'wb') as f:f.write(p.read_bytes())
 else:shutil.copy2(p,OUT/p.name)
shutil.copy2(ROOT/'output/refined-convergence/initial-native-rhs-probe-dt/build.json',OUT/'analysis-build.json')
result=dict(checkpoint_sha256=command['checkpoint_sha256'],initial_norm_match=checks,rates_band_09_095=rates,initial_fixed03_constraints=norms,inputs={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},qualification='N16 core/angles/initial geometry and gauge unchanged; only radial .90-.95 spacing reduced fourfold. Final .95-1 scri shell unchanged. Zero RK steps. Native analysis probe dt .00003125 vs baseline .000125; the physical bulk RHS in this band does not depend on the boundary timestep penalty, but this is not an evolved matched-timestep comparison. Baseline body-v1 and new checkpoint-dt body separately preserved. Original checkpoint hash unchanged. Regional reductions do not establish convergence order or long-time regularity.')
(OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for prefix,label,color in ((BASE/'n16','Original radial12','tab:blue'),(RAW/'rhs','Radially refined18','tab:orange')):
 rows=[r for p in prefix.parent.glob(prefix.name+'.rank*.csv') for r in csv.DictReader(p.open()) if r['mode']=='0' and int(r['j'])==8 and .875<float(r['r'])<.95];rows.sort(key=lambda r:float(r['r']))
 for ax,field,title in zip(axes.flat,('chi_rate','metric_rr_rate','K_rate','Lambda_r_rate'),('chi','radial conformal metric','K','radial Lambda')):
  ax.semilogy([float(r['r']) for r in rows],[abs(float(r[field])) or np.nan for r in rows],'.-',label=label,color=color,ms=3);ax.set(title=f'Initial |d_t {title}|',xlabel='Compact r',ylabel='Absolute native RHS');ax.axvspan(.9,.95,color='gray',alpha=.08);ax.grid(alpha=.2)
axes[0,0].legend();fig.suptitle('Initial native RHS: radial refinement only; face-center samples, no smoothing')
fig.savefig(OUT/'radial-control.png',dpi=170);fig.savefig(OUT/'radial-control.pdf');plt.close(fig)
print(json.dumps(dict(rates=rates,norms=norms),indent=2))
