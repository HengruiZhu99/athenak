"""Summarize initial production-RHS truncation; no evolution or state reset."""
from pathlib import Path
import json,hashlib,csv,gzip,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CMakeLists.txt').exists());DOC=Path(__file__).resolve().parent
RAW=ROOT/'output/refined-convergence/initial-native-rhs';OUT=DOC/'initial-native-rhs';OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
groups={'chi':[(0,1)],'metric':[(f,2 if f in (2,3,5) else 1) for f in range(1,7)],'A':[(f,2 if f in (8,9,11) else 1) for f in range(7,13)],'K':[(13,1)],'Lambda':[(f,1) for f in range(15,18)],'alpha':[(18,1)],'D':[(f,1) for f in range(25,28)]}
inputs={};rates={};profiles={}
for n in (12,14,16):
 command=json.loads((RAW/f'n{n}-command.json').read_text());assert command['returncode']==0 and command['initial_norms_match'] and command['checkpoint_unchanged']
 x=json.loads((RAW/f'n{n}.json').read_text());assert x['initial_reconstructed'] and x['steps_taken']==0
 rates[n]={};regions=[b for b in x['regions'] if b['lo']>=.9 and b['hi']<=.95];assert len(regions)==2
 for name,components in groups.items():
  values=[]
  for m in range(3):
   values.append(float(np.sqrt(sum(b['volume']*sum(w*b['modes'][m]['rms'][f]**2 for f,w in components) for b in regions)/sum(b['volume'] for b in regions))))
  rates[n][name]=dict(native=values[0],bulk_KO_zero=values[1],bulk_KO_contribution=values[2])
 profiles[n]=[r for p in sorted(RAW.glob(f'n{n}.rank*.csv')) for r in csv.DictReader(p.open()) if r['mode']=='0']
 files=[RAW/f'n{n}-command.json',RAW/f'n{n}.json',RAW/f'n{n}.log',*sorted(RAW.glob(f'n{n}.rank*.csv'))]
 inputs[n]={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
 for p in files:
  if p.suffix=='.csv':
   with gzip.open(OUT/(p.name+'.gz'),'wb') as f:f.write(p.read_bytes())
  else:shutil.copy2(p,OUT/p.name)
for name in ('build.json','body-v1.inc'):shutil.copy2(RAW/name,OUT/name)
orders={name:dict(pair12_14=float(np.log(rates[12][name]['native']/rates[14][name]['native'])/np.log(14/12)),pair14_16=float(np.log(rates[14][name]['native']/rates[16][name]['native'])/np.log(16/14)),fit=float(-np.polyfit(np.log([12,14,16]),np.log([rates[n][name]['native'] for n in (12,14,16)]),1)[0])) for name in groups}
angular={}
for n,rows in profiles.items():
 angular[n]={f:{ray:max(abs(float(r[f])) for r in rows if .9<float(r['r'])<.925 and (int(r['j'])==n//2)==(ray=='center')) for ray in ('center','corner')} for f in ('chi_rate','metric_rr_rate','K_rate','Lambda_r_rate','alpha_rate')}
result=dict(inputs=inputs,band=[.9,.95],rates=rates,orders=orders,center_corner_maxima_09_0925=angular,qualification='Actual native bulk RHS on analytically reconstructed initial geometry/gauge and checkpoint topology. Initial full/fixed native norms reproduced and original CP hashes unchanged. All rates should vanish in the continuum stationary outer region; this is a residual of the semidiscrete operator, not a measured evolved error. Removing bulk KO leaves the .9-.95 band unchanged (the original taper is already zero there). Scri closure/angular dissipation/halos retained in both analysis modes. N12/N14 Serial MPI1, N16 Serial MPI2; no GPU parity or evolved convergence claim. Baseline analysis body-v1 uses dt=.000125*16/N, equal to each input CP dt. New radial-control reader reads its separately admitted dt instead; campaign unchanged.')
(OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for ax,field,label in zip(axes.flat,('chi_rate','metric_rr_rate','K_rate','Lambda_r_rate'),('chi','radial conformal metric','K','radial Lambda')):
 for n,color in zip((12,14,16),('tab:blue','tab:orange','tab:green')):
  for center,style in ((True,'-'),(False,':')):
   rows=sorted([r for r in profiles[n] if (int(r['j'])==n//2)==center and float(r['r'])>.825],key=lambda r:float(r['r']))
   y=np.array([abs(float(r[field])) for r in rows]);y[y==0]=np.nan
   ax.semilogy([float(r['r']) for r in rows],y,style,color=color,marker='.' if center else None,ms=3,label=f'N{n} '+('face center' if center else 'corner'))
 ax.axvspan(.9,.95,color='gray',alpha=.1);ax.set(title=f'Initial |d_t {label}|',xlabel='Compact r',ylabel='Absolute native RHS');ax.grid(alpha=.2)
axes[0,0].legend(ncol=2,fontsize=7)
fig.savefig(OUT/'rhs-profiles.png',dpi=170);fig.savefig(OUT/'rhs-profiles.pdf');plt.close(fig)
print(json.dumps(dict(rates=rates,orders=orders),indent=2))
