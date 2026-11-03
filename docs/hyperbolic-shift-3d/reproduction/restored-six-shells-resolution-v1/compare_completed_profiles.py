"""Unsmoothed native rays at completed N16/N20/N24 endpoints; no interpolation."""
from pathlib import Path
import json,hashlib,subprocess,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from completed_three_data import HERE,ROOT,load_cases
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=HERE.parent/'restored-six-shells-v1/comparison/slices';prior=read(old/'provenance.json')['six']
binary=Path(prior['slices']['command'][0]);assert sha(binary)==prior['slices']['binary_sha256']
cases,_=load_cases();out=HERE/'n16-n20-n24-complete-profiles';out.mkdir(exist_ok=False)
records={};raw=ROOT/'output/restored-six-shells-resolution/aurora'
for ext in ['csv','json']:shutil.copy(old/f'six.{ext}',out/f'n16.{ext}')
records[16]=dict(reused_from=str(old),csv_sha256=sha(old/'six.csv'),json_sha256=sha(old/'six.json'),prior=prior)
for n in [20,24]:
 cp=raw/Path(cases[n]['records'][-1]['plan']).parent/'output/puncture.checkpoint'
 h=sha(cp);assert h==cases[n]['records'][-1]['checkpoint_sha256'];cmd=[str(binary),str(n),str(cp),str(.008/n),str(out/f'n{n}')]
 records[n]=dict(command=cmd,checkpoint_sha256=h,binary_sha256=sha(binary))
 (out/'provenance.json').write_text(json.dumps(records,indent=2)+'\n')
 with (out/f'n{n}.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 assert sha(cp)==h
 meta=read(out/f'n{n}.json');assert meta['n']==n and meta['schema']=='HYP3D_MPI_N61' and meta['t_over_M']==10 and meta['native_volume_integral_relative_difference']<1e-9
 records[n].update(csv_sha256=sha(out/f'n{n}.csv'),json_sha256=sha(out/f'n{n}.json'))
 (out/'provenance.json').write_text(json.dumps(records,indent=2)+'\n')
data={};summary={};colors={16:'#238b45',20:'#2166ac',24:'#d95f02'}
for n in [16,20,24]:
 a=np.genfromtxt(out/f'n{n}.csv',delimiter=',',names=True);a=a[a['i']<n];data[n]=a;summary[n]={}
 for ray in [0,1]:
  z=np.sort(a[a['ray']==ray],order='r');keep=(z['r']>=.6)&(z['r']<=.9);m={}
  for field in ['H','Mr','Mnorm','dr_chi','dr_alpha','dr_beta_dev_r','dr_D_r']:
   v=z[keep];i=np.argmax(abs(v[field]));m[field]=dict(max_abs=float(abs(v[field][i])),r_at_max=float(v['r'][i]))
  summary[n][ray]=dict(direction=[float(z[k][0]/z['r'][0]) for k in ['x','y','z']],rows=len(z),region_06_09=m)
def decorate(ax,label):
 ax.axvspan(.7,.775,color='gray',alpha=.12)
 for r in [.305,.45,.6,.75,.875]:ax.axvline(r,color='gray',ls=':',lw=.7)
 ax.set(xlabel='compactified radius r',ylabel=label,xlim=(.16,1));ax.grid(alpha=.2)
def footer(fig):
 fig.tight_layout(rect=(0,.085,1,.94));fig.text(.5,.025,'Native primary shell centers, no interpolation/smoothing. Ray angles vary slightly with N; no pointwise order inferred.\nSame operators and clipped-penalty family. N16 original policy failure retained; profiles do not prove regularity.',ha='center',fontsize=9)
for filename,fields,title in [('fields.png',['chi','alpha','metric_rr','beta_dev_r','D_r','K'],'Native field profiles at 10M, near face center'),('gradients.png',['dr_chi','dr_alpha','dr_beta_dev_r','dr_D_r'],'Native radial derivatives at 10M, near face center')]:
 fig,axes=plt.subplots(2,3 if len(fields)==6 else 2,figsize=(13,7.5))
 for ax,k in zip(axes.flat,fields):
  for n,a in data.items():
   z=np.sort(a[a['ray']==0],order='r');ax.plot(z['r'],z[k],'.-',ms=3,lw=1,label=f'N{n}',color=colors[n])
  decorate(ax,k)
 axes.flat[0].legend();fig.suptitle(title);footer(fig);fig.savefig(out/filename,dpi=180);plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(13,7.5))
for ray in [0,1]:
 for ax,k in zip(axes[ray],['H','Mr','Mnorm']):
  for n,a in data.items():
   z=np.sort(a[a['ray']==ray],order='r');ax.plot(z['r'],z[k],'.-',ms=3,lw=1,label=f'N{n}',color=colors[n])
  decorate(ax,k);ax.set_title('Near face center' if ray==0 else 'Near face corner / seam')
axes[0,0].legend();fig.suptitle('Native constraint profiles at 10M');footer(fig);fig.savefig(out/'constraints.png',dpi=180);plt.close(fig)
(out/'summary.json').write_text(json.dumps(dict(cases=summary,qualification='Ray directions differ across N; descriptive profiles, not pointwise self-convergence or regularity proof. Region maxima are sample dependent. Primary cells only, overlap rows remain in raw CSV.'),indent=2)+'\n')
print(json.dumps(summary,indent=2))
