"""Initial-only radial and interpolation controls, identities kept distinct."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from check_smr_restart import read
HERE=Path(__file__).resolve().parent;ROOT=next(p for p in HERE.parents if (p/'CMakeLists.txt').exists());RAW=ROOT/'output/refined-convergence';D=RAW/'endpoint-diagnostic';OUT=HERE/'initial-radial-controls';OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cases={};baseline=read('central20-n16-initial');core_count=baseline[0]['leaves']*16**3*31
for label,prefix in [('six','central20-n16-initial'),('eleven','central20-n16-radial11-initial'),('nine','central20-n16-radial9-initial'),('twelve','central20-n16-radial12-initial')]:
 report=json.loads((RAW/(prefix+'.report.json')).read_text());history=[json.loads(s) for s in (RAW/(prefix+'.history.jsonl')).read_text().splitlines()];meta,topology,ages,arrays=read(prefix)
 assert meta['time']==0 and meta['leaves']==baseline[0]['leaves'] and topology==baseline[1]
 assert np.array_equal(arrays[0][:core_count],baseline[3][0][:core_count])
 cases[label]=dict(checkpoint=meta,H=report['H'],M=report['M'],fixed=history[-1]['fixed'],core_primary_fields_bit_identical=True,core_cells=core_count//31,horizon_mass=history[-1]['mass'])
 for suffix in ['.report.json','.history.jsonl']:
  shutil.copy2(RAW/(prefix+suffix),OUT/(prefix+suffix))
 for suffix in ['-command.json']:
  if (RAW/(prefix+suffix)).exists():shutil.copy2(RAW/(prefix+suffix),OUT/(prefix+suffix))
 del arrays
controls={}
names=['central20-n16-initial','central20-n16-radial9-initial','central20-n16-radial12-initial-native','central20-n16-radial12-initial-analytic','central20-n16-radial9-initial-analytic','central20-n16-radial12-initial-cartesian-donors','central20-n16-radial12-initial-physical-frame','central20-n16-radial12-initial-legacy-frame']
for name in names:
 p=D/(name+'.json');q=json.loads(p.read_text());assert q['steps']==0;integrals=np.array(q['integrals']);norms=np.sqrt(integrals[:,1:]/integrals[:,0,None]);controls[name]=dict(full_norms=norms[0].tolist(),exterior_r03_norms=norms[1].tolist(),override=q.get('analysis_override','none'),ghost_count=q.get('replaced_ghosts',0),maximum_ghost_change=q.get('maximum_ghost_change',0),native_kernel_crosscheck=q['native_volume_integral_relative_difference'],metadata_sha256=sha(p));shutil.copy2(p,OUT/p.name)
 for p in D.glob(name+'-command.json'):shutil.copy2(p,OUT/p.name)
fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for label,name,color in [('six shells','central20-n16-initial','C0'),('nine: outer refinement','central20-n16-radial9-initial','C1'),('twelve: also inner refinement','central20-n16-radial12-initial-native','C3')]:
 b=np.array(json.loads((D/(name+'.json')).read_text())['radial_bins']);x=(b[:,0]+b[:,1])/2
 for k,col in enumerate([3,4]):ax[k].semilogy(x,np.where(b[:,2]>0,b[:,col],np.nan),'.-',ms=3,label=label,color=color)
for k,a in enumerate(ax):a.set(xlabel='compact r',ylabel=['Hamiltonian squared integral per bin','Momentum squared integral per bin'][k],xlim=(.3,1));a.axvspan(.4,.54,color='grey',alpha=.1,label='Omega transition');a.axvspan(.90,.93,color='C2',alpha=.1,label='late initial-slice transition');a.legend(fontsize=7);a.grid(alpha=.25)
fig.suptitle('Same initial geometry and Cartesian/angular resolution; different radial shell spacing');fig.savefig(OUT/'initial-radial-errors.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
labels=['Native 12-shell','Exact ghosts','Exact Cartesian→shell','Physical-frame interpolation','Legacy-frame interpolation'];keys=['central20-n16-radial12-initial-'+s for s in ['native','analytic','cartesian-donors','physical-frame','legacy-frame']]
for i in range(2):
 vals=[controls[k]['full_norms'][i] for k in keys];ax[i].bar(range(len(vals)),vals);ax[i].set_yscale('log');ax[i].set_xticks(range(len(vals)),labels,rotation=25,ha='right',fontsize=8);ax[i].set(ylabel=['Hamiltonian RMS','Momentum RMS'][i]);ax[i].grid(axis='y',alpha=.25)
fig.suptitle('t=0 diagnostic control only: same active fields, ghost interpolation changed explicitly');fig.savefig(OUT/'halo-controls.png',dpi=160);plt.close(fig)
result=dict(cases=cases,controls=controls,qualification='Initial-only controls, same native geometry/gauge/core/angular resolution; checkpoint identities differ by shell layout. Core nominal fields are bit-identical. Final shell spacing changes original scri stencil sampling and penalty CFL. Analytic ghosts identify interpolation error but are not a production evolution. Frame interpolation uses only stored donor values and known Omega. Radial11 intermediate binary hash was not preserved; it is a local exploratory negative control, not frozen production evidence. No long-time or convergence claim.')
(OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print('Saved controls and plots')
