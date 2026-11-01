"""Compare hash-verified history/report artifacts; no checkpoint-derived claims."""
from pathlib import Path
import json,hashlib,importlib.util,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
RAW=ROOT/'output/hyperbolic-core-shell/short-gauge-v1/aurora'
spec=importlib.util.spec_from_file_location('comparison',HERE.parent.parent/'short-gauge-v1/compare_resolutions.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
fig,axes=plt.subplots(1,3,figsize=(14,4.8))
fields=[('H_exterior_L2','Hamiltonian'),('M_exterior_L2','Momentum magnitude'),('Mr_exterior_L2','Radial momentum')]
colors={16:'#b55315',24:'#1769aa',32:'#228844'}
result={'scope':'Complete measured histories to10M. History/report artifacts verified against worker SHA256 and length. N32 large checkpoint downloads and common-cutoff radial reconstruction remain pending.','cutoff':'Each run uses its own complete candidate horizon; not a common-cutoff reconstruction.','resolutions':{}}
for n,segs in [(16,[1]),(24,[1]),(32,[1,2])]:
 histories=[];reports=[];identities=[]
 for seg in segs:
  rel=f'campaigns/n{n}-short-gauge-fresh10m-v1-segment{seg:02d}'
  folder=RAW/rel;worker=json.loads((folder/'worker-status.json').read_text())
  assert worker['status'] in ('passed','checkpointed')
  assert worker['source_manifest_sha256']==module.SOURCE
  assert worker['binaries']=={module.BINARY:module.BINARY_SHA}
  for suffix in ['history.csv','report.json']:
   name=rel+'/output/puncture.'+suffix;p=RAW/name;a=worker['cases'][0]['artifacts'][name]
   assert p.stat().st_size==a['bytes'] and module.digest(p)==a['sha256']
   dest=HERE/f'n{n}-segment{seg:02d}.{suffix}';shutil.copyfile(p,dest)
   identities.append(dict(file=dest.name,**a))
  reports.append(json.loads((folder/'output/puncture.report.json').read_text()))
  histories.append(np.atleast_1d(np.genfromtxt(folder/'output/puncture.history.csv',delimiter=',',names=True)))
 data,repeats=module.merge_histories(histories)
 assert reports[-1]['status']=='completed' and reports[-1]['t_over_M']==10 and abs(data['t_over_M'][-1]-10)<1e-12
 for ax,(field,title) in zip(axes,fields):
  ax.plot(data['t_over_M'],data[field],'.-',ms=3,lw=1.5,color=colors[n],label=f'N{n}')
  ax.set(xlabel='t / M',ylabel='Exterior RMS',title=title,xlim=(0,10));ax.grid(alpha=.22)
 last=data[-1]
 result['resolutions'][n]={'history_report_artifacts':identities,'restart_diagnostic_repeats':repeats,'rows':len(data),'endpoint':{k:float(last[k]) for k in ['t_over_M','H_exterior_L2','M_exterior_L2','Mr_exterior_L2','exterior_cutoff','horizon_found','horizon_irreducible_mass','null_surface','null_ghost']}}
axes[0].legend()
fig.suptitle('SG1: exterior constraints through 10M — N16 / N24 / N32',fontsize=16)
fig.tight_layout(rect=(0,.15,1,.91))
fig.text(.5,.035,'Same gauge and dt/dx = 0.0444444; direct chi/lapse from fresh wormhole data.\nEach run uses its own candidate horizon cutoff. Missing values remain gaps; no extrapolation.',ha='center',fontsize=10)
fig.savefig(HERE/'constraints.png',dpi=160)
(HERE/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({n:d['endpoint'] for n,d in result['resolutions'].items()},indent=2))
