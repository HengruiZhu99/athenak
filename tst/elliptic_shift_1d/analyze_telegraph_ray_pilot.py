#!/usr/bin/env python3
"""Complete-run ray interpolation comparison; preserves failures and raw errors."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from analyze_telegraph_time import load,controlled_inputs

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--legacy-root',type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
reports={}
for mode in ['linear','pchip']:
 cmd=[sys.executable,str(Path(__file__).with_name('analyze_telegraph_time.py'))]
 for flag,label in [('baseline','base'),('half','half'),('quarter','quarter')]:cmd+=['--'+flag,str(a.root/(mode+'-'+label))]
 if mode=='pchip':cmd+=['--spatial',str(a.root/'pchip-fine')]
 target=a.output/(mode+'-time.json');cmd+=['--output',str(target)]
 subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
 reports[mode]=json.loads(target.read_text())
left,right=[load(a.root/(mode+'-base')) for mode in ['linear','pchip']]
controlled_inputs(left,right,{'ray_interpolation'})
for t in left['data']:
 for key in ['q','q_physical','B','beta','diagnostic','gauge_speeds']:
  assert left['data'][t][key].tobytes()==right['data'][t][key].tobytes(),(t,key)
for label in ['base','half','quarter']:
 with np.load(a.root/('linear-'+label)/'checkpoint.npz') as x,np.load(a.root/('pchip-'+label)/'checkpoint.npz') as y:
  for key in ['q','B']:assert np.array_equal(x[key],y[key]),(label,key)
  n=len(x['r']);assert np.array_equal(x['compensation'][:9*n],y['compensation'][:9*n])
legacy_comparison=[]
if a.legacy_root:
 for case in ['linear-base','linear-half','linear-quarter','pchip-base','pchip-half','pchip-quarter','pchip-fine']:
  old,new=load(a.legacy_root/case),load(a.root/case)
  ignored={'output','resume','library','fast_library','ray_exit'}
  old_args={k:v for k,v in old['manifest']['arguments'].items() if k not in ignored}
  new_args={k:v for k,v in new['manifest']['arguments'].items() if k not in ignored}
  assert old_args==new_args and set(old['data'])==set(new['data'])
  for t in old['data']:
   for key in ['r','q','q_physical','B','beta','diagnostic','gauge_speeds']:
    assert old['data'][t][key].tobytes()==new['data'][t][key].tobytes(),('legacy-exit',case,t,key)
  old_error=max(abs(old['data'][t]['rays'][:,1]-new['data'][t]['rays'][:,1]).max() for t in old['data'])
  legacy_comparison.append(dict(case=case,geometry_flux_diagnostics_bitwise=True,reports=len(old['data']),max_logJ_difference=float(old_error),legacy_manifest=str((a.legacy_root/case/'manifest.json').resolve()),legacy_manifest_sha256=hashlib.sha256((a.legacy_root/case/'manifest.json').read_bytes()).hexdigest(),scope='Each frozen binding independently verified; code intentionally differs in passive exit processing. Direct bytewise geometry checks replace an assumption of code equivalence.'))
decomposition={}
for mode,base in [('linear',left),('pchip',right)]:
 quarter=load(a.root/(mode+'-quarter'));controlled_inputs(base,quarter,{'cfl'})
 rows=[]
 for t in sorted(base['data']):
  ra,rb=base['data'][t]['rays'],quarter['data'][t]['rays']
  aa=(ra[:,0]>0)&(ra[:,0]<1);ab=(rb[:,0]>0)&(rb[:,0]<1)
  row=dict(t=t)
  for key,mask in [('all',np.ones(len(ra),bool)),('active_both',aa&ab),('exited_both',~aa&~ab),('status_disagrees',aa!=ab)]:
   delta=np.where(mask,abs(ra[:,1]-rb[:,1]),-1);i=int(np.argmax(delta))
   row[key]=dict(error=float(delta[i]),ray=i,family=i%2,positions=[float(ra[i,0]),float(rb[i,0])]) if mask.any() else None
  rows.append(row)
 decomposition[mode]=dict(history=rows,peaks={key:max((dict(t=row['t'],**row[key]) for row in rows if row[key] is not None),key=lambda row:row['error'],default=None) for key in ['all','active_both','exited_both','status_disagrees']})
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
colors={'linear':'#b45238','pchip':'#186b94'}
for mode,report in reports.items():
 rows=report['history'];t=[row['t'] for row in rows]
 axes[0].semilogy(t,[max(row['base_quarter']['ray_logJ_max'],1e-16) for row in rows],label=mode+' all',color=colors[mode])
 active=[row['active_both']['error'] if row['active_both'] else np.nan for row in decomposition[mode]['history']]
 axes[0].semilogy(t,np.maximum(active,1e-16),':',color=colors[mode],label=mode+' active')
 metric=report['metrics']['ray_logJ_max'];peaks=metric['peaks']
 axes[1].loglog([1,.5],[peaks['base_half'],peaks['half_quarter']],'-o',label=mode,color=colors[mode])
 for family,style in [(0,'-'),(1,'--')]:
  run=left if mode=='linear' else right
  values=[]
  for key in sorted(run['data']):
   rr=run['data'][key]['rays'][family::2];active=(rr[:,0]>0)&(rr[:,0]<1);values.append(np.min(rr[active,1]) if np.any(active) else np.nan)
  axes[2].plot(sorted(run['data']),values,style,color=colors[mode],label=mode+(' −' if family==0 else ' +'))
axes[0].set(xlabel='Time',ylabel=r'Maximum $|\Delta\log J|$',title=r'Ray timestep sensitivity: $\Delta t$ vs $\Delta t/4$')
axes[1].set(xlabel=r'Coarser timestep / $\Delta t$',ylabel=r'Peak $|\Delta\log J|$',title='Successive timestep differences');axes[1].invert_xaxis()
axes[1].set_xticks([1,.5],[r'$\Delta t$',r'$\Delta t/2$']);axes[1].xaxis.set_minor_locator(NullLocator())
temporal_peaks=[report['metrics']['ray_logJ_max']['peaks'][key] for report in reports.values() for key in ['base_half','half_quarter']]
if max(temporal_peaks)>20*min(temporal_peaks):
 axes[1].set_yscale('log');axes[1].yaxis.set_minor_locator(NullLocator())
else:
 axes[1].set_yscale('linear');axes[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
axes[2].set(xlabel='Time',ylabel=r'Minimum $\log J$',title='Compression of active ray families')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
exit_mode=left['manifest']['arguments'].get('ray_exit','legacy')
fig.suptitle(f'Diagnostic interpolation pilot • N=513, refined scri map, KO8 • {exit_mode} exits',fontsize=12)
for ext in ['png','pdf']:fig.savefig(a.output/('comparison.'+ext))
summary=dict(geometry_and_flux_bitwise_all_base_reports=True,geometry_flux_compensation_bitwise_all_final_controls=True,
 legacy_exit_comparison=legacy_comparison,ray_exit_decomposition=decomposition,
 metrics={mode:report['metrics']['ray_logJ_max'] for mode,report in reports.items()},
 scope='Diagnostic t=3 pilot only. Finite interpolated rays cannot prove continuum noncrossing; no nonlinear candidate admitted.')
(a.output/'review.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
