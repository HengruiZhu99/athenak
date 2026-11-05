#!/usr/bin/env python3
"""Exterior-only constraint/cost frontier from controlled serial experiments."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from physical_constraints import evaluate

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 plan=json.loads((a.runs/'plan.json').read_text());assert (a.runs/'completion.json').exists()
 common=np.geomspace(25,100,301);runs=[];groups={};inputs={};execution_sources=None
 def read(path):
  inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
  return json.loads(path.read_text()) if path.suffix=='.json' else np.load(path)
 for item in plan['plan']:
  folder=a.runs/item['name'];result=read(folder/'result.json');manifest=read(folder/'manifest.json');cfg=manifest['config']
  assert manifest['binary_sha256']==plan['binary_sha256']
  current={key:manifest['sources'][key] for key in ['tst/spherical_optimization/run.py','tst/spherical_optimization/evolution.py','tst/spherical_optimization/kernel.py','tst/spherical_optimization/physical_constraints.py','tst/elliptic_shift_1d/numerics.py']}
  if execution_sources is None:execution_sources=current
  assert current==execution_sources,'Execution source changed during timing batch'
  assert cfg['duration']==10 and cfg['chi_reference_power']==2 and not cfg['probes'] and not cfg['floors']
  row=dict(name=item['name'],family=item['family'],n=item['n'],repeat=item['repeat'],status=result['status'],result=result)
  if result['status']=='completed_unqualified':
   initial=read(folder/'initial.npz');last=read(sorted(folder.glob('profile-*.npz'))[-1]);r=initial['r'];om=initial['omega']
   assert float(last['t_M'])==10
   phys=evaluate(r,last['q'],last['d'],last['dd'],om);phys[:,:2]/=9;phys[:,2:]/=3
   mask=r[:-1]>initial['configuration'][3]/6
   R=3*r[:-1][mask]/(om[:-1,0][mask]*np.sqrt(initial['q'][:-1,0][mask]))
   assert np.all(np.diff(R)>0) and R[0]<25 and R[-1]>100
   pchip=PchipInterpolator(R,phys[mask])(common);linear=np.column_stack([np.interp(common,R,phys[mask,f]) for f in range(4)])
   rms=np.sqrt(np.mean(pchip**2,axis=0));linear_rms=np.sqrt(np.mean(linear**2,axis=0));interp=np.sqrt(np.mean((pchip-linear)**2,axis=0))
   row.update(physical_rms=rms.tolist(),linear_physical_rms=linear_rms.tolist(),interpolation_disagreement_rms=interp.tolist(),indicator=float(max(rms)),screening_indicator=float(max(max(rms),max(linear_rms))),final_q_sha256=hashlib.sha256(last['q'].tobytes()).hexdigest())
  runs.append(row);groups.setdefault((item['family'],item['n']),[]).append(row)
 summary=[]
 for (family,n),rows in groups.items():
  ok=[x for x in rows if x['status']=='completed_unqualified'];s=dict(family=family,n=n,repeats=len(rows),completed=len(ok))
  if ok:
   assert len({x['final_q_sha256'] for x in ok})==1,'Repeated deterministic states differ'
   for key in ('stepping_seconds','setup_seconds','rhs_seconds','rhs_calls','steps','C_char_max_accepted','dt_over_M','min_dr_over_M','steps_per_M','worst_evaluated_speed'):
    values=[x['result'][key] for x in ok];s[key]=dict(min=min(values),median=float(np.median(values)),max=max(values))
   for key in ('indicator','screening_indicator','physical_rms','interpolation_disagreement_rms'):s[key]=ok[0][key]
  summary.append(s)
 complete=[s for s in summary if s['completed']==s['repeats']]
 frontier=[s for s in complete if not any(t['indicator']<=s['indicator'] and t['stepping_seconds']['median']<=s['stepping_seconds']['median'] and (t['indicator']<s['indicator'] or t['stepping_seconds']['median']<s['stepping_seconds']['median']) for t in complete)]
 targets=[]
 for tolerance in (.003,.001,.0001):
  selected={}
  for family in ('areal','compact'):
   eligible=[s for s in complete if s['family']==family and s['screening_indicator']<=tolerance]
   best=min(eligible,key=lambda s:s['stepping_seconds']['median']) if eligible else None
   selected[family]=dict(n=best['n'],median_seconds=best['stepping_seconds']['median'],rhs_calls=best['rhs_calls']['median'],indicator=best['indicator']) if best else None
  ratio=selected['areal']['median_seconds']/selected['compact']['median_seconds'] if all(selected.values()) else None
  targets.append(dict(tolerance=tolerance,selected=selected,areal_over_compact_cost_ratio=ratio))
 out=dict(domain='Reference R/M25–100, 301 uniform log-R samples, maximum dimensionless physical H/M/Theta/Z RMS at 10M',runs=runs,summary=summary,conditional_frontier=[dict(family=s['family'],n=s['n']) for s in frontier],fixed_indicator_targets=targets,admitted_frontier=[],execution_sources=execution_sources,inputs=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Same binary, gauge, central scale, r2 representation, target C_char=.4, diagnostics and 10M duration. Timings include stepping diagnostics/output but exclude setup and Python startup/imports. PCHIP/linear agreement is a sensitivity screen, not a certified error bound. Constraints alone are not solution error; boundary differential compatibility and late response fail, so no robust cost or timestep recommendation follows.')
 (a.output/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
 for family in ('areal','compact'):
  v=sorted([s for s in complete if s['family']==family],key=lambda s:s['n']);err=[s['indicator'] for s in v];cost=[s['stepping_seconds']['median'] for s in v]
  axes[0].loglog([s['n'] for s in v],err,'o-',label=family)
  axes[1].errorbar(cost,err,xerr=[[s['stepping_seconds']['median']-s['stepping_seconds']['min'] for s in v],[s['stepping_seconds']['max']-s['stepping_seconds']['median'] for s in v]],fmt='o-',capsize=3,label=family)
  for x,y,s in zip(cost,err,v):axes[1].annotate(str(s['n']),(x,y),xytext=(5,6),textcoords='offset points',fontsize=8)
 axes[0].set(xlabel='Radial points',ylabel='Exterior constraint indicator',title='Matched r2 representation and central scale')
 axes[1].set(xscale='log',yscale='log',xlabel='10M CPU stepping seconds (median; min–max)',ylabel='Exterior constraint indicator',title='Conditional comparison; no admitted candidate')
 for ax in axes:ax.grid(which='both',alpha=.25);ax.legend()
 fig.savefig(a.output/'accuracy-cost.png',dpi=160);plt.close(fig)
 print(json.dumps(dict(summary=summary,targets=targets),indent=2))
if __name__=='__main__':main()
