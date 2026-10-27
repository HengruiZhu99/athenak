#!/usr/bin/env python3
"""Export completed telegrapher evidence; no evolution and no partial spatial admission."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'tst/elliptic_shift_1d'))
from analyze_telegraph_time import load
from telegraph_analysis_io import validate_telegraph_profile,physical_lapse_gradient
from horizon import apparent_horizons
p=argparse.ArgumentParser();p.add_argument('--time-review',type=Path);a=p.parse_args()
D=HERE/'data/telegraph';D.mkdir(exist_ok=False)
E=ROOT/'docs/elliptic-shift-1d/evidence'
M=ROOT/'output/elliptic-shift-1d/telegraph-scri-pchip-convergence-v1'
receipts=[]
def record(path):
 receipts.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
def review(name,dest):
 path=E/name;record(path);shutil.copy2(path,D/dest);return json.loads(path.read_text())
spatial=review(Path('telegraph-scri-pchip-convergence-v1/review.json'),'spatial.json')
acceptance=review(Path('telegraph-scri-pchip-convergence-v1/acceptance.json'),'acceptance.json')
rays=review(Path('telegraph-scri-pchip-rays-v1/review.json'),'rays.json')
if a.time_review:
 timing=json.loads(a.time_review.read_text());record(a.time_review.resolve());shutil.copy2(a.time_review,D/'time.json')
else:timing=None
records=[];arrays={};selected=[0.,1.,2.5,4.,6.,8.,10.]
for n in [1025,2049,4097]:
 run=load(M/f'n{n}');manifest=run['manifest'];record(M/f'n{n}/manifest.json')
 (D/f'manifest-n{n}.json').write_text(json.dumps(manifest,indent=2)+'\n')
 history={round(x['t'],9):x for x in map(json.loads,(M/f'n{n}/history.jsonl').read_text().splitlines())}
 rows=[]
 for t,d in sorted(run['data'].items()):
  validate_telegraph_profile(d);r=d['r'];diag=d['diagnostic'];w=r*r*d['grid_jacobian'];C=diag[:,:4]
  norms=np.sqrt(np.sum(w[:,None]*C*C,axis=0)/sum(w));h=history[t]
  np.testing.assert_allclose(norms,[h[k+'_rms'] for k in ['H','M','Theta','Z']],rtol=2e-12,atol=1e-25)
  ah=apparent_horizons(r,diag[:,8],diag[:,11]);np.testing.assert_allclose(ah['apparent_horizon_r'],h['apparent_horizon_r'],rtol=2e-13,atol=1e-15)
  i=int(np.argmax(abs(C[:,0])));outside=r>=ah['apparent_horizon_r'];gradient=physical_lapse_gradient(d)
  rows.append(dict(t=t,**{k+'_rms':float(v) for k,v in zip(['H','M','Theta','Z'],norms)},H_max=float(abs(C[i,0])),H_peak_r=float(r[i]),H_bulk_max=float(max(abs(C[r<.8,0]))),H_outside_AH=float(max(abs(C[outside,0]))),AH_r=ah['apparent_horizon_r'],AH_mass_error=abs(ah['apparent_horizon_mass']-1/3),lapse_gradient=float(max(abs(gradient[r<.3])))))
  if t in selected:
   key=f'n{n}_t{round(t*100):04d}';O=(1-r*r)/2;A=(1+r*r)/2
   alpha=A*r*r/(r*r+O**4)*np.exp(d['q'][:,6]);np.testing.assert_allclose(alpha,diag[:,7],atol=1e-14,rtol=2e-13)
   arrays[key+'_r']=r;arrays[key+'_alpha']=alpha;arrays[key+'_physical']=np.divide(alpha,O,out=np.full_like(r,np.nan),where=O>0);arrays[key+'_H']=C[:,0]
  record(M/f'n{n}'/f'profile-{round(t/.05)+1:04d}.npz')
 if n==4097:
  times=np.array(sorted(run['data']));arrays['ray_times']=times;arrays['rays']=np.array([run['data'][t]['rays'] for t in times])
 records.append(dict(n=n,time=manifest['time'],arguments=manifest['arguments'],stages=manifest['all_RK_stages'],rows=rows))
np.savez_compressed(D/'profiles.npz',**arrays)
result=dict(records=records,spatial_status=acceptance['spatial_status'],time_control_complete=timing is not None,remaining_controls='Quadrature, KO-amplitude and adjacent fixed relaxation-parameter studies unrun at the user-requested stopping point.',scope='Completed three-grid spatial trajectories; all saved profiles independently checked against history. Radial RMS uses r^2 dr/dx; accepted-state maxima come from the integrator manifest and may exceed sampled peaks. No inner excision.')
(D/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
(D/'source-manifest.json').write_text(json.dumps(receipts,indent=2)+'\n')
print(json.dumps(dict(grids=[r['n'] for r in records],spatial_status=acceptance['spatial_status'],time_control_complete=timing is not None),indent=2))

# Typeset numbers are generated from the same assessed data as the figures.
def number(v):
 if v is None:return r'\text{unresolved}'
 if v==0:return '0'
 mantissa,exponent=f'{v:.3e}'.split('e')
 return mantissa+r'\times10^{'+str(int(exponent))+'}'
def macro(name,value,scientific=False):
 return '\\newcommand{\\'+name+'}{'+('\\ensuremath{'+number(value)+'}' if scientific else str(value))+'}\n'
comp=spatial['comparisons'][0];checks=acceptance['checks'];fine=records[-1];last=fine['rows'][-1]
values=''
for name,value in [('TeleBulkOrder',comp['bulk_peak_order']),('TeleEndpointOrder',comp['endpoint_peak_order'])]:values+=macro(name,f'{value:.2f}')
for name,key in [('TeleHOrder','H_rms'),('TeleMOrder','M_rms'),('TeleThetaOrder','Theta_rms'),('TeleZOrder','Z_rms')]:values+=macro(name,f"{comp['constraint_peak_orders'][key]['orders'][-1]:.2f}")
values+=macro('TeleGradientPercent',f"{100*checks['matched_gradient_discrepancy']['value']:.3f}")
failed=[k for k,v in checks.items() if not v['passed']]
assessment=('All listed spatial, boundary, and positivity checks pass for this fixed candidate. This is a finite-duration spatial result, not completion of the independent-control protocol.' if not failed else 'The spatial assessment fails the following unchanged checks: '+', '.join(r'\code{'+name+'}' for name in failed)+'. The candidate is not spatially accepted.')
values+=macro('TeleSpatialAssessment',assessment)
families=rays['histories'][-1]['rows'][-1]['families']
assert families[0]['active']==81 and families[1]['active']==0 and families[1]['terminal']==81, 'Update the manuscript ray-count statement to match evidence'
end=families[0];pair=rays['pairs'][-1];orders=rays['spatial_orders'][0]
for name,value in [('TeleFinalHMax',last['H_max']),('TeleFinalHRadius',last['H_peak_r']),('TeleAHMassError',max(r['AH_mass_error'] for r in fine['rows'])),('TeleRayGap',end['min_active_neighbor_gap']),('TeleRayPositionError',pair['peak_position']),('TeleRayLogError',pair['peak_logJ'])]:values+=macro(name,value,True)
values+=macro('TeleMSMassError',max(row['mass_absolute_error_max'] for row in acceptance['finest_invariants']),True)
values+=macro('TeleAFMassError',max(row['AF_mass_absolute_error'] for row in acceptance['finest_invariants']),True)
values+=macro('TeleMinLogJ',f"{end['min_active_logJ']:.6f}")
for name,key in [('TeleRayPositionOrder','position'),('TeleRayLogOrder','logJ')]:values+=macro(name,f'{orders[key]:.2f}' if orders[key] is not None else 'unresolved')
(D/'numbers.tex').write_text(values)
(D/'table.tex').write_text('\n'.join(str(rec['n'])+' & '+' & '.join('$'+number(rec['stages'][key])+'$' for key in ['max_accepted_H_rms','max_accepted_H_bulk','max_scri_pole'])+(r'\\' if i<2 else '') for i,rec in enumerate(records))+'\n')
half=review(Path('telegraph-combined-half-time-v1/review.json'),'half-time.json')
# Revalidate the complete control's numerical binding and saved profile receipts.
for rec in half['records']:
 for entry in rec['profile_receipts']:
  path=Path(entry['path'])
  if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Changed half-step evidence')
 from analyze_telegraph_time import numerical_binding
 if numerical_binding(Path(rec['folder']),rec['manifest'])!=rec['binding']:raise ValueError('Changed half-step numerical binding')
if timing:
 failed=[k for k,v in timing['metrics'].items() if not v['below_ten_percent']]
 prose='The full $N=2049$ sequence at $\\Delta t,\\Delta t/2,\\Delta t/4$ reaches $t=10$. '
 prose+=('Every measured timestep-to-spatial difference ratio is below $10\\%$. ' if not failed else 'The following timestep-to-spatial difference ratios fail the $10\\%$ gate: '+', '.join(r'\code{'+key+'}' for key in failed)+'. ')
 prose+='Observed temporal orders and raw differences are retained in the portable evidence; they are not certified error bounds. '
else:
 pause=review(Path('telegraph-wrapup-v1/status.json'),'wrapup.json')['quarter_control'];time=pause.get('manifest_time')
 prose='The baseline and independent half-step $N=2049$ trajectories both reach $t=10$. The quarter-step control was paused at the requested finest-grid stopping point, with saved reports through $t='+f'{time:g}'+'$. Consequently the required full-duration three-level temporal-order and timestep-to-spatial-error assessment remains incomplete. '
prose+=r'The completed baseline--half-step comparison gives maximum bulk and endpoint field differences of $4.04\times10^{-12}$ and $2.33\times10^{-12}$, a Hamiltonian pointwise difference of $5.71\times10^{-10}$, and ray-position and $\log\mathcal J$ differences of $4.27\times10^{-13}$ and $7.68\times10^{-8}$. Two timestep levels alone do not establish temporal order. These raw differences, and the earlier short diagnostic pilot, cannot replace the missing robustness controls.'
(D/'time-assessment.tex').write_text(prose+'\n')
for source,name in [('telegraph-formulation.md','formulation.md'),('telegraph-protocol-v1.json','protocol.json'),('telegraph-protocol-scri-pchip-convergence-v1.json','convergence-protocol.json'),('telegraph-protocol-scri-pchip-robustness-v1.json','unrun-robustness-protocol.json')]:
 path=ROOT/'docs/elliptic-shift-1d'/source;record(path);shutil.copy2(path,D/name)
for source,name in [('telegraph-exit-implementation-v1/receipt.json','implementation-receipt.json'),('telegraph-exit-implementation-v1/ctest.log','implementation-55-tests.log'),('telegraph-control-analysis-validation-v1/receipt.json','analysis-receipt.json'),('telegraph-control-analysis-validation-v1/ctest.log','analysis-targeted-tests.log')]:
 path=E/source;record(path);shutil.copy2(path,D/name)
(D/'source-manifest.json').write_text(json.dumps(receipts,indent=2)+'\n')
