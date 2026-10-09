"""Summarize completed native receipt and time-matched production control."""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
p=Path(__file__).resolve().parent
repo=p.parents[4]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
proof=json.loads((p/'native-experiment-report.json').read_text())
long=next(c for c in proof['cases'] if c['phase']=='long')
base=proof['baseline']['case']
h=np.array(long['history_rows']);b=np.atleast_2d(np.loadtxt(repo/'build-layer-research/clean-wide-kappa10-long/finite-angular-N24/hyp.z4c.user.hst'))
matched=[]
for t in [.02,.2,.5,.75,1.,1.25,1.5,1.75,2.]:
 if t>h[-1,0] or t>b[-1,0]:continue
 av={key:float(np.interp(t,h[:,0],h[:,j])) for key,j in [('H',2),('M',3),('Z',4),('Theta',5)]}
 bv={key:float(np.interp(t,b[:,0],b[:,j])) for key,j in [('H',2),('M',3),('Z',4),('Theta',5)]}
 matched.append({'time':t,'candidate':av,'base':bv,'ratios':{key:av[key]/bv[key] for key in av}})
spec=importlib.util.spec_from_file_location('budget',repo/'tst/hyperboloidal/analyze_native_constraints.py')
budget=importlib.util.module_from_spec(spec);spec.loader.exec_module(budget)
d=p/'native-long'/long['name']
paths=sorted((d/'bin').glob('*.con.*.bin'))
budgets=[]
for index in sorted(set([0, min(8,len(paths)-1),min(20,len(paths)-1),min(40,len(paths)-1),min(60,len(paths)-1),len(paths)-1])):
 budgets.append(budget.analyze(paths[index],[0,.25,.5,.75,.85,.9,.95,1]))
(p/'constraint-budgets.json').write_text(json.dumps(budgets,indent=2,allow_nan=False)+'\n')
receipt={'scope':proof['scope'],'implementation_commit':'27c19d20696ea6dd4704032c51dfd026218f64f2',
 'launch_head':'f615acf4356206eceddc59fa929fcc15a671fe09',
 'source_equation':{'xi':'1/a','eta':'rho*S/a^2','C':'(S/a)*(1-1/rho)','rho':1.5,
 'additional_shift_pole':'-eta*W*(beta^i-beta_ref^i+C*n^i*(G-Ghat)/Ghat)',
 'G':'chi*gtilde_inverse^ij*Omega_i*Omega_j','difference':'factored exact reference cancellation'},
 'native_executable_sha256':long['executable_sha256'],'parameters':long['input_parameters'],
 'exit_status':long['exit_status'],'wall_seconds':long['wall_seconds'],
 'final':long['diagnostics'],'ratios_to_matching_base':proof['long_to_base_ratios'],
 'matched_history':matched,'all_saved_snapshots_positive':all(x['all_active_fields_finite'] and x['alpha_min']>0 and x['chi_min']>0 and x['physical_metric_eigen_min']>0 for x in long['native_snapshots']),
 'snapshot_count':len(long['native_snapshots']),
 'reference_max_array_change':next(c['max_array_change_from_initial'] for c in proof['cases'] if c['phase']=='reference'),
 'outcome':'Rejected as pulse stabilization: finite fields through t2 and reduced M/Z, but H is1.9% worse than the matching control and all constraints still grow. Favorable frozen roots and initial BH compatibility do not establish nonlinear stability or regularity closure.',
 'timestep_scope':{'configured_pole_CFL':.03,'shared_initial_dt':float(h[0,1]),'candidate_final_cycle':long['diagnostics']['constraint_budget']['cycle'],'base_final_cycle':base['diagnostics']['constraint_budget']['cycle'],'explanation':'Derivative and live characteristic-speed bound are unchanged; the two evolutions need not take identical individual later steps. History samples alone do not record every step.'},
 'finite_Q_counterexample':proof['finite_Q_counterexample'],
 'BH_scope':'Verified instantaneous initial BH pole/first-jet/second-null-rate compatibility only. Original uncorrected complete jets have NrawDot=-29*Omega. No evolution-preservation proof or native BH integration.',
 'receipt_sha256':{x:sha(p/x) for x in ['native-experiment-report.json','native-build-receipt.json','native-overlay-gate-receipt.json','native.athinput','native_injection.hpp','spatial_norm_control.hpp','BH-gate-receipt-pass-20261009.json','BH-gate-complete-first-failure-20261009/receipt.json','constraint-budgets.json']}}
(p/'native-summary.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print('Final',receipt['final']);print('Ratios',receipt['ratios_to_matching_base']);print('Regular snapshots',receipt['snapshot_count'])
