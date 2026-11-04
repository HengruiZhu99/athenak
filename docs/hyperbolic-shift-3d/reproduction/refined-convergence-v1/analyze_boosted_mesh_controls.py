"""Compare fixed-domain initial and two-step native mesh controls."""
from pathlib import Path
import json,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent;O=D/'boosted-mesh-controls';O.mkdir(exist_ok=True)
assert json.loads((D/'shell-angular-checks/verification.json').read_text())['passed']
assert json.loads((D/'boosted-radial24-control/verification.json').read_text())['leaves']==64
specs=[('radial18_original_angles',D/'shell-angular-checks/baseline-old-binary.history.jsonl','18 layers\noriginal angles'),('radial18_angular2',D/'shell-angular-checks/angular-fixed-mpi1-step2.history.jsonl','18 layers\n2× angles'),('radial24_angular2',D/'boosted-radial24-control/state.history.jsonl','24 layers\n2× angles')]
cases=[]
for key,p,label in specs:
 rows=[json.loads(s) for s in p.read_text().splitlines()];assert [r['step'] for r in rows]==[0,1,2]
 assert all(r['leaves']==64 and r['cfl']<=.2+1e-12 for r in rows)
 vals=[]
 for r in rows:
  q=next(c for c in r['fixed'] if c['cut']==.3)
  vals.append(dict(step=r['step'],t_over_M=r['t_over_M'],H=q['H'],M=q['M'],Mr=q['Mr']))
 cases.append(dict(case=key,label=label,values=vals,max_cfl=max(r['cfl'] for r in rows),null_surface_max=max(r['null'] for r in rows),null_ghost_max=max(r['ghost_null'] for r in rows)))
assert all(c['values'][i]['t_over_M']==cases[0]['values'][i]['t_over_M'] for c in cases for i in range(3))
result=dict(cases=cases,initial_baseline_to_radial24_reduction={k:cases[0]['values'][0][k]/cases[2]['values'][0][k] for k in ['H','M','Mr']},step2_baseline_to_radial24_reduction={k:cases[0]['values'][2][k]/cases[2]['values'][2][k] for k in ['H','M','Mr']},initial_angular18_to_radial24_reduction={k:cases[1]['values'][0][k]/cases[2]['values'][0][k] for k in ['H','M','Mr']},nominal_full_smr_blocks=dict(core=400,radial18_original_angles=508,radial18_angular2=688,radial24_angular2=832),qualification='Fresh native initial data, same N12 uniform coarse Cartesian core, dt=1/24000, exact boost .1 and physical parameters. Original/core paths proven checkpoint-byte-identical between old/new binary. Corrected angular radial-plane donor policy used for both angular cases. Native M is conformal-metric norm. Independent fixed cut r>.3; only 0..0.00025M. The deliberately coarse core is not adequate for a production puncture evolution. Full candidate retains the separately qualified 400-leaf SMR tube. Two anisotropic mesh changes do not define a global h or establish fourth-order/evolved convergence, GPU parity or long-time stability.')
(O/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
fig,axs=plt.subplots(1,2,figsize=(10.6,4.6),constrained_layout=True)
for ax,k in zip(axs,['H','M']):
 ax.bar([i-.18 for i in range(3)],[c['values'][0][k] for c in cases],width=.36,label='Initial')
 ax.bar([i+.18 for i in range(3)],[c['values'][2][k] for c in cases],width=.36,label='After two steps')
 ax.set_yscale('log');ax.set_ylim((.0007,.03) if k=='H' else (.008,.4));ax.set_xticks(range(3),[c['label'] for c in cases]);ax.set(ylabel=f'Native {k} norm at r > 0.3',title='Hamiltonian' if k=='H' else 'Momentum (conformal norm)');ax.grid(axis='y',alpha=.25);ax.legend(fontsize=8)
fig.suptitle('Boosted initial resolution controls: targeted angular and radial refinement\nN12, unchanged coarse core; two steps = 0.00025M, not a stability test')
for ext in ['png','pdf']:fig.savefig(O/('initial-and-two-step-norms.'+ext),dpi=170)
print(json.dumps({k:v for k,v in result.items() if k not in ['cases','qualification']},indent=2))
