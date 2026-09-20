"""Compact evidence for the differential-boundary pilot, including limitations."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).parent
rows=[]
for n in [32,48,64,80]:
 s=json.loads((ROOT/f'p-only-n{n}.json').read_text());t=json.loads((ROOT/f'semigroup-n{n}-dt0.1.json').read_text());m=s['near_constraint_reference'][0]
 row={'degree':n,'max_real_spectrum_per_M':s['max_real'],'constraint_branch_per_M':m['real_per_M'],'constraint_branch_endpoint_PDE_relative':m['full_PDE_relative_residual_replaced_endpoint_rows'],'constraint_branch_actual_radiation_relative':m['radiative_constraint_dynamic_residuals_relative'],'offgrid':json.loads((ROOT/f'offgrid-n{n}.json').read_text()),'final_constraint_norm_over_initial_state':{k:v[-1]['constraint_L2_grid_over_initial_state'] for k,v in t['states'].items()},'all_finite_300M':all(x['finite'] for v in t['states'].values() for x in v),'Theta_pulse_boundary_absolute_at5M':t['states']['Theta_pulse'][3]['actual_radiation_boundary_absolute']}
 if n>=64:
  fine=json.loads((ROOT/f'semigroup-n{n}-dt0.05.json').read_text());a=np.array(t['final_state_real']);b=np.array(fine['final_state_real'])
  row['half_exponential_subdivision_final_relative_state_difference']=[float(np.linalg.norm(a[:,:,j]-b[:,:,j])/np.linalg.norm(b[:,:,j])) for j in range(3)]
  row['half_exponential_subdivision_final_constraint_norm']={k:v[-1]['constraint_L2_grid_over_initial_state'] for k,v in fine['states'].items()}
 rows.append(row)
out={'scope':'Linear spherical full-state radial annulus only; promising differential Bjorhus pilot, no 3D/nonlinear validation or exact finite-grid physical constraint preservation.','reference_constraint_branch_per_M':-.072790693446,'spectrum_neutral_mode_caveat':'Small positive eigenvalues near zero remain at double-precision resolution; no exact full-spectrum negativity proof. Fast N^2 modes of old algebraic closure are absent.','rows':rows}
(ROOT/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
fig,axs=plt.subplots(1,3,figsize=(12,3.4),layout='constrained');plt.rcParams.update({'font.size':10})
N=[r['degree'] for r in rows]
axs[0].semilogy(N,[abs(r['constraint_branch_per_M']+.072790693446) for r in rows],'o-',label='Constraint eigenvalue difference')
axs[0].semilogy(N,[r['offgrid']['relative_L2_full_PDE'] for r in rows],'s-',label='Off-grid PDE residual');axs[0].set(xlabel='Chebyshev degree',ylabel='Absolute / relative error');axs[0].legend(fontsize=8)
for k,label in [('pure_lapse','Pure lapse'),('Theta_pulse',r'$\Theta$ pulse'),('constraint_free_radial_coordinate','Physical coordinate variation')]:axs[1].semilogy(N,[r['final_constraint_norm_over_initial_state'][k] for r in rows],'o-',label=label)
axs[1].set(xlabel='Chebyshev degree',ylabel='Constraint norm / initial state at 300 M');axs[1].legend(fontsize=8)
for n in [32,48,64,80]:
 t=json.loads((ROOT/f'semigroup-n{n}-dt0.1.json').read_text())['states']['Theta_pulse'];axs[2].semilogy([x['time_M'] for x in t],[x['constraint_L2_grid_over_initial_state'] for x in t],'o-',markersize=3,label=f'N={n}')
axs[2].set(xlabel='Time [M]',ylabel=r'$\Theta$ pulse: constraint norm / initial state');axs[2].legend(fontsize=8)
for ax in axs:ax.grid(alpha=.2)
fig.savefig(ROOT/'differential-boundary-evidence.png',dpi=180);plt.close(fig)
print(json.dumps(out,indent=2))
