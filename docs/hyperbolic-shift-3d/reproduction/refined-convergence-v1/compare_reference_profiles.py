"""Compare native analytic t0/reference profiles with an admitted evolved sample."""
from pathlib import Path
import json,gzip,hashlib,shutil,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser();parser.add_argument('--segment',type=int,choices=[1,2],default=1);args=parser.parse_args()
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists());P=R/'output/refined-convergence/reference-profile-oracle';O=D/('reference-profile-regularity' if args.segment==1 else 'reference-profile-regularity-segment02');O.mkdir(exist_ok=True)
time_M=json.loads((D/f'stationary-reference-v2-segment{args.segment:02d}/spatial-analysis.json').read_text())['t_over_M']
a=np.genfromtxt(P/'initial.csv',delimiter=',',names=True);e=np.concatenate([np.genfromtxt(p,delimiter=',',names=True) for p in (R/'output/refined-convergence/reference-endpoint-reader').glob(f'segment{args.segment:02d}.rank*.csv')]);e=e[(e['ray']==0)&(e['r']>.85)&(e['r']<.96)];e.sort(order='r')
idx=np.searchsorted(a['r'],e['r']);assert np.max(abs(a['r'][idx]-e['r']))<1e-14;b=a[idx]
assert all(np.isfinite(a[n]).all() for n in a.dtype.names)
band=(a['r']>=.9)&(a['r']<=.95);derivative_error=float(a['derivative_step_error'].max());assert derivative_error<1e-5
# Widths use dense native S sampling in the monotone transition, not evolved interpolation.
width={}
for lo,hi in ((.1,.9),):
 q=a[(a['S']>=lo)&(a['S']<=hi)];width[f'{lo:g}-{hi:g}']=dict(first_sample_r=float(q['r'][0]),last_sample_r=float(q['r'][-1]),sampled_width=float(q['r'][-1]-q['r'][0]),)
for w in width.values():
 for n in (12,18):
  config=R/f'inputs/hyperbolic_shift_3d/refined_convergence/central20_stationary_smr025_radial{n}_reference.cfg'
  shell_line=next(x for x in config.read_text().splitlines() if x.startswith('shell_radii'))
  boundaries=[.35]+[float(x) for x in shell_line.split('=')[1].strip().split(',')]
  w[f'radial{n}_equivalent_cell_widths']=sum(max(0,min(hi,w['last_sample_r'])-max(lo,w['first_sample_r']))/((hi-lo)/16) for lo,hi in zip(boundaries,boundaries[1:]))
dense=a[(a['r']>=.85)&(a['r']<=.96)];peak=float(dense['dr_alpha'].max());half=dense[dense['dr_alpha']>=peak/2]
alpha_gradient_peak=dict(peak=peak,first_halfmax_sample=float(half['r'][0]),last_halfmax_sample=float(half['r'][-1]),sampled_halfmax_width=float(half['r'][-1]-half['r'][0]),radial12_cell_widths=float((half['r'][-1]-half['r'][0])/(.025/16)),radial18_cell_widths=float((half['r'][-1]-half['r'][0])/(.00625/16)))
assert .875<half['r'][0]<half['r'][-1]<.95
alpha_gradient_peak['radial18_cell_widths']=(max(0,.9-half['r'][0])/(.025/16)+max(0,half['r'][-1]-.9)/(.00625/16))
fields={}
for f in ('chi','alpha','beta_dev_r','K','metric_rr','dr_chi','dr_alpha','dr_beta_dev_r'):
 delta=e[f]-b[f];k=int(np.argmax(abs(delta)));fields[f]=dict(initial_max_abs=float(abs(b[f]).max()),evolved_max_abs=float(abs(e[f]).max()),max_abs_change=float(abs(delta[k])),change_at_r=float(e['r'][k]))
# Exact initial/reference normal norm g^tt=-1/alpha² in conformal evolution chart;
# initial Schwarzschild-chart spacelike margin is f/a². Neither is an evolved regularity proof.
result=dict(time_M=time_M,initial_derivative_probe_max_scaled_change=derivative_error,initial_spacelike_margin_band_min=float(a['spacelike_margin'][band].min()),initial_spacelike_margin_band_max=float(a['spacelike_margin'][band].max()),initial_dr_dR_band_min=float(a['dr_dR'][band].min()),initial_dr_dR_band_max=float(a['dr_dR'][band].max()),transition_width=width,initial_alpha_gradient_peak=alpha_gradient_peak,matched_sample_count=len(e),fields=fields,qualification='Native stationary analytic slice evaluated at exact sampled radii; evolved data unsmoothed from admitted GPU checkpoint. Local/Aurora native quadrature differs at roundoff as separately documented. Width is native S transition from dense samples (0.0001 spacing), not a fitted evolved feature. S10–90 width uses dense0.0001 sampling; equivalent cell widths sum actual configured shell spacing piecewise. Alpha-gradient halfmaximum straddles .9; radial18 cell-width count preserves original spacing below .9 and fourfold refinement above it. Counts measure sampling, not a universal resolution threshold. Positive initial spacelike margin does not prove evolved uniform regularity, and sharp gradients alone do not prove numerical failure.')
(O/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');shutil.copy2(P/'command.json',O/'command.json');(O/'initial.csv.gz').write_bytes(gzip.compress((P/'initial.csv').read_bytes(),mtime=0))
fig,axes=plt.subplots(3,2,figsize=(11,9),layout='constrained')
q=a[(a['r']>.85)&(a['r']<.96)]
for ax,f in zip(axes.flat,('alpha','chi','beta_dev_r','K','dr_alpha','dr_beta_dev_r')):
 ax.plot(q['r'],q[f],label='Native initial/reference',lw=1.1);ax.plot(e['r'],e[f],'.',label=f'Evolved {time_M:.4f}M, +x ray',ms=4);ax.set(xlabel='Compact r',ylabel=f);ax.axvspan(.9,.95,color='gray',alpha=.08);ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8);fig.suptitle('Initial versus evolved outer profiles: no smoothing or evolved resampling');fig.savefig(O/'initial-versus-evolved.png',dpi=170);fig.savefig(O/'initial-versus-evolved.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,f in zip(axes,('metric_rr','K')):
 ax.plot(e['r'],e[f]-b[f],'.-');ax.set(xlabel='Compact r',ylabel='Evolved − initial '+f);ax.axvspan(.9,.95,color='gray',alpha=.08);ax.grid(alpha=.2)
fig.suptitle(f'Native-center deviations from stationary initial geometry at{time_M:.4f}M');fig.savefig(O/'geometry-deviations.png',dpi=170);fig.savefig(O/'geometry-deviations.pdf');plt.close(fig)
print(json.dumps(result,indent=2))
