"""Write report results from verified terminal campaign assessments."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--assessment',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
s=json.loads(a.assessment.read_text());ref=json.loads((a.evidence/'advective-summary.json').read_text());pun=s['puncture_assessments']
def tex(x):return str(x).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def sci(x):
 if not np.isfinite(x):return r'\mathrm{unavailable}'
 if x==0:return '0'
 power=int(np.floor(np.log10(abs(x))));return f'{x/10**power:.3f}\\times10^{{{power}}}'
if s['reaches_100M']:
 summary=r'\begin{abstract}At least one frozen nonadvective trajectory reaches the aligned $100M$ target. Its finite-resolution comparison and qualification limits are reported below. Single-grid survival does not establish convergence.\end{abstract}'
else:
 summary=r'\begin{abstract}The proposed bulk equations and their characteristic speeds are implemented and independently checked. Exact spherical nullness and the new driver compatibility hold to floating-point precision in the tests. The first incoming-characteristic boundary discretization nevertheless fails consistency with the metric-reconstructed connection constraint and has reproducible growing numerical modes. The short pulse and puncture tests fail the scri-numerator guard, so no qualified $100M$ result is obtained for this implementation, and no late-time improvement in horizon drift or settling is established. The completed coarse advective reference supplies the comparison baseline.\end{abstract}'
(a.report/'results-summary.tex').write_text(summary+'\n')
lines=[r'\section{Trajectory outcomes and the advective comparison}',r'Trajectories use immutable source snapshots. Failed and bounded outcomes remain terminal; no failed state is restarted or relabeled. Crosses in the plots mark the last reported diagnostic, which can precede the actual failure time below. Table times are coordinate times; multiply by three to obtain $t/M$. The numerator guard is sampled approximately every $0.025$ coordinate-time units, so a reported stop is not an exact onset time.',r'\begin{center}\small\begin{tabular}{llrrl}\toprule',r'Candidate / test & $N$ & Target $t$ & Accepted $t$ & Outcome\\\midrule']
for run in s['runs']:
 m=json.loads((Path(run['case'])/'manifest.json').read_text());cfg=m['configuration'];label=('Direct' if cfg['reference_damping']==0 else 'Adaptive')+(' puncture' if cfg['puncture'] else ' pulse')
 lines.append(f"{label} & {cfg['n']} & {m['target']:.4g} & {m['t']:.6g} & {tex(m['status'])}\\\\")
lines.extend([r'\bottomrule\end{tabular}\end{center}'])
failures=[run for run in s['runs'] if run['error']]
if failures and len({run['error'] for run in failures})==1:
 lines.append('All '+str(len(failures))+' failed segments stop on the '+tex(failures[0]['error']).lower()+'. Their last accepted times and resolutions are retained in the table.')
else:
 for run in failures:lines.append(r'\textbf{'+tex(run['name'])+r':} '+tex(run['error'])+'.')
if not s['reaches_100M']:
 lines.append(r'No candidate with a failed regular-center gate is promoted to a long puncture evolution. Consequently the $100M$ comparison, final-$20M$ drift fit for a new candidate, and a statement of improved late-time settling are unavailable. This is a failed qualification of the tested boundary implementation, not a measured puncture failure at $100M$ or a rejection of every discretization of the bulk equations.')
lines.append(r'The advective $N=257$ trajectory completes $100.0002M$. Its final relative horizon-mass error is $'+sci(ref['final_relative_mass_error'])+r'$, with peak sampled drift $'+sci(ref['peak_mass_drift'])+r'$. Its compact horizon radius changes from '+f"{ref['initial_horizon_r']:.6f}"+' to '+f"{ref['final_horizon_r']:.6f}"+r'. The least-squares slope over $80.025$--$100.0002M$ is $'+sci(ref['late_horizon_slope_per_M'])+r'$ per unit $t/M$, with fit RMS $'+sci(ref['late_horizon_fit_RMS'])+r'$. Its final exterior mixed field-rate RMS is $'+sci(ref['final_exterior_field_rate_RMS'])+r'$. This surviving reference has not reached a stationary coordinate state by this diagnostic.')
for r in pun:
 lines.append(r'For the '+r['name'].lower()+r' puncture test, the last accepted time is $'+f"{r['t_over_M']:.6g}"+r'M$. Peak sampled relative horizon-mass drift is $'+sci(r['peak_relative_mass_drift'])+r'$ and the last reported absolute relative mass error is $'+sci(r['final_absolute_relative_mass_error'])+r'$. There are '+str(len(r['exact_matched_diagnostic_samples']))+r' exact shared diagnostic times with the old history. The complete histories retain their native sample spacing; no unequal-time profile is used as a refinement comparison.')
for r in pun:
 samples=[x for x in r['exact_matched_diagnostic_samples'] if x['t']>0]
 if samples:
  match=samples[-1];lines.append('At the latest exact shared diagnostic time, $t/M='+f"{3*match['t']:.6g}"+'$, the '+r['name'].lower()+r' has the following finite-resolution comparison:')
  lines.extend([r'\begin{center}\small\begin{tabular}{lrrr}\toprule',r'Metric & New & Advective & New / old\\\midrule'])
  for field,title in [('H_exterior_rms','Exterior H RMS'),('M_exterior_rms','Exterior M RMS'),('Theta_exterior_rms','Exterior Theta RMS'),('Z_exterior_rms','Exterior Z RMS')]:
   x,y=match['new'][field],match['advective'][field];ratio=f'{x/y:.4g}' if y else 'undefined'
   lines.append(title+' & $'+sci(x)+'$ & $'+sci(y)+'$ & '+ratio+r'\\')
  lines.extend([r'\bottomrule\end{tabular}\end{center}'])
  lines.append('The compact horizon radii at this same time are '+f"{match['new']['apparent_horizon_r']:.8f}"+' (new) and '+f"{match['advective']['apparent_horizon_r']:.8f}"+' (advective). This short comparison cannot establish a reduced late-time drift rate.')
lines.append(r'\clearpage\subsection{Diagnostic plots}')
figures=[('pulse-gates','Regular-center pulse gates, in coordinate time. The numerator and derivative-reduction defects expose boundary growth separately from exact nullness.'),('exterior-constraints','Apparent-horizon-exterior RMS Hamiltonian, momentum, Theta, and connection constraints. The primary Z curves use independent metric finite differences on saved profiles and terminal states; the other three diagnostics agree with the independently recomputed values there. Norms are normalized over exterior radial weights. Zero values are omitted only on logarithmic axes.'),('mass-horizon-settling','Outermost-horizon mass error, drift from each numerical initial mass, compact horizon radius, and candidate field-rate RMS. The time window shows the actual short candidate evidence; it does not imply a late-time comparison. Reference field rates at the short common times come from read-only diagnostics on the exactly recovered sampling replay; its 100M value is reported separately.'),('boundary-health','Maximum stagewise relative nullness, current numerator residual, raw lapse-clock compatibility, and the new driver compatibility residual. Roundoff gauge identities can coexist with growing regularity errors.'),('regularity-traces','Approximate regularity traces, distinct from exact nullness. These use the evolved boundary jet, whose inconsistency with metric reconstruction is quantified separately.'),('advective-duration-context','The preserved advective duration context. The late horizon-coordinate fit uses the final $20M$; no new candidate reaches this interval. Short matched new profiles and histories are shown separately.'),('gauge-positivity-floor','New sufficient characteristic-separation margin, geometry minima at accepted diagnostic samples, and maximum stage floor radius. The old advective gap has a different definition and is not superposed.'),('interior-errors','Interior constraint errors are retained separately and do not enter the exterior accuracy budget. No positive-radius neighborhood has been silently removed.'),('constraint-definition','The independent constraint-definition audit. Dashed curves use evolved boundary jets; solid markers reconstruct the derivative from metric values. Their difference exposes the reduction-consistency failure instead of suppressing its endpoint contribution.'),('reduction-and-settling','Boundary derivative-reduction defects, reference gap, shift rate, and independent connection residual. These distinguish coordinate settling from simple horizon-radius flattening.'),('matched-profiles','Lapse, shift, and radial metric at an exact common saved time. Every positive-radius node is retained; the central coefficient endpoint is not a physical puncture sample.')]
for stem,caption in figures:
 source=a.assessment.parent/'figures'/(stem+'.pdf')
 if source.exists():
  shutil.copy2(source,a.report/'figures'/source.name)
  lines.extend([r'\begin{figure}[htbp]\centering',r'\includegraphics[width=\linewidth]{figures/'+source.name+'}',r'\caption{'+caption+r'}\end{figure}'])
lines.append(r'\FloatBarrier\subsection{Boundary timestep and diagnostic checks}')
for eta,name in [(0,'direct'),(1,'adaptive')]:
 full=json.loads((a.evidence/'boundary-step-n33-v1.json').read_text());half=json.loads((a.evidence/'boundary-step-half-dt-n33-v1.json').read_text())
 f=next(r for r in full['rows'] if r['eta']==eta and r['epsilon']==5e-7);h=next(r for r in half['rows'] if r['eta']==eta and r['epsilon']==5e-7)
 gf=np.log(f['spectral_radius'])/f['dt'];gh=np.log(h['spectral_radius'])/h['dt']
 lines.append('For the '+name+r' control at $N=33$, halving the timestep changes the dominant growth rate from '+f'{gf:.5f}'+' to '+f'{gh:.5f}'+r' per unit coordinate time. The growing frozen mode therefore persists under this timestep reduction; the result is not explained by a large RK step alone.')
check=a.evidence/'constraint-sampling-audit.json'
if check.exists():
 d=json.loads(check.read_text());lines.append('An independent recomputation of the native constraints replaces the evolved endpoint derivative reductions by ordinary finite-difference derivatives on every saved profile and last accepted state. Maximum changes in the H, M, Theta, and Z diagnostics are respectively $'+',\ '.join(sci(v) for v in d['max_constraint_difference'])+r'$. The Z difference is material and the metric-reconstructed Z norm is therefore used for the primary comparison. This exposes the boundary consistency failure; it does not repair the evolution.')
lines.append(r'\subsection{Answers to the comparison questions}')
if not s['reaches_100M']:
 lines.append(r'The tested implementation has no qualified $100M$ result. Reduced late-time horizon drift and competitive long-time exterior constraints or mass conservation are therefore unestablished. Exact nullness is assessed on the actual evaluated stages, not extrapolated to an unperformed long run. The numerical boundary growth prevents a claim of improved settling or robustness. No nearby reference-rate scan is justified by a failure already localized in the boundary discretization.')
(a.report/'results-detail.tex').write_text('\n\n'.join(lines)+'\n')
print('Generated report result sources from terminal evidence.')
