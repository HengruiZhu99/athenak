#!/usr/bin/env python3
"""All-cell and common-radius diagnostics for related classical-RK4 controls.
Observed ratios are retained; changing puncture cutoff/volume prevents treating
all-cell volume-RMS ratios as a proof of uniform-domain convergence.
"""
import argparse,hashlib,json,math,os,re,sys
from pathlib import Path
import numpy as np
import compare_coupled_family as c

def main():
 p=argparse.ArgumentParser();p.add_argument('--family',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();family=a.family.resolve();out=a.output.resolve();m=json.loads((family/'manifest.json').read_text());complete=[n for n,row in m['cases'].items() if 'result' in row];assert complete
 c.PROFILE_GRID_NOTE='related N6/N8/N10 meshes; all-cell puncture cutoff and Penrose volume change'
 c.CASES=tuple((f'n{n}',f'N{n}: fixed50blocks /7shells',str(family/f'n{n}'),'coupled.csv',f'../n{n}-coupled.stdout','../manifest.json',[.45,.55,.60,.65,.70,.75,1]) for n in complete)
 sys.argv=[__file__,'--output',str(out)];c.main();path=out/'comparison.json';summary=json.loads(path.read_text());summary.pop('prior_AMR_localization_record',None);summary.pop('reported_independent_controls',None);summary['scope']='Related actual full3D classical-RK4 evolutions, fixed50blocks /7shells /7surfacepoints at t=.0002, dt1e-4,tol1e-10; supported positivejets+Hermite fixed; no nativeDriver task claim, no AMR/cache, every scientific gate explicitly assessed'
 summary['family_manifest']=dict(path=str(family/'manifest.json'),sha256=c.digest(family/'manifest.json'));summary['failed_or_incomplete']={n:row for n,row in m['cases'].items() if 'result' not in row};summary['predeclared_spherical_reference_targets']=dict(source='docs/elliptic-shift-1d/telegraph-protocol-v1.json',bulk_field_and_accepted_constraint_order_min=3.5,endpoint_field_order_min=3,finest_H_rms_max=1e-5,finest_H_bulk_max=1e-3,required_main_duration=6,required_extension_duration=10,note='Current t=.0002 pilot does not qualify late lapse-front or original duration/independent-control gates. No H_bulk pass inferred without matching its original definition.')
 summary['spatial_qualification']='N6/8/10 are related fixed block maps and identical physical schemes. Actual-node volume RMS includes every cell, but puncture minimum radius and quadrature volume change with N. All-cell RMS apparent exponents below are diagnostics, not uniform-domain convergence proofs. Common-radius64direction samples have exactly matched locations, while their interpolation error remains included.'
 summary['three_related_resolutions_complete']=len(complete)==3;profiles={}
 for n in complete:
  key='n'+n;row=summary['cases'][key];record=m['cases'][n];row['recorded_exit']=record['exit_code'];row['execution_completed_with_bitwise_restart']=record['completed_with_bitwise_restart'];row['gates'].update(record['gates']);row['recorded_scientific_status']='FAIL' if not all(v['passes'] for v in row['gates'].values()) else 'PROVISIONAL_SHORT_PILOT_ONLY';row['scientific_acceptance']=False
  data=np.genfromtxt(family/key/'coupled.csv',delimiter=',',names=True);profiles[n]=data
  solve_rows=[]
  for match in re.finditer(r'shift_solve iterations=(\d+) seconds=([\d.eE+-]+) physical_rms=([\d.eE+-]+) point_max=([\d.eE+-]+)',(family/f'n{n}-coupled.stderr').read_text()):solve_rows.append(dict(iterations=int(match[1]),seconds=float(match[2]),physical_rms=float(match[3]),point_max=float(match[4])))
  row['solve_cost']=dict(calls=len(solve_rows),iteration_sum=sum(x['iterations'] for x in solve_rows),wall_sum=sum(x['seconds'] for x in solve_rows),rows=solve_rows,includes_midpoint_restart=True,scaling_claim=False)
  # Preserve every actual nodal direction/value as well as the existing shared
  # radius64 probes. Radius grouping is diagnostic only, never evolution.
  xyz=np.column_stack([data[f] for f in 'xyz']);radius=data['r'];group=np.round(radius,14);physical=np.column_stack([data[f] for f in c.FIELDS]);np.savez_compressed(out/(key+'-raw-all-node-angular.npz'),points=xyz,radius=radius,group_radius=group,directions=xyz/radius[:,None],fields=np.array(c.FIELDS),values=physical,weights=data['weight'])
  row['actual_node_angular']={}
  unique,inverse=np.unique(group,return_inverse=True)
  for i,f in enumerate(c.FIELDS):
   means=[];sd=[];counts=[]
   for j,r in enumerate(unique):v=physical[inverse==j,i];means.append(float(v.mean()));sd.append(float(v.std()));counts.append(len(v))
   row['actual_node_angular'][f]=dict(radii=unique.tolist(),counts=counts,mean=means,scatter=sd,maximum_scatter=max(sd),grouping='round(r,14); every node retained')
 orders={}
 def observed(vals,ns):
  return [math.log(abs(vals[i]/vals[i+1]))/math.log(ns[i+1]/ns[i]) if vals[i]>0 and vals[i+1]>0 else None for i in range(len(vals)-1)]
 if len(complete)==3:
  ns=[int(n) for n in complete];rows=[summary['cases']['n'+n] for n in complete]
  for channel in ['scri_tensor_pole','H_rms','H_max','M_rms','M_max','Theta_rms','Z_rms']:
   vals=[row['reported'][channel] for row in rows];orders[channel]=dict(values=vals,adjacent_observed_exponents=observed(vals,ns),formal_convergence_established=False)
  for f in c.FIELDS:
   vals=[row['field_errors'][f]['penrose_volume_rms'] for row in rows];common=[max(abs(np.array(row['angular'][f]['mean_error']))) for row in rows];scat=[row['angular'][f]['maximum_scatter'] for row in rows]
   orders[f]=dict(all_cell_volume_RMS=vals,all_cell_apparent_exponents=observed(vals,ns),common_radius_mean_error_max=common,common_radius_mean_error_exponents=observed(common,ns),common_radius_scatter_max=scat,common_radius_scatter_exponents=observed(scat,ns),formal_convergence_established=False)
 summary['observed_resolution_controls']=orders;summary['analysis_wrapper_sha256']=c.digest(Path(__file__));summary['reference_control_role']='N1025−N2049 physical errors evaluated on each actual grid and the same64ray points; kept distinct from spatial/evolution/elliptic/interface errors.'
 path.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
 # Explicit related-grid plot avoids implying that falling initial/elliptic
 # residuals establish the original coupled boundary/scientific gates.
 os.environ['MPLCONFIGDIR']=str(out/'plot-cache-related');os.environ['XDG_CACHE_HOME']=str(out/'plot-cache-related');import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained');ns=[int(n) for n in complete]
 for field in ['scri_tensor_pole','shift_residual_max','shift_residual_point_max']:axes[0].loglog(ns,[summary['cases']['n'+n]['reported'][field] for n in complete],'o-',label=field)
 axes[0].axhline(1e-7,color='black',ls='--',label='Pole/point target1e-7');axes[0].axhline(1e-10,color='gray',ls=':',label='Original RMS target1e-10');axes[0].set_title('Original vector and tensor boundary equations');axes[0].legend(fontsize=6)
 for field in ['H_rms','M_rms','Theta_rms','Z_rms']:axes[1].loglog(ns,[summary['cases']['n'+n]['reported'][field] for n in complete],'o-',label=field)
 axes[1].axhline(1e-5,color='black',ls='--',label='Spherical H RMS target1e-5');axes[1].set_title('All-cell Penrose-volume constraint RMS');axes[1].legend(fontsize=7)
 for f in c.FIELDS:axes[2].loglog(ns,[summary['cases']['n'+n]['field_errors'][f]['penrose_volume_rms'] for n in complete],'o-',label=f)
 axes[2].set_title('All-cell physical-field error vs radial N2049');axes[2].legend(fontsize=7,ncol=2)
 for ax in axes:ax.set_xlabel('N per logical block direction');ax.grid(alpha=.2)
 fig.suptitle('Actual classical-RK4 coupled controls at t=.0002; every scientific milestone remains incomplete');fig.savefig(out/'related-resolution-gates.png',dpi=160);fig.savefig(out/'related-resolution-gates.pdf');plt.close(fig)
 print(json.dumps({key:dict(H=row['reported']['H_rms'],pole=row['reported']['scri_tensor_pole'],alphaRMS=row['field_errors']['alpha']['penrose_volume_rms'],betaRMS=row['field_errors']['beta_r']['penrose_volume_rms'],volume=row['penrose_volume_quadrature'],scientific=row['recorded_scientific_status']) for key,row in summary['cases'].items()}))
if __name__=='__main__':main()
