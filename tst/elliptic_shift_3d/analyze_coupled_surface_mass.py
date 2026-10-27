#!/usr/bin/env python3
"""Finite Hawking-surface mass error budget; retain all raw samples/failures."""
import argparse,hashlib,json,math,os
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();src=a.input.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((src/'manifest.json').read_text());ref=json.loads((src/'reference/reference-manifest.json').read_text());radii=np.array(ref['radii']);mass0=1/3;ref_fine=np.array(ref['reference']['2049']['mass_at_requested']);ref_coarse=np.array(ref['reference']['1025']['mass_at_requested']);summary=dict(scope='Finite coordinate-sphere Hawkingmass fromactual full3Dacceptedfields; no generic3DMisnerSharp/Bondi/late-timevalidation claim',source_manifest_sha256=sha(src/'manifest.json'),reference_manifest_sha256=sha(src/'reference/reference-manifest.json'),reference_parameters=ref['selected_parameters'],radii=radii.tolist(),normalization='Nullnormals n±s have innerproduct−2; spherical normalR=−R*p/2, spatial=R*k/2, M=R/2*(1+normalR²−spatial²)',normalization_checks=ref['reference'],reference_N1025_minus_N2049=(ref_coarse-ref_fine).tolist(),all_requested_radii_retained=True,failed_samples=0,total_raw_samples=0,results={},owner_side_diagnostics={},scientific_acceptance=False,extraction_qualified=False,limits=['OneN10spatialgrid; angularquadraturevariation is a control, not a rigorous error bound','Static initial extraction bias does not necessarily cancel for evolved fields','r=.7 exactshellinterface has owner-side derivative jump; no radiusoffset/ownerselection/filter applied','No Omega0 or puncture-endpoint mass extracted'])
raw={};max_recompute=0;mass_table={}
for mode in ['mink','analytic_initial','initial','final']:
 summary['results'][mode]={};raw[mode]={}
 for points in ([128] if mode=='analytic_initial' else [128,512,2048]):
  records=[(name,row) for name,row in m['runs'].items() if name.startswith(f'{mode}-{points}-')];surfaces=[];arrays=[]
  for name,row in records:
   assert row['exit_code']==0;surfaces+=row['result']['surfaces'];f=src/(name+'.csv.gz');assert sha(f)==row['gzip_sha256'];arrays.append(np.genfromtxt(f,delimiter=',',names=True))
  q=np.concatenate(arrays);raw[mode][points]=q;summary['total_raw_samples']+=len(q);summary['failed_samples']+=int(np.sum(q['valid']!=1));assert len(q)==8*points and np.all(q['valid']==1);surfaces=sorted(surfaces,key=lambda row:row['radius']);assert np.array_equal([v['radius'] for v in surfaces],radii);mass_table[(mode,points)]=np.array([v['mass'] for v in surfaces]);rows=[]
  for radius,entry in zip(radii,surfaces):
   s=q[q['radius']==radius];assert len(s)==points;area=math.fsum(float(w*d) for w,d in zip(s['solid_weight'],s['area_per_solid_angle']));integral=math.fsum(float(w*d*v) for w,d,v in zip(s['solid_weight'],s['area_per_solid_angle'],s['expansion_product']));R=math.sqrt(area/(4*math.pi));mass=R/2*(1+integral/(16*math.pi));difference=abs(mass-entry['mass']);max_recompute=max(max_recompute,difference);assert difference<1e-10
   row=dict(entry);row.update(independent_integral_mass=mass,independent_integral_difference=difference,normalization_kp_error_if_spherical='p and k retained locally; fields never radially averaged',angular_scatter={field:dict(minimum=float(s[field].min()),maximum=float(s[field].max()),unweighted_std=float(s[field].std())) for field in ['area_per_solid_angle','mean_curvature','curvature_trace','theta_plus','theta_minus','expansion_product']});rows.append(row)
  summary['results'][mode][str(points)]=rows
summary['max_independent_integral_mass_difference']=max_recompute;summary['error_budget']={}
for points in [128,512,2048]:
 initial=mass_table[('initial',points)];final=mass_table[('final',points)];summary['error_budget'][str(points)]=dict(initial_same3Dsampling_minus_exact_mass=(initial-mass0).tolist(),final_minus_selected1D=(final-ref_fine).tolist(),final_minus_sampled_initial=(final-initial).tolist(),note='Last difference includes physical/numerical evolution AND changed extraction error; not a pure physical mass drift estimate',mink_control=mass_table[('mink',points)].tolist(),double_vs_extended_sum_mass=[abs(row['mass']-row['mass_extended_summation']) for row in summary['results']['final'][str(points)]])
summary['angular_refinement_changes']={mode:{f'{old}_to_{new}':(mass_table[(mode,new)]-mass_table[(mode,old)]).tolist() for old,new in [(128,512),(512,2048)]} for mode in ['mink','initial','final']}
for mode in ['initial','final']:
 summary['owner_side_diagnostics'][mode]={}
 for points in [128,512,2048]:
  q=raw[mode][points];s=q[q['radius']==.7];groups={}
  for layer in sorted(set(s['layer'])):
   g=s[s['layer']==layer];w=g['solid_weight'];groups[str(int(layer))]=dict(samples=len(g),solid_angle_fraction=float(w.sum()/(4*np.pi)),proper_area_fraction=float(np.sum(w*g['area_per_solid_angle'])/np.sum(s['solid_weight']*s['area_per_solid_angle'])),area_density_weighted_mean=float(np.average(g['area_per_solid_angle'],weights=w)),mean_curvature_weighted_mean=float(np.average(g['mean_curvature'],weights=w)),curvature_trace_weighted_mean=float(np.average(g['curvature_trace'],weights=w)),mean_curvature_range=[float(g['mean_curvature'].min()),float(g['mean_curvature'].max())])
  summary['owner_side_diagnostics'][mode][str(points)]=groups
summary['controls']=dict(mink_max_abs_mass=max(abs(v).max() for (mode,n),v in mass_table.items() if mode=='mink'),analytic_original_wormhole_max_abs_mass_error=float(abs(mass_table[('analytic_initial',128)]-mass0).max()),no_evolution_or_solve=True,all_private_sources_stable=m['private_sources_unchanged'],all_external_dependencies_stable=m['external_dependencies_unchanged'])
(out/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n');np.savez_compressed(out/'mass-arrays.npz',radii=radii,reference_N2049=ref_fine,reference_N1025=ref_coarse,**{mode+'_'+str(n):v for (mode,n),v in mass_table.items()});(out/'.gitignore').write_text('/plot-cache/\n');os.environ['MPLCONFIGDIR']=str(out/'plot-cache');os.environ['XDG_CACHE_HOME']=str(out/'plot-cache');import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');colors=['tab:blue','tab:orange','tab:green']
for points,color in zip([128,512,2048],colors):
 axes[0].plot(radii,mass_table[('final',points)],'o-',color=color,label=f'Final, {points} angular points');axes[0].plot(radii,mass_table[('initial',points)],'s--',color=color,alpha=.65,label=f'Initial sampled, {points}');axes[1].semilogy(radii,abs(mass_table[('final',points)]-ref_fine),'o-',color=color,label=f'Final−1D, {points}');axes[1].semilogy(radii,abs(mass_table[('initial',points)]-mass0),'s--',color=color,alpha=.65,label=f'Initial extraction error, {points}')
axes[0].axhline(mass0,color='black',lw=1,label='Exact initial / matched radial ≈1/3');axes[0].set(xlabel='Coordinate sphere radius',ylabel='Finite Hawking mass',title='All radii retained; r=.7 seam extraction failure');axes[0].legend(fontsize=6);axes[1].set(xlabel='Coordinate sphere radius',ylabel='Absolute mass discrepancy',title='Interpolation + angular quadrature + evolved-state error');axes[1].legend(fontsize=6);fig.savefig(out/'mass-error-budget.png',dpi=160);fig.savefig(out/'mass-error-budget.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for mode,marker in [('initial','s'),('final','o')]:
 for ri,radius in enumerate(radii):axes[0].semilogx([128,512,2048],[mass_table[(mode,n)][ri]-mass0 for n in [128,512,2048]],marker+'-',label=f'{mode} r={radius:g}')
axes[0].set(xlabel='Angular quadrature points',ylabel='Extracted mass −1/3',title='Quadrature control; no extrapolated accuracy claim');axes[0].legend(fontsize=5,ncol=2)
q=raw['initial'][2048];s=q[q['radius']==.7]
for layer,color in [(4,'tab:blue'),(5,'tab:orange')]:
 g=s[s['layer']==layer];axes[1].scatter(g['mu'],g['mean_curvature'],s=5,color=color,label=f'Initial r=.7: donor layer {layer}',alpha=.7)
axes[1].set(xlabel='Angular μ',ylabel='Local physical mean curvature k',title='Same exact sphere, two one-sided interpolation limits');axes[1].legend(fontsize=7);fig.savefig(out/'quadrature-and-seam-control.png',dpi=160);fig.savefig(out/'quadrature-and-seam-control.pdf');plt.close(fig)
print(json.dumps(dict(raw_samples=summary['total_raw_samples'],failed_samples=summary['failed_samples'],controls=summary['controls'],initial_finest=mass_table[('initial',2048)].tolist(),final_finest=mass_table[('final',2048)].tolist(),max_integral_recompute_difference=max_recompute)))
