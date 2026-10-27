#!/usr/bin/env python3
"""Audit complete interface qualification and every-cell initial-source comparison."""
import argparse,collections,hashlib,json,os,tempfile
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--failed-preflight',type=Path,required=True);p.add_argument('--balancing',type=Path,required=True);p.add_argument('--qualification',type=Path,required=True);p.add_argument('--norms',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(exist_ok=False);hashes={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def track(p):hashes[str(p)]=sha(p);return p
def manifest(p):return json.loads(track(p/'manifest.json').read_text())
failed=manifest(a.failed_preflight);qualified=manifest(a.qualification);norm_manifest=manifest(a.norms);assert qualified['complete_interface_coverage'] and len(qualified['face_summaries'])==6 and all(r['queries']==72 and r['selected_field_plans']==144 and r['unsupported']==r['control_failures']==0 and r['passed'] for r in qualified['face_summaries'].values());supports={};counts=collections.Counter();geometries=collections.Counter()
for face in range(6):
 lines=track(a.qualification/f'face{face}.supports').read_text().splitlines();assert lines[0]=='INTERFACE_SUPPORTS_V2 144'
 for line in lines[1:]:
  point,field,count,geometry=map(int,line.split());assert (point,field) not in supports;supports[point,field]=(count,geometry);counts[count]+=1;geometries[geometry]+=1
assert len(supports)==864 and len(set(p for p,f in supports))==432
balancing=[json.loads(line) for line in track(a.balancing/'probe.stdout').read_text().splitlines()];fd=[r for r in balancing if r['phase']=='mapped_FD_amplification'];norms=norm_manifest['norms'];assert norms['all_active_cells']==47088 and norms['overridden_scalar_jets']==864 and norms['actual_unknown_values_unchanged'];csv=track(a.norms/'all-cell.csv');header=csv.open().readline().strip().split(',');cols=[i for i,h in enumerate(header) if h!='method'];fields=[header[i] for i in cols];data=np.loadtxt(csv,delimiter=',',skiprows=1,usecols=cols).reshape(47088,2,len(cols));lookup={f:i for i,f in enumerate(fields)};assert np.all(np.isfinite(data));points=data[:,0,lookup['point']];assert np.array_equal(points,np.arange(47088)) and np.all(data[:,:,lookup['point']]==points[:,None]);weight=data[:,0,lookup['weight']];assert np.all(weight>0) and np.array_equal(data[:,0,lookup['weight']],data[:,1,lookup['weight']]);method_names=['supported_baseline','qualified_interface_basis'];localization={}
for method,name in enumerate(method_names):
 localization[name]={}
 for field in ['H','M']:
  values=data[:,method,lookup[field]];rms=float(np.sqrt(np.sum(weight*values**2)/np.sum(weight)));assert abs(rms-norms['regions']['all'][name][field]['RMS'])<1e-12;p0=int(abs(values).argmax());localization[name][field]={'point':p0,'value':float(values[p0]),'xyz':[float(data[p0,method,lookup[f]]) for f in ['x','y','z']],'radius':float(data[p0,method,lookup['r']]),'block':int(data[p0,method,lookup['block']]),'face':int(data[p0,method,lookup['face']]),'layer':int(data[p0,method,lookup['layer']]),'logical_ijk':[int(data[p0,method,lookup[f]]) for f in ['i','j','k']]}
source_columns=['error_'+str(i) for i in range(24)];np.savez_compressed(out/'complete-all-cell-initial-sources.npz',fields=np.array(fields),methods=np.array(method_names),data=data)
source_mms=collections.defaultdict(list);uncertainty=collections.defaultdict(list)
for face in range(6):
 rows=[json.loads(line) for line in track(a.qualification/f'face{face}.stdout').read_text().splitlines()]
 for r in rows:
  if r['phase']=='MMS':source_mms[(r['field'],r['case'])].append(r)
  if r['phase']=='uncertainty':uncertainty[r['variant']].append(r)
for key,rows in source_mms.items():assert len(rows)==432
for variant,rows in uncertainty.items():assert len(rows)==864 and all(r['status']=='ready' for r in rows)
summary={'scope':'Complete432-cell i0/i1 interface and864 scalar-log plans, then all47088-cell initial comparison; still no coupled/evolution/scientific acceptance.','failed_original_preflight':failed['preflight_summary'],'global_active_donors':'Every candidate includes global active Cartesian and shell cells with one common scalar-log convention. No Cartesian-only extrapolation or patch-by-patch solve.','explicit_catalog':qualified['predeclared_budget']['explicit_geometry_catalog'],'coverage':{'query_cells':432,'scalar_fields':864,'unsupported':0,'control_failures':0,'supports':dict(counts),'geometry_counts':{'Euclidean':geometries[0],'target_Jacobian':geometries[1]}},'unchanged_guards':{'condition':1e8,'reproduction':1e-10,'physical_functional_l1':[16,32,64]},'mapped_FD_and_shape_audits':fd,'MMS_errors':{str(key):{'queries':len(rows),'maximum_H_error':max(abs(r['H_error']) for r in rows),'maximum_physical_source_error':max(r['source_error_max'] for r in rows),'maximum_scaled_unknown_noise':max(r['scaled_unknown_noise'] for r in rows)} for key,rows in source_mms.items()},'background_uncertainty':{str(key):{'meaning':{1:'Independent quadrature192, unknown96 fixed',2:'Background nodal noise1e-12, query jets fixed',3:'Background query value/gradient/Hessian noise1e-12 in scaled jet units, nodal data fixed'}[key],'plans':len(rows),'maximum_physical_source_change':max(r['source_change_max'] for r in rows),'maximum_scaled_jet_change':max(r['scaled_jet_change_max'] for r in rows)} for key,rows in uncertainty.items()},'face_summaries':qualified['face_summaries'],'all_cell_initial_norms':norms,'maxima_localization':localization,'scientific_acceptance':False,'production_adopted':False,'limitations':['Independent exact radial shift shared for derivative/source validation; actual elliptic/coupled RK gate remains.','Only first two transition radial planes overridden; momentum, other shells/core/scri failures remain visible.','Initial data and local manufactured profiles do not establish generic wormhole-to-trumpet or moving-puncture safety.']}
with tempfile.TemporaryDirectory(prefix='complete-transition-plot-',dir=out) as cache:
 os.environ['MPLCONFIGDIR']=cache;os.environ['XDG_CACHE_HOME']=cache
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 regions=['all','qualified_interface_i0_i1','remaining_transition_i2_i5','Cartesian','puncture_r_lt_01','other_shells','scri_r_ge_09'];labels=['all','interface i0/i1','transition rest','Cartesian','r<.1','other shells','r>=.9'];fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
 for method,name in enumerate(method_names):
  for ax,field in zip(axes[0],['H','M']):ax.semilogy(range(len(regions)),[norms['regions'][reg][name][field]['RMS'] for reg in regions],'.-',label=name)
  values=data[:,method,lookup['H']];axes[1,0].scatter(data[:,method,lookup['r']],abs(values),s=1,alpha=.18,label=name,rasterized=True)
 for ax,field in zip(axes[0],['H','M']):ax.set_xticks(range(len(regions)),labels,rotation=35,ha='right');ax.set_title(field+' RMS: all cells and overlapping regions');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=7);axes[1,0].set_yscale('symlog',linthresh=1e-12);axes[1,0].set_xscale('log');axes[1,0].set(xlabel='compact radius: every active cell',ylabel='|H|',title='No puncture/interface/scri exclusions');axes[1,0].legend(fontsize=7)
 ax=axes[1,1];ax.bar([0,1,2],[counts[125],counts[175],counts[256]]);ax.set_xticks([0,1,2],[125,175,256]);ax.set_xlabel('actual global active donors');ax.set_ylabel('scalar field plans');ax.set_title('All864 field plans qualify under unchanged guards')
 fig.suptitle('Complete interface derivative qualification: no production/evolution claim\nMinkowski gauge/values/core scheme unchanged; remaining failures retained');fig.savefig(out/'complete-interface-initial-errors.png',dpi=150);plt.close(fig)
summary['input_sha256']=hashes;summary['inputs_unchanged']=all(sha(Path(p))==h for p,h in hashes.items());assert summary['inputs_unchanged'];summary['script_sha256']=sha(Path(__file__));(out/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps({'coverage':summary['coverage'],'all_H_before':norms['regions']['all'][method_names[0]]['H']['RMS'],'all_H_after':norms['regions']['all'][method_names[1]]['H']['RMS'],'interface_H_after':norms['regions']['qualified_interface_i0_i1'][method_names[1]]['H']['RMS'],'new_H_max':localization[method_names[1]]['H']}))
