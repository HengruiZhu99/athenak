"""Audit all actual native field snapshots and near-boundary metrics after run completion."""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
root=Path(__file__).resolve().parents[3];work=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reader',root/'vis/python/bin_convert.py');reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
spec=importlib.util.spec_from_file_location('budget',root/'tst/hyperboloidal/analyze_native_constraints.py');budget=importlib.util.module_from_spec(spec);spec.loader.exec_module(budget)
rows=[]
for prefix in [root/'build-layer-research/clean-wide-kappa10-long',work/'native-kappa10-t2']:
 directory=prefix/'finite-angular-N24';case=[]
 for path in sorted((directory/'bin').glob('hyp.z4c.*.bin')):
  data=reader.read_binary(str(path));fields={k:np.asarray(v) for k,v in data['mb_data'].items()};mask=fields['z4c_active'][0].astype(bool)
  coordinates=[]
  for axis in range(3):
   lower,upper=data['mb_geometry'][0,2*axis:2*axis+2];step=(upper-lower)/data[f'nx{axis+1}_mb'];start=data['mb_index'][0,2*axis];first=lower+(start+.5)*step;coordinates.append(first+np.arange(mask.shape[2-axis])*step)
  zz,yy,xx=np.meshgrid(*coordinates[::-1],indexing='ij');radius2=xx*xx+yy*yy+zz*zz
  assert np.array_equal(mask,radius2<1)
  radius=np.sqrt(radius2[mask]);count=mask.sum();metric=np.zeros((count,3,3));A=np.zeros_like(metric)
  for i,j,suffix in [(0,0,'xx'),(0,1,'xy'),(0,2,'xz'),(1,1,'yy'),(1,2,'yz'),(2,2,'zz')]:
   metric[:,i,j]=metric[:,j,i]=fields['z4c_g'+suffix][0][mask];A[:,i,j]=A[:,j,i]=fields['z4c_A'+suffix][0][mask]
  chi=fields['z4c_chi'][0][mask];alpha=fields['z4c_alpha'][0][mask];regular=metric/chi[:,None,None];eig=np.linalg.eigvalsh(regular);det=np.linalg.det(metric);trace=np.einsum('pab,pba->p',np.linalg.inv(metric),A)
  checks={'all_active_fields_finite':all(np.isfinite(v[0][mask]).all() for v in fields.values()),'alpha_positive':bool((alpha>0).all()),'chi_positive':bool((chi>0).all()),'regular_metric_spd':bool((eig[:,0]>0).all())}
  record={'time':data['time'],'cycle':data['cycle'],'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'checks':{k:bool(v) for k,v in checks.items()},'alpha_min':float(alpha.min()),'chi_min':float(chi.min()),'active_full_det_residual_max_float32':float(abs(det-1).max()),'active_full_traceA_max_float32':float(abs(trace).max()),'regular_metric_eigen_min':float(eig[:,0].min()),'regular_metric_eigen_max':float(eig[:,-1].max())}
  for region,selected in [('bulk_r_lt_.9',radius<.9),('outer_r_gt_.9',radius>.9)]:
   record[region]={'nodes':int(selected.sum()),'regular_metric_eigen_min':float(eig[selected,0].min()),'regular_metric_eigen_max':float(eig[selected,-1].max()),'regular_metric_stretch_max':float((eig[selected,-1]/eig[selected,0]).max())}
  con=directory/'bin'/path.name.replace('hyp.z4c.','hyp.con.');record['constraints']=budget.analyze(con,[0,.05,.9,.95,1]);case.append(record)
 rows.append({'directory':str(directory),'snapshots':case,'all_snapshots_pass':all(all(c['checks'].values()) for c in case)})
receipt={'scope':'All saved native active field snapshots (float32 outputs), positivity/SPD/finite checks plus regular metric eigenvalues and exact native constraint spatial budgets; no interpolated fields or repaired matrices. Tiny active determinant/trace residuals include float32 output rounding.','cases':rows}
(work/'evolution-field-audit.json').write_text(json.dumps(receipt,indent=2)+'\n');print([(c['directory'],len(c['snapshots']),c['all_snapshots_pass']) for c in rows])
