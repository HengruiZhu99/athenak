"""Scale-invariant common-region readback of the two frozen/saved modes."""
from pathlib import Path
import hashlib,json
import numpy as np
W=Path(__file__).resolve().parent;B=W.parent
P=B/'mode-subsidiary-defect/immutable-mode-subsidiary-defect-20261009'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(P/'index.json')=='0ccc0eb70207cfdc7ba14d7da156063902c0850119690215f65bc0c8b3c3321b'
idx=json.loads((P/'index.json').read_text())
for name,row in idx['files'].items():assert sha(P/name)==row['sha256'],name
old_arrays=B/'mode-subsidiary-defect/diagnostic-arrays.npz'
old_record=next(r for r in idx['large_artifacts_metadata_only'] if r['path']==str(old_arrays))
assert sha(old_arrays)==old_record['sha256']
radius=json.loads((P/'results.json').read_text())['masks']['fully_nested_native']['r_max']
cases=[('N16',P,old_arrays,B/'full-tensor-propagator/full22-v2/spatialnorm-cache0.0001-metadata.json'),
       ('N20',W,W/'diagnostic-arrays.npz',B/'full-tensor-C0-N20-20261009/full22/spatialnorm-cache0.0001-metadata.json')]
out=[]
for name,p,array,meta in cases:
 results=json.loads((p/'results.json').read_text());geometry=json.loads((p/'comparator-metadata.json').read_text())
 a=np.load(array);coords=np.array(json.loads(meta.read_text())['xyz_omega_volume_ginv_chi']);r=np.linalg.norm(coords[:,:3],axis=1)
 G=np.zeros((len(r),3,3))
 for f,(i,j) in enumerate(((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))):G[:,i,j]=G[:,j,i]=coords[:,5]*coords[:,6+f]
 mask=r<=radius+1e-12;nested=np.array(geometry['fully_nested_primitive_active_stencil'],dtype=bool);assert np.all(nested[mask])
 def norm2(x):
  z=np.empty((len(r),4));z[:,0]=abs(x[:,0])**2;z[:,3]=abs(x[:,7])**2
  for k,s in enumerate((1,4),1):z[:,k]=np.einsum('pi,pij,pj->p',x[:,s:s+3].conj(),G,x[:,s:s+3],optimize=False).real
  return np.maximum(z,0)
 def rms(x):return np.sqrt(np.mean(norm2(x)[mask],axis=0)).tolist()
 q=a['q_Cv'];actual=a['r_CJv'];kc=a['strict_active_only_Kc'];matched=kc+a['strict_active_only_Uc']+a['strict_active_only_Qc']
 def case(k):
  d=actual-k
  return {'actual_rms_H_M_Z_Theta':rms(actual),'subsidiary_rms_H_M_Z_Theta':rms(k),'defect_rms_H_M_Z_Theta':rms(d),
          'defect_over_actual':np.sqrt(np.sum(norm2(d)[mask],axis=0)/np.sum(norm2(actual)[mask],axis=0)).tolist()}
 out.append({'grid':name,'cells':int(mask.sum()),'radius_min':float(r[mask].min()),'radius_max':float(r[mask].max()),
             'h':geometry['spacing'],'Omega_min':float(coords[:,3].min()),'mode_lambda':results['candidate_lambda'],
             'mode_constraint_squared_fraction':(np.sum(norm2(q)[mask],axis=0)/np.sum(norm2(q),axis=0)).tolist(),
             'centered':case(kc),'matched_Lx_KO':case(matched),
             'input_sha256':{str(x):sha(x) for x in [p/'results.json',p/'comparator-metadata.json',array,meta]}})
record={'scope':'Same physical ball r<=N16 full-nested maxradius. Different approximate modes, grids, closest-shell phase and Omega_min forbid an order or continuum growth claim.',
        'common_radius':radius,'cases':out,'source_sha256':sha(__file__)}
(W/'common-region-comparison.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
print(json.dumps(record,indent=2),flush=True)
