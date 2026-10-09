from pathlib import Path
import hashlib,json,numpy as np
p=Path(__file__).resolve().parent
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def cm(v):a=np.asarray(v);return a[:,:,0]+1j*a[:,:,1]
report={'scope':'Stationary Einstein-reference linearization, actual20 physical-P norm-gauge RHS and coefficient-aware C1/blend subsidiary; no global spectrum or scri energy assertion.','cases':[],'convergence':[]}
for mode in ['C1','blend']:
 rows=json.loads((p/f'tangent-{mode}-kappa10.json').read_text());assert len(rows)==1000
 errors=[];gauge=[];roots=[];omissions=[];selected=[]
 for x in rows:
  q,d,s,f,g,ng=map(cm,[x['Q'],x['D'],x['subsidiary'],x['frozen'],x['constraint_generator'],x['without_dC']])
  assert np.isfinite(d).all() and abs(q[:,[0,4,5,6]]).max()==0
  rel=float(abs(d-s).max()/(1+abs(d).max()+abs(s).max()))
  if x['level']==4:
   assert rel<1e-7;errors.append(rel)
   gauge.append({'r':x['r'],'k':x['k'],'oblique':x['oblique'],
    'absolute':float(abs(d[:,[0,4,5,6]]).max()),
    'scaled':float(abs(d[:,[0,4,5,6]]).max()/(1+abs(d).max())),
    'frozen_absolute':float(abs(f[:,[0,4,5,6]]).max())})
   # Frequency weights improve the mixed-order eigensolve; they impose no data falloff.
   weights=np.array([1.]*4+[max(x['k'],1.)]*4)
   a=np.einsum('i,ij,j->ij',weights,g[:,:8],1/weights)
   ev,v=np.linalg.eig(a);z=ev[ev.real.argmax()]
   residual=max(float(abs(np.einsum('ij,j->i',a,v[:,j])-ev[j]*v[:,j]).max()/((np.max(np.sum(abs(a),axis=1))+abs(ev[j]))*abs(v[:,j]).max())) for j in range(8))
   assert residual<1e-12
   roots.append({k:x[k] for k in ['r','k','oblique']}|{'max_real':float(z.real),'imag':float(z.imag),'backward_error':residual})
   omitted=float(abs(d-ng).max()/(1+abs(d).max()+abs(ng).max()))
   omissions.append({k:x[k] for k in ['r','k','oblique']}|{'relative_closure_error_without_dC':omitted})
  if x['r'] in [.65,.75,.95,.992845500317144,.9983726993838523] and x['k']==64 and not x['oblique']:
   report['convergence'].append({'mode':mode,'r':x['r'],'level':x['level'],'h':x['h'],
    'relative_identity_error':rel,'gauge_Cdot_absolute':float(abs(d[:,[0,4,5,6]]).max()),
    'frozen_gauge_Cdot_absolute':float(abs(f[:,[0,4,5,6]]).max())})
 worst=max(roots,key=lambda x:x['max_real']);assert worst['max_real']<0
 if mode=='blend':assert max(x['relative_closure_error_without_dC'] for x in omissions)>1e-3
 report['cases'].append({'mode':mode,'kappa':10,'full20_rows':len(rows),'local_generator_rows':len(roots),
  'max_finest_relative_error':max(errors),'static_gauge_columns_exact_zero':True,
  'max_finest_gauge_absolute':max(x['absolute'] for x in gauge),
  'max_finest_gauge_scaled':max(x['scaled'] for x in gauge),
  'worst_local_root':worst,'roots':roots,'gauge':gauge,
  'worst_omitted_dC_control':max(omissions,key=lambda x:x['relative_closure_error_without_dC'])})
gate=json.loads((p/'blend-gate.json').read_text());debug=json.loads((p/'blend-gate-debug.json').read_text())
assert gate==debug
assert gate['rows']==4004 and gate['einstein_max']==gate['outer_offconstraint_max']==gate['outer_cutoff_jets_max']==0
assert gate['support_Omega_bound_margin']>=0 and gate['reference_addition_max']<1e-12
report['blend_tensor_gate']=gate
report['provenance']={f:sha(p/f) for f in ['constraint_tangent.cpp','subsidiary.hpp','subsidiary_c0.hpp','blend.hpp','bulk_c1_additions.hpp','blend_gate.cpp','principal_blend.cpp','kernel_symbol_copy.cpp']}
(p/'check-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
for c in report['cases']:print(c['mode'],c['max_finest_relative_error'],c['worst_local_root'],c['worst_omitted_dC_control'])
print('PASS actual20 C1/blend coefficient-aware subsidiary and cutoff/Einstein gates; no global acceptance')
