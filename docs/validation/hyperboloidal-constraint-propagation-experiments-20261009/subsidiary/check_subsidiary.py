"""Checked full20 chain rule, coefficient convergence and local subsidiary roots."""
from pathlib import Path
import hashlib,json,numpy as np
p=Path(__file__).resolve().parent
def cm(v):a=np.array(v);return a[:,:,0]+1j*a[:,:,1]
report={'scope':'Actual C_Z4c0 geometric kernel dual chain rule, analytic phase and fourth-order smooth coefficient derivatives. Local subsidiary matrices are not global spectra or energy bounds. Gauge RHS does not enter these instantaneous physical constraints.', 'cases':[],'high_frequency':[],'convergence':[]}
for kap in [5,10]:
 rows=json.loads((p/f'tangent-kappa{kap}.json').read_text())
 errors=[];gauge=[];roots=[]
 for r in rows:
  q,d,s,f,g=map(cm,[r['Q'],r['D'],r['subsidiary'],r['frozen'],r['constraint_generator']])
  assert abs(q[:,[0,4,5,6]]).max()==0
  if r['h']==.000125:
   rel=float(abs(d-s).max()/(1+abs(d).max()+abs(s).max()));errors.append(rel);assert rel<1e-7
   gauge.append(float(abs(d[:,[0,4,5,6]]).max()));a=g[:,:8];weights=np.array([1.]*4+[max(r['k'],1.)]*4)
   balanced=np.einsum('i,ij,j->ij',weights,a,1/weights);e,v=np.linalg.eig(balanced)
   back=max(float(abs(np.einsum('ij,j->i',balanced,v[:,i])-e[i]*v[:,i]).max()/((np.max(np.sum(abs(balanced),axis=1))+abs(e[i]))*abs(v[:,i]).max())) for i in range(8));assert back<1e-12
   z=e[e.real.argmax()];roots.append({'r':r['r'],'k':r['k'],'oblique':bool(r['oblique']),'max_real':float(z.real),'imag':float(z.imag),'backward_residual':back})
  if kap==10 and r['r'] in [.75,.85,.95,.98] and r['k']==64 and not r['oblique']:
   report['convergence'].append({'r':r['r'],'h':r['h'],'gauge_Cdot_max':float(abs(d[:,[0,4,5,6]]).max()),'frozen_gauge_Cdot_max':float(abs(f[:,[0,4,5,6]]).max()),'relative_full_closure_error':float(abs(d-s).max()/(1+abs(d).max()+abs(s).max()))})
 worst=max(roots,key=lambda r:r['max_real']);report['cases'].append({'kappa':kap,'full20_chain_rule_samples':len(rows),'finest_h':.000125,'max_relative_closure_error':max(errors),'max_gauge_Cdot_absolute':max(gauge),'static_constraint_gauge_columns_exact_zero':True,'local_constraint_generator_samples':len(roots),'worst_root':worst,'roots':roots})
 if kap==10:assert worst['max_real']<0
 else:assert worst['max_real']>32
 rows=json.loads((p/f'generator-kappa{kap}.json').read_text())
 for r in rows:
  a=cm(r['G'])[:,:8];w=np.array([1.]*4+[r['k']]*4);a=np.einsum('i,ij,j->ij',w,a,1/w);e=np.linalg.eigvals(a);z=e[e.real.argmax()]
  report['high_frequency'].append({key:r[key] for key in ['r','k','oblique','beta_n','light_speed']}|{'kappa':kap,'max_real':float(z.real),'imag':float(z.imag),'phase_generator_over_k':float(z.imag/r['k'])})
report['energy_obstruction']={'scope':'Obstructs direct uniform monotone unweighted M²+Theta² damping estimate; does not exclude an adapted energy/symmetrizer or justified regularity estimate.','C0_trace_isotropic_term':'B_Theta=-(6*alpha*w+3*kappa_input)/Omega','outer_S1_a0.5_kappa10':'M_r,t contains +(8*r/Omega²)*Theta; damping quadratic terms are only O(1/Omega).','needed_for_uniform_absorption':'Additional controlled energy/Hardy regularity or cross terms; no live Theta falloff or invariant nonlinear regularity condition is imposed here.'}
report['provenance']={'receipt_sha256':hashlib.sha256((p/'receipt.json').read_bytes()).hexdigest(),'derived_source_sha256':hashlib.sha256((p/'subsidiary.hpp').read_bytes()).hexdigest(),'dual_kernel_driver_sha256':hashlib.sha256((p/'constraint_tangent.cpp').read_bytes()).hexdigest()}
(p/'subsidiary-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
for c in report['cases']:print(c['kappa'],'full20samples',c['full20_chain_rule_samples'],'relative identity error',c['max_relative_closure_error'],'worst local root',c['worst_root'])
print('PASS exact zero static gauge columns; full20 subsidiary chain rule; h refinement and eigen backward checks. No global stability claim.')
