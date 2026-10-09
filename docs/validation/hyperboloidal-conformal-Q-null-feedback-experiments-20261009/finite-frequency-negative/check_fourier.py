"""Retain every positive primitive frozen root; no subsidiary/global inference."""
from pathlib import Path
import json
import numpy as np
P=Path(__file__).resolve().parent;data=json.loads((P/'fourier.json').read_text());assert len(data)==1120
matrices={};roots=[];outer_error=core_error=k0_error=onset_error=0.
for d in data:
 m=np.array(d['M']);m=m[:,:,0]+1j*m[:,:,1];assert np.isfinite(m).all();key=(d['a'],d['r'],d['k'],d['dir'],d['form']);matrices[key]=m
 e=np.linalg.eigvals(m);j=e.real.argmax();roots.append({k:d[k]for k in ['a','r','Omega','k','dir','form']}|{'max_real':float(e[j].real),'imag':float(e[j].imag),'positive_roots':int(np.sum(e.real>1e-8))})
for (a,r,k,n,f),m in matrices.items():
 if r>=.85 and f==1:outer_error=max(outer_error,float(np.max(np.abs(m-matrices[a,r,k,n,2]))))
 if r==.45 and f==2:core_error=max(core_error,float(np.max(np.abs(m-matrices[a,r,k,n,3]))))
 if k==0:k0_error=max(k0_error,float(np.max(np.abs(m.imag))),float(np.max(np.abs(m-matrices[a,r,k,1-n,f]))))
 if r==.85 and f==1:onset_error=max(onset_error,float(np.max(np.abs(m-matrices[a,r,k,n,0]))))
assert max(outer_error,core_error,k0_error,onset_error)<1e-10
summary=[]
for a in [.5,.75,1,2]:
 for f in range(4):summary.append(max([r for r in roots if r['a']==a and r['form']==f],key=lambda r:r['max_real']))
report={'matrix_count':1120,'forms':{'0':'originalQ preferred sigma0','1':'globalQ preferred sigma5 v.85-.95','2':'physical-inner alpha blend Q preferred sigma5 v.85-.95 (currentcandidate)','3':'physicalP sourceoff spatialnorm xi1/a baseline'},'k0_direction_imaginary_error':k0_error,'outer_two_variant_equality_error':outer_error,'inner_W0_baseline_equality_error':core_error,'onset_feedback_zero_identity_error':onset_error,'worst_by_a_and_form':summary,'interpretation':'Positive primitive roots retained. Reference frozen full20 generator includes local coefficient values/reference jets but does not differentiate the perturbation coefficient amplitudes or supply global energy, boundary or constraint-subsidiary classification. No global/native stability accepted.','scope':'Fresh separate finite-Omega reference screen; no native run, no change to frozen core helper or original382inputs.'}
(P/'fourier-roots.json').write_text(json.dumps(roots,indent=2)+'\n');(P/'fourier-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
