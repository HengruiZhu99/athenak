from pathlib import Path
import numpy as np,json
from frozen_symbol import Frozen
from principal_symbol import TF
out=[]
for h in [.5,.25,.125,.0625]:
 f=Frozen([h/2]*3,h=h);A=f.evaluate([0,0,0])[0];e,V=np.linalg.eig(A);ix=np.argsort(-e.real);n=np.ones(3)/np.sqrt(3);q=[]
 for k in ix[:8]:
  v=V[:,k];v=v/max(abs(v));T=np.einsum('q,qij->ij',v[7:12],TF);Tn=T@n;nn=n@Tn;Tv=np.outer(n,Tn-n*nn)+np.outer(Tn-n*nn,n)
  q.append({'lambda':[e[k].real,e[k].imag],'chi_abs':float(abs(v[0])),'Khat_abs':float(abs(v[6])),'Theta_abs':float(abs(v[15])),'lapse_abs':float(abs(v[16])),'Gamma_norm':float(np.linalg.norm(v[12:15])),'Gamma_radial_fraction':float(abs(n@v[12:15])/max(1e-30,np.linalg.norm(v[12:15]))),'A_radial_vector_fraction':float(np.linalg.norm(Tv)/max(1e-30,np.linalg.norm(T)))})
 out.append({'h':h,'xi':[0,0,0],'note':'Local diagonal frozen point; radial projection classifies the fastest eigenspace only, not global perturbations.','largest_eigenmodes':q})
Path('nearest-mode-classification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
