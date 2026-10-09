"""Exact alpha-row value-column attribution for actual raw22/projected20 matrices."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.sparse import load_npz,coo_matrix
w=Path(__file__).resolve().parent;old=w.parents[2]/'full-tensor-propagator/full22-v2';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();mx=lambda A:float(abs(A.data).max()) if A.nnz else 0.
coef=np.array(json.loads((w/'reference-coefficients.json').read_text()));xyz=coef[:,:3];O=coef[:,3];r=coef[:,4];h=coef[:,5];beta=coef[:,6:9];G=coef[:,9:12];dh=coef[:,12:15];c=coef[:,15];n=len(h);B=np.sum(beta*G,axis=1)/h;alpha=3*c*(h+2*c)*B/O;betac=-3*c[:,None]*(h+2*c)[:,None]*G/O[:,None]
combined=True
if combined:alpha-=c*np.sum(beta*dh,axis=1)/h;betac-=c[:,None]*dh
expected=np.column_stack([alpha,betac]);res={'variant':'combined','reference_formula':'Falpha=3c(h+2c)(beta_ref.gradOmega)/(hOmega); Fbeta=−3c(h+2c)gradOmega/Omega; combined adds−c(beta_ref.gradh)/h and−cgradh','combined':combined,'coefficient_schema':['x','y','z','Omega','r','alpha_ref','beta_x','beta_y','beta_z','dOmega_x','dOmega_y','dOmega_z','dalpha_x','dalpha_y','dalpha_z','1-Wgauge'],'coefficients_sha256':sha(w/'reference-coefficients.json'),'outer_r_ge085_points':int(np.sum(r>=.85)),'core_r_le005_points':int(np.sum(r<=.05)),'expected_coefficients_max_abs':float(abs(expected).max()),'matrices':{}}
for kind,size,alphacol,file in [('raw22',22,18,'spatialnorm-cache0.0001-J22.npz'),('projected20',20,16,'spatialnorm-projected-J20.npz')]:
 J=load_npz(w/file);base=load_npz(old/file);rows=np.repeat(size*np.arange(n)+alphacol,4);cols=np.ravel(size*np.arange(n)[:,None]+alphacol+np.arange(4)[None,:]);E=coo_matrix((expected.ravel(),(rows,cols)),shape=J.shape).tocsr();E.eliminate_zeros();D=J-base;error=D-E;outside=D.tolil();outside[rows,cols]=0.;outside=outside.tocsr();outside.eliminate_zeros();outer=np.ravel(size*np.where(r>=.85)[0][:,None]+np.arange(size));outererr=mx(D[outer,:]);record={'matrix_sha256':sha(w/file),'C0_matrix_sha256':sha(old/file),'actual_changed_entries':D.nnz,'expected_entries':E.nnz,'max_absolute_error_vs_exact':mx(error),'max_unexpected_entry':mx(outside),'outer_allrows_changed_entry_max':outererr,'passed':mx(error)<1e-8 and mx(outside)<1e-8 and outererr==0};assert record['passed'];res['matrices'][kind]=record
 if kind=='projected20':
  v=np.load(w.parent/'spatialnorm-validation-vectors.npz')['gauge_pulse'];change=(D@v).reshape(n,20);other=change.copy();other[:,16]=0;res['initial_gauge_changed_action_l2']=float(np.linalg.norm(change));res['initial_gauge_nonalpharows_changed_action_linf']=float(abs(other).max());assert not other.any()
(w/'actual-matrix-trace-attribution.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
