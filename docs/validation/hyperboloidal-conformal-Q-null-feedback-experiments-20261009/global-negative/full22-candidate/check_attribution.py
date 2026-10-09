"""Independent algebraic reference Jacobian delta, never fit from cached rows."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.sparse import block_diag,load_npz
w=Path(__file__).resolve().parent;old=w.parents[1]/'full-tensor-propagator/full22-v2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();mx=lambda A:float(abs(A.data).max()) if A.nnz else 0.
c=np.asarray(json.loads((w/'reference-coefficients.json').read_text()));N=len(c);B=np.zeros((N,22,22));axes=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
for p,z in enumerate(c):
 r,O,h,chi,K,wo,W,nu,eta,v=z[3:13];beta=z[13:16];G=z[16:19];dh=z[19:22];dc=z[22:25];lam=z[25:28];inv=z[28:37].reshape(3,3);H=z[37:46].reshape(3,3);T=chi*inv;q=G@G;Bh=beta@G;Ghat=G@T@G;n=-G/np.sqrt(q);L=dc/(2*chi)-dh/h;fhat=beta@dh/h-h*K
 for field in [0,1,2,3,4,5,6,18,19,20,21]:
  da=float(field==18);db=np.zeros(3);dg=np.zeros((3,3));dchi=float(field==0)
  if 19<=field<=21:db[field-19]=1
  if 1<=field<=6:i,j=axes[field-1];dg[i,j]=dg[j,i]=1
  dT=dchi*inv-chi*inv@dg@inv;dG=G@dT@G;D=Bh/h*da-db@G
  alpha=W*(-beta@dh/h*da-db@dh+(3*(h+2*(1-W))*D+2*W*2*h*da+W*(h*db@G+Bh*da))/O)
  df=db@dh/h+nu*da/h-K*da
  ds=2*h*da*chi*lam+h*h*dchi*lam+eta*db-db*fhat-beta*df+2*h*da*(T@L)+h*h*dT@L
  dDelta=2*h*da*np.sum(T*H)+h*h*np.sum(dT*H)-2*db@H@beta-2*h*da*O*wo-G@ds
  dNull=h*h*dG+2*Bh*D
  dBeta=-W*G*dDelta/q+v*5*G*dNull/(q*O)+6*W*(db+(2/3)*n*dG/Ghat)/O
  B[p,18,field]=alpha;B[p,19:22,field]=dBeta
E=block_diag(B,format='csr');E.eliminate_zeros();base=load_npz(old/'spatialnorm-cache0.0001-J22.npz');J=load_npz(w/'spatialnorm-cache0.0001-J22.npz');D=J-base;error=D-E
rows=np.ravel(22*np.arange(N)[:,None]+np.arange(18));outside=D[rows,:]
# Geometry rows are bitwise original. Gauge residual tolerance allows cached
# fourth-order finite-difference coefficient roundoff, not a fitted delta.
res={'scope':'exact value-only local reference delta of Q physical-inner/preferred/null gauge minus original C0 spatialnorm, expanded into native global ordering; no changed stencil/principal','formula_notes':['deltaT=deltaChi*ginv-chi*ginv*deltaGmetric*ginv','alpha_delta=W*(Qalpha-Palpha), including regular reference-advection difference and xi2 physical source','beta_delta=preferred regular projection + sigma5 weighted-null pole - original spatialnorm pole','all delta input derivatives are zero; only value columns chi/g/alpha/beta change; physical P/Theta/A/Lambda rows unchanged'],'coefficient_schema':['x','y','z','r','Omega','alpha','chi','Kbar','wOmega','Wgauge','nu','eta','Vnull','beta[3]','dOmega[3]','dalpha[3]','dchi[3]','Lambda_ref[3]','ginv[9]','OmegaHessian[9]'],'coefficients_sha256':sha(w/'reference-coefficients.json'),'raw22':{'matrix_sha256':sha(w/'spatialnorm-cache0.0001-J22.npz'),'original_matrix_sha256':sha(old/'spatialnorm-cache0.0001-J22.npz'),'actual_changed_entries':int(D.nnz),'expected_value_entries':int(E.nnz),'max_absolute_error_vs_independent_formula':mx(error),'unchanged_geometry_rows_max':mx(outside),'max_expected_entry':mx(E),'tolerance':1e-7}}
L=np.fromfile(w/'spatialnorm-cache0.0001-lift.bin','<f8').reshape(N,22,20);P=np.fromfile(w/'spatialnorm-cache0.0001-restrict.bin','<f8').reshape(N,20,22);F=np.einsum('pij,pjk,pkl->pil',P,B,L);E20=block_diag(F,format='csr');E20.eliminate_zeros();j=load_npz(w/'spatialnorm-projected-J20.npz');b=load_npz(old/'spatialnorm-projected-J20.npz');res['projected20']={'matrix_sha256':sha(w/'spatialnorm-projected-J20.npz'),'max_absolute_error_vs_independent_formula':mx(j-b-E20),'actual_changed_entries':int((j-b).nnz),'expected_value_entries':int(E20.nnz),'max_expected_entry':mx(E20),'tolerance':1e-7}
res['passed']=res['raw22']['max_absolute_error_vs_independent_formula']<1e-7 and res['projected20']['max_absolute_error_vs_independent_formula']<1e-7 and res['raw22']['unchanged_geometry_rows_max']==0
(w/'actual-matrix-Q-attribution.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2));assert res['passed']
