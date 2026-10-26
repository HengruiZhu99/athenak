#!/usr/bin/env python3
"""Full50 symmetric first-order reduction of actual frozen native20, no evolution."""
import ast,json,hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix
np.seterr(all='raise')
root=Path(__file__).resolve().parents[2];base=root/'docs/validation/goal1-minkowski-20261018';out=base/'first-order-reduction-v1';work=root/'build-goal1-minkowski-20261018/full-native-graph-v1';assert not(out/'review.json').exists()
for filename,names in [('check_constraint_incoming_lift.py',['mm','symbols']),('check_goal1_outer_port.py',['gram']),('check_goal1_native_graph_energy.py',['coeffs','readgraph','ball'])]:
 p=root/'tst/cubed_sphere'/filename
 for node in ast.parse(p.read_text()).body:
  if isinstance(node,ast.FunctionDef)and node.name in names:exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),globals())
native=list(symbols(base/'oblique-symbol.txt'))[:30];requests=[list(map(float,s.split()))for s in(base/'oblique-symbol.samples').read_text().splitlines()][:30]
I=np.eye(10);rng=np.random.default_rng(20261018);kappa=1.;states=[];points=[];charts=[]
for state in range(5):
 A,Ae,B,Bi,F,C,beta,G,a,K,werr=coeffs(state);H=block_diag(G,G,np.kron(a,G));T=np.linalg.cholesky(H).T;Ti=np.linalg.inv(T);M=np.zeros((3,50,50));L=np.zeros((50,50));L[:10,:10]=-kappa*I;L[:10,10:20]=I;L[10:20,:10]=-kappa*kappa*I;L[10:20,10:20]=kappa*I
 for i in range(3):
  gi=slice(20+10*i,30+10*i);L[:10,gi]=beta[i]*I;L[gi,gi]=-kappa*I;M[i,10:20,10:20]=beta[i]*I;M[i,gi,10:20]=I
  for j in range(3):
   gj=slice(20+10*j,30+10*j);M[i,10:20,gj]=a[i,j]*I;M[i,gj,gj]=beta[i]*I
 Mb=np.stack([mm(mm(T,m),Ti)for m in M]);Lb=mm(mm(T,L),Ti);sym=float(np.max(abs(Mb-Mb.swapaxes(1,2)))/max(1,np.max(abs(Mb))));sourcebound=float(np.linalg.eigvalsh(Lb+Lb.T)[-1]);charerr=0
 for case in native[6*state:6*state+6]:
  n=case['normal'];bn=float(n.dot(beta));c=float(np.sqrt(n.dot(a.dot(n))));raw=np.einsum('i,ijk->jk',n,Mb);vals=np.linalg.eigvalsh((raw+raw.T)/2);expected=np.sort(np.r_[np.zeros(10),np.full(10,bn-c),np.full(10,bn+c),np.full(20,bn)]);charerr=max(charerr,float(np.max(abs(vals-expected))/max(1,np.max(abs(expected)))))
 pmax=0;qmax=0;cmax=0
 for trial in range(100):
  u=np.einsum('ij,j->i',Ti,rng.standard_normal(50));du=np.einsum('ij,kj->ki',Ti,rng.standard_normal((3,50)));rhs=np.einsum('iab,ib->a',M,du)+np.einsum('ab,b->a',L,u);q=u[:10];P=u[10:20];g=u[20:].reshape(3,10);dq=du[:,:10];dP=du[:,10:20];dg=du[:,20:].reshape(3,3,10);pi=P-kappa*q;dpi=dP-kappa*dq;c=g-dq;p=np.einsum('ab,b->a',Bi,pi-sum(np.einsum('ab,b->a',Ae[i],g[i])for i in range(3)));dp=np.stack([np.einsum('ab,b->a',Bi,dpi[i]-sum(np.einsum('ab,b->a',Ae[j],dg[i,j])for j in range(3)))for i in range(3)])
  nativeq=sum(np.einsum('ab,b->a',A[i],g[i])for i in range(3))+np.einsum('ab,b->a',B,p);nativep=sum(np.einsum('ab,b->a',C[i,j],.5*(dg[i,j]+dg[j,i]))for i in range(3)for j in range(3))+sum(np.einsum('ab,b->a',F[i],dp[i])for i in range(3));extra=np.zeros(10)
  for i in range(3):
   extra+=kappa*np.einsum('ab,b->a',beta[i]*I-Ae[i],c[i])
   for j in range(i+1,3):extra-=.5*np.einsum('ab,b->a',mm(Ae[i],Ae[j])-mm(Ae[j],Ae[i]),dg[i,j]-dg[j,i])
  reducedp=np.einsum('ab,b->a',Bi,rhs[10:20]-kappa*rhs[:10]-sum(np.einsum('ab,b->a',Ae[i],rhs[20+10*i:30+10*i])for i in range(3)));corrected=nativep+np.einsum('ab,b->a',Bi,extra);pmax=max(pmax,float(np.linalg.norm(reducedp-corrected)/max(1,np.linalg.norm(reducedp),np.linalg.norm(corrected))));qmax=max(qmax,float(np.linalg.norm(nativeq-rhs[:10])/max(1,np.linalg.norm(nativeq))));expectedc=np.stack([sum(beta[j]*(dg[j,i]-dg[i,j])for j in range(3))-kappa*c[i]for i in range(3)]);dc=rhs[20:].reshape(3,10)-(sum(beta[j]*dg[:,j]for j in range(3))+dP-kappa*dq);cmax=max(cmax,float(np.max(abs(dc-expectedc))/max(1,np.max(abs(expectedc)))))
 record={'state':state,'native_wave_identity':werr,'principal_symmetry_scaled':sym,'full50_characteristic_scaled_defect':charerr,'native_p_reconstruction_scaled':pmax,'native_q_reconstruction_scaled':qmax,'reduction_constraint_time_identity_scaled':cmax,'symmetric_source_growth_bound':sourcebound,'minimum_symmetrizer_eigenvalue':float(np.linalg.eigvalsh(H)[0]),'symmetrizer_condition':float(np.linalg.cond(H)),'passed':bool(max(werr,sym,charerr,pmax,qmax,cmax)<1e-10)};states.append(record);charts.append((Mb,Lb,beta,a,sourcebound))
for na,nr in[(16,24),(24,36)]:
 _,_,W,x,metric,D,S=readgraph(work/f'N{na}-R{nr}.bin');N=len(W);f=ball(x,[.22,.04,-.03],.25);g=ball(x,[.45,.45,.1],.2);h=ball(x,[.25,.125,.03125],.18,4);values=np.column_stack([f,g,h,np.sin(1.7*x[:,0]-.4*x[:,1]+.6*x[:,2])*.001]);weights=rng.standard_normal((4,50));y=mm(values,weights);dy=np.stack([d.dot(y)for d in D]);Ey=float(np.einsum('n,ni,ni->',W,y,y));surface=[s.dot(y)for s in S]
 for state,(Mb,Lb,beta,a,sourcebound)in enumerate(charts):
  rhs=sum(mm(dy[i],Mb[i].T)for i in range(3))+mm(y,Lb.T);direct=2*float(np.einsum('n,ni,ni->',W,y,rhs));boundary=sum(float(np.einsum('ni,ni->',y,mm(surface[i],Mb[i].T)))for i in range(3));volume=-2*sum(float(np.einsum('n,n,ni,ni->',W,metric[i],y,mm(y,Mb[i].T)))for i in range(3));source=2*float(np.einsum('n,ni,ni->',W,y,mm(y,Lb.T)));defect=abs(direct-boundary-volume-source)/max(Ey,abs(direct),abs(boundary),abs(volume),abs(source));geo=max(0,float(np.max(-2*np.einsum('i,in->n',beta,metric)+2*np.sqrt(np.maximum(0,np.einsum('in,ij,jn->n',metric,a,metric))))));bound=(geo+sourcebound)*Ey;excess=(direct-boundary-bound)/Ey;points.append({'bulk_nodes':na,'radial_nodes':nr,'state':state,'all50_energy':Ey,'direct_work':direct,'exterior_work':boundary,'geometric_work':volume,'source_work':source,'energy_identity_scaled_defect':defect,'geometric_growth_bound':geo,'source_growth_bound':sourcebound,'total_growth_bound':geo+sourcebound,'actual_log_rate_after_exterior':(direct-boundary)/Ey,'bound_excess_over_energy':excess,'passed':bool(defect<1e-10 and excess<1e-10)})
 print('completed graph',na,flush=True)
report={'scope':__doc__,'actual_native_principal_reduction_pass':all(s['passed']for s in states),'full50_frozen_graph_energy_bound_pass':all(p['passed']for p in points),'states':states,'global_energy_checks':points,'point_jet_tests':500,'independent_gradient_variables':True,'arbitrary_gradient_resets':False,'same_RWM_gauge':True,'native_nonlinear_source_implemented':False,'native_evolution_started':False,'exact_scri_implemented':False,'full_method_adopted':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(out/'review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))
