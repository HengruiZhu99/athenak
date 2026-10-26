#!/usr/bin/env python3
"""Full actual frozen-native20 energy on a mapped true-face graph; no evolution."""
import ast,hashlib,json,math
from pathlib import Path
import numpy as np
import scipy
from scipy.sparse import csr_matrix
np.seterr(all='raise')
root=Path(__file__).resolve().parents[2];base=root/'docs/validation/goal1-minkowski-20261018';out=base/'full-native-graph-v1';work=root/'build-goal1-minkowski-20261018/full-native-graph-v1';assert not(out/'review.json').exists()
for filename,names in [('check_constraint_incoming_lift.py',['mm','symbols']),('check_goal1_outer_port.py',['gram'])]:
 p=root/'tst/cubed_sphere'/filename
 for node in ast.parse(p.read_text()).body:
  if isinstance(node,ast.FunctionDef)and node.name in names:exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),globals())
native=list(symbols(base/'oblique-symbol.txt'))[:30];requests=[list(map(float,s.split()))for s in(base/'oblique-symbol.samples').read_text().splitlines()][:30]
def coeffs(state):
 cases=native[6*state:6*state+6];M=[x['matrix']for x in cases];B=M[0][:10,10:];Bi=np.linalg.inv(B);beta=np.array([-(x['lo']+x['hi'])/2 for x in cases[:3]]);A=[x[:10,:10]for x in M[:3]];Ae=[A[i]-beta[i]*np.eye(10)for i in range(3)];F=[x[10:,10:]for x in M[:3]];G=gram(cases[0]['basis'][:11,:10],cases[0]['inverse']);a=requests[6*state][25]**2*requests[6*state][7]*cases[0]['inverse'];C=np.zeros((3,3,10,10))
 for i in range(3):C[i,i]=M[i][10:,:10]
 design=[];residual=[]
 for case,m in zip(cases[3:],M[3:]):
  n=case['normal'];design.append([2*n[0]*n[1],2*n[0]*n[2],2*n[1]*n[2]]);residual.append(m[10:,:10]-sum(n[i]*n[i]*C[i,i]for i in range(3)))
 cross=np.linalg.solve(np.array(design),np.array(residual).reshape(3,100)).reshape(3,10,10)
 for pair,val in zip([(0,1),(0,2),(1,2)],cross):C[pair[0],pair[1]]=C[pair[1],pair[0]]=val
 K={};errors=[]
 for i in range(3):
  errors.append(np.linalg.norm(mm(Ae[i],B)+mm(B,F[i])-beta[i]*B)/max(1,np.linalg.norm(B)))
  for j in range(3):errors.append(np.linalg.norm((mm(Ae[i],Ae[j])+mm(Ae[j],Ae[i]))/2+mm(B,C[i,j])-a[i,j]*np.eye(10))/max(1,abs(a[i,j])))
  for j in range(i+1,3):K[i,j]=beta[j]*Ae[i]-beta[i]*Ae[j]+.5*(mm(Ae[i],Ae[j])-mm(Ae[j],Ae[i]))
 return A,Ae,B,Bi,F,C,beta,G,a,K,float(max(errors))
def readgraph(path):
 with path.open('rb')as f:
  header=np.fromfile(f,dtype='<u8',count=4);assert header[0]==0x31465342504f5347;N,na,nr=map(int,header[1:]);W=np.fromfile(f,dtype='<f8',count=N);x=np.fromfile(f,dtype='<f8',count=3*N).reshape(N,3);metric=np.fromfile(f,dtype='<f8',count=3*N).reshape(3,N);D=[];B=[]
  for d in range(3):
   for target in[D,B]:
    nnz=int(np.fromfile(f,dtype='<u8',count=1)[0]);ptr=np.fromfile(f,dtype='<i4',count=N+1);col=np.fromfile(f,dtype='<i4',count=nnz);v=np.fromfile(f,dtype='<f8',count=nnz);target.append(csr_matrix((v,col,ptr),shape=(N,N)))
  assert f.read()==b'' and np.all(W>0)
 return na,nr,W,x,metric,D,B
def ball(x,center,width,power=16):
 z=np.sum(((x-np.array(center))/width)**2,axis=1);return np.maximum(0,1-z)**power*.001
records=[];audits=[];wave=[];spatial=[]
for na0,nr0 in[(16,24),(24,36)]:
 na,nr,W,x,metric,D,Surf=readgraph(work/f'N{na0}-R{nr0}.bin');N=len(W);assert na==na0 and nr==nr0
 parity=np.empty(N);start=0
 for block in range(7):
  nx=na if block==0 else nr;size=nx*na*na;local=np.arange(size);parity[start:start+size]=1-2*((local%nx+(local//nx)%na+local//(nx*na))%2);start+=size
 values=np.column_stack([ball(x,[.22,.04,-.03],.25),ball(x,[.45,.45,.1],.2),ball(x,[.25,.125,.03125],3*.5/na,4),ball(x,[.25,.125,.03125],.18,4)*parity])
 df=np.stack([d.dot(values)for d in D],axis=0);dd=np.stack([[d.dot(df[j])for j in range(3)]for d in D],axis=0)
 for profile in range(4):
  g=df[:,:,profile];h=dd[:,:,:,profile];T=np.einsum('n,in,jkn->ijk',W,g,h);U=np.stack([mm(g,s.dot(g.T))for s in Surf]);V=np.einsum('n,dn,in,jn->dij',W,metric,g,g);H=mm(g*W[None,:],g.T);comm=h-h.swapaxes(0,1)
  spatial.append({'nodes_per_bulk_axis':na,'profile':profile,'value_sha256':hashlib.sha256(values[:,profile].tobytes()).hexdigest(),'maximum_commutator_absolute':float(np.max(abs(comm))),'commutator_W_RMS':float(np.sqrt(np.einsum('n,ijn,ijn->',W,comm,comm)/sum(W)))})
  for state in range(5):
   A,Ae,B,Bi,F,C,beta,G,a,K,werr=coeffs(state)
   if na==16 and profile==0:wave.append({'state':state,'native_wave_identity_scaled':werr,'passed':werr<1e-10})
   basis=np.linalg.inv(np.linalg.cholesky(G).T);kappa=np.array([1.,.4,-.3]);kappa/=np.linalg.norm(kappa);cdir=float(np.sqrt(kappa.dot(a.dot(kappa))));vgdef=0
   gcl_bound=2*float(np.dot(abs(beta),np.max(abs(metric),axis=1)))+2*float(np.max(np.sqrt(np.maximum(0,np.einsum('in,ij,jn->n',metric,a,metric)))))
   for mode in range(10):
    v=basis[:,mode];vv=float(v.dot(G.dot(v)));choices=[('parallel',v)]
    candidates=[(pair,np.einsum('ij,j->i',kk,v))for pair,kk in K.items()];pair,kv=max(candidates,key=lambda kv:float(kv[1].dot(G.dot(kv[1]))));kn=float(kv.dot(G.dot(kv)))
    if kn>1e-24:choices.append(('K'+str(pair[0])+str(pair[1]),kv/math.sqrt(kn)))
    else:choices.append(('K-zero-control',v))
    for direction,w in choices:
     ww=float(w.dot(G.dot(w)));wv=float(w.dot(G.dot(v)))
     for sign in[-1,1]:
      s=sign*cdir;P=np.stack([np.einsum('ij,j->i',Bi,s*kappa[i]*w-np.einsum('ij,j->i',Ae[i],v))for i in range(3)]);qd=np.stack([beta[i]*v+s*kappa[i]*w for i in range(3)]);NativePi=np.zeros((3,3,10));ideal=np.zeros_like(NativePi);correction=np.zeros_like(NativePi)
      qerr=0
      for i in range(3):
       raw=np.einsum('ij,j->i',A[i],v)+np.einsum('ij,j->i',B,P[i]);qerr=max(qerr,float(np.linalg.norm(raw-qd[i])))
       for j in range(3):NativePi[i,j]=np.einsum('ab,b->a',Ae[i],qd[j])+np.einsum('ab,b->a',B,np.einsum('ab,b->a',C[i,j],v)+np.einsum('ab,b->a',F[i],P[j]));ideal[i,j]=beta[i]*s*kappa[j]*w+a[i,j]*v
      for(i,j),kk in K.items():correction[i,j]=np.einsum('ab,b->a',kk,v);correction[j,i]=-correction[i,j]
      action=float(np.linalg.norm(NativePi-ideal-correction)/max(1,np.linalg.norm(NativePi)))
      piwork=2*s*np.einsum('a,aij,b,bc,ijc->',kappa,T,w,G,NativePi);gradwork=2*np.einsum('ij,ijr,b,bc,rc->',a,T,v,G,qd);direct=float(piwork+gradwork)
      E=float(ww*s*s*kappa.dot(H.dot(kappa))+vv*np.einsum('ij,ij->',a,H));assert E>0
      boundary=float(np.einsum('d,dij->',beta,ww*s*s*np.einsum('i,j,dij->dij',kappa,kappa,U)+vv*a[None,:,:]*U)+2*s*wv*np.einsum('dj,i,dij->',a,kappa,U))
      volume=float(-2*np.einsum('d,dij->',beta,ww*s*s*np.einsum('i,j,dij->dij',kappa,kappa,V)+vv*a[None,:,:]*V)-4*s*wv*np.einsum('dj,i,dij->',a,kappa,V))
      cp=0.
      for(i,j),kk in K.items():cp+=2*s*float(w.dot(G.dot(np.einsum('ab,b->a',kk,v))))*float(np.einsum('a,a->',kappa,T[:,i,j]-T[:,j,i]))
      cg=float(2*vv*np.einsum('ij,k,ijk->',a,beta,T-T.swapaxes(1,2)));balance=abs(direct-boundary-volume-cp-cg)/max(E,abs(direct),abs(boundary),abs(volume),abs(cp)+abs(cg));record={'bulk_nodes':na,'radial_nodes':nr,'profile':profile,'state':state,'q_mode':mode,'Pi_direction':direction,'Pi_sign':sign,'energy':E,'direct_work':direct,'boundary_work':boundary,'geometric_volume_work':volume,'Pi_commutator_work':cp,'gradient_advection_commutator_work':cg,'balance_scaled_defect':float(balance),'canonical_action_scaled_defect':action,'q_action_absolute_defect':qerr,'total_logarithmic_rate':direct/E,'commutator_logarithmic_rate':(cp+cg)/E,'scalar_geometry_logarithmic_bound':gcl_bound,'identity_pass':bool(max(balance,action,qerr,werr)<1e-10)};records.append(record)
      # Independent all-grid field action for one chart direction and both Pi
      # choices/signs on every profile; direct CSR operations, not T factorization.
      if state==0 and mode==0:
       q=values[:,profile,None]*v;pfield=mm(g.T,P);dg=np.stack([d.dot(q)for d in D]);dp=np.stack([d.dot(pfield)for d in D]);qdot=sum(mm(dg[i],A[i].T)for i in range(3))+mm(pfield,B.T);pdot=sum(mm(.5*(D[i].dot(dg[j])+D[j].dot(dg[i])),C[i,j].T)for i in range(3)for j in range(3))+sum(mm(dp[i],F[i].T)for i in range(3));dqdot=np.stack([d.dot(qdot)for d in D]);Pi=mm(pfield,B.T)+sum(mm(dg[i],Ae[i].T)for i in range(3));Pidot=mm(pdot,B.T)+sum(mm(dqdot[i],Ae[i].T)for i in range(3));Edot=2*np.einsum('n,ni,ij,nj->',W,Pi,G,Pidot)+2*np.einsum('n,ab,ani,ij,bnj->',W,a,dg,G,dqdot);Ei=np.einsum('n,ni,ij,nj->',W,Pi,G,Pi)+np.einsum('n,ab,ani,ij,bnj->',W,a,dg,G,dg);defect=max(abs(Edot-direct),abs(Ei-E))/max(E,abs(Edot),abs(direct));audits.append({'bulk_nodes':na,'profile':profile,'direction':direction,'sign':sign,'all_grid_native_matrix_action_energy_scaled':float(defect),'passed':bool(defect<1e-10)})
  print('completed',na,profile,flush=True)
summary=[]
for profile in range(4):
 for na in[16,24]:
  rows=[r for r in records if r['profile']==profile and r['bulk_nodes']==na];best=max(rows,key=lambda r:r['commutator_logarithmic_rate']);summary.append({'bulk_nodes':na,'profile':profile,'maximum_positive_commutator_logarithmic_rate':max(0,best['commutator_logarithmic_rate']),'witness':{k:best[k]for k in['state','q_mode','Pi_direction','Pi_sign','energy','boundary_work','geometric_volume_work','Pi_commutator_work','gradient_advection_commutator_work','scalar_geometry_logarithmic_bound']}})
report={'scope':__doc__,'native_coefficients_and_energy_identities_pass':all(r['identity_pass']for r in records)and all(r['passed']for r in audits),'wave_checks':wave,'independent_all_grid_action_audits':audits,'spatial_profiles':spatial,'records':records,'summary':summary,'commuting_derivative_shortcut_valid':bool(max(abs(r['commutator_logarithmic_rate'])for r in records)<1e-10),'native_coupled_evolution_started':False,'full_method_adopted':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scipy_version':scipy.__version__};(out/'review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print({k:v for k,v in report.items()if k not in['records','summary','spatial_profiles','wave_checks','independent_all_grid_action_audits']});print(json.dumps(summary,indent=2))
