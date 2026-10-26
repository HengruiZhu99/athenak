#!/usr/bin/env python3
"""Independent MP nonlinear native scri leading-pole fixtures, no evolved chart."""
from pathlib import Path
import argparse,json,random,subprocess,hashlib,time
import mpmath as m
p=argparse.ArgumentParser();p.add_argument('binary',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=True);m.mp.dps=80;rng=random.Random(16271);pairs=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)];records=[];lines=[]
def rand(scale):return m.mpf(str(rng.uniform(-scale,scale)))
def dot(x,y):return sum(x[i]*y[i]for i in range(3))
def tr(x):return sum(x[i,i]for i in range(3))
def scalar(v,d=None,dd=None):
 d=d if d is not None else [m.mpf(0)]*3;dd=dd if dd is not None else m.zeros(3);return[v,*d,*[dd[i,j]for i,j in pairs]]
for idx in range(96):
 if idx<6:n=m.matrix([int(k==idx//2)*(-1 if idx%2 else 1)for k in range(3)])
 else:n=m.matrix([rand(1)for _ in range(3)]);n/=m.sqrt(dot(n,n))
 nu=-2*n;M=m.eye(3)
 for i in range(3):
  for j in range(3):M[i,j]+=rand(.045)
 b=M*M.T;C=b**-1;chi=m.det(b)**(-m.mpf(1)/3);g=chi*b;gi=g**-1;db=[];dg=[];dc=[];dlog=[]
 for d in range(3):
  B=m.zeros(3)
  for i in range(3):
   for j in range(i,3):B[i,j]=B[j,i]=rand(.08)
  db.append(B);dlog.append(-tr(C*B)/3);dc.append(chi*dlog[-1]);dg.append(dc[-1]*b+chi*B)
 ddchi=m.matrix(3);ddg=[[None]*3 for _ in range(3)]
 for d in range(3):
  for e in range(3):ddchi[d,e]=chi*(tr(C*db[e]*C*db[d])/3+dlog[d]*dlog[e]);ddg[d][e]=ddchi[d,e]*b+dc[d]*db[e]+dc[e]*db[d]
 v=C*n;c=dot(n,v);T=tr(C)-dot(v,v)/c;root=m.sqrt(c);alpha=4*root/T;beta=-4*v/T;P=-6*root;theta=m.mpf(0);w=-2*root;kappa=10/alpha;p1=rand(.15);tau=rand(.04);a1=rand(.1);b1=m.matrix([rand(.1)for _ in range(3)]);dP=[];dtheta=[];dalpha=[];dbeta=[]
 for d in range(3):
  dn=m.matrix([int(i==d)-n[i]*n[d]for i in range(3)]);Cd=-C*db[d]*C;vd=Cd*n+C*dn;cd=2*dot(dn,v)+dot(n,Cd*n);Td=tr(Cd)-2*dot(v,vd)/c+dot(v,v)*cd/c**2;dP.append(-3*cd/root+nu[d]*p1);dtheta.append(nu[d]*tau);dalpha.append(2*cd/(root*T)-4*root*Td/T**2+nu[d]*a1);dbeta.append(-4*vd/T+4*v*Td/T**2+nu[d]*b1)
 # Independent ordinary Christoffels of Penrose and twice-conformal metrics.
 def connection(inv,der):return[[[sum(inv[k,l]*(der[i][l,j]+der[j][l,i]-der[l][i,j])/2 for l in range(3))for j in range(3)]for i in range(3)]for k in range(3)]
 cb=connection(C,db);cg=connection(gi,dg);gamma=m.matrix([sum(gi[i,j]*cg[k][i][j]for i in range(3)for j in range(3))for k in range(3)]);H=m.matrix(3)
 for i in range(3):
  for j in range(3):H[i,j]=-2*int(i==j)-sum(cb[k][i][j]*nu[k]for k in range(3))
 H-=g*tr(gi*H)/3;q=gi*nu;J=gi*H*q;rhs=-m.mpf(2)/3*gi*m.matrix([2*dP[j]+dtheta[j]for j in range(3)])+4*chi/w*J;matrix=2*kappa*m.eye(3)-(4*chi/(3*w))*q*nu.T;zu=m.lu_solve(matrix,rhs);zc=g*zu;A=-chi/w*(H+zc*nu.T+nu*zc.T-m.mpf(2)/3*g*dot(zu,nu));lam=gamma+2*zu
 flat=scalar(chi,dc,ddchi)+scalar(alpha,dalpha)+scalar(P,dP)+scalar(theta,dtheta,-2*tau*m.eye(3))
 for i in range(3):flat+=scalar(beta[i],[dbeta[d][i]for d in range(3)])+scalar(lam[i])
 for i,j in pairs:flat+=scalar(g[i,j],[dg[d][i,j]for d in range(3)],m.matrix([[ddg[d][e][i,j]for e in range(3)]for d in range(3)]))+[A[i,j],0,0,0]
 assert len(flat)==184
 # Independent geometric/scalar/gauge and A/Lambda residue identities.
 K=P+2*theta;G=dot(nu,C*nu);L=alpha**2*C-beta*beta.T;gauge_alpha=-alpha*(alpha*P+dot(beta,nu)+tr(L));gauge_beta=-8*alpha**2*C*n+beta*(4*dot(beta,n)-tr(L));resA=2*alpha*chi*(H+zc*nu.T+nu*zc.T-m.mpf(2)/3*g*dot(zu,nu))+alpha*A*(K-w);resL=-alpha*(m.mpf(4)/3*K+2*kappa)*zu-m.mpf(2)/3*alpha*gi*m.matrix([2*dP[j]+dtheta[j]for j in range(3)])-4*alpha*gi*A*gi*nu
 residual=max([abs(K-3*w),abs(G-w*w),abs(gauge_alpha),abs(m.det(g)-1),abs(tr(gi*A)),*[abs(v)for v in gauge_beta],*[abs(v)for v in resA],*[abs(v)for v in resL]])
 records.append({'id':idx,'n':[float(v)for v in n],'max_MP_identity_error':float(residual),'alpha':float(alpha),'beta':[float(v)for v in beta],'P':float(P),'T_over_c':float(T/c),'native_target_sha256':hashlib.sha256((' '.join(str(float(v))for v in flat)).encode()).hexdigest()});lines.append(' '.join(str(float(v))for v in [idx,*n,*flat]))
fixtures=a.output/'fixtures.txt';fixtures.write_text('\n'.join(lines)+'\n');(a.output/'independent-fixtures.json').write_text(json.dumps({'precision_digits':80,'scope':'Leading point manifold; first tangential alpha/beta/P/Theta jets compatible with chosen native metric on sphere; no complete higher-jet or vacuum/time hierarchy claim.','records':records},indent=2)+'\n');start=time.time()
with fixtures.open()as inp:run=subprocess.run([str(a.binary.resolve())],stdin=inp,capture_output=True,text=True)
(a.output/'native.jsonl').write_text(run.stdout);(a.output/'native.stderr').write_text(run.stderr);(a.output/'native-run.json').write_text(json.dumps({'exit':run.returncode,'seconds':time.time()-start,'binary_sha256':hashlib.sha256(a.binary.read_bytes()).hexdigest(),'fixture_sha256':hashlib.sha256(fixtures.read_bytes()).hexdigest()},indent=2)+'\n');assert run.returncode==0,run.stderr
rows=[json.loads(s)for s in run.stdout.splitlines()];keys=['value_error','actual_native22_pole','A_trace'];mx={k:max(r[k]for r in rows)for k in keys};neg={k:max(r[k]for r in rows)for k in ['negative_no_A','negative_reference_leading','negative_no_Z_feedback']};passed=len(rows)==96 and max(mx.values())<1e-10 and max(r['max_MP_identity_error']for r in records)<1e-70 and min(neg.values())>.01;report={'scope':'Actual full native22 nonlinear leading pole manifold only; no evolved chart or numerical closure.','samples':len(rows),'maxima':mx,'independent_MP_max':max(r['max_MP_identity_error']for r in records),'negative_controls_max':neg,'minimum_absolute_divisors':{k:min(abs(r[k])for r in rows)for k in ['tangential_divisor','normal_divisor','scalar_gap']},'passed_declared_point_gates':passed,'time_hierarchy_admitted':False,'general_characteristics_admitted':False,'exact_scri_numerical_admitted':False,'added_evolved_fields':0};(a.output/'review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert passed
