#!/usr/bin/env python3
"""Independent80-digit full velocity momentum, split source and inverse jets."""
import ast,hashlib,json
from pathlib import Path
import mpmath as mp
mp.mp.dps=80
root=Path(__file__).resolve().parents[2];out=root/'docs/validation/goal1-minkowski-20261018/velocity-momentum-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=root/'tst/cubed_sphere/check_goal1_typed_ad.py';tree=ast.parse(old.read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),str(old),'exec'))
def metric_from_velocity(g,v):
 L=chol(g);Ld=mp.matrix([[L[0,0]*v[0],0,0],[v[2],L[1,1]*v[1],0],[v[3],v[4],-L[2,2]*(v[0]+v[1])]])
 return Ld*L.T+L*Ld.T
def source_parts(u,r,ograd,c):
 third=mp.mpf(1)/3;g=tensor(u[2:8]);ig=g**-1;gh=tensor(r[2:8]);igh=gh**-1;a=u[1];h=r[1];chi=u[0];ch=r[0];beta=u[8:11];bh=r[8:11];db=[v-v0 for v,v0 in zip(beta,bh)]
 C=chi*ig;Ch=ch*igh;V=a*a*C;Vh=h*h*Ch;L=V-mp.matrix(beta)*mp.matrix(beta).T;Lh=Vh-mp.matrix(bh)*mp.matrix(bh).T;dL=L-Lh;dV=V-Vh
 div=sum(r[28+3*i+i]for i in range(3));w=-sum(beta[i]*ograd[i]for i in range(3))/a
 regular=[-2*third*chi*div+sum(beta[i]*r[22+i]for i in range(3))]+[mp.mpf(0)]*9
 pole=[2*third*a*chi*(r[11]+2*r[12]-3*w)]+[mp.mpf(0)]*9
 M=-2*a*tensor(r[13:19])-2*third*g*div
 for d in range(3):M+=beta[d]*metric_from_velocity(g,velocity(gh,tensor(r[37+6*d:43+6*d])))
 for i in range(3):
  for j in range(3):M[i,j]+=sum(g[k,i]*r[28+3*j+k]+g[k,j]*r[28+3*i+k]for k in range(3))
 regular[1:6]=velocity(g,M)
 regular[6]=-(a-h)*sum(bh[i]*r[25+i]for i in range(3))/h+sum(db[i]*r[25+i]for i in range(3))
 pole[6]=-a*((a-h)*r[11]+sum(db[i]*ograd[i]for i in range(3)))-a*sum(dL[i,j]*c[0][i][j]for i in range(3)for j in range(3))
 for i in range(3):
  regular[7+i]=(a*a*chi-h*h*ch)*r[19+i]+sum(db[j]*r[28+3*j+i]+(a*a*ig[i,j]-h*h*igh[i,j])*r[22+j]/2-(a*C[i,j]-h*Ch[i,j])*r[25+j]for j in range(3))
  pole[7+i]=2*sum(dV[i,j]*ograd[j]for j in range(3))-sum(dL[j,k]*(c[i+1][j][k]+beta[i]*c[0][j][k])+Lh[j,k]*db[i]*c[0][j][k]for j in range(3)for k in range(3))
 return regular,pole
rows=[]
for run in json.loads((out/'runs.json').read_text()):
 file=out/(run['label']+'.jsonl');assert sha(file)==run['stdout_sha256'];rows.extend(json.loads(l)for l in file.read_text().splitlines()if l.startswith('{')and 'input_u'in l)
assert len(rows)==8;oracle=[]
for row in rows:
 reference=[mp.mpf(v)for v in row['input_ref']];c=[[[mp.mpf(row['connection'][9*k+3*i+j])for j in range(3)]for i in range(3)]for k in range(4)];cache={}
 def evaluate(x):
  key=(mp.mp.prec,tuple(v._mpf_ for v in x))
  if key not in cache:
   u=[poly(v,x)for v in row['input_u']];o=poly(row['input_Omega'],x);ograd=[poly(v,x)for v in row['input_Omega_gradient']];fr,fp=source_parts(u,reference,ograd,c);F=[a+b/o for a,b in zip(fr,fp)];pi=momentum(u,reference,o);forward=[a+b for a,b in zip(pi,F)];target=[poly(v,x)for v in row['target_W']];native=inverse(u,reference,o,[w-f for w,f in zip(target,F)]);cache[key]=fr+fp+forward+native
  return cache[key]
 maxima={};offset=0
 for label in ['source_regular','source_pole','forward_W','inverse_p']:
  maximum=0
  for i,cpp in enumerate(row[label]):
   truth=derivatives(lambda x:evaluate(x)[offset+i]);maximum=max(maximum,max(float(abs(mp.mpf(v)-w)/max(1,abs(w)))for v,w in zip(cpp,truth)))
  maxima[label]=maximum;offset+=len(row[label])
 oracle.append({'kind':row['kind'],'ordinal':row['ordinal'],'case':row['case'],'maxima':maxima,'passed':max(maxima.values())<=1e-10});print('oracle',row['kind'],row['ordinal'],maxima,flush=True)
report={'scope':__doc__,'precision_digits':80,'points':len(oracle),'maxima':{k:max(r['maxima'][k]for r in oracle)for k in oracle[0]['maxima']},'all_independent_map_checks_pass':all(r['passed']for r in oracle),'oracle':oracle,'reuse_old_independent_helpers_sha256':sha(old),'source_sha256':sha(__file__),'native_trajectories':0};(out/'independent-review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))
