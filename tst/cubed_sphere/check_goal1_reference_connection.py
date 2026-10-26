#!/usr/bin/env python3
"""Independent80-digit reference connection first gradient/Omega third jets."""
import hashlib,json,subprocess,time
from pathlib import Path
import mpmath as mp
mp.mp.dps=80
root=Path(__file__).resolve().parents[2];out=root/'docs/validation/goal1-minkowski-20261018/reference-connection-gradient-v1';work=root/'build-goal1-minkowski-20261018/reference-connection-gradient-v1';assert not(out/'review.json').exists();binary=work/'probe';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();assert sha(binary)==json.loads((out/'build.json').read_text())['binary_sha256'];requests=[];index=0
for radius in[0,.049,.05,.050001,.1,.2,.45,.7,.9,.949999,.95,.98,.999999]:
 for n in[[1,0,0],[.8,-.48,.36],[1/3**.5]*3]:requests.append(str(index)+' '+' '.join(format(radius*a,'.17g')for a in n));index+=1
(out/'samples.txt').write_text('\n'.join(requests)+'\n');t=time.time();done=subprocess.run([str(binary)],input=(out/'samples.txt').read_text(),text=True,capture_output=True,timeout=60);(out/'points.jsonl').write_text(done.stdout);(out/'points.stderr').write_text(done.stderr);(out/'run.json').write_text(json.dumps({'command':[str(binary)],'exit_code':done.returncode,'seconds':time.time()-t,'binary_sha256':sha(binary),'stdin_sha256':sha(out/'samples.txt'),'stdout_sha256':sha(out/'points.jsonl')},indent=2)+'\n');assert done.returncode==0;rows=[json.loads(l)for l in done.stdout.splitlines()if l.startswith('{')];assert len(rows)==39
lo=mp.mpf(.05);hi=mp.mpf(.95);a=mp.mpf(.5)
def omega(r):
 if r<=lo:return mp.mpf(1)
 outer=(1-r*r)/(2*a)
 if r>=hi:return outer
 s=(r-lo)/(hi-lo);t=(hi-r)/(hi-lo);w=1/(1+mp.exp(1/s-1/t));return 1+w*(outer-1)
def b(r):
 if r<=lo:return mp.mpf(0)
 if r>=hi:return r/a
 s=(r-lo)/(hi-lo);t=(hi-r)/(hi-lo);return r/a/(1+mp.exp(1/s-1/t))
def connection(x):
 r=mp.sqrt(sum(v*v for v in x))
 if r<=lo:return[mp.mpf(0)]*36
 n=[v/r for v in x];o=omega(r);op=mp.diff(omega,r);opp=mp.diff(omega,r,2);bb=b(r);bp=mp.diff(b,r);L=o-r*op;alpha=mp.sqrt(o*o+bb*bb);ar=L*(o*bp-bb*op)/alpha**3;at=bb/(alpha*r);H=[[opp*n[i]*n[j]+op/r*((1 if i==j else 0)-n[i]*n[j])for j in range(3)]for i in range(3)];g=[op*v for v in n];out=[]
 for k in range(4):
  for i in range(3):
   for j in range(3):out.append(ar*n[i]*n[j]+at*((1 if i==j else 0)-n[i]*n[j])if k==0 else -(g[j]if k-1==i else 0)-(g[i]if k-1==j else 0)-x[k-1]*o*H[i][j]/L)
 return out
results=[]
for row in rows:
 x=tuple(mp.mpf(v)for v in row['x']);err=0;third=0
 for component,cpp in enumerate(row['connection']):
  truth=[connection(x)[component]]
  for d in range(3):truth.append(mp.diff(lambda z:connection(tuple(z if i==d else x[i]for i in range(3)))[component],x[d]))
  err=max(err,max(float(abs(mp.mpf(c)-t)/max(1,abs(t)))for c,t in zip(cpp,truth)))
 for i in range(3):
  for j in range(3):
   for k in range(3):
    order=tuple(int(d==i)+int(d==j)+int(d==k)for d in range(3));truth=mp.diff(lambda *y:omega(mp.sqrt(sum(v*v for v in y))),x,order);cpp=row['omega_third'][9*i+3*j+k];third=max(third,float(abs(mp.mpf(cpp)-truth)/max(1,abs(truth))))
 results.append({'case':row['case'],'connection_value_first_scaled':err,'omega_third_scaled':third,'passed':bool(row['valid']and max(err,third)<=1e-10)});print('oracle',row['case'],err,third,flush=True)
report={'scope':__doc__,'precision_digits':80,'points':len(rows),'all_new_reference_gradient_checks_pass':all(x['passed']for x in results),'maximum_connection_first_scaled':max(x['connection_value_first_scaled']for x in results),'maximum_omega_third_scaled':max(x['omega_third_scaled']for x in results),'results':results,'reference_source_modified':False,'connection_second_derivatives_admitted':False,'exact_scri_source_admitted':False,'source_sha256':sha(Path(__file__))};(out/'review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print({k:v for k,v in report.items()if k!='results'})
