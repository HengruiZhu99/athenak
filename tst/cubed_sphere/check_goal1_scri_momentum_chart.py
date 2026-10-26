#!/usr/bin/env python3
"""Independent exterior harmonic-time Minkowski chart fixtures; no trajectories."""
import hashlib,json,subprocess,time
from pathlib import Path
import mpmath as m
m.mp.dps=80
root=Path(__file__).resolve().parents[2]
out=root/'docs/validation/goal1-minkowski-20261018/scri-momentum-chart-v1'
work=root/'build-goal1-minkowski-20261018/scri-momentum-chart-v1'
protocol=json.loads((out/'protocol-v1.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=m.mpf('.5');one=m.mpf(1)
def radial(r,e):
 o=1-r*r;L=1+r*r;b=2*r
 B=1-2*e*L*b/r**2-e**2*L**2*o**2/r**4
 kr=-(1/a-2*e*L**3/r**3)/B**m.mpf('1.5')
 kt=-(1/a+e*L*o**2/r**3)/m.sqrt(B)
 delta_over_o=e*L*o/r**3*(2+B-e*L/(a*r))/B**m.mpf('1.5')
 return o,L,B,kr,kt,delta_over_o
def fields(x,e):
 r=m.sqrt(sum(t*t for t in x));n=[t/r for t in x];o,L,B,kr,kt,dko=radial(r,e);chi=B**(-one/3)
 bp=m.diff(lambda s:radial(s,e)[2],r)
 lr=(one*2/3)*B**(-one*5/3)*bp+2*B**(-one*2/3)*(B-1)/r
 beta=-(2*r+e*L*o**2/r**2)/B
 g=[[chi*((i==j)+(B-1)*n[i]*n[j])for j in range(3)]for i in range(3)]
 A=[[chi*dko/3*(2*B*n[i]*n[j]-((i==j)-n[i]*n[j]))for j in range(3)]for i in range(3)]
 return [chi,L/m.sqrt(B),kr+2*kt,m.mpf(0)], [beta*t for t in n],[lr*t for t in n],g,A
def derivative(fn,x,indices):
 orders=[indices.count(i)for i in range(3)]
 return m.diff(lambda *y:fn(y),tuple(x),tuple(orders)) if indices else fn(x)
def jet(fn,x,second=True):
 return [fn(x)]+[derivative(fn,x,[i])for i in range(3)]+[derivative(fn,x,[i,j]) if second else m.mpf(0)for i in range(3)for j in range(i,3)]
fixture=work/'fixtures.txt'
if fixture.exists():
 oracle=json.loads((out/'independent-oracle.json').read_text())['rows']
else:
 fixtures=[];oracle=[]
 for e0 in protocol['epsilon']:
  e=m.mpf(str(e0))
  for r0 in protocol['radii']:
   r=m.mpf(str(r0))
   for direction in [(1,0,0),(m.mpf(2)/3,-m.mpf(2)/3,one/3)]:
    x=[r*t for t in direction];idx=len(oracle);data=[]
    for k in range(4):data+=jet(lambda y,k=k:fields(y,e)[0][k],x,second=k<2)
    for i in range(3):
     data+=jet(lambda y,i=i:fields(y,e)[1][i],x)
     data+=jet(lambda y,i=i:fields(y,e)[2][i],x,False)
    for i in range(3):
     for j in range(i,3):
      data+=jet(lambda y,i=i,j=j:fields(y,e)[3][i][j],x)
      data+=[fields(x,e)[4][i][j]]+[derivative(lambda y,i=i,j=j:fields(y,e)[4][i][j],x,[d])for d in range(3)]
    fixtures.append(' '.join(m.nstr(t,45)for t in [idx,e,*x,*data]))
    o,L,B,kr,kt,dko=radial(r,e);R=r/o;h=2*r/L+e/R**2;s=o**2*B/L**2
    hp=m.diff(lambda t:2*t/(1+t*t)+e*(1-t*t)**2/t**2,r)*o**2/L
    errk=max(abs(kr+hp/s**m.mpf('1.5')),abs(kt+h/(R*m.sqrt(s))),abs(kr-kt-o*dko))
    # Physical Hamiltonian and momentum for spherical Minkowski embedding.
    metric_R=1-h*h
    scalar3=2*(1-1/metric_R)/R**2+2*(2*h*hp)/(-metric_R**2*R)
    hamiltonian=scalar3+(kr+2*kt)**2-(kr**2+2*kt**2)
    momentum=m.diff(lambda t:radial(t,e)[4],r)*o**2/L-(kr-kt)/R
    harmonic=m.diff(lambda t:e/t,R,2)+2/R*m.diff(lambda t:e/t,R)
    oracle.append({'case':idx,'epsilon':e0,'radius':r0,'extrinsic_consistency':float(errk),'physical_H':float(hamiltonian),'physical_M':float(momentum),'harmonic_inverse_map':float(harmonic),'spacelike_B':float(B),'physical_trace':float(kr+2*kt)})
    print('fixture',idx,flush=True)
 fixture=work/'fixtures.txt';assert not fixture.exists();fixture.write_text('\n'.join(fixtures)+'\n')
 (out/'independent-oracle.json').write_text(json.dumps({'precision_digits':80,'rows':oracle},indent=2)+'\n')
build=json.loads((root/'docs/validation/goal1-minkowski-20261018/native-reduced-source-v1/build.json').read_text());cmd=build['command'][:]
cmd[cmd.index(str(root/'tst/cubed_sphere/goal1_native_reduced_probe.cpp'))]=str(root/'tst/cubed_sphere/goal1_scri_momentum_probe.cpp')
cmd[cmd.index('-o')+1]=str(work/'chart_probe')
t=time.time()
with(out/'build.log').open('w')as f:done=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
(out/'build.json').write_text(json.dumps({'command':cmd,'exit_code':done.returncode,'seconds':time.time()-t,'source_sha256':sha(root/'tst/cubed_sphere/goal1_scri_momentum_probe.cpp'),'binary_sha256':sha(work/'chart_probe')if done.returncode==0 else None},indent=2)+'\n')
assert done.returncode==0
t=time.time()
with fixture.open()as fi,(out/'native.jsonl').open('w')as fo,(out/'native.stderr').open('w')as er:done=subprocess.run([str(work/'chart_probe')],stdin=fi,stdout=fo,stderr=er,timeout=60)
(out/'run.json').write_text(json.dumps({'exit_code':done.returncode,'seconds':time.time()-t,'fixture_sha256':sha(fixture),'stdout_sha256':sha(out/'native.jsonl')},indent=2)+'\n');assert done.returncode==0
rows=[json.loads(l)for l in(out/'native.jsonl').read_text().splitlines()if l.startswith('{')];assert len(rows)==30
endpoint=[]
for e0 in protocol['epsilon']:
 e=m.mpf(str(e0));o,L,B,kr,kt,dko=radial(one,e);alpha=L/m.sqrt(B);beta=-2/B;chi=B**(-one/3);g_rr=chi*B
 endpoint.append({'epsilon':e0,'B':float(B),'alpha':float(alpha),'chi':float(chi),'beta_r':float(beta),'physical_trace':float(kr+2*kt),'nullness_absolute':float(abs(beta**2-alpha**2*chi/g_rr)),'predicted_Omega_Pi_alpha_limit':float(-alpha**2*(kr+2*kt+3/a)),'A_radial':float(chi*dko*2*B/3)})
report={'scope':__doc__,'points':len(rows),'independent_physical_identity_max':max(abs(p[k])for p in oracle for k in ['extrinsic_consistency','physical_H','physical_M','harmonic_inverse_map']),'conditioned_native_rhs_max':max(p['conditioned_rhs']for p in rows),'native_raw_rhs_max':max(p['native_rhs_max']for p in rows),'q_decomposition_max':max(p['q_decomposition_absolute']for p in rows),'endpoint':endpoint,'native_stationary_roundoff_conditioned_pass':max(p['conditioned_rhs']for p in rows)<=2e-10,'physical_oracle_pass':max(abs(p[k])for p in oracle for k in ['extrinsic_consistency','physical_H','physical_M','harmonic_inverse_map'])<=1e-60,'new_trajectories':0,'exact_endpoint_RHS_implemented':False,'source_sha256':sha(__file__)}
(out/'review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
