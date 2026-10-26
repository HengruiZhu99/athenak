#!/usr/bin/env python3
"""Independent geometric native trace targets and actual full native chart tests."""
from pathlib import Path
import argparse,json,hashlib,subprocess,sys,time,random
import mpmath as m
p=argparse.ArgumentParser();p.add_argument('binary',type=Path);p.add_argument('output',type=Path);a=p.parse_args();R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent));from goal1_global_clock_oracle import Clock
from goal1_tensor_native_oracle import TensorNative
m.mp.dps=80;E=R/'docs/validation/goal1-minkowski-20261018';W=R/'build-goal1-minkowski-20261018';cache=json.loads((E/'native-global-clock-v1/reused-height-cache.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();lines=[];records=[];start=time.time();allzero=[m.mpf(0)]*22

def add(label,mode,t,x,native,Q,dQ,tau,dtau,target,source,checks=None):
 idx=len(records);lines.append(' '.join(format(float(v),'.17g')for v in [idx,mode,t,*x,*native,Q,*dQ,tau,*dtau,*target]));records.append({'id':idx,'label':label,'mode':mode,'source':str(source.relative_to(R)),'source_sha256':sha(source),'new_geometric_trace':str(Q),'new_trace_gradient':list(map(str,dQ)),'new_trace_time':str(target[7]),'tau':str(tau),'checks':checks})
for name in ['fixture-r.98-t.75','fixture-r1-t.75','fixture-r1-t0','fixture-e-5e-6-r1-t.75','fixture-e0-r1-t.75','fixture-e5e-6-r1-t1.5']:
 meta=json.loads((E/'native-global-clock-v1'/(name+'.json')).read_text());source=W/'native-global-clock-v1'/meta.get('file',name+'.txt');z=list(map(m.mpf,source.read_text().split()));assert len(z)==251;eps=m.mpf(meta['epsilon']);t=m.mpf(meta['t']);r=m.mpf(meta['r']);x=z[3:6];native=z[6:190];target=z[190:212];oracle=Clock(r,cache)
 def geometric_q(rr,tt):
  c=oracle.core(rr,tt,eps);G=c['G'];L=c['L'];alpha=c['q']*L/m.sqrt(G)
  def beta(v):
   a=oracle.core(v,tt,eps);return -a['q']*a['v']*a['L']/a['G']
  b=beta(rr);Gr=m.diff(lambda v:oracle.core(v,tt,eps)['G'],rr);Gt=m.diff(lambda v:oracle.core(rr,v,eps)['G'],tt);br=m.diff(beta,rr)
  return(-Gt+b*Gr+2*G*br)/(2*alpha*G)+2*b/(alpha*rr)
 Q=geometric_q(r,t);Qr=m.diff(lambda rr:geometric_q(rr,t),r);Qt=m.diff(lambda tt:geometric_q(r,tt),t);dQ=[v/r*Qr for v in x];target[7]=Qt;target[17]=0;checks={}
 if r<1:
  c=oracle.core(r,t,eps);b=-c['q']*c['v']*c['L']/c['G'];alpha=c['q']*c['L']/m.sqrt(c['G']);normal=2*r*b/alpha;checks['independent_inverse_trace_error']=str(abs((native[20]-3*normal)/c['o']-Q))
 # Native flat scalar order: chi0,alpha10,P20,Theta30. Compare geometric Q only.
 add(name,1 if r==1 else 0,t,x,native,Q,dQ,0,[0]*3,target,source,checks)
 if r<1:
  for k,tau in enumerate([m.mpf('.01'),m.mpf('-.02'),m.mpf('.07')]):add(name+'-nonzero-tau'+str(k),2,t,x,native,Q,dQ,tau,[m.mpf('.03'),m.mpf('-.02'),m.mpf('.01')],allzero,source)
# Arbitrary finite source identities using existing nonreference positive native jets.
source=E/'native-scri-leading-v1/fixtures.txt';rng=random.Random(7171)
for i,row in enumerate(source.read_text().splitlines()[:12]):
 z=list(map(m.mpf,row.split()));x=[m.mpf('.98')*v for v in z[1:4]];native=z[4:];Q=m.mpf(str(rng.uniform(-4,1)));tau=m.mpf(str(rng.uniform(-.1,.1)));dQ=[m.mpf(str(rng.uniform(-.2,.2)))for _ in range(3)];dtau=[m.mpf(str(rng.uniform(-.1,.1)))for _ in range(3)];add('arbitrary-finite-'+str(i),2,0,x,native,Q,dQ,tau,dtau,allzero,source)
# New geometric LINEAR TT trace/mixed-time targets, no division by Omega.
r=m.mpf(1);t=m.mpf('.4');tt=TensorNative(r,cache)
def tt_trace(rr,ttime):
 data=tt.metric(rr,ttime);p,s,b,ar,dx,da,ba,bc=data[:8];L=1+rr*rr;B=-2*rr;Kref=-6/L
 def trh(z,tv):v=tt.metric(z,tv);return-v[0]+2*v[1]+v[3]
 lt=m.diff(lambda tv:trh(rr,tv),ttime);lr=m.diff(lambda z:trh(z,ttime),rr);bcr=m.diff(lambda z:tt.metric(z,ttime)[7],rr);return(-lt/2+bcr+(2*bc-3*ba)/rr+B*lr/2)/L-da*Kref/L
q=tt_trace(r,t);qr=m.diff(lambda rr:tt_trace(rr,t),r);qt=m.diff(lambda tv:tt_trace(r,tv),t)
for sector,pol in [('plus',m.diag([1,-1,0])),('cross',m.matrix([[0,1,0],[1,0,0],[0,0,0]]))]:
 source=E/f'native-tensor-source-v1/fixture-{sector}-r1-t.4.json';j=json.loads(source.read_text());x=list(map(m.mpf,j['xyz']));n=m.matrix(x);en=(n.T*pol*n)[0];vp=pol*n-en*n;native=list(map(m.mpf,j['native184']));target=list(map(m.mpf,j['native22_time']));target[7]=en*qt;target[17]=0;add('linear-TT-'+sector,3,t,x,native,en*q,[n[i]*en*qr+2*q*vp[i]for i in range(3)],0,[0]*3,target,source)
f=a.output/'fixtures.txt';f.write_text('\n'.join(lines)+'\n');(a.output/'independent-targets.json').write_text(json.dumps({'precision':80,'seconds':time.time()-start,'records':records,'new_targets':'Only geometric Penrose trace/value/first/time; existing native184 and all other22 time targets reused. No native RHS imported.','source_sha256':sha(Path(__file__))},indent=2)+'\n');print('targets',len(records),time.time()-start,flush=True)
with f.open()as inp:run=subprocess.run([str(a.binary.resolve())],stdin=inp,capture_output=True,text=True)
(a.output/'native.jsonl').write_text(run.stdout);(a.output/'native.stderr').write_text(run.stderr);(a.output/'native-run.json').write_text(json.dumps({'exit':run.returncode,'binary_sha256':sha(a.binary),'fixture_sha256':sha(f)},indent=2)+'\n');assert run.returncode==0,run.stderr;rows=[json.loads(s)for s in run.stdout.splitlines()];keys=['native22_chain_scaled','oracle22_scaled','numerator_value','numerator_first_compatibility','scalar_second_compatibility','unprovided_Q_second_invariance'];mx={k:max(r[k]for r in rows)for k in keys};negative=max(r['negative_omit_w_time']for r in rows);passed=len(rows)==len(records)and max(mx.values())<1e-10 and negative>.001;report={'scope':'Two native fields replaced, no extra state: full22 exact finite native chain and regular endpoint source on full clock/LINEAR TT analytic controls. Not general nonlinear invariant hierarchy or numerical closure.','samples':len(rows),'mode_counts':{str(k):sum(r['mode']==k for r in rows)for k in range(4)},'maxima':mx,'negative_omit_w_time_max':negative,'passed':passed,'new_evolved_fields':0,'active_numerical_candidate_changed':False,'general_time_invariance_admitted':False,'stable_numerical_endpoint_admitted':False};(a.output/'review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert passed
