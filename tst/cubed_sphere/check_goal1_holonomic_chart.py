#!/usr/bin/env python3
"""Independent high-precision chart, jet and native-time comparisons."""
import hashlib,json
from pathlib import Path
import mpmath as mp
mp.mp.dps=80
root=Path(__file__).resolve().parents[2];out=root/'docs/validation/goal1-minkowski-20261018/holonomic-chart-v1';assert not(out/'review.json').exists();runs=json.loads((out/'runs.json').read_text());rows=[]
for run in runs:
 p=out/(run['label']+'.jsonl');assert hashlib.sha256(p.read_bytes()).hexdigest()==run['stdout_sha256'];rows.extend(json.loads(line)for line in p.read_text().splitlines()if line.startswith('{'))
fixture=json.loads((out/'fixture-protocol-v1.json').read_text());expected={(x['kind'],x['case']):x for x in fixture['cases']};admission=[]
for row in rows:
 e=expected.get((row['kind'],row['case']),{'admit':True});admission.append(row['valid']==e['admit'])
def polynomial(j,x):
 return mp.mpf(j[0])+sum(mp.mpf(j[1+i])*x[i]for i in range(3))+sum(mp.mpf(j[4+3*i+k])*x[i]*x[k]/2 for i in range(3)for k in range(3))
def decode(q):
 l00=mp.exp(q[1]);l11=mp.exp(q[2]);l22=mp.exp(-q[1]-q[2]);a,b,c=q[3:6];return [l00*l00,l00*a,l00*b,a*a+l11*l11,a*b+l11*c,b*b+c*c+l22*l22]
def encode(m):
 l00=mp.sqrt(m[0]);a=m[1]/l00;b=m[2]/l00;l11=mp.sqrt(m[3]-a*a);c=(m[4]-a*b)/l11;return[mp.log(l00),mp.log(l11),a,b,c]
def jet(f):
 z=(mp.mpf(0),)*3;orders=[(0,0,0)]+[tuple(int(i==a)for i in range(3))for a in range(3)]+[tuple(int(i==a)+int(i==b)for i in range(3))for a in range(3)for b in range(3)];return [mp.diff(lambda *x:f(x),z,o)for o in orders]
comparisons=[];worst=0;timeworst=0
for row in rows:
 if not row['valid']or row['kind']=='metric-input':continue
 native=row['kind'].startswith('native-')
 if native and row['case']%23!=0:continue
 maxima={}
 if row['chart_input']is not None:
  target=row['chart_input'];cpp=row['metric'];err=0
  for component in range(6):
   truth=jet(lambda x:decode([polynomial(q,x)for q in target])[component]);err=max(err,max(float(abs(mp.mpf(a)-b)/max(1,abs(b)))for a,b in zip(cpp[component],truth)))
  maxima['independent_decode_scaled']=err
 # Native metric jets are input data for an independent inverse-map derivative;
 # this validates the chart, not a repeated admission of reference geometry.
 m=row['metric'];cpp=row['chart'];err=0
 for component in range(5):
  truth=jet(lambda x:encode([polynomial(q,x)for q in m])[component]);err=max(err,max(float(abs(mp.mpf(a)-b)/max(1,abs(b)))for a,b in zip(cpp[1+component],truth)))
 maxima['independent_encode_scaled']=err
 v=row['native_config_velocity'];qdot=row['chart_velocity'];expected_time=[mp.mpf(v[0])]+[mp.diff(lambda t:encode([mp.mpf(m[i][0])+t*mp.mpf(v[1+i])for i in range(6)])[j],0)for j in range(5)]+[mp.mpf(v[7])]+[mp.mpf(a)for a in v[8:]];terr=max(float(abs(mp.mpf(a)-b)/max(1,abs(b)))for a,b in zip(qdot,expected_time));maxima['independent_time_scaled']=terr;timeworst=max(timeworst,terr);worst=max(worst,max(maxima.values()));comparisons.append({'kind':row['kind'],'case':row['case'],**maxima,'passed':max(maxima.values())<=1e-10})
 print('oracle',row['kind'],row['case'],flush=True)
groups=[]
for kind in['config','reference','native-stationary4','native-pulse8','metric-input']:
 rs=[r for r in rows if r['kind']==kind and r['valid']];keys=['roundtrip_value','roundtrip_first','roundtrip_second','time_roundtrip','det_value_error','det_first','det_second','det_time'];maxima={k:max(abs(r[k])for r in rs)for k in keys};groups.append({'kind':kind,'points':len(rs),'maxima':maxima,'value_time_roundtrip_pass':max(maxima['roundtrip_value'],maxima['time_roundtrip'])<=1e-10,'differential_roundtrip_pass':max(maxima['roundtrip_first'],maxima['roundtrip_second'])<=1e-10})
negative=[r for r in rows if r['kind']=='metric-input'];negative_pass=bool(len(negative)==2 and all(max(r['roundtrip_first'],r['roundtrip_second'],r['time_roundtrip'])>1e-3 and max(r['det_first'],r['det_second'])>1e-3 for r in negative))
report={'scope':__doc__,'precision_digits':80,'all_expected_value_admissions_pass':all(admission),'independent_oracle_pass':all(c['passed']for c in comparisons),'independent_oracle_points':len(comparisons),'maximum_independent_scaled_error':worst,'off_differential_constraint_negative_control_detected':negative_pass,'groups':groups,'comparisons':comparisons,'native_trajectories_run':0,'native_nonlinear_source_implemented':False,'exact_scri_equations_implemented':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};(out/'review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print({k:v for k,v in report.items()if k not in['comparisons','groups']});print(groups)
