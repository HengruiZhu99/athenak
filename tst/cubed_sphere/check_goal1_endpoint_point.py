#!/usr/bin/env python3
"""Exact native22 endpoint point equations on the verified stationary family."""
import ast,hashlib,json,subprocess,time
from pathlib import Path
import mpmath as m
m.mp.dps=80
root=Path(__file__).resolve().parents[2];out=root/'docs/validation/goal1-minkowski-20261018/scri-momentum-chart-v1';work=root/'build-goal1-minkowski-20261018/scri-momentum-chart-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();a=m.mpf('.5');one=m.mpf(1)
source=root/'tst/cubed_sphere/check_goal1_scri_momentum_chart.py'
tree=ast.parse(source.read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),str(source),'exec'))
fixture=work/'endpoint-fixtures.txt'
if not fixture.exists():
 lines=[]
 for e0 in [0,.001,-.001]:
  e=m.mpf(str(e0))
  for direction in [(1,0,0),(one*2/3,-one*2/3,one/3)]:
   x=list(direction);data=[]
   for k in range(4):data+=jet(lambda y,k=k:fields(y,e)[0][k],x)
   for i in range(3):
    data+=jet(lambda y,i=i:fields(y,e)[1][i],x)
    data+=jet(lambda y,i=i:fields(y,e)[2][i],x)
   for i in range(3):
    for j in range(i,3):
     data+=jet(lambda y,i=i,j=j:fields(y,e)[3][i][j],x)
     data+=[fields(x,e)[4][i][j]]+[derivative(lambda y,i=i,j=j:fields(y,e)[4][i][j],x,[d])for d in range(3)]
   lines.append(' '.join(m.nstr(t,45)for t in [len(lines),e,*x,*data]));print('endpoint fixture',len(lines),flush=True)
 fixture.write_text('\n'.join(lines)+'\n')
build=json.loads((out/'build.json').read_text());cmd=build['command'][:];cmd[cmd.index(str(root/'tst/cubed_sphere/goal1_scri_momentum_probe.cpp'))]=str(root/'tst/cubed_sphere/goal1_endpoint_point_probe.cpp');cmd[cmd.index('-o')+1]=str(work/'endpoint_probe');t=time.time()
with(out/'endpoint-build.log').open('w')as f:done=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
(out/'endpoint-build.json').write_text(json.dumps({'command':cmd,'exit_code':done.returncode,'seconds':time.time()-t,'source_sha256':{str(p.relative_to(root)):sha(p)for p in [source,Path(__file__),root/'tst/cubed_sphere/goal1_endpoint_point.hpp',root/'tst/cubed_sphere/goal1_endpoint_point_probe.cpp']},'binary_sha256':sha(work/'endpoint_probe')if done.returncode==0 else None},indent=2)+'\n');assert done.returncode==0
t=time.time()
with fixture.open()as fi,(out/'endpoint-native.jsonl').open('w')as fo,(out/'endpoint.stderr').open('w')as er:done=subprocess.run([str(work/'endpoint_probe')],stdin=fi,stdout=fo,stderr=er,timeout=60)
(out/'endpoint-run.json').write_text(json.dumps({'exit_code':done.returncode,'seconds':time.time()-t,'fixture_sha256':sha(fixture),'stdout_sha256':sha(out/'endpoint-native.jsonl')},indent=2)+'\n');assert done.returncode==0
rows=[json.loads(l)for l in(out/'endpoint-native.jsonl').read_text().splitlines()if l.startswith('{')];assert len(rows)==6
maxima={key:max(r[key]for r in rows)for key in ['pole_max','tangent_max','endpoint_rhs_max','unused_jet_invariance']}
old=json.loads((out/'review.json').read_text());native=[json.loads(l)for l in(out/'native.jsonl').read_text().splitlines()if l.startswith('{')];pole=[]
for endpoint in old['endpoint']:
 if endpoint['epsilon']==0:continue
 limit=endpoint['predicted_Omega_Pi_alpha_limit'];sample=[r for r in native if r['epsilon']==endpoint['epsilon']][::2]
 errors=[abs(r['Omega_Pi_alpha']-limit)for r in sample];ratios=[abs(r['Pi_alpha'])for r in sample]
 pole.append({'epsilon':endpoint['epsilon'],'nonzero_limit':limit,'absolute_limit_errors':errors,'Pi_alpha_absolute':ratios,'fine_limit_error_below_1e-4':errors[-1]<1e-4,'convergent_limit':all(y<x for x,y in zip(errors,errors[1:]))})
report={'scope':__doc__,'points':len(rows),'maxima':maxima,'all_endpoint_point_checks_pass':all(r['valid']and r['invalid_trace_rejected']for r in rows)and max(maxima[k]for k in ['pole_max','tangent_max','endpoint_rhs_max'])<=1e-10 and maxima['unused_jet_invariance']<=1e-12,'canonical_bounded_chart_rejected_for_general_scri':all(r['fine_limit_error_below_1e-4']and r['convergent_limit']for r in pole),'canonical_pole':pole,'inference':'Actual native22 regular-plus-pole first-jet quotient admits this regular exterior family; bounded old Pi excludes it. Not a general RWM gauge obstruction.','endpoint_time_compatibility_verified':False,'full_endpoint_characteristic_system_verified':False,'numerical_endpoint_closure_implemented':False,'new_evolution_trajectories':0,'source_sha256':sha(__file__)}
(out/'endpoint-review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
