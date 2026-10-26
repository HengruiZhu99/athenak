import ast,json,hashlib
from pathlib import Path
import mpmath as m
m.mp.dps=80
root=Path('/Users/hz0693/research/hyperboloidal');out=root/'docs/validation/goal1-minkowski-20261018/scri-momentum-chart-v1';src=root/'tst/cubed_sphere/check_goal1_scri_momentum_chart.py';one=m.mpf(1);a=m.mpf('.5');exec(compile(ast.Module(body=[n for n in ast.parse(src.read_text()).body if isinstance(n,ast.FunctionDef)],type_ignores=[]),str(src),'exec'))
rows=[json.loads(s)for s in(out/'native.jsonl').read_text().splitlines()if s.startswith('{')];oracle=json.loads((out/'independent-oracle.json').read_text())['rows'];results=[]
for d,o in zip(rows,oracle):
 r=m.mpf(str(o['radius']));e=m.mpf(str(o['epsilon']));Omega,L,B,*_=radial(r,e);beta=-(2*r+e*L*Omega**2/r**2)/B;deriv=m.diff(lambda s:radial(s,e)[1]/m.sqrt(radial(s,e)[2]),r)-2*r;W=-beta*deriv
 limit=4/(1-8*e)*(1/m.sqrt(1-8*e)-1)
 results.append({'case':d['case'],'epsilon':o['epsilon'],'radius':o['radius'],'W_alpha_independent':float(W),'W_alpha_old_Pi_plus_F':d['Pi_alpha']+d['F_alpha'],'agreement_absolute':abs(float(W)-d['Pi_alpha']-d['F_alpha']),'endpoint_limit':float(limit),'limit_error':float(abs(W-limit))})
p={'scope':'Preliminary source-inclusive velocity chart value test, lapse component only','definition':'W = Pi + F(q,x); q_t = beta_i*g_i + W; P_new = W + kappa*q_deviation','new_gauge':False,'rows':results,'agreement_absolute_max':max(r['agreement_absolute']for r in results),'agreement_pass_1e-10':max(r['agreement_absolute']for r in results)<=1e-10,'finite_independent_endpoint_lapse_values':list({(r['epsilon'],r['endpoint_limit'])for r in results}),'full_chart_inverse_source_implemented':False,'warnings':['Naive Pi+F cancellation is not a stable endpoint implementation.','Use scaled native regular/pole parts for trace/Theta inversion; beta source quotient still requires derived gauge compatibility.','Full actual endpoint characteristics, time compatibility and stable numerical closures remain mandatory.']}
(out/'velocity-chart-value-review.json').write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({k:v for k,v in p.items()if k!='rows'},indent=2))
