from pathlib import Path
from decimal import Decimal,localcontext
from collections import Counter
import hashlib,json
P=Path(__file__).resolve().parent
B=P.parent/'continuum'
N=B/'native-angular-pulse-flat-derivatives-compact-root-launch-held-20261009/comparison-attempt001'
O=B/'native-angular-pulse-flat-derivatives-timing-launch-held-20261009/timing-attempt001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
paths=[N/'receipt.json',N/'rays.json',N/'checks.json',N/'groups.json',O/'receipt.json',O/'rays.json']
before={str(p):sha(p) for p in paths}
assert before[str(N/'receipt.json')]=='45b9967263a81cc09e0234f88f6fedf8c6a45561b10277b2cfcd0f643a0cf86f'
assert before[str(O/'rays.json')]=='07474689a98f05fff14fc42539d5170fce8dab64d274d81fd8657f506669fccc'
receipt=load(N/'receipt.json');original=load(O/'receipt.json')
assert receipt['passed_compact_root_comparison_gate'] and receipt['sources_unchanged'] and receipt['failed_checks']==[]
assert (receipt['ray_rows'],receipt['group_rows'],receipt['checks'])==(480,24,38344)
for path,digest in receipt['output_pins'].items():assert sha(path)==digest,path
key=lambda r:tuple(r[k] for k in ('dps','level','event','boost','polar_index','azimuth_index'))
new=load(N/'rays.json');old=load(O/'rays.json');oldby={key(r):r for r in old}
assert len(new)==len(oldby)==480 and len({key(r) for r in new})==480
checks=load(N/'checks.json');assert len(checks)==38344 and len({r['name'] for r in checks})==38344
assert len(load(N/'groups.json'))==24
methods=Counter();maximum_jet=Decimal(0);maximum_metric=Decimal(0);maximum_root=Decimal(0)
with localcontext() as ctx:
 ctx.prec=120
 for c in checks:
  error,tol=Decimal(c['error']),Decimal(c['tolerance'])
  assert error.is_finite() and tol.is_finite() and 0<=error<=tol and c['passed'] is True,c['name']
 for row in new:
  prior=oldby[key(row)];audit=row['compact_root'];methods[audit['method']]+=1
  assert audit['accepted'] is True
  for a,b in zip(row['jet'],prior['jet']):
   assert len(a)==len(b)==15
   for x,y in zip(a,b):
    x,y=Decimal(x),Decimal(y);err=abs(x-y)/max(Decimal(1),abs(x),abs(y))
    assert err<=Decimal('1e-30');maximum_jet=max(maximum_jet,err)
  for x,y in zip(row['k'],prior['k']):
   x,y=Decimal(x),Decimal(y);assert abs(x-y)/max(Decimal(1),abs(x),abs(y))<=Decimal('1e-35')
  for name,x in row['metrics'].items():
   x,y=Decimal(x),Decimal(prior['metrics'][name]);tol=Decimal('1e-40' if name=='root' else '1e-35')
   err=abs(x-y)/max(Decimal(1),abs(x),abs(y));assert err<=tol;maximum_metric=max(maximum_metric,err)
  assert Decimal(row['metrics']['minimum_D'])>0
  maximum_root=max(maximum_root,Decimal(audit['original_g_residual']))
  assert Decimal(audit['original_g_residual'])<=Decimal('1e-40')
  if audit['method'].startswith('compact_'):
   assert Decimal(audit['F_left'])<=0<=Decimal(audit['F_right'])
   assert 0<Decimal(audit['radius_width'])<Decimal('1e-50')
   assert 0<=Decimal(audit['lambda_width'])<Decimal('1e-50')
   assert Decimal(audit['lambda_left'])>=Decimal(audit['lambda_right'])>=0
  else:
   assert Decimal(audit['radius_width'])==Decimal(audit['lambda_width'])==0
assert all(sha(Path(path))==digest for path,digest in before.items())
result={'passed_independent_saved_decimal_readback':True,'pins':before,'ray_rows':480,'checks':38344,'jet_components_independently_compared':28800,'methods':dict(methods),'maximum_scaled_jet_difference':str(maximum_jet),'maximum_scaled_metric_difference':str(maximum_metric),'maximum_original_root_residual':str(maximum_root),'new_child_seconds':receipt['seconds'],'old_child_seconds':original['seconds'],'scope':'Saved scalar arithmetic/complete fixed ray sample only; no oracle rerun, angular accuracy, interval enclosure, inverse or PDE acceptance.'}
with (P/'saved-readback001.json').open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result))
