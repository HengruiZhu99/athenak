"""Root independent standard-library readback; no scientific module imports."""
from pathlib import Path
from decimal import Decimal, localcontext
import hashlib,json
from collections import Counter
P=Path(__file__).resolve().parent; R=P.parents[1]
S=R/'build-layer-research/continuum/native-angular-pulse-flat-IVP-held-20261009'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
expected={'values-attempt001/receipt.json':'b054132e54cbe1e63b217f2cb1d416ea87e955f5ef1dfa2c945936e127730493','values-invocation001/receipt.json':'f6e5ee874b40543841c520f398a12c5e9f4609af29520848d72199c576e2f267','values-attempt001/checks.json':'230adc8bed460a364ddf8e41c8a34111fbb12a21829a0ae9c4c65d78aa521fae','values-recipe.json':'664b5da4f11a1f2c483cd05ec3fb79eb863262f02d6be81c4352b53a22b0cb77'}
for n,h in expected.items():assert sha(S/n)==h,n
child=load(S/'values-attempt001/receipt.json');outer=load(S/'values-invocation001/receipt.json')
assert child['passed_scalar_values_only'] and not child['accepted_native'] and not child['failed_checks']
assert outer['returncode']==0 and outer['passed_outer_process'] and not outer['accepted_native']
allpins={}
for record in [child,outer]:
 assert record['sources_unchanged'] and record['source_before']==record['source_after']
 for p,h in record['source_before'].items():assert sha(p)==h,p;allpins[p]=h
for n,h in child['output_pins'].items():
 p=Path(n);p=p if p.is_absolute() else S/'values-attempt001'/p
 assert sha(p)==h,str(p);allpins[str(p)]=h
for n in ['stdout','stderr']:assert sha(S/'values-invocation001'/(n+'.log'))==outer[n+'_sha256']
recipe=load(S/'values-recipe.json'); checks=load(S/'values-attempt001/checks.json')
assert len(checks)==282
counts=Counter(x['name'] for x in checks)
assert len(counts)==198 and all(n==(4 if k.startswith('initial/') else 1) for k,n in counts.items())
initial=load(S/'values-attempt001/initial-data.json')
assert {(x['dps'],tuple(x['xyz']),tuple(x['amplitudes'])) for x in initial}=={(d,tuple(p),tuple(a)) for d in recipe['precisions'] for p in recipe['initial_points'] for a in recipe['initial_amplitude_controls']}
initial_checks=[x for x in checks if x['name'].startswith('initial/')]
expected_initial=[('initial/%s/%s/%s'%(x['dps'],x['xyz'],k),x[k]) for x in initial for k in ['normal_data_scaled_error','determinant_scaled_error']]
assert [(x['name'],x['error']) for x in initial_checks]==expected_initial
groups={}
for x in checks:
 g=x['name'].split('/')[0]
 tol={'coarea_convergence':recipe['tolerances']['value_convergence'],'precision':recipe['tolerances']['precision'],'initial':recipe['tolerances']['initial_data'],'control':recipe['tolerances']['value_convergence'],'height':recipe['tolerances']['height_convergence'],'ray_convergence':recipe['tolerances']['ray_comparison'],'ray_coarea':recipe['tolerances']['ray_comparison'],'zero_pulse':'0'}[g]
 e,t=Decimal(x['error']),Decimal(x['tolerance'])
 assert e.is_finite() and t.is_finite() and e>=0 and t==Decimal(tol) and e<=t and x['passed'],x['name']
 groups[g]=max(groups.get(g,Decimal(0)),e)
values=load(S/'values-attempt001/values.json')
assert len(values)==96
assert {(x['dps'],x['level'],x['name']) for x in values}=={(d,l['name'],e['name']) for d in recipe['precisions'] for l in recipe['levels'] for e in recipe['events']}
for x in values:
 assert 'no native target inverse' in x['interpretation']
 for q in x['u']+x['phi']:assert Decimal(q).is_finite()
 assert Decimal(x['maximum_root_absolute_residual'])<=Decimal('1e-40')
worst=Decimal(0)
with localcontext() as ctx:
 ctx.prec=120
 for ev in recipe['events']:
  a=next(x for x in values if x['name']==ev['name'] and x['dps']==110 and x['level']=='full128')
  b=next(x for x in values if x['name']==ev['name'] and x['dps']==110 and x['level']=='full64')
  for field in ['u','phi']:
   err=max(abs(Decimal(u)-Decimal(v))/(1+abs(Decimal(u))) for u,v in zip(a[field],b[field]))
   worst=max(worst,err);assert err<=Decimal('1e-10')
for n,k in [('initial-data.json',56),('controls.json',72),('height.json',108),('rays.json',18)]:assert len(load(S/'values-attempt001'/n))==k
for p,h in allpins.items():assert sha(p)==h,p
out={'passed_saved_readback':True,'source_sha256':sha(__file__),'receipt_and_recipe_pins':expected,'unique_inputs_outputs_rehashed':len(allpins),'checks':282,'actual_grid_values':96,'all_fixed_tolerances_rechecked':True,'groups':{k:str(v) for k,v in groups.items()},'independently_recomputed_110digit_full64_full128_max_u_phi':str(worst),'no_scientific_imports_or_queries':True,'scope':'Finite reference-event values only; no Jacobian/inverse/native-target/global-caustic acceptance.'}
dest=P/'report003.json'
with dest.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(out,indent=2))
