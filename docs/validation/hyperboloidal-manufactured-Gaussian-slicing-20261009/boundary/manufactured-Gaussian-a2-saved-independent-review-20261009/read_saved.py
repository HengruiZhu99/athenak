#!/usr/bin/env python3
"""Independent stdlib-only source/provenance and saved scalar reductions."""
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()
def pin(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
def read(path):return json.loads(Path(path).read_text())
def write(path,obj):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def require(ok,why):
    if not ok:raise RuntimeError(why)
def guard(rows):
    for row in rows:require(pin(row['path'])==row,'Protected drift '+row['path'])

def review(recipe):
    owner=Path(recipe['owner']);old=Path(recipe['old_owner']);root=Path(recipe['root_release'])
    current=read(owner/'recipe.json');prior=read(old/'recipe.json')
    require(current['a']=='2' and prior['a']=='1/2','Exact a control')
    other=[k for k in set(current)|set(prior) if k not in ('a','pins') and current.get(k)!=prior.get(k)]
    require(not other,'Non-parameter recipe change '+str(other))
    require((owner/'screen.py').read_bytes()==(old/'screen.py').read_bytes(),'Scientific code equality')
    require(sha(owner/'screen.py')==recipe['scientific_source_sha256'],'Expected frozen scientific source')
    for path,digest in prior['pins'].items():require(current['pins'].get(path)==digest,'Prior pin changed '+path)
    for path,digest in current['pins'].items():require(sha(path)==digest,'Current recipe input drift '+path)
    index=read(owner/'source-index.json')
    guard(index['files'])
    require(index['parameter_control']=={'a_before':'1/2','a_after':'2'},'Source index parameter labels')
    require(index['scientific_source_byte_identical'] is True,'Source equality label')
    auth=read(root/'authorization.json');release=read(root/'release.json')
    require(auth['finite_Gaussian_screen_authorized'] is True,'Actual root release')
    for key,path in [('recipe_sha256',owner/'recipe.json'),('source_index_sha256',owner/'source-index.json'),('screen_source_sha256',owner/'screen.py'),('root_review_sha256',root/'source-review001.json')]:
        require(auth[key]==sha(path),'Authorization binding '+key)
    require(release['authorization_sha256']==sha(root/'authorization.json'),'Release authorization binding')
    for path,digest in release['pins'].items():require(sha(path)==digest,'Release input drift '+path)
    child=owner/'attempts/screen001';outer=root/'invocation001'
    receipt=read(child/'receipt.json');out=read(outer/'receipt.json');result=read(child/'result.json')
    require(receipt['completed'] is True and receipt['returncode']==0 and receipt['checks_passed'] is True and receipt['inputs_unchanged'] is True,'Actual child success')
    require(out['completed'] is True and out['returncode']==0 and out['accepted_finite_screen_identity'] is True and out['inputs_unchanged'] is True,'Actual outer success')
    require(out['child_receipt_sha256']==sha(child/'receipt.json') and out['result_sha256']==sha(child/'result.json'),'Outer child/result binding')
    guard(out['output_inventory'])
    require((outer/'stderr.log').read_bytes()==b'','Unexpected outer stderr')
    before=read(child/'pins-before.json')
    require(before==dict(release['pins'],**{str(root/'authorization.json'):sha(root/'authorization.json')}),'Child before pin map exact')
    for path,digest in read(outer/'pins-before.json').items():require(sha(path)==digest,'Outer before input drift '+path)
    prior_receipt=read(old/'attempts/screen001/receipt.json');prior_result=read(old/'attempts/screen001/result.json')
    require(prior_receipt['completed'] is True and prior_receipt['returncode']==0 and prior_receipt['inputs_unchanged'] is True and prior_result['checks_passed'] is True,'Prior actual consistency gate')
    require(prior_result['all_sampled_D_positive'] is False,'Prior negative screen retained')
    history=read(root/'preparation-history.json')
    require(history['first_tool_invocation_rejected_before_execution'] is True and history['filesystem_or_scientific_command_executed'] is False,'Recorded mechanical preparation history')

    # Independent grid-label enumeration; no Gaussian or owner functions are called.
    time_sets={}
    for radius in current['radius_over_sigma']:
        r=Fraction(radius)
        times={Fraction(s) for s in current['time_over_sigma']}
        times.update(r+Fraction(s) for s in current['retarded_time_over_sigma'] if r+Fraction(s)>=0)
        time_sets[radius]=times
    expected_events=sum(len(v) for v in time_sets.values())*len(current['sigma'])*len(current['epsilon'])
    require(expected_events==current['anticipated_records_per_precision']==28032,'Independent expected count')
    rows=read(child/'samples.json')
    require(len(rows)==56064,'Complete saved rows')
    expected_keys={(digits,sigma,epsilon,radius,t) for digits,_ in current['precision_and_terms'] for sigma in current['sigma'] for epsilon in current['epsilon'] for radius,times in time_sets.items() for t in times}
    seen=set();counts=Counter();branches=Counter();low={};high={};all_positive=True;all_J_positive=True
    identity=Decimal(0);max_precision=Decimal(0)
    with localcontext() as ctx:
        ctx.prec=190
        for row in rows:
            numeric={name:Decimal(row[name]) for name in ['time_over_sigma','D','D_over_reference_D','minimizing_p','J_at_minimum','angular_identity_residual']}
            require(all(v.is_finite() for v in numeric.values()),'Saved nonfinite scalar')
            time_value=numeric['time_over_sigma'];canonical=Fraction(str(time_value.quantize(Decimal('0.000001'))))
            exact_time=Decimal(canonical.numerator)/Decimal(canonical.denominator)
            require(abs(time_value-exact_time)<=Decimal('1e-70'),'Original key-time guard')
            key=(row['digits'],row['sigma'],row['epsilon'],row['radius_over_sigma'],canonical)
            require(key in expected_keys and key not in seen,'Unexpected/duplicate saved event')
            seen.add(key)
            require(row['terms']==dict(current['precision_and_terms'])[row['digits']],'Series term binding')
            r=Fraction(row['radius_over_sigma'])
            branch='origin' if r==0 else 'origin_series' if r<=Fraction(1,8) else 'advanced_retarded'
            require(row['branch']==branch,'Branch label')
            require(row['passed_positive'] is (numeric['D']>0),'Saved positivity flag')
            require(abs(numeric['minimizing_p'])<=Decimal('.5'),'Saved angular domain')
            require(numeric['angular_identity_residual']>=0,'Negative identity residual')
            counts[(row['digits'],row['sigma'],row['epsilon'])]+=1;branches[(row['digits'],branch)]+=1
            all_positive=all_positive and numeric['D']>0;all_J_positive=all_J_positive and numeric['J_at_minimum']>0
            identity=max(identity,numeric['angular_identity_residual'])
            without_precision=key[1:]
            (low if row['digits']==current['precision_and_terms'][0][0] else high)[without_precision]=row
        require(seen==expected_keys,'Missing expected events')
        require(len(low)==len(high)==28032,'Both precision grids')
        for key,x in low.items():
            y=high[key]
            for name in ['D','D_over_reference_D','J_at_minimum']:
                xx,yy=Decimal(x[name]),Decimal(y[name])
                max_precision=max(max_precision,abs(xx-yy)/max(Decimal(1),abs(xx),abs(yy)))
        require(abs(max_precision-Decimal(result['precision_comparison_max']))<=Decimal('1e-100'),'Saved precision aggregate serialization')
        require(identity==Decimal(result['angular_identity_max']),'Saved identity maximum')
        require(max_precision<=Decimal(current['precision_tolerance']) and identity<=Decimal(current['identity_tolerance']),'Original saved consistency thresholds')
        profile_report=[]
        for saved in result['profiles']:
            group=[row for row in high.values() if row['sigma']==saved['sigma'] and row['epsilon']==saved['epsilon']]
            require(len(group)==saved['samples']==3504,'Profile count')
            worst=min(group,key=lambda row:Decimal(row['D_over_reference_D']))
            require(worst==saved['worst'],'Exact saved worst row')
            require(saved['all_sampled_D_positive'] is all(row['passed_positive'] for row in group),'Profile positive aggregate')
            profile_report.append({'sigma':saved['sigma'],'epsilon':saved['epsilon'],'samples_per_precision':len(group),'worst_D_over_reference_D':worst['D_over_reference_D'],'worst_D':worst['D'],'radius_over_sigma':worst['radius_over_sigma'],'physical_time_over_sigma':worst['time_over_sigma'],'minimizing_p':worst['minimizing_p'],'J_at_minimum':worst['J_at_minimum']})
        require(len(profile_report)==8 and {(q['sigma'],q['epsilon']) for q in profile_report}=={(s,e) for s in current['sigma'] for e in current['epsilon']},'All eight profile aggregates')
    require(result['checks_passed'] is True and result['samples_per_precision']==28032,'Compact consistency result')
    require(result['all_sampled_D_positive'] is all_positive is True and all_J_positive,'All saved D/J positive')
    require(result['global_positivity_proven'] is False and result['continuum_or_native_stability_accepted'] is False,'Required finite-screen limitations')
    return {'passed':True,'source_byte_identical':True,'only_scientific_recipe_change':'a:1/2->2','additive_recipe_pin_count':len(set(current['pins'])-set(prior['pins'])),'saved_records':len(rows),'events_per_precision':expected_events,'profile_count':len(profile_report),'profiles':profile_report,'branches':[{'digits':d,'branch':b,'count':n} for (d,b),n in sorted(branches.items())],'precision_max_reduced_from_saved_strings':str(max_precision),'identity_max_reduced_from_saved_strings':str(identity),'all_saved_D_positive':all_positive,'all_saved_J_positive':all_J_positive,'actual_child_seconds':receipt['seconds'],'actual_outer_seconds':out['seconds'],'recorded_pre_command_preparation_failure_preserved':True,'scientific_targets_recomputed':False,'screen_imported_or_reexecuted':False,'global_positivity_native_inverse_jets_PDE_or_stability_admitted':False,'scope':'Exact source/provenance and saved finite physical-event reductions only; no Gaussian/angular target recomputation.'}

def main():
    require(sys.flags.isolated and sys.dont_write_bytecode and not sys.flags.optimize,'Require -I -B unoptimized')
    recipe=read(HERE/'recipe.json');index=read(HERE/'source-index.json')
    attempt=HERE/'attempt001';attempt.mkdir(exist_ok=False)
    started=time.monotonic();receipt={'completed':False,'passed':False,'returncode':1,'scientific_execution':False}
    try:
        guard(recipe['input_pins']);guard(index['files'])
        report=review(recipe);write(attempt/'report.json',report)
        guard(recipe['input_pins']);guard(index['files'])
        receipt.update(completed=True,passed=True,returncode=0,inputs_unchanged=True,report=pin(attempt/'report.json'))
    except BaseException as error:
        receipt['failure']={'type':type(error).__name__,'message':str(error)}
        try:guard(recipe['input_pins']);guard(index['files']);receipt['inputs_unchanged']=True
        except BaseException as drift:receipt['input_guard_failure']=str(drift)
    finally:
        receipt.update(seconds=time.monotonic()-started,review_source=pin(__file__),source_index=pin(HERE/'source-index.json'))
        write(attempt/'receipt.json',receipt)
    print(json.dumps({'passed':receipt['passed'],'receipt':pin(attempt/'receipt.json')}))
    return receipt['returncode']

if __name__=='__main__':raise SystemExit(main())
