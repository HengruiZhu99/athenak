#!/usr/bin/env python3
"""Compact saved-only audit. Never opens a JSONL or imports a target."""
from pathlib import Path
from fractions import Fraction
from decimal import Decimal, localcontext
import json, hashlib, sys
HERE=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def require(v,why):
    if not v: raise ValueError(why)

def reject_constant(x): raise ValueError('nonfinite JSON '+x)
def load(p): return json.loads(Path(p).read_text(),parse_constant=reject_constant)
def show(q):
    with localcontext() as c:
        c.prec=24
        return str(Decimal(q.numerator)/Decimal(q.denominator))

m=load(HERE/'capture.json')
by_origin={}
for x in m['inputs']:
    p=HERE/x['copy']; require(p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],'captured input drift')
    require(Path(x['origin']).stat().st_size==x['bytes'] and sha(x['origin'])==x['sha256'],'original input drift')
    by_origin[x['origin']]=p
for x in m['external_metadata_only']:
    require(Path(x['origin']).stat().st_size==x['bytes'] and sha(x['origin'])==x['sha256'],'external metadata input drift')

def pick(suffix):
    matches=[p for s,p in by_origin.items() if s.endswith(suffix)]
    require(len(matches)==1,'ambiguous or missing compact origin '+suffix)
    return matches[0]

child=load(pick('/attempts/timing001/receipt.json'))
result=load(pick('/attempts/timing001/result.json'))
outer=load(pick('/timing-outer001/receipt.json'))
root=load(pick('/timing-invocation001/receipt.json'))
auth=load(pick('/timing-authorization.json'))
release=load(pick('/timing-release.json'))
require(child['stage']=='timing' and child['completed'] is True and child['passed'] is False,'child failed-stage disposition')
require(child['sources_unchanged'] is True and child['scientific_stage_authorized'] is True,'child provenance disposition')
require(result['passed'] is False and result['full_registry'] is False,'result scope')
for name,value in [('source_index_sha256','92a412f36e668bb2e2414bd65e5dee381e42534d94e2df712fd7e9f3a52ce5d3'),('recipe_sha256','b3fdabcb50016d8bba414f7a4999ff51438d3a1e8e00f6b8a15ca20e1cd76ea2'),('driver_sha256','78ce07e5ce78ab317afbc47948de15ceb9d7ea4bb2b88565273f565be3bb4d14')]:
    require(child[name]==value and auth[name]==value,'source/recipe/driver binding '+name)
require(auth['Gaussian_third_jet_oracle_stage_authorized']=='timing' and release['stage']=='timing','actual stage authorization')
for invsuffix in ['/timing-outer001/invocation.json','/timing-invocation001/invocation.json']:
    inv=load(pick(invsuffix));require('-I' in inv['command'] and '-B' in inv['command'],'isolated command')
    require(inv['environment']['PYTHONOPTIMIZE']=='0' and inv['environment']['PYTHONDONTWRITEBYTECODE']=='1','optimization/bytecode guards')
for label,r in [('outer',outer),('root',root)]:
    require(r['completed'] is False and r['accepted_stage'] is False and r['returncode']==1 and r['inputs_unchanged'] is True,label+' rejected failed stage')
require(root['child_receipt_sha256']==sha(pick('/attempts/timing001/receipt.json')) and root['result_sha256']==sha(pick('/attempts/timing001/result.json')) and root['outer_receipt_sha256']==sha(pick('/timing-outer001/receipt.json')),'root exact child/result/outer binding')
require(root['elapsed_seconds']<root['root_process_group_cap_seconds'],'not a resource timeout')
require('actual child completion/pass/unchanged and result pass required' in pick('/timing-outer001/failure.txt').read_text(),'outer fails on science result')

before=load(pick('/attempts/timing001/source-before.json'))
after=load(pick('/attempts/timing001/source-after.json'))
require(before==after,'child source before/after')
require(before==load(pick('/timing-outer001/pins-before.json'))==load(pick('/timing-outer001/pins-after.json')),'outer child pin agreement')
pins={}
for d in [before,load(pick('/timing-invocation001/pins-before.json')),release['pins'],auth['review_pins']]:
    for p,h in d.items():
        require(p not in pins or pins[p]==h,'conflicting saved pin')
        pins[p]=h
for p,h in pins.items(): require(sha(p)==h,'source/runtime/review pin drift '+p)
for name,h in child['output_hashes'].items():
    p=Path(auth['output'])/name;require(sha(p)==h,'child output hash '+name)
for r,key in [(outer,'outputs'),(root,'output_inventory')]:
    for x in r[key]:
        p=Path(x['path']);require(p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],'actual output inventory drift')

lines=pick('/timing-outer001/stdout.log').read_text().splitlines()
require(len(lines)==40,'twenty begin/complete pairs')
progress=[json.loads(line,parse_constant=reject_constant) for line in lines]
completed=[]; failure_events=[]; prior=0
for n in range(20):
    begin,end=progress[2*n:2*n+2]
    require(begin['event']=='begin' and end['event']=='complete' and (begin['key'],begin['digits'])==(end['key'],end['digits']),'ordered event pair')
    require(end['records']==n+1 and end['failed']>=prior,'record and cumulative failure monotonicity')
    require(begin['elapsed']<=end['elapsed'],'per-event elapsed ordering')
    delta=end['failed']-prior
    if delta: failure_events.append({'record':n+1,'key':end['key'],'digits':end['digits'],'new_failures':delta,'prior_failed':prior,'final_failed':end['failed']})
    prior=end['failed'];completed.append(end)
require(result['records']==20==len(completed) and prior==result['failed_total']==6,'record/failure count agreement')
require([x['digits'] for x in completed]==[110]*10+[150]*10,'precision blocks')
require([x['key'] for x in completed[:10]]==[x['key'] for x in completed[10:]],'same events in both precision blocks')
require(failure_events==[{'record':9,'key':'eps3/4/r999999999999999999/1000000000000000000/t1/10/2,-3,6','digits':110,'new_failures':6,'prior_failed':0,'final_failed':6}],'unique failure event association')
failed=result['failed'];require(len(failed)==6 and result['failed_summary_truncated'] is False,'all compact failures retained')
require([x['name'] for x in failed]==['connection_conformal_011','connection_conformal_012','connection_conformal_013','connection_conformal_022','connection_conformal_023','connection_conformal_033'],'exact component association')
require(all(x['branch']=='bounded_outer_conformal' for x in failed),'branch association')
rounding=[];scales=[]
for x in failed:
    a,b,s=map(Fraction,[x['absolute'],x['term_sum'],x['scaled']])
    require(a>=0 and b>=0 and s>Fraction('1e-55'),'actual fixed tolerance failure')
    err=abs(s-a/max(Fraction(1),b));require(err<=Fraction('1e-100'),'saved decimal scaling serialization agreement')
    rounding.append(err);scales.append(s)
require(result['identity_checks']==11816 and result['precision_checks']==1692,'reported aggregate counts')
require(all(pick(s).stat().st_size==0 for s in ['/timing-outer001/stderr.log','/timing-invocation001/stderr.log','/timing-invocation001/stdout.log']),'empty stderr/root stdout')

out={'saved_only_review_passed':True,'scientific_timing_passed':False,'full_stage_admitted':False,'source_index_sha256':child['source_index_sha256'],'recipe_sha256':child['recipe_sha256'],'driver_sha256':child['driver_sha256'],'child_receipt_sha256':sha(pick('/attempts/timing001/receipt.json')),'result_sha256':sha(pick('/attempts/timing001/result.json')),'outer_receipt_sha256':sha(pick('/timing-outer001/receipt.json')),'root_receipt_sha256':sha(pick('/timing-invocation001/receipt.json')),'progress_stdout_sha256':sha(pick('/timing-outer001/stdout.log')),'copied_compact_inputs':len(m['inputs']),'metadata_only_inputs':len(m['external_metadata_only']),'unique_protected_pins_rehashed':len(pins),'complete_records_independently_associated':len(completed),'reported_identity_checks_not_payload_recounted':result['identity_checks'],'reported_precision_checks_not_payload_recounted':result['precision_checks'],'failure_events':failure_events,'failed_components':[x['name'] for x in failed],'scaled_minimum':show(min(scales)),'scaled_maximum':show(max(scales)),'saved_decimal_scaling_max_absolute_difference':show(max(rounding)),'unchanged_scientific_tolerance':'1e-55','saved_decimal_serialization_only_tolerance':'1e-100','root_elapsed_seconds':root['elapsed_seconds'],'child_elapsed_seconds':child['elapsed_seconds'],'all_originals_and_protected_pins_unchanged':True,'JSONL_decoded':False,'height_context_decoded':False,'candidate_import_or_target_rerun':False,'limitations':['Aggregate identity and precision counts are checked as reported; the JSONL rows were not decoded or independently recounted.','Cumulative progress localizes all failures to the single 110-digit near-scri event; no new failures at 150 digits is a scoped saved observation.','This audit does not prove cancellation as the cause, bound unreported residuals, change precision or tolerances, or admit the full stage.']}
(HERE/'saved-readback.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
