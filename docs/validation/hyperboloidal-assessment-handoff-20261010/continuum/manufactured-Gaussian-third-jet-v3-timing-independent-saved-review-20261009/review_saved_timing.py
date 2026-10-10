#!/usr/bin/env python3
"""Saved compact provenance/count audit and empirical cost bookkeeping only."""
from pathlib import Path
from fractions import Fraction
from decimal import Decimal, localcontext
import hashlib, json, os, sys
HERE=Path(__file__).resolve().parent
def require(v,s):
    if not v: raise ValueError(s)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite '+s)))
def show(q):
    q=Fraction(q)
    with localcontext() as c:
        c.prec=24
        return str(Decimal(q.numerator)/Decimal(q.denominator))
def write(name,x):
    with (HERE/name).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
require(sys.flags.isolated==1 and sys.flags.dont_write_bytecode==1 and sys.flags.optimize==0,'stdlib isolated nonoptimized route')
require(os.environ.get('PYTHONOPTIMIZE')=='0','explicit optimization guard')
m=load(HERE/'capture.json');by_origin={}
for x in m['inputs']:
    p=HERE/x['copy'];require(p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],'captured drift')
    require(Path(x['origin']).stat().st_size==x['bytes'] and sha(x['origin'])==x['sha256'],'original drift')
    by_origin[x['origin']]=p
for x in m['external_metadata_only']:
    p=Path(x['origin']);require(p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],'external metadata drift')
def pick(suffix):
    out=[p for s,p in by_origin.items() if s.endswith(suffix)]
    require(len(out)==1,'missing/ambiguous origin '+suffix)
    return out[0]
child=load(pick('/attempts/timing001/receipt.json'));result=load(pick('/attempts/timing001/result.json'))
outer=load(pick('/timing-outer001/receipt.json'));root=load(pick('/timing-invocation001/receipt.json'))
auth=load(pick('/timing-authorization.json'));release=load(pick('/timing-release.json'))
recipe=load(pick('/manufactured-angular-Gaussian-third-jet-oracle-v3-held-20261009/recipe.json'))
expected={'source_index_sha256':'41a050d288076d6b54ff396b09851835dac03dd3518dac03dd3518f74dca65890f6fd025a8a'}
# Exact literal is deliberately written independently of any actual result.
expected['source_index_sha256']='41a050d288076d6b54ff396b09851835dac03dd3518f74dca65890f6fd025a8a'
expected.update(recipe_sha256='847cd099b7d98f16963fbbdfc6f8a0eb6f7913529b9a2d72dcabb909c287c8cf',driver_sha256='de0b83ba44f1a837c2b477d6d60f5a8f2a947b8a885159a638bbbaeee94e7915')
for k,v in expected.items():require(child[k]==v and auth[k]==v,'exact source/recipe/driver binding '+k)
require(child['stage']=='timing' and child['completed'] is True and child['passed'] is True and child['sources_unchanged'] is True and child['scientific_stage_authorized'] is True,'child PASS disposition')
require(result['passed'] is True and result['records']==20 and result['failed_total']==0 and result['failed']==[] and result['failed_summary_truncated'] is False,'compact no-failure result')
require(result['full_registry'] is False and result['no_native_queries'] is True and result['no_inverse_coverage_or_BH_adoption'] is True,'scoped timing result')
require(auth['Gaussian_third_jet_oracle_stage_authorized']=='timing' and release['stage']=='timing','timing-only release')
for label,r in [('outer',outer),('root',root)]:
    require(r['completed'] is True and r['accepted_stage'] is True and r['returncode']==0 and r['inputs_unchanged'] is True,label+' PASS and unchanged')
require(root['child_receipt_sha256']==sha(pick('/attempts/timing001/receipt.json')) and root['result_sha256']==sha(pick('/attempts/timing001/result.json')) and root['outer_receipt_sha256']==sha(pick('/timing-outer001/receipt.json')),'root output bindings')
require(root['root_process_group_cap_seconds']==690 and root['elapsed_seconds']<690,'root process cap')
for s in ('/timing-outer001/invocation.json','/timing-invocation001/invocation.json'):
    inv=load(pick(s));require('-I' in inv['command'] and '-B' in inv['command'],'actual isolated command')
    require(inv['environment']['PYTHONOPTIMIZE']=='0' and inv['environment']['PYTHONDONTWRITEBYTECODE']=='1','actual optimization/bytecode guards')
before=load(pick('/attempts/timing001/source-before.json'));after=load(pick('/attempts/timing001/source-after.json'))
require(before==after==load(pick('/timing-outer001/pins-before.json'))==load(pick('/timing-outer001/pins-after.json')),'child and outer pin identity')
require(load(pick('/timing-invocation001/pins-before.json'))==load(pick('/timing-invocation001/pins-after.json')),'root pre/post identity')
pins={}
for d in (before,load(pick('/timing-invocation001/pins-before.json')),release['pins'],auth['review_pins']):
    for p,h in d.items():require(p not in pins or pins[p]==h,'inconsistent pin');pins[p]=h
for p,h in pins.items():require(sha(p)==h,'protected source/runtime/review drift '+p)
for name,h in child['output_hashes'].items():require(sha(Path(auth['output'])/name)==h,'child output drift')
for r,key in ((outer,'outputs'),(root,'output_inventory')):
    for x in r[key]:
        p=Path(x['path']);require(p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],'output inventory drift')
for s in ('/timing-outer001/stderr.log','/timing-invocation001/stdout.log','/timing-invocation001/stderr.log'):
    require(pick(s).stat().st_size==0,'empty actual stream '+s)
prior=load(pick('/manufactured-Gaussian-third-jet-v2-timing-failure-independent-saved-review-20261009/saved-readback.json'))
require(prior['scientific_timing_passed'] is False and prior['reported_identity_checks_not_payload_recounted']==11816 and len(prior['failed_components'])==6,'original six-failure lineage remains failed')
progress=[json.loads(s,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x))) for s in pick('/timing-outer001/stdout.log').read_text().splitlines()]
require(len(progress)==40,'twenty progress pairs')
events=[]
for n in range(20):
    b,e=progress[2*n:2*n+2]
    require(b['event']=='begin' and e['event']=='complete' and (b['key'],b['digits'])==(e['key'],e['digits']),'ordered matching progress pair')
    require(e['records']==n+1 and e['failed']==0,'completed count and no reported failures')
    dt=Fraction(str(e['elapsed']))-Fraction(str(b['elapsed']));require(dt>=0,'elapsed ordering')
    events.append({'key':b['key'],'digits':b['digits'],'seconds':dt})
require([x['digits'] for x in events]==[180]*10+[220]*10,'fixed precision blocks')
require([x['key'] for x in events[:10]]==recipe['timing_keys']==[x['key'] for x in events[10:]],'exact timing key coverage')
require(result['identity_checks']==11816 and result['precision_checks']==1692,'reported aggregate counts unchanged')
require(recipe['identity_tolerance']==recipe['component_tolerance']=='1e-55','unchanged numeric gates')
require(recipe['stage_seconds']['full']==14400 and recipe['outer_seconds']['full']==14460,'held full caps unchanged')
# Reconstruct keys and branch classes with Fraction/string bookkeeping only;
# do not call the target registry or construct numerical coordinates.
dirs=['p='+p for p in recipe['angular_p']]+['e2','e3','111','2,-3,6']
def category(r):
    return 'origin' if r==0 else 'nonzero_core' if r<=Fraction('.05') else 'transition' if r<Fraction('.95') else 'outer'
full={};counts={}
for eps in recipe['epsilon']:
    for r in recipe['radii']:
        for t in recipe['times']:
            for d in (['origin'] if Fraction(r)==0 else dirs):
                key='eps'+eps+'/r'+r+'/t'+t+'/'+d
                full[key]=category(Fraction(r));counts[full[key]]=counts.get(full[key],0)+1
negative='negative/sigma1/2/eps3/4/r3/4/t1/10/p1/2';full[negative]='negative_control';counts['negative_control']=1
require(len(full)==recipe['records_per_level']==2505 and counts=={'origin':8,'nonzero_core':312,'transition':1248,'outer':936,'negative_control':1},'independent exact full-registry keys/counts')
require(len(dirs)==13 and len(set(dirs))==13,'fixed thirteen direction labels')
require(all(e['key'] in full for e in events),'timing keys included in full')
valid=2504;require(valid*188==recipe['expected_counts']['full_component_checks']==470752,'fixed component-count bookkeeping')
eligible=sum(v!='negative_control' and Fraction(k.split('/r')[1].split('/t')[0])<=Fraction(recipe['graph_radius']) for k,v in full.items())
require(eligible==recipe['expected_counts']['nominal_native_eligible_valid_cases']==1880,'exact nominal eligible count')
tables=[];predicted_min=predicted_mean=predicted_max=Fraction(0)
for digits in (180,220):
    for cat,count in counts.items():
        sample=[e['seconds'] for e in events if e['digits']==digits and full[e['key']]==cat]
        require(bool(sample),'missing branch cost representative')
        low,high=min(sample),max(sample);mean=sum(sample,Fraction(0))/len(sample)
        predicted_min+=count*low;predicted_mean+=count*mean;predicted_max+=count*high
        tables.append({'digits':digits,'branch':cat,'full_record_count':count,'sample_record_count':len(sample),'sample_min_seconds':show(low),'sample_mean_seconds':show(mean),'sample_max_seconds':show(high)})
event_total=sum((e['seconds'] for e in events),Fraction(0))
non_event_child=Fraction(str(child['elapsed_seconds']))-event_total
non_event_root=Fraction(str(root['elapsed_seconds']))-event_total
require(non_event_child>=0 and non_event_root>=non_event_child,'setup/provenance overhead bookkeeping')
# Retain measured two-level setup once, rather than multiplying its height work
# by 5010/20. The sampled event extrema are empirical, not certified bounds.
cost={'saved_only':True,'source_index_sha256':expected['source_index_sha256'],'timing_receipt_sha256':sha(pick('/attempts/timing001/receipt.json')),'timing_result_sha256':sha(pick('/attempts/timing001/result.json')),'full_stage_source_review_passed':True,'full_stage_cost_admission':False,'root_cost_decision_required':True,'full_records':5010,'timing_records':20,'full_registry_records_per_level':counts,'branch_empirical_table':tables,'timed_event_seconds':show(event_total),'child_non_event_seconds':show(non_event_child),'root_non_event_seconds':show(non_event_root),'naive_root_total_times_5010_over20_seconds':show(Fraction(str(root['elapsed_seconds']))*5010/20),'branch_weighted_min_sample_model_seconds':show(predicted_min+non_event_root),'branch_weighted_mean_sample_model_seconds':show(predicted_mean+non_event_root),'branch_weighted_max_sample_model_seconds':show(predicted_max+non_event_root),'mean_model_with_25percent_event_inflation_seconds':show(predicted_mean*Fraction(5,4)+non_event_root),'max_model_with_25percent_event_inflation_seconds':show(predicted_max*Fraction(5,4)+non_event_root),'full_soft_cap_seconds':14400,'full_outer_cap_seconds':14460,'max_model_event_inflation_factor_fitting_soft_cap':show((Fraction(14400)-non_event_root)/predicted_max),'limitations':['These are empirical branch-weighted scenarios, not prediction intervals or a worst-case runtime proof.','No nonorigin epsilon=0 event was timed; its reference-rate checks and zero-jet costs are unmeasured.','Only one nonzero core radius and one direction there were timed; unsampled directions, native times, nonlinear roots, Gaussian regime switching and scheduling can change cost.','Every admitted event still forms raw graph and physical-curvature diagnostics, even where their admission_gate is false; those diagnostics were not omitted from the cost model.','The full output volume and 470752 precision-component rows are larger than the timing output; this audit does not decode payloads to measure serialization scaling.','No full stage is authorized by this audit. Root must make a separate bounded cost decision using exact source/timing pins.']}
write('cost-assessment.json',cost)
out={'passed':True,'saved_only':True,'inputs_unchanged':True,'scientific_timing_passed':True,'full_stage_admitted':False,**expected,'child_receipt_sha256':sha(pick('/attempts/timing001/receipt.json')),'result_sha256':sha(pick('/attempts/timing001/result.json')),'outer_receipt_sha256':sha(pick('/timing-outer001/receipt.json')),'root_receipt_sha256':sha(pick('/timing-invocation001/receipt.json')),'progress_stdout_sha256':sha(pick('/timing-outer001/stdout.log')),'records':20,'reported_identity_checks_not_payload_recounted':11816,'reported_precision_checks_not_payload_recounted':1692,'reported_failed_total':0,'complete_progress_records_independently_associated':20,'compact_inputs':len(m['inputs']),'metadata_only_inputs':len(m['external_metadata_only']),'unique_protected_pins_rehashed':len(pins),'original_v2_timing_failed_total':6,'prior_failure_status_preserved':True,'full_registry_count_bookkeeping_passed':True,'full_nominal_native_eligible_valid_cases':1880,'root_elapsed_seconds':root['elapsed_seconds'],'child_elapsed_seconds':child['elapsed_seconds'],'JSONL_decoded':False,'height_context_decoded':False,'target_import_or_rerun':False,'cost_assessment_sha256':sha(HERE/'cost-assessment.json'),'limitations':['Aggregate identity and precision counts and absence of failures are audited as reported, without decoding/recounting JSONL rows.','No maximum passing residual is available in the compact result; no numerical zero or absolute uniform forward-error bound is inferred.','This finite analytic geometry/source oracle does not solve the coupled inner BM helper and gives no native evolution, global slicing, inverse-coverage or BH stability acceptance.']}
write('saved-readback.json',out)
print(json.dumps({'passed':True,'records':20,'identity_checks_reported':11816,'precision_checks_reported':1692,'pins':len(pins),'cost':{k:cost[k] for k in cost if 'model_seconds' in k or 'inflation' in k or k=='naive_root_total_times_5010_over20_seconds'}},indent=2))
