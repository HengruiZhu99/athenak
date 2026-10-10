"""Saved finite JSON/hash/scalar readback; no imports or reruns of targets."""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import hashlib
import json
import sys

HERE=Path(__file__).resolve().parent
ROOT=Path('/Users/hz0693/research/hyperboloidal')
OWNER=ROOT/'build-layer-research/continuum/manufactured-angular-Gaussian-third-jet-oracle-v2-held-20261009'
RELEASE=ROOT/'build-layer-research/Gaussian-third-jet-oracle-root-release-20261009'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def load(p):
    return json.loads(Path(p).read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))

def captured(p):
    return HERE/'captured'/Path(p).relative_to(ROOT)

def require(ok,message):
    if not ok:raise RuntimeError(message)

def main():
    out=HERE/'saved-readback.json'
    require(not out.exists(),'one-shot fresh readback')
    inv=load(HERE/'inputs-captured-before-further-review.json')
    json_count=0
    for row in inv['inputs']:
        for key in ('original','captured'):
            require(sha(row[key])==row['sha256'] and Path(row[key]).stat().st_size==row['bytes'],'captured/original drift')
        if Path(row['captured']).suffix=='.json':load(row['captured']);json_count+=1
    childdir=OWNER/'attempts/units001'
    child=load(captured(childdir/'receipt.json'));result=load(captured(childdir/'result.json'))
    outer=load(captured(RELEASE/'units-outer001/receipt.json'))
    root=load(captured(RELEASE/'units-invocation001/receipt.json'))
    require(child['stage']=='units' and all(child[k] is True for k in ('completed','passed','sources_unchanged','scientific_stage_authorized')),'child admission')
    require(child['source_index_sha256']=='92a412f36e668bb2e2414bd65e5dee381e42534d94e2df712fd7e9f3a52ce5d3','child source index')
    require(result=={'passed':True,'records':0,'checks':318,'failed':[],'failed_total':0},'result summary')
    for receipt in (outer,root):
        require(receipt['returncode']==0 and all(receipt[k] is True for k in ('completed','accepted_stage','inputs_unchanged')),'enclosing completion')
        require(receipt['child_receipt_sha256']==sha(childdir/'receipt.json') and receipt['result_sha256']==sha(childdir/'result.json'),'receipt output binding')
    require(root['outer_receipt_sha256']==sha(RELEASE/'units-outer001/receipt.json'),'root outer binding')
    require(root['root_process_group_cap_seconds']==60 and root['elapsed_seconds']<60,'root actual time cap')
    before=load(captured(childdir/'source-before.json'));after=load(captured(childdir/'source-after.json'))
    require(before==after,'child before/after maps')
    combined=dict(before)
    release=load(captured(RELEASE/'units-release.json'))
    for p,h in release['pins'].items():
        if p in combined:require(combined[p]==h,'overlapping pin mismatch')
        combined[p]=h
    for p,h in combined.items():require(sha(p)==h,'current protected input drift '+p)
    for rel,h in child['output_hashes'].items():require(sha(childdir/rel)==h,'child output hash')
    for row in root['output_inventory']:
        require(sha(row['path'])==row['sha256'] and Path(row['path']).stat().st_size==row['bytes'],'root output inventory')
    auth=load(captured(RELEASE/'units-authorization.json'))
    require(sha(RELEASE/'units-authorization.json')==release['authorization_sha256'],'exact authorization')
    require(auth['Gaussian_third_jet_oracle_stage_authorized']=='units','stage authorization')
    require(auth['source_index_sha256']==child['source_index_sha256'] and auth['recipe_sha256']==child['recipe_sha256'] and auth['driver_sha256']==child['driver_sha256'],'auth source binding')
    for folder in ('units-invocation001','units-outer001'):
        invocation=load(captured(RELEASE/folder/'invocation.json'))
        require('-I' in invocation['command'] and '-B' in invocation['command'],'isolated no-bytecode command')
        require(invocation['environment']['PYTHONOPTIMIZE']=='0','unoptimized environment')
        for stream in ('stdout.log','stderr.log'):
            require((RELEASE/folder/stream).stat().st_size==0,'nonempty run log')
    rows=load(captured(childdir/'unit-checks.json'))
    require(len(rows)==318 and len({r['name'] for r in rows})==318,'fixed unique row count')
    groups=Counter(r['name'].split('/')[0] for r in rows)
    max_scaled=Fraction(0);max_saved_arithmetic=Fraction(0);max_name=None
    for row in rows:
        require(row['admission_gate'] is True,'ungated unit row')
        terms=list(map(Fraction,row['terms']));signed=Fraction(row['signed']);absolute=Fraction(row['absolute'])
        total=Fraction(row['term_sum']);scaled=Fraction(row['scaled'])
        require(absolute>=0 and total>=0 and scaled>=0 and absolute==abs(signed),'invalid signed/absolute fields')
        require(scaled<=Fraction('1e-55'),'original scientific tolerance failed')
        exact_sum=sum(terms,Fraction(0));exact_total=sum(map(abs,terms),Fraction(0))
        scale=max(Fraction(1),exact_total)
        error=max(abs(exact_sum-signed),abs(exact_total-total))/scale
        # Readback serialization allowance only, not a changed scientific gate.
        require(error<=Fraction('1e-75'),'saved operand sum mismatch')
        denominator=Fraction(row['component_scale']) if 'component_scale' in row else max(Fraction(1),total)
        expected_scaled=absolute/denominator
        require(abs(expected_scaled-scaled)<=Fraction('1e-75')*max(Fraction(1),expected_scaled),'saved scaled residual mismatch')
        max_saved_arithmetic=max(max_saved_arithmetic,error)
        if scaled>max_scaled:max_scaled=scaled;max_name=row['name']
    output={'passed':True,'saved_only':True,'candidate_imported_or_targets_rerun':False,
      'target_arithmetic_CAS_compile_queries_eigen_propagation':False,'checks':318,'failed':0,
      'groups':dict(sorted(groups.items())),'original_scientific_tolerance':'1e-55',
      'saved_scalar_serialization_allowance':'1e-75; not an oracle tolerance change',
      'maximum_saved_scaled_fraction':str(max_scaled),'maximum_scaled_row':max_name,
      'maximum_saved_operand_sum_error_fraction':str(max_saved_arithmetic),
      'child_protected_pins':len(before),'unique_child_plus_root_protected_pins':len(combined),
      'captured_inputs':len(inv['inputs']),'finite_captured_JSON':json_count,
      'root_seconds':root['elapsed_seconds'],'child_seconds':child['elapsed_seconds'],
      'root_group_cap_seconds':60,'all_original_inputs_unchanged':True,
      'child_receipt_sha256':sha(childdir/'receipt.json'),'outer_receipt_sha256':sha(RELEASE/'units-outer001/receipt.json'),
      'root_receipt_sha256':sha(RELEASE/'units-invocation001/receipt.json'),
      'first_capture_preflight_failure_preserved':True,'source_sha256':sha(__file__),'argv':sys.argv}
    out.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:output[k] for k in ('passed','checks','failed','child_protected_pins','unique_child_plus_root_protected_pins','groups','root_seconds','child_seconds')},sort_keys=True))

if __name__=='__main__':main()
