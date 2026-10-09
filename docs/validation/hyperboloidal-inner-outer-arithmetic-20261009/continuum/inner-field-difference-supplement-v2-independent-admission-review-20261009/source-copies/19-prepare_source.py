#!/usr/bin/env python3
"""Admission-only source preparation; no compile/import/query/Fraction run."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import shutil

ROOT=Path('/Users/hz0693/research/hyperboloidal')
HERE=Path(__file__).resolve().parent
OLD=ROOT/'build-layer-research/boundary/inner-field-difference-arithmetic-supplement-held-20261009'
MAIN=ROOT/'build-layer-research/boundary/inner-joint-nonlinear-helper-source003-held-20261009'
SAVED=ROOT/'build-layer-research/boundary/inner-source003-failure-saved-readback-20261009'


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')


def main():
    assert not (HERE/'source-index.json').exists() and not (HERE/'inputs').exists()
    protected=read(OLD/'input-pins.json')
    protected += [pin(p) for p in OLD.rglob('*') if p.is_file()]
    protected += [pin(MAIN/'attempts/Release001'/n) for n in ['receipt.json','oracle-report.json']]
    protected += [pin(SAVED/n) for n in ['index.json','review-receipt.json','attempt001/summary.json','attempt001/receipt.json','outer_identity/attempt001/summary.json','outer_identity/attempt001/receipt.json']]
    for p in protected:assert sha(Path(p['path']))==p['sha256']
    shutil.copytree(OLD/'inputs',HERE/'inputs');(HERE/'history').mkdir()
    for name in ['PLAN.md','recipe.json','gate_context.py','run_once.py','authorization-schema.json','source-index.json']:
        shutil.copyfile(OLD/name,HERE/'history'/name)
    for name in ['probe.cpp','oracle.py','negative-expression-bindings.json']:
        shutil.copyfile(OLD/name,HERE/name)
    plan=(OLD/'PLAN.md').read_text()
    first_end=plan.index('The scientific inputs are byte-copied')
    newplan='''# Held field-difference arithmetic supplement v2: actual main FAIL retained

SOURCE ONLY; no compile/import/query/arithmetic gate execution is admitted.
The original supplementv1 remains byte-exact and ineligible under its original
full003PASS prerequisite. This separately named v2 does not require or invent
that PASS: it binds the exact actual003 FAILED Release receipt and saved oracle
report, plus the completed independent saved association and W1 identity proof.
Every remaining source003 failure occurs in the frozen W1 RWM branch, before
any of the named arithmetic helpers can be evaluated. No W<1 comparison failed
in the fixed completed main suite. Overall003FAIL remains mandatory evidence.

This129-record gate tests only FieldSquareDifference,
FieldLogGradientDifference/NearFieldValue with the exact003 helper bytes,
registered relative field duals, original Fraction closed targets and negative
controls. Its mathematical targets do not depend on the old outer gauge's
high-contrast accuracy. A later unit PASS cannot qualify nonlinear gauge,
source003overall, new outer arithmetic, native evolution or BH acceptance.
Root must separately release this exact v2 source index and recipe.

'''+plan[first_end:]
    (HERE/'PLAN.md').write_text(newplan)
    oldguard=(OLD/'gate_context.py').read_text()
    newguard='''"""Stdlib exact FAILED-main and independent saved-association admission."""
import hashlib,json
from pathlib import Path

def pin(path):
    path=Path(path).resolve();h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return {'path':str(path),'sha256':h.hexdigest(),'bytes':path.stat().st_size}
def read(path):return json.loads(Path(path).read_text())
def write_new(path,obj):
    with Path(path).open('x') as stream:stream.write(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\\n')
def guard(rows):
    for row in rows:
        if pin(row['path'])!=row:raise RuntimeError('Protected input drift '+row['path'])
def admitted(here,authorization,build):
    recipe=read(here/'recipe.json');index=read(here/'source-index.json');auth=read(authorization)
    if not(auth.get('field_difference_units_execution_admitted') is True and auth.get('recipe_sha256')==pin(here/'recipe.json')['sha256'] and auth.get('source_index_sha256')==pin(here/'source-index.json')['sha256'] and build in auth.get('allowed_builds',[])):
        raise RuntimeError('Exact fresh root field-unit release required')
    protected=read(here/'input-pins.json');guard(protected);guard(index['files'])
    for key in ('actual_main003_failed_receipt','actual_main003_failed_oracle','saved_association_summary','saved_association_receipt','saved_outer_identity_summary','saved_outer_identity_receipt'):
        guard([recipe[key]])
    main=read(recipe['actual_main003_failed_receipt']['path'])
    if not(main.get('completed') is False and main.get('passed') is False and main.get('returncode')==1 and main.get('source_inputs_unchanged') is True and main.get('build')=='release' and main.get('source_index_sha256')==recipe['main003_source_index_sha256']):
        raise RuntimeError('Exact actual003 FAILED Release evidence required; no overall PASS claimed')
    oracle=read(recipe['actual_main003_failed_oracle']['path'])
    if not(oracle.get('passed') is False and oracle.get('counts')==recipe['actual_main_fixed_counts']):
        raise RuntimeError('Complete failed actual main report required')
    association=read(recipe['saved_association_summary']['path']);ar=read(recipe['saved_association_receipt']['path'])
    identity=read(recipe['saved_outer_identity_summary']['path']);ir=read(recipe['saved_outer_identity_receipt']['path'])
    if not(ar.get('completed') is True and ar.get('passed') is True and ar.get('returncode')==0 and ar.get('inputs_unchanged') is True and ir.get('completed') is True and ir.get('passed') is True and ir.get('returncode')==0 and ir.get('inputs_unchanged') is True):
        raise RuntimeError('Successful independent saved association/identity receipts required')
    case=[x for x in association['cases'] if x['source']=='source003']
    if len(case)!=1:raise RuntimeError('Unique actual003 association required')
    case=case[0]
    if not(association.get('passed_saved_readback') is True and association.get('original_source002_and003_still_failed') is True and case['association']['ambiguous_failure_count']==0 and case['counts']==oracle['counts'] and case['failure_count']==len(oracle['failures'])==identity['failures_checked'] and identity.get('passed_saved_identity_readback') is True and identity.get('source003_original_failed') is True and identity.get('all_failures_W_exactly_one') is True and identity['failure_native_bits_equal_source002']==identity['failures_checked'] and identity['failure_query_all_split_parts_equal_frozen_RWM']==identity['failures_checked']):
        raise RuntimeError('Complete unambiguous W1-only failure qualification required')
    # No oracle targets are recomputed here. These exact saved receipts qualify
    # the prerequisite for helper arithmetic units only, never the gauge.
    protected=protected+[pin(authorization)]+main['output_inventory']
    guard(protected)
    return recipe,index,protected
'''
    (HERE/'gate_context.py').write_text(newguard)
    (HERE/'admission-only.diff').write_text(''.join(difflib.unified_diff(oldguard.splitlines(True),newguard.splitlines(True),fromfile='held-v1-full-main-PASS',tofile='held-v2-exact-main-FAIL-W1-qualification')))
    runner=(OLD/'run_once.py').read_text().replace('after actual-main PASS.','actual main FAIL preserved; helper units only.')
    (HERE/'run_once.py').write_text(runner)
    recipe=read(OLD/'recipe.json')
    def rebind(v):
        if isinstance(v,str):return v.replace(str(OLD),str(HERE))
        if isinstance(v,list):return [rebind(x) for x in v]
        if isinstance(v,dict):return {k:rebind(x) for k,x in v.items()}
        return v
    recipe=rebind(recipe)
    recipe.update(actual_main_release_PASS_required_before_compile_or_arithmetic=False,
        actual_main_failed_receipt_and_complete_saved_W1_only_qualification_required=True,
        actual_main003_failed_receipt=pin(MAIN/'attempts/Release001/receipt.json'),
        actual_main003_failed_oracle=pin(MAIN/'attempts/Release001/oracle-report.json'),
        actual_main_fixed_counts=read(MAIN/'recipe.json')['expected_record_counts'],
        saved_association_summary=pin(SAVED/'attempt001/summary.json'),
        saved_association_receipt=pin(SAVED/'attempt001/receipt.json'),
        saved_outer_identity_summary=pin(SAVED/'outer_identity/attempt001/summary.json'),
        saved_outer_identity_receipt=pin(SAVED/'outer_identity/attempt001/receipt.json'),
        original_v1_source_index=pin(OLD/'source-index.json'),
        original_v1_ineligible_unchanged=True,actual_main003_overall_passed=False,
        scope='named exact003 field-difference arithmetic units only; no nonlinear gauge acceptance',
        execution_admitted=False)
    # The actual failed-main path remains external, never rebound to a fake
    # local receipt. All scalar helper/probe/Fraction oracle bytes are exact.
    recipe['actual_main003_release_receipt']=str(MAIN/'attempts/Release001/receipt.json')
    save(HERE/'recipe.json',recipe)
    save(HERE/'authorization-schema.json',{'field_difference_units_execution_admitted':False,
        'recipe_sha256':'ROOT_BINDS_EXACT_V2','source_index_sha256':'ROOT_BINDS_EXACT_V2',
        'allowed_builds':[],'actual_main003_overall_passed':False,
        'no_nonlinear_gauge_or_outer_accuracy_acceptance':True})
    for name in ['gate_context.py','run_once.py','oracle.py']:
        ast.parse((HERE/name).read_text(),filename=name)
    for name in ['probe.cpp','oracle.py','negative-expression-bindings.json']:
        assert (HERE/name).read_bytes()==(OLD/name).read_bytes()
    assert (HERE/'inputs/inner_gauge.hpp').read_bytes()==(MAIN/'inner_gauge.hpp').read_bytes()
    save(HERE/'source-preparation.json',{'source_only':True,'original_v1_ineligible_unchanged':True,
        'actual003_overall_failed':True,'named003_helper_exact':pin(HERE/'inputs/inner_gauge.hpp'),
        'probe_and_Fraction_targets_exact_v1':True,'129_fixed_records_unchanged':True,
        'no_import_compile_query_arithmetic_gate_executed':True})
    inputs={x['path']:x for x in protected}
    for p in HERE.rglob('*'):
        if p.is_file() and p.name not in ['input-pins.json','source-index.json']:inputs[str(p.resolve())]=pin(p.resolve())
    save(HERE/'input-pins.json',[inputs[k] for k in sorted(inputs)])
    files=[p for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='source-index.json']
    save(HERE/'source-index.json',{'source_only':True,'execution_admitted':False,'files':[pin(p) for p in files],
        'protected_inputs':len(inputs),'scope':recipe['scope'],'original_main003_failed':True,'original_v1_ineligible':True})
    for p in protected:assert sha(Path(p['path']))==p['sha256']
    print(json.dumps({'index':pin(HERE/'source-index.json'),'protected':len(inputs),'no_science_executed':True}))


if __name__=='__main__':main()
