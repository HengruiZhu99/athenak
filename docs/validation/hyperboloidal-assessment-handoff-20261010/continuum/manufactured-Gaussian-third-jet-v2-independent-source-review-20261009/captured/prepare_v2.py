"""One-shot stdlib source/AST freeze; never imports analytic candidate modules."""
from pathlib import Path
import ast
import difflib
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/'build-layer-research/continuum/manufactured-angular-Gaussian-third-jet-oracle-held-20261009'
REVIEW=ROOT/'build-layer-research/continuum/manufactured-Gaussian-third-jet-independent-source-review-20261009'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()

def load(path):return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def replace_once(s,a,b):
    if s.count(a)!=1:raise RuntimeError('exact source anchor count: '+a)
    return s.replace(a,b)
def ast_text(s):return ast.dump(ast.parse(s),include_attributes=False)

def main():
    if (HERE/'source-index.json').exists():raise RuntimeError('one-shot freeze already exists')
    if sha(OLD/'source-index.json')!='2348bbf4391444606dc2f946067a3a75eb55d472cc0c738795d8f0617f161d88':raise RuntimeError('v1 index drift')
    if sha(REVIEW/'index.json')!='d61ddb62752163f938540f23d02aeb49b65147174f6ebb0edaaef1851ff7ba57':raise RuntimeError('v1 failed review drift')
    old_index=load(OLD/'source-index.json')
    for name,digest in old_index['files'].items():
        if sha(name)!=digest:raise RuntimeError('v1 source drift: '+name)
    recipe=load(OLD/'recipe.json')
    old_protected=dict(recipe['protected_inputs'])
    protected={**old_protected,**old_index['files'],str(OLD/'source-index.json'):sha(OLD/'source-index.json')}
    for p in sorted(REVIEW.rglob('*')):
        if p.is_file():protected[str(p)]=sha(p)
    for p,digest in protected.items():
        if sha(p)!=digest:raise RuntimeError('external drift '+p)
    recipe['status']='HELD source-only Gaussian v2: physical reference normalization, nominal admission, measured timing guard'
    recipe['protected_inputs']=protected
    recipe['v1_ineligible_source_index_sha256']=sha(OLD/'source-index.json')
    recipe['v1_failed_source_review_index_sha256']=sha(REVIEW/'index.json')
    recipe['graph_admission_selector']='exact Fraction nominal registry radius <= exact Fraction graph_radius; geometry evaluation unchanged'
    recipe['full_measured_timing_review_required_fields']=['source_index_sha256','timing_receipt_sha256','timing_result_sha256','full_stage_source_review_passed=true','full_stage_cost_admission=true']
    save(HERE/'recipe.json',recipe)
    immutable_keys=set(recipe)-{'status','protected_inputs','v1_ineligible_source_index_sha256','v1_failed_source_review_index_sha256','graph_admission_selector','full_measured_timing_review_required_fields'}
    if any(recipe[k]!=load(OLD/'recipe.json')[k] for k in immutable_keys):raise RuntimeError('scientific recipe drift')
    schema=load(OLD/'authorization-schema.json')
    schema['measured_timing_review']='full only: path/sha256 of explicit same-index timing-receipt/result-bound source and measured-cost approval record'
    schema['review_pins']='exact root and independent source review path-to-sha256 map, protected by BOTH child and outer'
    save(HERE/'authorization-schema.json',schema)
    save(HERE/'measured-timing-review-schema.json',{
        'source_index_sha256':'exact current v2 source-index hash',
        'timing_receipt_sha256':'exact same-source successful timing child receipt hash',
        'timing_result_sha256':'exact timing receipt output_hashes[result.json]',
        'full_stage_source_review_passed':True,'full_stage_cost_admission':True,
        'measured_elapsed_seconds_and_cost_assessment':'root records actual timings and reason for full-stage admission; no extrapolated measurement claim',
        'scope':'full analytic stage only; no inverse coverage/native/BH/source adoption'})
    plan=(OLD/'PLAN.md').read_text()+'''\n\n## Fresh v2 correction and admission history\n\nThe original index2348bbf4 is preserved, unexecuted and ineligible. Its independent source review d61ddb62 reports three blocking issues. This sibling makes only the following corrections.\n\n1. The independent reference ADM route first constructs the conformal four-metric, then divides every jet component by Omega squared before computing the physical reference Christoffel connection. The embedding connection is physical already. The geometric fields, implicit solve and compact source formulas are unchanged.\n2. Graph/configuration, physical-curvature and reference-connection admission uses the exact Fraction nominal registry radius compared with the exact Fraction .98 cutoff. It does not use the norm of rounded angular MP coordinates. Those rounded coordinates still define the actual geometry evaluation, unchanged. Each record retains nominal_radius and finite_graph_admission; the planned 1,880 native-eligible valid case count refers to nominal admission only, with native queries still held. Units that construct exact known core/negative radii do not invoke all_checks and are unchanged.\n3. Full-stage child and outer require a pinned measured_timing_review whose source-index, timing receipt and timing result hashes match the exact successful same-source timing prerequisite, and whose explicit source-review and measured-cost admissions are true. Both child and outer protect this record, authorization and review_pins before/after execution. A generic passing timing receipt does not authorize the full stage.\n\nNumerical thresholds, precision, 5,010/3,330,320/470,752 full counts, 318 units, 20 timing records, geometry and source implementation are otherwise unchanged. Exact source transformation/reverse-byte and AST proofs are recorded by stdlib preparation. No candidate was imported, compiled or evaluated in preparing this sibling. New unit/timing/full executions remain separately held for exact root releases, and no v1 execution or result is relabeled.\n'''
    (HERE/'PLAN.md').write_text(plan)
    (HERE/'SCHEMA.md').write_text((OLD/'SCHEMA.md').read_text()+'''\n\nV2 records add exact nominal_radius (rational string) and finite_graph_admission (boolean). Admission uses this exact nominal radius; evaluated event coordinates and all physical geometry remain unchanged. The reference ADM connection comparator is formed from physical ghat=Omega^-2 bar(ghat), matching the inertial-embedding route. Full stage additionally requires the explicit pinned measured-timing-review schema.\n''')
    transforms={}
    transforms['diagnostics.py']=[
        ('def all_checks(data,graph_radius):','def all_checks(data,nominal_radius,graph_radius):'),
        ('    finite = ref["rvalue"] <= graph_radius','    finite = nominal_radius <= graph_radius'),
        ('    reference_metric = metric_from_adm(*reference_adm(ref))','    reference_bar = metric_from_adm(*reference_adm(ref))\n    reference_metric = [[entry/omega**2 for entry in row] for row in reference_bar]')]
    driver=(HERE/'run_oracle.py').read_text()
    guard_start=driver.index('\n            review_pin=auth["measured_timing_review"]')
    guard_end=driver.index('\n        if Path(sys.executable)',guard_start)
    transforms['run_oracle.py']=[
        ('yield key,[rat(tlabel)]+[r*v for v in n],rat("7/20"),rat(eps),False','yield key,[rat(tlabel)]+[r*v for v in n],rat("7/20"),rat(eps),False,Fraction(rlabel)'),
        ('rat("1/2"),rat("3/4"),True','rat("1/2"),rat("3/4"),True,Fraction("3/4")'),
        ('for key,event,sigma,epsilon,negative in events:','for key,event,sigma,epsilon,negative,nominal_radius in events:'),
        ('rows,aux=diagnostics.all_checks(data,mp.mpf(recipe["graph_radius"]))','rows,aux=diagnostics.all_checks(data,nominal_radius,Fraction(recipe["graph_radius"]))'),
        ('                record["elapsed_seconds"]=time.monotonic()-started','                record["nominal_radius"]=str(nominal_radius)\n                record["finite_graph_admission"]=not negative and nominal_radius<=Fraction(recipe["graph_radius"])\n                record["elapsed_seconds"]=time.monotonic()-started'),
        ("            prerequisite_pins.update({str(timing_path.parent/rel):h for rel,h in timing['output_hashes'].items()})","            prerequisite_pins.update({str(timing_path.parent/rel):h for rel,h in timing['output_hashes'].items()})"+driver[guard_start:guard_end]),
        ('pins={**recipe["protected_inputs"],**index["files"],**prerequisite_pins}','pins={**recipe["protected_inputs"],**index["files"],**auth["review_pins"],**prerequisite_pins}')]
    outer=(HERE/'outer_once.py').read_text()
    outer_start=outer.index("        if stage=='full':\n            timing=load(auth['timing_receipt']['path'])")
    outer_end=outer.index("        pins={**recipe['protected_inputs']",outer_start)
    transforms['outer_once.py']=[
        ("        pins={**recipe['protected_inputs'],**index['files'],**auth['review_pins']}",outer[outer_start:outer_end]+"        pins={**recipe['protected_inputs'],**index['files'],**auth['review_pins']}"),
        ("for name in ('unit_receipt','timing_receipt'):","for name in ('unit_receipt','timing_receipt','measured_timing_review'):")]
    reverse=[]
    for name,operations in transforms.items():
        old=(OLD/name).read_text();new=(HERE/name).read_text();expected=old
        for a,b in operations:expected=replace_once(expected,a,b)
        if expected!=new:raise RuntimeError('unapproved source change '+name)
        restored=new
        for a,b in reversed(operations):restored=replace_once(restored,b,a)
        if restored!=old or ast_text(restored)!=ast_text(old):raise RuntimeError('reverse source/AST failure '+name)
        reverse.append({'path':name,'old_sha256':sha(OLD/name),'new_sha256':sha(HERE/name),'exact_forward_bytes':True,'exact_reverse_bytes':True,'reverse_AST':True,'replacement_count':len(operations)})
    unchanged=[]
    for name in ('taylor3.py','reference3.py','gaussian3.py','geometry.py','oracle.py','units.py','values_context.py'):
        if (HERE/name).read_bytes()!=(OLD/name).read_bytes():raise RuntimeError('unaffected body changed '+name)
        unchanged.append({'path':name,'sha256':sha(HERE/name),'byte_identical_v1':True})
    diff=''.join(''.join(difflib.unified_diff((OLD/name).read_text().splitlines(True),(HERE/name).read_text().splitlines(True),fromfile='v1/'+name,tofile='v2/'+name)) for name in ('diagnostics.py','run_oracle.py','outer_once.py','recipe.json','authorization-schema.json','PLAN.md','SCHEMA.md'))
    (HERE/'v1-to-v2.diff').write_text(diff)
    modules=[]
    for p in sorted(HERE.glob('*.py')):
        ast.parse(p.read_text(),filename=str(p));modules.append({'path':str(p),'sha256':sha(p),'AST_parse_only':True})
    save(HERE/'source-preparation-v2.json',{'source_only':True,'candidate_imports':False,'arithmetic_calls':False,'compiler_calls':False,'launch_HEAD':'e1313dac526611214d2b50d285ca517e526a7f54','v1_source_index_sha256':sha(OLD/'source-index.json'),'v1_review_index_sha256':sha(REVIEW/'index.json'),'reverse_source_proofs':reverse,'unaffected_numerical_bodies':unchanged,'all_scientific_recipe_values_unchanged':True,'nominal_admission_exact_fraction':True,'reference_comparator_both_physical':True,'full_measured_timing_guard_child_and_outer':True,'protected_external_inputs':len(protected),'AST_only_modules':modules,'no_execution_relabel':True})
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and p!=HERE/'source-index.json')
    mapping={str(p):sha(p) for p in files}
    save(HERE/'source-index.json',{'status':'immutable SOURCE-ONLY held Gaussian v2 correction','source_only':True,'execution_admitted':False,'file_count':len(files),'files':mapping})
    for p,h in {**mapping,**protected}.items():
        if sha(p)!=h:raise RuntimeError('final pin drift '+p)
    print(json.dumps({'index':sha(HERE/'source-index.json'),'recipe':sha(HERE/'recipe.json'),'driver':sha(HERE/'run_oracle.py'),'outer':sha(HERE/'outer_once.py'),'diagnostics':sha(HERE/'diagnostics.py'),'diff':sha(HERE/'v1-to-v2.diff'),'files':len(files),'external_inputs':len(protected),'candidate_imports':False}))

if __name__=='__main__':main()
