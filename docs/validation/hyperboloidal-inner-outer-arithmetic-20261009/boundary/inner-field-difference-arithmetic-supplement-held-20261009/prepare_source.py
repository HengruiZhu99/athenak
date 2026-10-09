#!/usr/bin/env python3
"""One-shot stdlib metadata/source preparation only; never imports gate code."""
import hashlib,json
from pathlib import Path

ROOT=Path('/Users/hz0693/research/hyperboloidal')
HERE=Path(__file__).resolve().parent
MAIN=ROOT/'build-layer-research/boundary/inner-joint-nonlinear-helper-source003-held-20261009'
OLD=ROOT/'build-layer-research/boundary/inner-joint-nonlinear-helper-source002-held-20261009'
PLAN=ROOT/'build-layer-research/boundary/inner-joint-nonlinear-helper-cancellation-plan-held-20261009'
REVIEW=ROOT/'build-layer-research/continuum/inner-joint-nonlinear-helper-source003-independent-review-20261009'

def pin(path):
    path=Path(path).resolve();h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return {'path':str(path),'sha256':h.hexdigest(),'bytes':path.stat().st_size}
def read(path):return json.loads(Path(path).read_text())
def write(path,obj):
    with Path(path).open('x') as stream:stream.write(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def main():
    if (HERE/'source-index.json').exists():raise RuntimeError('Already source-indexed')
    fixed={MAIN/'source-index.json':'6d0f91c4b7bf701f83d723f81c89df8265f76fe839ef4f522e6e056b9544d671',MAIN/'inner_gauge.hpp':'b26357cc3d25822b8c2b7f60bdf4bb10163bd904babbbe3231ccf3e5c0aeae4b',PLAN/'index.json':'deb3b2e8491c41ff8a02460a3730f5a9359a4906f66724e87fe2cb28d2f3de94',REVIEW/'index.json':'412dddbbc1c7f77b820b9a4000bf4c17f5c1ea7fa199d908b208cc479007cca9'}
    for path,digest in fixed.items():
        if pin(path)['sha256']!=digest:raise RuntimeError('Fixed source drift '+str(path))
    protected={}
    def add(row):
        if pin(row['path'])!=row:raise RuntimeError('Input drift '+row['path'])
        protected[row['path']]=row
    for row in read(MAIN/'input-pins.json')+read(MAIN/'source-index.json')['files']+[pin(MAIN/'source-index.json')]:add(row)
    for tree in [PLAN,REVIEW]:
        for path in sorted(tree.rglob('*')):
            if path.is_file():add(pin(path))
    # No read of or pin to the potentially live main003 attempt is made here.
    (HERE/'inputs').mkdir(exist_ok=False)
    for source in [MAIN/'inner_gauge.hpp']+sorted((MAIN/'inputs').iterdir()):
        if source.is_file():(HERE/'inputs'/source.name).write_bytes(source.read_bytes())
    old_text=(OLD/'inner_gauge.hpp').read_text();new_text=(MAIN/'inner_gauge.hpp').read_text()
    start='inline double ProductValue('
    old_product=old_text[old_text.index(start):old_text.index('template<class T> T Coefficient(')]
    new_product=new_text[new_text.index(start):new_text.index('// Exact [1/2,2]')]
    if old_product!=new_product:raise RuntimeError('Original Product bodies changed')
    expressions=[
        '  const T dA=Product<T>({a+h,da,chi},audit)+Product<T>({h,h,dchi},audit);',
        '    dc[i]=(u.chi.d[i]-p.state.chi.d[i])-Product<T>({dchi,p.state.chi.d[i]/ch},audit);',
        '    dal[i]=(u.alpha.d[i]-p.dalpha[i])-Product<T>({da,p.dalpha[i]/h},audit);']
    probe=(HERE/'probe.cpp').read_text()
    for expression in expressions:
        if old_text.count(expression)!=1 or probe.count(expression)!=1:raise RuntimeError('Original negative expression byte identity')
    write(HERE/'negative-expression-bindings.json',{'original_source002_helper':pin(OLD/'inner_gauge.hpp'),'source003_helper':pin(MAIN/'inner_gauge.hpp'),'unchanged_Product_and_ProductValue_body_sha256':hashlib.sha256(old_product.encode()).hexdigest(),'exact_original_expression_lines':expressions,'all_three_lines_copied_byte_exact_into_probe':True,'scalar_negative_control_only_not_old_actual_source_rerun':True})
    main_recipe=read(MAIN/'recipe.json')
    flags={key:[value.replace(str(MAIN),str(HERE)) for value in values] for key,values in main_recipe['compile_flags'].items()}
    recipe={'source_only':True,'execution_admitted':False,'scope':'Separate fixed scalar arithmetic supplement only; no main suite/kernel/Grid/PDE changes','repository':str(ROOT),'compiler':main_recipe['compiler'],'python':main_recipe['python'],'compile_flags':flags,'environment':dict(main_recipe['environment'],PYTHONOPTIMIZE='0'),'attempt_names':{'release':'Release001','debug':'Debug001'},'actual_main003_release_receipt':str(MAIN/'attempts/Release001/receipt.json'),'main003_source_index_sha256':pin(MAIN/'source-index.json')['sha256'],'actual_main_release_PASS_required_before_compile_or_arithmetic':True,'actual_main_receipt_exact_hash_bound_by_future_root_authorization':True,'main003_oracle_unchanged':pin(MAIN/'oracle.py'),'cancellation_plan_index':pin(PLAN/'index.json'),'source003_independent_review_index':pin(REVIEW/'index.json'),'expected_counts':{'witness':18,'negative-old-near':3,'near-bound':108,'total':129},'relative_seeds':[[0,0,0],[1,0,0],[0,1,0],[1,1,0],[1,-2,0],[0,0,1]],'near_reference_exponents':[-400,0,400],'near_ratios':['1/2-2^-40','1/2','1/2+2^-40','2-2^-40','2','2+2^-40'],'nonzero_normal_target_relative_threshold':'2e-10','zero_target_absolute_threshold':'2e-10','near_bound_scope':'bool, dA and both log-gradient kind values/full field duals; no C1 continuity claim','dV_extension':False,'old_source001_and_source002_failures_preserved':True,'outer_stdout_stderr_returncode_capture_required':True}
    write(HERE/'recipe.json',recipe);write(HERE/'input-pins.json',sorted(protected.values(),key=lambda row:row['path']))
    write(HERE/'authorization-schema.json',{'arithmetic_supplement_execution_admitted':False,'recipe_sha256':pin(HERE/'recipe.json')['sha256'],'source_index_sha256':'ROOT_BINDS_AFTER_FINAL_INDEX','allowed_builds':['release','debug'],'actual_main003_release_receipt':{'path':recipe['actual_main003_release_receipt'],'sha256':'ROOT_BINDS_COMPLETED_SUCCESSFUL_MAIN003_RELEASE_ONLY','bytes':'ROOT_BINDS_ACTUAL_BYTES'},'no_supplement_execution_before_actual_main003_release_PASS':True})
    write(HERE/'source-preparation.json',{'source_only':True,'no_import_compile_query_or_scientific_arithmetic':True,'preparation_only_stdlib_text_copy_hash_JSON':True,'protected_inputs':len(protected),'source003_helper_and_support_copies_byte_identical':all(pin(HERE/'inputs'/source.name)['sha256']==pin(source)['sha256'] for source in [MAIN/'inner_gauge.hpp']+sorted((MAIN/'inputs').iterdir()) if source.is_file()),'original_Product_bodies_and_three_negative_lines_byte_identical':True,'no_main003_live_attempt_read_or_frozen':True,'fixed_main15740_oracle_cases_thresholds_not_modified':True,'derivative_targets_are_independent_closed_Fraction_formulas_not_helper_replays':True})
    for row in protected.values():add(row)
    files=[pin(path) for path in sorted(HERE.rglob('*')) if path.is_file()]
    write(HERE/'source-index.json',{'source_only':True,'execution_admitted':False,'files':files,'file_count':len(files),'protected_input_count':len(protected),'protected_inputs_unchanged':True,'scope':'Held129-record additive scalar arithmetic supplement; actual main003 Release PASS plus fresh root release required'})
    print(json.dumps({'source_index':pin(HERE/'source-index.json'),'probe':pin(HERE/'probe.cpp'),'oracle':pin(HERE/'oracle.py'),'runner':pin(HERE/'run_once.py'),'recipe':pin(HERE/'recipe.json'),'files':len(files),'protected_inputs':len(protected)},sort_keys=True))

if __name__=='__main__':main()
