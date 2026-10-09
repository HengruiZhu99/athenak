#!/usr/bin/env python3
"""One-shot SOURCE-ONLY extraction, metadata and exact-byte pin preparation."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent
OUTER=HERE.parent/'reference-wave-map-outer-arithmetic-source001-held-20261009'
PLAN=HERE.parent/'reference-wave-map-far-dual-plan-held-20261009'
READBACK=HERE.parent/'reference-wave-map-outer001-saved-readback-v2-20261009'

def pin(path):
    p=Path(path).absolute();h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as out:json.dump(value,out,indent=2,sort_keys=True,allow_nan=False);out.write('\n')
def check(row):assert pin(row['path'])=={k:row[k] for k in ('path','bytes','sha256')},row['path']
def extract(text,start,end):return text[text.index(start):text.index(end,text.index(start))]

def main():
    start=time.time();assert not (HERE/'source-index.json').exists()
    assert pin(PLAN/'index.json')['sha256']=='1da6c8208103786560bc7c9ec73c8ce3bbdfeafdbbf0eb6d5e4c6abb44a1af00'
    assert pin(OUTER/'source-index.json')['sha256']=='9da21a96d243441c95f276d1c076d3a4e561fc20921858be33f911cb3464f27e'
    assert pin(READBACK/'index.json')['sha256']=='0066ac951df874f0ae9987790d49e31acd4ee59d0a4eda46722741d1c803bbb8'
    protected={}
    def add(row):
        existing=protected.get(row['path']);simple={k:row[k] for k in ('path','bytes','sha256')}
        assert existing is None or existing==simple,row['path'];protected[row['path']]=simple
    for root,filename in ((OUTER,'source-index.json'),(PLAN,'index.json'),(READBACK,'index.json')):
        index=read(root/filename)
        for row in index['files']:add(row)
        add(pin(root/filename))
    for row in read(OUTER/'input-pins.json')+read(PLAN/'input-pins.json')+read(READBACK/'input-pins.json'):add(row)
    for row in protected.values():check(row)
    inputs=HERE/'inputs';inputs.mkdir(exist_ok=False)
    (inputs/'plan').mkdir()
    for name in ('PLAN.md','ORACLES.md','CASE-REGISTRY.json','recipe.json','index.json'):
        shutil.copyfile(PLAN/name,inputs/'plan'/name)
    for name in ('reference_wave_map.hpp','reference_wave_map_legacy.hpp','arithmetic_traits.hpp','dual_helpers.hpp','nonlinear_values.hpp'):
        shutil.copyfile(OUTER/'inputs'/name,inputs/name)
        assert pin(OUTER/'inputs'/name)['sha256']==pin(inputs/name)['sha256']
    original=(OUTER/'probe.cpp').read_text()
    state=extract(original,'hyp::Z4cJet<double> State(', 'template<class T>void EmitGauge(')
    cast=extract(original,'hyp::LayerPoint<D> CastPoint(', 'Jet Direction(')
    (inputs/'original_state.hpp').write_text(state)
    (inputs/'original_cast.hpp').write_text(cast)
    assert (inputs/'original_state.hpp').read_text()==state
    assert (inputs/'original_cast.hpp').read_text()==cast
    runner=(OUTER/'run_once.py').read_text()
    old_runner=runner
    runner=runner.replace("outer_arithmetic_local_execution_admitted","far_complete_dual_local_execution_admitted")
    dependency="    dependency=read(recipe['principal_receipt'])\n    if not(dependency['completed'] and dependency['passed'] and dependency['returncode']==0 and dependency['inputs_unchanged']):\n        raise RuntimeError('Successful principal outer receipt required')\n"
    assert dependency in runner
    runner=runner.replace(dependency,"    # Exact source/context pins and fresh root authorization govern this separate supplement.\n")
    runner=runner.replace('finite-Omega local gauge/helper plus actual C0 analytic-jet20-dual only','direct finite-Omega RWM complete field-dual arithmetic supplement only')
    runner=runner.replace("exe=attempt/('inner-probe-'+args.build)","exe=attempt/('far-dual-probe-'+args.build)")
    (HERE/'run_once.py').write_text(runner)
    (HERE/'runner-context-only.diff').write_text(''.join(difflib.unified_diff(old_runner.splitlines(True),runner.splitlines(True),fromfile=str(OUTER/'run_once.py'),tofile=str(HERE/'run_once.py'))))
    # AST/syntax/source metadata only: never import or execute oracle/runner.
    for name in ('oracle.py','run_once.py','prepare_sources.py'):ast.parse((HERE/name).read_text())
    probe=(HERE/'probe.cpp').read_text()
    assert 'ConformalRHS(' not in probe and 'inner::Gauge(' not in probe and 'Consistent(u)' not in probe
    main_recipe=read(OUTER/'recipe.json')
    flags={build:[token.replace(str(OUTER),str(HERE)) for token in values] for build,values in main_recipe['compile_flags'].items()}
    registry=read(PLAN/'CASE-REGISTRY.json')
    recipe={'source_only':True,'execution_admitted':False,
        'scope':'direct RWM finite-Omega field-dual arithmetic only, not compound inner/full22/native/evolution',
        'repository':main_recipe['repository'],'compiler':main_recipe['compiler'],
        'python':main_recipe['python'],'python_flags':['-I','-B'],
        'compile_flags':flags,'environment':main_recipe['environment'],
        'mpmath_parent':main_recipe['mpmath_parent'],'mpmath_init':main_recipe['mpmath_init'],
        'modes':['direct','closed'],'attempt_names':{'release':'Release001','debug':'Debug001'},
        'source_index':str(HERE/'source-index.json'),'input_pins':str(HERE/'input-pins.json'),
        'case_registry':str(inputs/'plan/CASE-REGISTRY.json'),
        'approved_plan_index':pin(PLAN/'index.json'),'outer_source_index':pin(OUTER/'source-index.json'),
        'outer_saved_local_readback_index_context_only':pin(READBACK/'index.json'),
        'source003_FAIL_preserved':True,'main_registry_and_oracle_changed':False,
        'fixed_counts':registry['counts'],'precision_decimal_digits':[480,560],
        'thresholds':{'precision_entrywise_scaled':'1e-220','native_entrywise_scaled':'2e-10',
            'closed_nonzero_normal_relative':'2e-10','closed_zero_absolute':'2e-10',
            'input_seed_formula_nonzero_relative':'2e-14','input_seed_formula_zero_absolute':'0',
            'unused_source_tangent_absolute':'0','FD_final_entrywise_scaled':'5e-7',
            'FD_first_over_final_min':'2','FD_all_level_floor':'5e-9'},
        'FD_scope':'16 original alpha-relative representative rows,5levels,160 new-helper sides embedded in direct rows',
        'error_scale':'max(1,abs(native entry),abs(target entry)); primal/tangent separately',
        'closed_negative_scope':'reuse three zero-seed legacy outputs; no extra helper calls',
        'no_operator_spectrum_propagation_or_native_admission':True}
    write(HERE/'recipe.json',recipe)
    write(HERE/'authorization-schema.json',{'far_complete_dual_local_execution_admitted':False,
        'recipe_sha256':'exact frozen recipe','source_index_sha256':'exact frozen source-index',
        'allowed_builds':[]})
    add(pin(Path(sys.executable)))
    rows=sorted(protected.values(),key=lambda row:row['path'])
    for row in rows:check(row)
    write(HERE/'input-pins.json',rows)
    write(HERE/'source-extraction-identities.json',{'source_only':True,'no_scientific_execution':True,
        'original_State_body_exact':True,'original_CastPoint_body_exact':True,
        'direct_rwm_helper_exact':pin(inputs/'reference_wave_map.hpp')['sha256'],
        'LegacyGauge_exact':pin(inputs/'reference_wave_map_legacy.hpp')['sha256'],
        'Product_traits_exact':pin(inputs/'arithmetic_traits.hpp')['sha256'],
        'main_15740_registry_and_oracle_unchanged':True,
        'case_registry_exact':pin(inputs/'plan/CASE-REGISTRY.json')['sha256'],
        'helper_counts_planned':registry['counts']['total_helper_evaluations'],
        'oracle_source_AST_only':True,'oracle_imported':False,'compiler_run':False,'queries_run':False})
    write(HERE/'preparation-receipt.json',{'completed':True,'returncode':0,'inputs_unchanged':True,
        'source_only':True,'protected_input_count':len(rows),'wall_seconds':time.time()-start,
        'scientific_imports_or_numerical_targets':False,'compiler_or_queries':False})
    files=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()]
    write(HERE/'source-index.json',{'files':files,'source_only':True,'execution_admitted':False,
        'scope':recipe['scope'],'protected_input_count':len(rows),'fixed_counts':registry['counts']})
    print(json.dumps({'source_only':True,'prepared':True,'files':len(files),'protected_inputs':len(rows),
                      'source_index_sha256':pin(HERE/'source-index.json')['sha256']},sort_keys=True))

if __name__=='__main__':main()
