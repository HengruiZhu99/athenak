"""Stdlib text/AST/hash/compact-receipt review only; never import candidate code."""
import ast
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
OWNER = BASE/'continuum/manufactured-Gaussian-a2-cache-v6-instrumented-held-20261009'
ERRATUM = BASE/'continuum/Gaussian-a2-cache-v6-inventory-addendum-20261009'
INDEX = 'cafc4df240ba9988919f2f4d97522bda2f71a19f1899a625d2be98afa2144397'
RECIPE = '9a7ee237b6fd4d764bac38e62ff4307248fbc4beb57b7a0dbd307e690d0e0742'
ERRATUM_INDEX = '62be282de52536b2e5f3f5c82bf43f9de309efcbe3e38ab4b146325717aac4f2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1 << 20),b''):
            h.update(block)
    return h.hexdigest()


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def load(path):
    path = Path(path)
    assert path.suffix != '.jsonl' and path.stat().st_size <= 1 << 20
    def reject(token):
        raise ValueError('nonfinite JSON '+token)
    return json.loads(path.read_text(),parse_float=Decimal,parse_constant=reject)


def save(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def erase_observations(text):
    result = []
    depth = 0
    for line in text.splitlines(keepends=True):
        if line.strip() == '# BEGIN_OBSERVATION_ONLY':
            assert depth == 0
            depth = 1
        elif line.strip() == '# END_OBSERVATION_ONLY':
            assert depth == 1
            depth = 0
        elif not depth:
            result.append(line)
    assert depth == 0
    return ''.join(result)


def main():
    started = time.monotonic()
    assert sha(OWNER/'source-index.json') == INDEX
    assert sha(OWNER/'recipe.json') == RECIPE
    assert sha(ERRATUM/'index.json') == ERRATUM_INDEX
    idx,recipe = load(OWNER/'source-index.json'),load(OWNER/'recipe.json')
    erridx = load(ERRATUM/'index.json')
    hist = load(ERRATUM/'additional-protected-history.json')
    assert idx['file_count'] == len(idx['files']) == 67
    assert idx['source_only'] and idx['execution_authorized'] is False
    assert hist['historical_additions_only'] == 3 and hist['candidate_source_modified'] is False
    assert hist['all_active_candidate_sources_previously_indexed'] is True
    assert len(hist['files']) == 4
    entries = idx['files']+recipe['protected_inputs']+erridx['files']+hist['files']
    entries += [pin(OWNER/'source-index.json'),pin(ERRATUM/'index.json'),recipe['python']]
    entries += [pin(HERE/'PLAN.md'),pin(HERE/'review_source.py')]
    protected = {}
    for entry in entries:
        actual = pin(entry['path'])
        assert actual['bytes'] == entry['bytes'] and actual['sha256'] == entry['sha256'],entry['path']
        assert actual['path'] not in protected or actual == protected[actual['path']]
        protected[actual['path']] = actual
    save('inputs-before.json',list(protected.values()))
    listed = {Path(x['path']).relative_to(OWNER).as_posix() for x in idx['files']}
    historical = {Path(x['path']).relative_to(OWNER).as_posix() for x in hist['files']}
    actual_files = {x.relative_to(OWNER).as_posix() for x in OWNER.rglob('*') if x.is_file()}
    assert actual_files == listed|historical
    assert len(actual_files) == 71

    equalities = []
    old = OWNER/'history/v5'
    for name in ['interval.py','interval_uncached.py','producer_bounds.py','replay_bounds.py',
                 'replay_stage.py','unit_stage.py','cache_units.py','run_once.py']:
        new_text,old_text = (OWNER/name).read_text(),(old/name).read_text()
        assert new_text == old_text,name
        assert ast.dump(ast.parse(new_text)) == ast.dump(ast.parse(old_text))
        equalities.append(dict(file=name,byte_equal=True,AST_equal=True))
    stripped = erase_observations((OWNER/'certificate_stage.py').read_text())
    assert stripped == (old/'certificate_stage.py').read_text()
    assert ast.dump(ast.parse(stripped)) == ast.dump(ast.parse((old/'certificate_stage.py').read_text()))
    admission = erase_observations((OWNER/'admission.py').read_text()).replace(
        'if stage != "certificate":\n        raise RuntimeError("cache v6 instrumentation admits only a fresh diagnostic producer")',
        'if stage not in ("certificate", "replay"):\n        raise RuntimeError("cache v5 admits certificate/replay only after exact release")')
    assert admission == (old/'admission.py').read_text()
    for x in idx['files']:
        if Path(x['path']).suffix == '.py':
            ast.parse(Path(x['path']).read_text())
    original_recipe = load(old/'recipe.json')
    math_keys = ['a','bits','replay_bits','series_order','sigma','epsilon_endpoint','max_depth',
                 'max_leaves','max_certificate_bytes','cache_endpoint_cap','cache_eviction',
                 'expected_unit_count','expected_cache_unit_count','expected_combined_unit_count',
                 'replay_wall_seconds','python','python_version','scientific_arrays_or_native_inputs']
    assert all(recipe[k] == original_recipe[k] for k in math_keys)
    assert original_recipe['domain_wall_seconds'] == 1800 and recipe['domain_wall_seconds'] == 600
    assert original_recipe['stage_timeouts']['certificate'] == 1860 and recipe['stage_timeouts']['certificate'] == 660
    assert recipe['stage_timeouts']['replay'] == original_recipe['stage_timeouts']['replay'] == 960
    assert recipe['stage_timeouts']['units'] == original_recipe['stage_timeouts']['units'] == 120
    assert recipe['instrumentation_only'] is True and recipe['replay_admitted_by_this_candidate'] is False
    assert recipe['instrumentation_cadence_nodes'] == 128 and recipe['no_resume'] is True
    assert recipe['no_completion_fraction_claim'] is True

    unit = load(recipe['cached_v3_units_receipt']['path'])
    report = load(recipe['cached_v3_units_report']['path'])
    cache_report = load(recipe['cached_v3_cache_report']['path'])
    root_review = load(recipe['cached_v3_root_source_review']['path'])
    assert unit['completed'] is unit['passed'] is unit['inputs_unchanged'] is True
    assert type(unit['returncode']) is int and unit['returncode'] == 0 and unit['stage'] == 'units'
    assert unit['source_index_sha256'] == recipe['cached_v3_source_index_sha256']
    assert unit['recipe_sha256'] == recipe['cached_v3_recipe_sha256']
    assert recipe['cached_v3_units_report'] in unit['outputs'] and recipe['cached_v3_cache_report'] in unit['outputs']
    assert report['passed'] is report['cache_units_passed'] is True
    assert (report['case_count'],report['cache_unit_count'],report['combined_unit_count']) == (144,35,179)
    assert cache_report['passed'] is True and cache_report['case_count'] == len(cache_report['cases']) == 35
    assert all(row['passed'] is True for row in cache_report['cases'])
    assert root_review['passed'] is root_review['root_full_cache_source_math_and_admission_review'] is True
    assert root_review['original_uncached_endpoint_body_AST_identical'] is True
    failures = {}
    for version in ['v4','v5']:
        receipt_pin = recipe['cached_'+version+'_timeout_receipt']
        failure = load(receipt_pin['path'])
        assert failure['completed'] is failure['passed'] is False
        assert failure['inputs_unchanged'] is True and type(failure['returncode']) is int and failure['returncode'] == 1
        assert failure['stage'] == 'certificate'
        assert failure['source_index_sha256'] == recipe['cached_'+version+'_source_index_sha256']
        assert failure['recipe_sha256'] == recipe['cached_'+version+'_recipe_sha256']
        for suffix in ['progress','stderr','command','partial_certificate_metadata']:
            assert recipe['cached_'+version+'_'+suffix] in failure['outputs']
        assert 'UNRESOLVED: declared domain time limit' in Path(recipe['cached_'+version+'_stderr']['path']).read_text()
        assert not (Path(receipt_pin['path']).parent/'report.json').exists()
        failures[version] = dict(receipt_sha256=receipt_pin['sha256'],passed=False,
                                 inputs_unchanged=True,partial_payload_decoded=False)
    progress = load(recipe['cached_v5_progress']['path'])
    assert (progress['nodes'],progress['leaves'],progress['pending']) == (24448,12217,18)

    observer = (OWNER/'instrumentation.py').read_text()
    # These are static source-protocol assertions, not executed dict/interval tests.
    for text in ['present=super().__contains__(key)','return present','super().__delitem__(key)',
                 'for root_id,path,box,depth in reversed(stack):',
                 "'coverage_fraction_claimed':False","'global_slicing_accepted':False",
                 "'most_recent_node':self.last","'completion_requires_full_producer_footer_and_independent_replay':True"]:
        assert text in observer,text
    ctree = ast.parse(observer)
    cls = next(n for n in ctree.body if isinstance(n,ast.ClassDef) and n.name=='ObservedEndpointCache')
    assert {n.name for n in cls.body if isinstance(n,ast.FunctionDef)} == {'__init__','__contains__','__delitem__','counters'}
    body = (OWNER/'certificate_stage.py').read_text()
    assert body.index('ctx._exp_endpoint_cache = cache') < body.index('result = coefficients(')
    assert body.index('check(args.recipe, args.authorization') < body.index('from interval import')
    guard = (OWNER/'admission.py').read_text()
    assert guard.index('if stage != "certificate":') < guard.index('if stage == "replay":')
    assert 'out.mkdir(parents=True, exist_ok=False)' in (OWNER/'run_once.py').read_text()

    save('source-equality-review.json',dict(unchanged_sources=equalities,
        producer_observation_erased_byte_and_AST_equal=True,admission_reverse_byte_equal=True,
        mathematical_recipe_fields_equal=math_keys,active_source_inventory67=True,
        additive_historical_indices3=True,all71_owner_files_accounted=True))
    # Capture reviewed source text only. Scientific JSONL remains metadata/hash only.
    for entry in idx['files']:
        src = Path(entry['path'])
        dst = HERE/'reviewed-source'/src.relative_to(OWNER)
        dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_bytes(src.read_bytes())
    for entry in erridx['files']+hist['files']:
        src = Path(entry['path'])
        if src.is_relative_to(ERRATUM):
            dst = HERE/'inventory-addendum'/src.relative_to(ERRATUM)
        else:
            dst = HERE/'reviewed-source'/src.relative_to(OWNER)
        dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_bytes(src.read_bytes())
    after = [pin(row['path']) for row in protected.values()]
    assert after == list(protected.values())
    save('inputs-after.json',after)
    result = dict(passed=True,reviewed_source_index_sha256=INDEX,reviewed_recipe_sha256=RECIPE,
        inventory_erratum_index_sha256=ERRATUM_INDEX,protected_unique_files=len(protected),inputs_unchanged=True,
        candidate_imported=False,interval_arithmetic_executed=False,certificate_JSONL_decoded=False,
        targets_or_scientific_calls=False,source_review_only=True,execution_authorized_by_this_review=False,
        prerequisite_combined_units=179,current_v5_failed=True,failures=failures,
        producer_seconds=600,wrapper_seconds=660,replay_admitted=False,
        findings=[
            'Marked observations reverse to the v5 producer exactly; bounds, arithmetic, DFS, splits, positivity and caps are unchanged except time limits.',
            'Dict membership delegates once and returns the identical boolean; deletion delegates before counting. Lookup, assignment, iteration, FIFO and exact Context-owned cached values are unchanged.',
            'Observers report the most recent completed node and reversed pending stack in future DFS order, between nodes. Failure inside a node may leave the prior checkpoint only.',
            'Directed bounds, regular parity remainder, separated formulas and full-sphere quadratic lower minimization remain source-identical; no numerical proof evaluation was performed.',
            'Active admission allows certificate only, requires successful 179 units and exact failed v4/v5 histories, and refuses replay before interval imports.',
            'Historical recipe/plan labels mentioning v4/v5 and admitted certificate/replay are superseded by the explicit v6 status and executable certificate-only guard.',
            'The additive inventory erratum protects three omitted historical index copies without changing owner bytes; it is separately bound by this review.',
            'No partial-region completion, coverage percentage, termination, global slicing or PDE/native acceptance is inferred.'
        ],blockers=[],seconds=time.monotonic()-started)
    save('receipt.json',result)
    print(json.dumps(result,sort_keys=True))


if __name__ == '__main__':
    main()
