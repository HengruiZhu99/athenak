#!/usr/bin/env python3
"""Independent metadata/text/AST review, never imports candidate modules."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
OWNER = Path('/Users/hz0693/research/hyperboloidal/build-layer-research/continuum/manufactured-Gaussian-a2-cache-v7-overlap-held-20261009')
INDEX_SHA = 'e0b82ef9a9e1615c921e30571524900b070c826606713a6052a59ff7cebb9aae'
RECIPE_SHA = 'cc2bf9fa1e67206f55cd34925f830bed4c01dd2d5fe3f053165443b97c7def5a'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    def reject(value):
        raise ValueError('Nonfinite JSON metadata: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def save(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def meta(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def ast_text(text):
    return ast.dump(ast.parse(text), include_attributes=False)


def main():
    started = time.perf_counter()
    if not sys.flags.isolated or not sys.dont_write_bytecode or sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE') != '0':
        raise RuntimeError('Require isolated unoptimized bytecode-off stdlib interpreter')
    if (HERE / 'receipt.json').exists() or (HERE / 'inputs').exists():
        raise RuntimeError('Fresh one-shot review only')
    index_path, recipe_path = OWNER/'source-index.json', OWNER/'recipe.json'
    assert digest(index_path) == INDEX_SHA
    assert digest(recipe_path) == RECIPE_SHA
    index, recipe = load(index_path), load(recipe_path)
    assert index['execution_authorized'] is False and index['source_only'] is True
    assert recipe['source_only'] is True and recipe['certificate_and_replay_remain_held'] is True
    pins = {}
    for record in index['files'] + recipe['protected_inputs'] + [recipe['python'], meta(index_path), meta(recipe_path)]:
        path = record['path']
        if path in pins and pins[path] != record:
            assert (pins[path]['sha256'],pins[path]['bytes']) == (record['sha256'],record['bytes'])
        pins[path] = record
    before = []
    for path, record in sorted(pins.items()):
        actual = meta(path)
        assert (actual['sha256'],actual['bytes']) == (record['sha256'],record['bytes']), path
        before.append(actual)
    science = load(OWNER/'source-equalities.json')['proofs']
    proofs = []
    for proof in science:
        name = proof['file']
        new, old = (OWNER/name).read_text(), (OWNER/'history/v6'/name).read_text()
        restored = new
        for original, changed in proof.get('exact_replacements', []):
            assert restored.count(changed) == 1, (name,changed)
            restored = restored.replace(changed, original, 1)
        assert restored == old and ast_text(restored) == ast_text(old), name
        proofs.append({'file':name, 'reverse_bytes_equal':True, 'reverse_AST_equal':True})
    def erase(text):
        return re.sub(r'    # BEGIN_V7_ADMISSION_OR_UNIT_ONLY\n.*?    # END_V7_ADMISSION_OR_UNIT_ONLY\n','',text,flags=re.S)
    restored = erase((OWNER/'unit_stage.py').read_text())
    original = (OWNER/'history/v6/unit_stage.py').read_text()
    assert restored == original and ast_text(restored) == ast_text(original)
    proofs.append({'file':'unit_stage.py','named_additions_erased_byte_AST_equal':True})
    restored = erase((OWNER/'admission.py').read_text()).replace(
        'if stage not in ("units", "certificate", "replay"):\n        raise RuntimeError("cache v7 admits only exact separately released stages")',
        'if stage != "certificate":\n        raise RuntimeError("cache v6 instrumentation admits only a fresh diagnostic producer")')
    original = (OWNER/'history/v6/admission.py').read_text()
    assert restored == original and ast_text(restored) == ast_text(original)
    proofs.append({'file':'admission.py','named_additions_and_stage_change_reverse_byte_AST_equal':True})
    restored = (OWNER/'run_once.py').read_text().replace(
        '("units_receipt", "new_units_receipt", "new_units_report", "new_overlap_report", "new_cache_report", "certificate_receipt", "certificate_payload")',
        '("units_receipt", "certificate_receipt", "certificate_payload")')
    original = (OWNER/'history/v6/run_once.py').read_text()
    assert restored == original and ast_text(restored) == ast_text(original)
    proofs.append({'file':'run_once.py','dynamic_key_extension_reverse_byte_AST_equal':True})
    old_recipe = load(OWNER/'history/v6/recipe.json')
    unchanged_keys = ['a','bits','replay_bits','series_order','epsilon_endpoint','sigma','domain_wall_seconds',
                      'replay_wall_seconds','stage_timeouts','cache_endpoint_cap','cache_eviction',
                      'max_certificate_bytes','max_depth','max_leaves','instrumentation_cadence_nodes']
    assert all(recipe[k] == old_recipe[k] for k in unchanged_keys)
    assert (recipe['expected_unit_count'],recipe['expected_cache_unit_count'],recipe['expected_overlap_unit_count'],recipe['expected_combined_unit_count']) == (144,35,58,237)
    assert recipe['series_interface_sigma_multiple'] == 2
    prior = load(recipe['cached_v6_timeout_receipt']['path'])
    assert prior['completed'] is False and prior['passed'] is False and prior['returncode'] == 1 and prior['inputs_unchanged'] is True
    assert prior['source_index_sha256'] == recipe['cached_v6_source_index_sha256']
    assert prior['recipe_sha256'] == recipe['cached_v6_recipe_sha256']
    assert not (Path(recipe['cached_v6_timeout_receipt']['path']).parent/'report.json').exists()
    selected = ['PLAN.md','recipe.json','source-index.json','source-equalities.json','unit-registry.json',
                'producer_bounds.py','replay_bounds.py','certificate_stage.py','replay_stage.py',
                'admission.py','run_once.py','unit_stage.py','overlap_units.py','interval.py',
                'interval_uncached.py','cache_units.py','instrumentation.py','v6-cap-history.json']
    pencil = Path(recipe['reviewed_Taylor_overlap_note']['path'])
    extra = [(pencil,'Taylor-overlap-ASSESSMENT.md'),(Path(recipe['reviewed_Taylor_overlap_index']['path']),'Taylor-overlap-index.json')]
    output = HERE/'inputs'; output.mkdir()
    copies = []
    for source, name in [(OWNER/name,name) for name in selected]+extra:
        target = output/name; shutil.copyfile(source,target)
        assert digest(source) == digest(target)
        copies.append({'source':str(source),'copy':str(target.relative_to(HERE)),'sha256':digest(target),'bytes':target.stat().st_size})
    after = [meta(record['path']) for record in before]
    assert before == after
    save(HERE/'pin-readback.json',{'before':before,'after':after,'inputs_unchanged':True,'copies':copies})
    save(HERE/'source-proof-readback.json',{'proofs':proofs,'unchanged_recipe_keys':unchanged_keys,
                                         'no_candidate_import_or_evaluation':True,'all_metadata_pins_unchanged':True})
    save(HERE/'receipt.json',{'passed':True,'inputs_unchanged':True,'source_review_only':True,
                             'reviewed_source_index_sha256':INDEX_SHA,'reviewed_recipe_sha256':RECIPE_SHA,
                             'unique_protected_inputs':len(before),'source_copies':len(copies),
                             'mathematical_source_review_no_blocker':True,'candidate_execution':False,
                             'interval_or_Fraction_arithmetic_executed':False,'JSONL_decoded':False,
                             'old_v6_failed_preserved':True,'future_unit_count':237,'future_units_executed':False,
                             'producer_or_replay_admitted_by_this_review':False,
                             'elapsed_seconds':time.perf_counter()-started,
                             'runtime':{'executable':sys.executable,'isolated':sys.flags.isolated,'dont_write_bytecode':sys.dont_write_bytecode,'optimize':sys.flags.optimize}})
    records=[]
    for path in sorted(HERE.rglob('*')):
        if path.is_file():
            entry=meta(path);entry['path']=str(path.relative_to(HERE));records.append(entry)
    save(HERE/'index.json',{'files':records,'scope':'independent_source_math_admission_review_only',
                            'candidate_execution':False,'inputs_unchanged':True})
    print(json.dumps({'passed':True,'index_sha256':digest(HERE/'index.json'),
                      'receipt_sha256':digest(HERE/'receipt.json'),'unique_inputs':len(before)},allow_nan=False))


if __name__ == '__main__':
    main()
