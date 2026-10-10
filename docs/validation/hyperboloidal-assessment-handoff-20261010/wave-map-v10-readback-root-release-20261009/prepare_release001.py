"""UNEXECUTED: one-shot stdlib source/review/pin preparation for three v10 cases."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SUITE = BASE/'boundary/reference-wave-map-J0-bulk008-retained-independent-readback-v10-held-20261009'
OLD = BASE/'boundary/reference-wave-map-J0-bulk008-retained-independent-readback-v9-held-20261009'
EXPECTED = {
    'source-index.json': 'a67f587c07e18ee98e6aaaedfba1249fc334f51905a84c64aa3751df150408ff',
    'recipe.json': '05ba2b41afdcb39aeecfc7dbf659670b3e0011d7a6635a25cf42106f617e3fb2',
    'verify_retained.py': 'ee7de62d0faa1f221a636d03d1ba43901cb4b28c0bfbc63ce946fe9c5fc10b11',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda s:
                      (_ for _ in ()).throw(ValueError(s)))


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def add(pins, path, digest):
    path = str(Path(path).resolve())
    if pins.setdefault(path, digest) != digest:
        raise RuntimeError('conflicting protected input '+path)


def verify(pins):
    for path, digest in pins.items():
        if sha(path) != digest:
            raise RuntimeError('changed protected input '+path)


def add_index(pins, path):
    path = Path(path).resolve()
    index = load(path)
    add(pins, path, sha(path))
    files = index['files']
    if isinstance(files, dict):
        for name, item in files.items():
            add(pins, Path(name) if Path(name).is_absolute() else path.parent/name, item['sha256'])
    else:
        for item in files:
            add(pins, item['path'], item['sha256'])
    for item in index.get('external_inputs', []):
        add(pins, item['path'], item['sha256'])


def reverse_diff(current, diff):
    """Independently apply the saved unified diff backwards, checking every line."""
    lines = current.splitlines(True)
    patch = diff.splitlines(True)
    output = []
    pos = 0
    at = 2
    if not (patch[0].startswith('--- ') and patch[1].startswith('+++ ')):
        raise RuntimeError('expected full unified source diff')
    while at < len(patch):
        match = re.fullmatch(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*\n?', patch[at])
        if match is None:
            raise RuntimeError('unexpected source diff hunk')
        target = int(match.group(3))-1
        if target < pos:
            raise RuntimeError('overlapping source diff')
        output.extend(lines[pos:target])
        pos = target
        at += 1
        while at < len(patch) and not patch[at].startswith('@@ '):
            text = patch[at]
            prefix = text[:1]
            if prefix in (' ', '+'):
                if pos >= len(lines) or lines[pos] != text[1:]:
                    raise RuntimeError('source diff does not bind actual candidate line')
                if prefix == ' ':
                    output.append(lines[pos])
                pos += 1
            elif prefix == '-':
                output.append(text[1:])
            else:
                raise RuntimeError('unexpected unified diff record')
            at += 1
    output.extend(lines[pos:])
    return ''.join(output)


def source_checks(recipe):
    for name, digest in EXPECTED.items():
        if sha(SUITE/name) != digest:
            raise RuntimeError('exact frozen v10 identity differs: '+name)
    prior = load(OLD/'recipe.json')
    for key in ('cases', 'context', 'gates', 'environment', 'python', 'no_threshold_changes'):
        if recipe[key] != prior[key]:
            raise RuntimeError('changed inherited scientific/runtime contract '+key)
    if set(recipe['cases']) != {'primary', 'radial_pair', 'angular_pair'}:
        raise RuntimeError('fixed three cases required')
    new = (SUITE/'verify_retained.py').read_text()
    old = (OLD/'verify_retained.py').read_text()
    reverse = reverse_diff(new, (SUITE/'five-outer-measure-and-audits.diff').read_text())
    if reverse != old or ast.dump(ast.parse(reverse), include_attributes=False) != ast.dump(ast.parse(old), include_attributes=False):
        raise RuntimeError('complete byte/AST reverse does not restore v9')
    tree = ast.parse(new)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == 'mul'
             and isinstance(node.func.value, ast.Subscript)
             and isinstance(node.func.value.value, ast.Name)
             and node.func.value.value.id == 'outer_measure_arithmetic']
    labels = sorted(node.args[2].value for node in calls)
    if labels != sorted('outer_measure:'+name for name in ('E','Ks','Kw','G','loads')):
        raise RuntimeError('exact five final-product labels required')
    stores = [node for node in ast.walk(tree) if isinstance(node, ast.Name)
              and node.id == 'outer_measure_arithmetic' and isinstance(node.ctx, ast.Store)]
    if len(stores) != 1:
        raise RuntimeError('adapter dictionary shadowed')
    for name in ('tiny_normalization.py', 'fast_weighting.py', 'column_norm.py'):
        if (SUITE/name).read_bytes() != (OLD/name).read_bytes():
            raise RuntimeError('changed arithmetic helper '+name)
    evidence = recipe['outer_measure_correction']['evidence']
    diag = load(evidence['diagnostic_receipt']['path'])
    result = load(evidence['diagnostic_result']['path'])
    saved = load(evidence['saved_review_receipt']['path'])
    failure = load(evidence['preserved_v9_radial_failure']['path'])
    if not (diag.get('completed') is True and diag.get('returncode') == 0
            and diag.get('inputs_unchanged') is True and diag.get('diagnostic_completed') is True
            and result.get('counts') == recipe['outer_measure_correction']['diagnosed_E_counts']
            and saved.get('passed') is True and saved.get('inputs_unchanged') is True
            and failure.get('completed') is False and failure.get('returncode') == 1
            and failure.get('inputs_unchanged') is True):
        raise RuntimeError('completed E-only diagnosis/saved audit and old FAIL required')
    runtime = recipe['runtime_inventory_addendum']
    inventory = load(runtime['manifest']['path'])
    if not (runtime['files'] == len(inventory['files']) == 1332
            and inventory['bytecode_excluded'] is True
            and inventory['binaries_and_scientific_payloads_metadata_only'] is True):
        raise RuntimeError('exact SciPy runtime inventory required')
    for path, digest in inventory['files'].items():
        if recipe['pins'].get(path) != digest:
            raise RuntimeError('SciPy runtime file missing from recipe '+path)
    return {'complete_source_reverse_bytes_AST': True, 'five_named_products_only': True,
            'one_unshadowed_adapter_dictionary': True, 'helpers_byte_exact': True,
            'scientific_cases_gates_runtime_unchanged': True,
            'SciPy_runtime1332_manifest_and_files_protected': True,
            'original_v9_radial_FAIL_immutable': True, 'other_four_occurrences_not_inferred': True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--independent-review-receipt', type=Path, required=True)
    ap.add_argument('--independent-review-receipt-sha256', required=True)
    ap.add_argument('--independent-review-index', type=Path, required=True)
    ap.add_argument('--independent-review-index-sha256', required=True)
    ap.add_argument('--cap-seconds', type=int, required=True)
    args = ap.parse_args()
    if not (sys.flags.isolated and sys.dont_write_bytecode and sys.flags.optimize == 0):
        raise RuntimeError('root preparer requires -I -B and optimize0')
    if args.cap_seconds != 900:
        raise RuntimeError('exact newly declared900-second process-group cap required')
    for name in ('review.json','authorization.json','release.json','pins-prepared001.json'):
        if (HERE/name).exists():
            raise RuntimeError('fresh one-shot release only')
    attempt = HERE/'preparation001'
    attempt.mkdir(exist_ok=False)
    started = time.monotonic()
    record = {'completed': False, 'returncode': 1, 'source_review_preparation_only': True,
              'scientific_execution': False, 'process_group_cap_seconds': 900,
              'cap_is_new_not_historical': True}
    pins = {}
    try:
        recipe = load(SUITE/'recipe.json')
        pins.update(recipe['pins'])
        add_index(pins, SUITE/'source-index.json')
        add_index(pins, recipe['context']['owner_source_index'])
        add_index(pins, HERE/'launcher-source-index.json')
        add(pins, Path(recipe['python']).resolve(), sha(Path(recipe['python']).resolve()))
        independent = args.independent_review_receipt.resolve()
        independent_index = args.independent_review_index.resolve()
        add(pins, independent, args.independent_review_receipt_sha256)
        add(pins, independent_index, args.independent_review_index_sha256)
        add_index(pins, independent_index)
        verify(pins)
        write(attempt/'pins-before.json', pins)
        rv = load(independent)
        if not (rv.get('passed') is True and rv.get('inputs_unchanged') is True
                and rv.get('reviewed_source_index_sha256') == EXPECTED['source-index.json']
                and rv.get('source_review_only') is True):
            raise RuntimeError('completed exact independent source review required')
        checks = source_checks(recipe)
        for case in recipe['cases']:
            if (SUITE/'attempts'/('independent-'+case+'001')).exists() or (HERE/(case+'-invocation001')).exists():
                raise RuntimeError('fresh disjoint per-case destinations required')
        verify(pins)
        write(HERE/'review.json', {'passed': True, 'source_review_only': True,
            'reviewed_source_index_sha256': EXPECTED['source-index.json'],
            'independent_review_receipt': str(independent),
            'independent_review_receipt_sha256': args.independent_review_receipt_sha256,
            'independent_review_index': str(independent_index),
            'independent_review_index_sha256': args.independent_review_index_sha256,
            'checks': checks, 'no_numerical_execution': True,
            'all_three_fresh_actual_readbacks_required': True,
            'independent_parallel_cases_share_only_readonly_inputs': True,
            'process_group_cap_seconds': 900, 'cap_is_new_not_historical': True})
        auth = load(SUITE/'authorization-schema.json')
        auth.update(independent_saved_matrix_readback_authorized=True,
                    recipe_sha256=EXPECTED['recipe.json'], source_index_sha256=EXPECTED['source-index.json'],
                    readback_source_sha256=EXPECTED['verify_retained.py'],
                    root_review_sha256=sha(HERE/'review.json'),
                    independent_review_receipt=str(independent),
                    independent_review_receipt_sha256=args.independent_review_receipt_sha256,
                    independent_review_index=str(independent_index),
                    independent_review_index_sha256=args.independent_review_index_sha256,
                    process_group_cap_seconds=900, cap_is_new_not_historical=True,
                    child_review_gate_required=True, generator_eigenvalues_authorized=False,
                    no_automatic_retry_or_cap_increase=True)
        for name in ('fast_weighting_wrapper_units','column_norm_units'):
            add(pins, auth[name]['path'], auth[name]['sha256'])
        verify(pins)
        write(HERE/'authorization.json', auth)
        write(HERE/'pins-prepared001.json', pins)
        write(HERE/'release.json', {'suite': str(SUITE), 'pins': pins,
            'authorization_sha256': sha(HERE/'authorization.json'), 'review_sha256': sha(HERE/'review.json'),
            'launcher_source_index_sha256': sha(HERE/'launcher-source-index.json'),
            'independent_review_receipt': str(independent),
            'independent_review_receipt_sha256': args.independent_review_receipt_sha256,
            'independent_review_index': str(independent_index),
            'independent_review_index_sha256': args.independent_review_index_sha256,
            'process_group_cap_seconds': 900, 'cap_is_new_not_historical': True,
            'cases': ['primary','radial_pair','angular_pair'],
            'case_dependency': 'Independent fresh cases after this one frozen release; no shared mutable case output',
            'generator_spectrum_or_propagation_authorized': False})
        record.update(completed=True, returncode=0, pins=len(pins),
                      authorization_sha256=sha(HERE/'authorization.json'), release_sha256=sha(HERE/'release.json'))
    except BaseException as exc:
        record.update(error=type(exc).__name__+': '+str(exc))
        (attempt/'failure.txt').write_text(traceback.format_exc())
    finally:
        try:
            verify(pins)
            record['inputs_unchanged'] = True
        except BaseException as exc:
            record.update(inputs_unchanged=False, completed=False, returncode=1, post_pin_failure=str(exc))
        write(attempt/'pins-after.json', pins)
        record['seconds'] = time.monotonic()-started
        write(attempt/'receipt.json', record)
    print(json.dumps(record))
    if not (record['completed'] and record['inputs_unchanged']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
