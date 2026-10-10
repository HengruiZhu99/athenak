"""One-shot SOURCE ONLY text/AST/hash preparation; never imports a candidate."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import time

ROOT = Path('/Users/hz0693/research/hyperboloidal')
BASE = ROOT / 'build-layer-research'
OLD = BASE / 'boundary/reference-wave-map-J0-bulk008-retained-independent-readback-v9-held-20261009'
NEW = BASE / 'boundary/reference-wave-map-J0-bulk008-retained-independent-readback-v10-held-20261009'
PREP = Path(__file__).resolve().parent
MASS = BASE / 'boundary/reference-wave-map-v9-mass-measure-diagnostic-held-20261009'
REVIEW = BASE / 'boundary/reference-wave-map-v9-mass-measure-independent-saved-review-20261009'
MASS_ROOT = BASE / 'wave-map-v9-mass-measure-diagnostic-root-release-20261009'


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
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def row(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def index_pins(index, parent):
    files = index['files']
    if isinstance(files, dict):
        return {str((Path(path) if Path(path).is_absolute() else parent/path).resolve()): item['sha256']
                for path, item in files.items()}
    return {item['path']: item['sha256'] for item in files}


def verify(pins):
    for path, digest in pins.items():
        if sha(path) != digest:
            raise RuntimeError('changed protected source/input ' + path)


def main():
    started = time.monotonic()
    if NEW.exists() or (PREP / 'receipt.json').exists():
        raise RuntimeError('fresh source preparation only; no overwrite or retry')
    required = {
        OLD/'source-index.json': '0f276fc8cd813dee704134aac4d91e3f033457982db7fcb00ef445f4620e5241',
        OLD/'verify_retained.py': '345d5d5a87278c8a793a9ce8fcfcaeb1a50171a6ba68206b4df27d5de4f8bdb6',
        OLD/'recipe.json': '5addb0fb49669ca2f240087e4d8291065c19c613f35eab91a85b78479631105e',
        OLD/'tiny_normalization.py': 'd27a49639ffc20de508e8b72bf63cfea68b6a703fdcd130f25c8689291a59272',
        OLD/'fast_weighting.py': '5781ed73f9da31e0be3164e141125ad7b0d436333ecacee1ecbc124cf61ab095',
        OLD/'column_norm.py': 'f768c26318712035452c6d45d5a9fd88b5c147a16161b9ac671317ebf9bf0f01',
        MASS/'attempts/diagnostic001/receipt.json': '80c93fe0ab717c8cbaaedd7ab14bfd0dc77b905f001de8b783418382a6e57d98',
        MASS/'attempts/diagnostic001/result.json': '862459d97b8ae5704f376d6cf5758826da863187ea5fd30cad797da48ff10926',
        MASS/'outer-invocation001/receipt.json': '92d16b76f4ada57eea4f6a8dc8571d7d5b0fce8b78d6f48b0ef954a43e2badec',
        REVIEW/'index.json': '096522a55c256a2a7cea77117ddbdd728208ca8db1316a3c227e11f46b53f73c',
        REVIEW/'receipt.json': '3460bed12cc0747765f7e88c6850fcac15760df97de92dbce0f19740d7a49721',
        REVIEW/'summary.json': '6b252f9ca5d39f49b2a0645f666d593ef97dbeca1010b13ea7456a2601b2d5cc',
        OLD/'attempts/independent-radial_pair001/receipt.json': 'bfab682d017693316e885f22bb00b13707aa6f211bf851198a54e05bf03dc4d1',
    }
    pins = {str(path.resolve()): digest for path, digest in required.items()}
    old_recipe = load(OLD/'recipe.json')
    for source in (old_recipe['pins'], index_pins(load(OLD/'source-index.json'), OLD),
                   index_pins(load(REVIEW/'index.json'), REVIEW)):
        for path, digest in source.items():
            if pins.setdefault(path, digest) != digest:
                raise RuntimeError('conflicting source history pin')
    runtime_path = MASS_ROOT/'scipy-runtime-pins001.json'
    runtime = load(runtime_path)
    if not (runtime['imports_executed'] is False and runtime['bytecode_excluded'] is True
            and runtime['binaries_and_scientific_payloads_metadata_only'] is True
            and len(runtime['files']) == 1332):
        raise RuntimeError('exact already inventoried SciPy runtime metadata required')
    pins[str(runtime_path)] = sha(runtime_path)
    for path, digest in runtime['files'].items():
        if pins.setdefault(path, digest) != digest:
            raise RuntimeError('conflicting SciPy runtime pin')
    for name in ('primary', 'angular_pair'):
        path = OLD/'attempts'/('independent-'+name+'001')/'receipt.json'
        pins[str(path)] = sha(path)
    pins[str(OLD/'attempts/independent-radial_pair001/failure.txt')] = sha(OLD/'attempts/independent-radial_pair001/failure.txt')
    pins[str(MASS_ROOT/'outer-invocation001/receipt.json')] = sha(MASS_ROOT/'outer-invocation001/receipt.json')
    pins[str(MASS_ROOT/'release.json')] = sha(MASS_ROOT/'release.json')
    evidence_paths = {
        'diagnostic_source_index': MASS/'source-index.json',
        'diagnostic_recipe': MASS/'recipe.json',
        'diagnostic_receipt': MASS/'attempts/diagnostic001/receipt.json',
        'diagnostic_result': MASS/'attempts/diagnostic001/result.json',
        'diagnostic_wrapper_receipt': MASS/'outer-invocation001/receipt.json',
        'root_receipt': MASS_ROOT/'outer-invocation001/receipt.json',
        'saved_review_index': REVIEW/'index.json',
        'saved_review_receipt': REVIEW/'receipt.json',
        'saved_review_summary': REVIEW/'summary.json',
        'preserved_v9_radial_failure': OLD/'attempts/independent-radial_pair001/receipt.json',
        'preserved_v9_radial_trace': OLD/'attempts/independent-radial_pair001/failure.txt',
    }
    for path in evidence_paths.values():
        pins[str(path)] = sha(path)
    pins[str(Path(__file__).resolve())] = sha(__file__)
    verify(pins)
    write(PREP/'input-pins-before.json', pins)
    NEW.mkdir()
    (NEW/'history').mkdir()
    for name in ('verify_retained.py', 'recipe.json', 'source-index.json', 'PLAN.md'):
        (NEW/'history'/('v9-'+name)).write_bytes((OLD/name).read_bytes())
    for name in ('tiny_normalization.py', 'fast_weighting.py', 'column_norm.py'):
        (NEW/name).write_bytes((OLD/name).read_bytes())

    source = (OLD/'verify_retained.py').read_text()
    changes = []

    def replace(old, new, scope):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError('nonunique intended edit: '+scope)
        source = source.replace(old, new)
        changes.append({'scope': scope, 'old': old, 'new': new})

    replace('BASIS_DIVISION_AUDIT = None\n',
            'BASIS_DIVISION_AUDIT = None\nOUTER_MEASURE_AUDITS = None\n', 'audit global only')
    replace("                                 'weak-sum addition and final measure multiplication unchanged')",
            "                                 'weak-sum addition unchanged; v10 final measure products have separate audits')",
            'metadata clarification: the historical weak operand correction itself is unchanged')
    summary = '''def outer_measure_summary(name):
    if OUTER_MEASURE_AUDITS is None:
        return None
    result = OUTER_MEASURE_AUDITS[name].summary()
    result['scope'] = ('Only the final rounded radial measure times the existing ' +
                       name + ' point-sum array; separate instance/audit for each of five sites')
    result['operation_label'] = 'outer_measure:' + name
    result['operand_scope'] = ('measure=weights[ir]/c and every original inner sum, '
                               'bilinear, H/Gamma action and incoming/boundary expression are unchanged')
    result['operation_order'] = ('Only this final multiplication uses the unchanged tested '
                                 'fast proof/exact-tiny adapter; the original strict augmented '
                                 'addition and all preceding operations retain their order')
    result['unchanged_scope'] = ('No BLAS, einsum, reduction, sqrt, SVD, derivative, source, '
                                'quadrature, trial, solve, gate or tolerance change')
    result['measured_scope'] = ('Saved radial E-only diagnostic observed 28 tiny products '
                               'at627 in609..640,20 rounded zero and8 nonzero subnormal; '
                               'no occurrence is asserted for the other four products or grids')
    result['error_scope'] = ('Exact local rounding counts/bounds only; no propagated matrix, '
                            'solve, continuum or native error certificate')
    return result


'''
    replace('def scientific_readback(case, context, out):\n',
            summary+'def scientific_readback(case, context, out):\n', 'new summary function only')
    replace('    global NORMALIZATION_AUDIT, WEIGHTING_AUDIT, DERIVATIVE_AUDIT, WEAK_AUDIT, BASIS_DIVISION_AUDIT\n',
            '    global NORMALIZATION_AUDIT, WEIGHTING_AUDIT, DERIVATIVE_AUDIT, WEAK_AUDIT, BASIS_DIVISION_AUDIT, OUTER_MEASURE_AUDITS\n',
            'audit global declaration only')
    replace('    BASIS_DIVISION_AUDIT = basis_division_arithmetic\n',
            "    BASIS_DIVISION_AUDIT = basis_division_arithmetic\n    outer_measure_arithmetic = {name: FastWeightingArithmetic(np)\n                                for name in ('E', 'Ks', 'Kw', 'G', 'loads')}\n    OUTER_MEASURE_AUDITS = outer_measure_arithmetic\n",
            'five independent unchanged helper instances; no arithmetic performed here')
    originals = {
        'E': '        E+=measure*(bilinear(y,hy,angles)+bilinear(u,u,angles))\n',
        'Ks': '        Ks+=measure*(bilinear(y,hyt,angles)+bilinear(u,ut,angles))\n',
        'Kw': '        Kw+=measure*(-bilinear(action(H,dy)[:,qidx]+weak_arithmetic.mul(div,hy[:,qidx],"line394:div_times_hy_q"),ut,angles)+\n                     bilinear(hy[:,vidx],vt,angles)+bilinear(u,ut,angles))\n',
        'G': '        G+=measure*(sourcepart+sourcepart.T-bilinear(y,action(gamma,y),angles)+umass+umass.T)\n',
        'loads': '        loads+=measure*(bilinear(y,action(H,fy),angles)+bilinear(u,fu,angles))\n',
    }
    for name, old in originals.items():
        operand = old.split('+=measure*', 1)[1].rstrip('\n')
        new = "        "+name+"+=outer_measure_arithmetic['"+name+"'].mul(measure,"+operand+",'outer_measure:"+name+"')\n"
        replace(old, new, 'SOLE numerical expression '+name+': final outer measure multiplication')
    replace("        'tiny_retained_basis_division_arithmetic':basis_division_summary(),\n",
            "        'tiny_retained_basis_division_arithmetic':basis_division_summary(),\n        'tiny_outer_measure_arithmetic':{name:outer_measure_summary(name)\n            for name in ('E','Ks','Kw','G','loads')},\n",
            'result metadata: five separate summaries')
    admission = '''        evidence = recipe['outer_measure_correction']['evidence']
        diagnostic = load(evidence['diagnostic_receipt']['path'])
        diagnostic_result = load(evidence['diagnostic_result']['path'])
        saved_review = load(evidence['saved_review_receipt']['path'])
        old_radial = load(evidence['preserved_v9_radial_failure']['path'])
        expected_counts = recipe['outer_measure_correction']['diagnosed_E_counts']
        if not (diagnostic.get('completed') is True and diagnostic.get('returncode') == 0
                and diagnostic.get('inputs_unchanged') is True
                and diagnostic.get('diagnostic_completed') is True
                and diagnostic_result.get('diagnostic_completed') is True
                and diagnostic_result.get('counts') == expected_counts
                and diagnostic_result.get('input_map_radius_window') == [609, 640]
                and diagnostic_result.get('E_accumulated') is False
                and diagnostic_result.get('SVD_executed') is False
                and diagnostic_result.get('query_or_generator_executed') is False
                and saved_review.get('passed') is True
                and saved_review.get('inputs_unchanged') is True
                and saved_review.get('no_scientific_reexecution') is True
                and saved_review.get('original_v9_radial_FAIL_preserved') is True
                and old_radial.get('completed') is False
                and old_radial.get('returncode') == 1
                and old_radial.get('inputs_unchanged') is True
                and old_radial.get('failure') == 'FloatingPointError: underflow encountered in multiply'):
            raise RuntimeError('completed E-only diagnosis/saved review and preserved v9 FAIL required')
'''
    replace("        write(out/'pins-before.json',protected)\n        result=scientific_readback(case,recipe['context'],out)\n",
            admission+"        write(out/'pins-before.json',protected)\n        result=scientific_readback(case,recipe['context'],out)\n",
            'pre-import evidence guard only; no numeric import or target arithmetic')
    replace("            write(out/'tiny-retained-basis-division-rounding.json',basis_division_summary())\n",
            "            write(out/'tiny-retained-basis-division-rounding.json',basis_division_summary())\n        if OUTER_MEASURE_AUDITS is not None:\n            for name in ('E','Ks','Kw','G','loads'):\n                write(out/('tiny-outer-measure-'+name+'-rounding.json'),outer_measure_summary(name))\n",
            'success/failure audit persistence only')
    (NEW/'verify_retained.py').write_text(source)

    old_source = (OLD/'verify_retained.py').read_text()
    reversed_source = source
    for item in reversed(changes):
        if reversed_source.count(item['new']) != 1:
            raise RuntimeError('nonunique reverse patch')
        reversed_source = reversed_source.replace(item['new'], item['old'])
    old_ast = ast.dump(ast.parse(old_source), include_attributes=False)
    reverse_ast = ast.dump(ast.parse(reversed_source), include_attributes=False)
    if reversed_source != old_source or old_ast != reverse_ast:
        raise RuntimeError('exact source/AST reversal failed')
    tree = ast.parse(source)
    calls = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'mul' and isinstance(node.func.value, ast.Subscript)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == 'outer_measure_arithmetic'):
            calls.append(node)
    stores = [node for node in ast.walk(tree) if isinstance(node, ast.Name)
              and isinstance(node.ctx, ast.Store) and node.id == 'outer_measure_arithmetic']
    if len(calls) != 5 or len(stores) != 1:
        raise RuntimeError('exact five sites/one unshadowed adapter dictionary required')
    labels = sorted(node.args[2].value for node in calls)
    if labels != sorted('outer_measure:'+name for name in originals):
        raise RuntimeError('distinct fixed five labels required')
    (NEW/'five-outer-measure-and-audits.diff').write_text(''.join(difflib.unified_diff(
        old_source.splitlines(True), source.splitlines(True), fromfile=str(OLD/'verify_retained.py'),
        tofile=str(NEW/'verify_retained.py'))))
    write(NEW/'exact-reverse-proof.json', {
        'source_only': True, 'candidate_imported': False, 'scientific_execution': False,
        'baseline': row(OLD/'verify_retained.py'), 'candidate': row(NEW/'verify_retained.py'),
        'reverse_source_sha256': hashlib.sha256(reversed_source.encode()).hexdigest(),
        'byte_exact_reverse': True, 'AST_exact_reverse': True,
        'new_final_measure_multiply_calls': len(calls), 'adapter_dictionary_Store_count': len(stores),
        'distinct_labels': labels, 'other_scientific_expressions_unchanged': True,
        'helpers_byte_exact': {name: sha(NEW/name) == sha(OLD/name)
                              for name in ('tiny_normalization.py', 'fast_weighting.py', 'column_norm.py')},
        'changes': [{'scope': item['scope']} for item in changes],
    })

    recipe = old_recipe
    # Preserve all historical paths/pins. Only active local helper metadata and future destinations move.
    recipe['column_norm_correction']['helper'] = str(NEW/'column_norm.py')
    recipe['derivative_product_correction']['helper'] = row(NEW/'tiny_normalization.py')
    recipe['derivative_product_correction']['wrapper'] = row(NEW/'fast_weighting.py')
    recipe['fast_weighting_proof']['source'] = str(NEW/'fast_weighting.py')
    recipe['retained_basis_division_correction']['helper'] = row(NEW/'tiny_normalization.py')
    recipe['weak_product_correction']['helper'] = row(NEW/'fast_weighting.py')
    recipe['weak_product_correction']['unchanged_fallback'] = row(NEW/'tiny_normalization.py')
    recipe['weak_product_correction']['source'] = str(NEW/'verify_retained.py')
    recipe['weak_product_correction']['unmodified'] = ('All weak operand/addition/derivative/contraction/gate definitions unchanged; '
                                                      'final outer measure multiplication has a separate v10 audit')
    recipe['scope'] = 'Independent saved point/coefficient reconstruction; five named final outer measure products only; no query, generator spectrum or propagation'
    recipe['status'] = 'HELD v10 source only; five separate final-measure adapters; root and independent source review/release required'
    recipe['readiness'] = 'SOURCE-ONLY stdlib text/JSON/hash/AST preparation; no candidate import, scientific arithmetic, array/map/JSONL decode or verifier execution'
    recipe['scientific_execution'] = False
    recipe['source_only'] = True
    recipe['future_commands'] = [[recipe['python'], '-B', '-s', str(NEW/'verify_retained.py'),
        '--authorization', 'ROOT_EXACT_AUTHORIZATION.json', '--authorization-sha256', 'ROOT_HASH',
        '--case', name, '--output', str(NEW/'attempts'/('independent-'+name+'001'))]
        for name in ('primary', 'radial_pair', 'angular_pair')]
    diagnostic_result = load(MASS/'attempts/diagnostic001/result.json')
    recipe['outer_measure_correction'] = {
        'sites': [{'accumulator': name, 'label': 'outer_measure:'+name,
                   'audit': 'tiny-outer-measure-'+name+'-rounding.json',
                   'shape': [64, 33] if name == 'loads' else [64, 64]}
                  for name in ('E','Ks','Kw','G','loads')],
        'numerical_expression_changes': 5,
        'each_instance': 'unchanged FastWeightingArithmetic with unchanged NormalizationArithmetic exact fallback',
        'helper': row(NEW/'tiny_normalization.py'), 'fast_proof': row(NEW/'fast_weighting.py'),
        'normal_zero_path': 'Original np.multiply on unchanged binary64 operands; zero signs and strict overflow/invalid behavior retained',
        'tiny_path': 'Only possible-tiny constituent products use exact Fraction binary64 nearest-even rounding; separate per-site counts and exact local error bound',
        'outside_scope': 'All contractions, sums, augmented additions, source maps, coefficients, quadrature, boundaries, solves, SVDs and thresholds unchanged',
        'no_occurrence_inference': 'Only E was diagnosed in the bounded radial window. Other four sites share the operation contract but their actual occurrence is unmeasured.',
        'diagnosed_E_counts': diagnostic_result['counts'],
        'evidence': {key: row(path) for key, path in evidence_paths.items()},
        'all3_fresh_v10_readbacks_required': True,
        'generator_admission': False,
        'local_error_not_matrix_error_certificate': True,
    }
    recipe['history_v10'] = {
        'v9_index': row(OLD/'source-index.json'), 'v9_source': row(OLD/'verify_retained.py'),
        'v9_primary_pass': row(OLD/'attempts/independent-primary001/receipt.json'),
        'v9_angular_pass': row(OLD/'attempts/independent-angular_pair001/receipt.json'),
        'v9_radial_FAIL': row(OLD/'attempts/independent-radial_pair001/receipt.json'),
        'v9_overall_generator_ineligible': True, 'v8_FAIL_immutable': True,
        'no_old_result_relabelled': True, 'all3_fresh_v10_readbacks_required': True,
    }
    recipe['runtime_inventory_addendum'] = {
        'manifest': row(runtime_path), 'files': len(runtime['files']), 'roots': runtime['roots'],
        'bytecode_excluded': True, 'all_binaries_metadata_only': True,
        'reason': 'Original saved readback uses scipy.linalg.blas.dgemm; bind the actual first-PYTHONPATH SciPy package/distribution metadata and any inventoried sibling library directory',
    }
    recipe['weighting_rounding_prerequisite']['scope'] = 'Unchanged exact96 helper units; unchanged exact20 fast-wrapper units separately guarded; exact24 column-norm units retained'
    recipe['remaining_limits'].extend([
        'Only E outer measure products were diagnosed; no actual occurrence claim for Ks/Kw/G/loads.',
        'Five local rounding audits do not certify propagated matrix, solve, finite-generator or continuum error.',
        'Historical weak_product_correction measure-unchanged wording is superseded only by five explicit v10 final products.',
        'v9 primary/angular successes do not substitute for any of the three fresh v10 readbacks.',
    ])
    for path in evidence_paths.values():
        pins[str(path)] = sha(path)
    for path in (OLD/'source-index.json', OLD/'recipe.json', OLD/'PLAN.md', OLD/'verify_retained.py'):
        pins[str(path)] = sha(path)
    recipe['pins'] = dict(sorted(pins.items()))
    write(NEW/'recipe.json', recipe)
    (NEW/'recipe-and-admission.diff').write_text(''.join(difflib.unified_diff(
        (OLD/'recipe.json').read_text().splitlines(True), (NEW/'recipe.json').read_text().splitlines(True),
        fromfile=str(OLD/'recipe.json'), tofile=str(NEW/'recipe.json'))))
    auth = load(OLD/'authorization-schema.json')
    auth.update(independent_saved_matrix_readback_authorized=False,
                readback_source_sha256=sha(NEW/'verify_retained.py'),
                recipe_sha256=sha(NEW/'recipe.json'), source_index_sha256='EXACT_ROOT_REQUIRED',
                scope='Saved-only v10 five final outer measure products; three fresh cases required; no generator/query/propagation')
    write(NEW/'authorization-schema.json', auth)
    (NEW/'PLAN.md').write_text('''# Held v10: five final radial-measure products

This is source preparation only. Do not run a verifier or import a scientific package until exact independent source review and root authorization. All three primary/radial/angular readbacks require fresh v10 attempts. The original v8 and v9 failures remain failures; their primary/angular passes do not qualify the radial case or a generator.

The admitted E-only diagnostic observed 28 inexact tiny products at radial index627 in the fixed609..640 window:20 round to signed zero and8 to nonzero subnormals. Its32 radii contain131072 products and one whole-array underflow. The independently audited compact saved evidence is a prerequisite, not a matrix acceptance. No occurrence is claimed for any other outer product.

Exactly five existing final products change: measure times the E mass sum, Kstrong sum, Kweak sum, Gvolume sum and33-field pointwise forcing load sum. The same unchanged FastWeightingArithmetic implementation is used in five distinct instances with labels outer_measure:E/Ks/Kw/G/loads. Each instance writes its own audit on success and failure. The old scalar measure=weights[ir]/c and every inner operand expression are unchanged. All augmented additions, sums, actions, bilinears, BLAS/einsum/reduction order, incoming/boundary work, trial, source, quadrature, SVD, solves and gates are unchanged.

The sufficient min-nonzero frexp proof and all-zero/empty paths call the original NumPy multiply on the original rounded operands. Only possible-tiny products delegate to the byte-exact, independently96-unit-tested exact nearest-even helper. The unchanged fast proof has its separately admitted20 units. The column-norm24-unit prerequisite remains. Overflow/invalid and all outside arithmetic stay strict. There is no blanket warning suppression, floor, clipping, changed threshold or fallback for sums, BLAS or SVD. Local rounding error audit bounds are not matrix/solve error certificates.

The new pre-import guard requires the completed E-only diagnostic, compact independent saved review and exact failed v9 radial receipt. The historical v9 verifier/recipe/index/plan and all original history remain protected. The reverse proof restores the entire original v9 source bytes and AST after removing the five products plus their explicitly listed bookkeeping/guard additions. The metadata wording of the weak audit is corrected to acknowledge the separate v10 final-measure adapters; its weak operand remains unchanged.

The actual first-PYTHONPATH SciPy package and distribution metadata are added through the completed root runtime inventory (1332 files, bytecode excluded, binaries metadata only). Existing producer dependencies and runtime pins remain protected before and after execution. Original scientific payloads are metadata-only during this preparation; none is decoded. The inherited source/normal/weak-strong/volume/forcing/rank/condition thresholds are unchanged.

Future execution is limited to three separately authorized saved-data readbacks. No PDE query, source compile, new operator assembly, generator eigenvalue, propagation, native or BH admission is granted here. A later generator stage can depend only on actual completed, unchanged-input successful results for all three fresh v10 cases and an independent provenance/results review.
''')
    verify(pins)
    write(PREP/'input-pins-after.json', pins)
    record = {
        'completed_source_preparation': True, 'inputs_unchanged': True, 'source_only': True,
        'candidate_imported': False, 'scientific_execution': False,
        'arrays_maps_JSONL_decoded': False, 'targets_recomputed': False,
        'source_inputs': len(pins), 'numeric_expression_changes': 5,
        'byte_exact_reverse': True, 'AST_exact_reverse': True,
        'fresh_destination': str(NEW), 'preparation_source': row(__file__),
        'seconds': time.monotonic()-started,
    }
    write(PREP/'receipt.json', record)
    write(NEW/'source-preparation.json', record)
    local_files = [row(path) for path in sorted(NEW.rglob('*')) if path.is_file()]
    write(NEW/'source-index.json', {
        'files': local_files, 'file_count': len(local_files), 'source_only': True,
        'scientific_execution': False, 'no_original_mutation': True,
        'scope': 'HELD v10 five outer measure products plus separate audits/admission/runtime metadata only',
    })
    write(PREP/'index.json', {'source_only': True, 'scientific_execution': False,
        'files': [row(path) for path in sorted(PREP.iterdir()) if path.is_file()],
        'prepared_source_index': row(NEW/'source-index.json')})
    print(json.dumps({'source_only': True, 'scientific_execution': False,
        'source_index': row(NEW/'source-index.json'), 'recipe': row(NEW/'recipe.json'),
        'verifier': row(NEW/'verify_retained.py'), 'receipt': row(PREP/'receipt.json')}))


if __name__ == '__main__':
    main()
