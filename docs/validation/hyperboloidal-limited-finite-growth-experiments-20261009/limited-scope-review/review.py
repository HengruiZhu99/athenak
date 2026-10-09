"""Static-only diff/scope review; never import analyzer or load a matrix."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
P = HERE.parent
ROOT = P.parents[2]
OLD = ROOT/'build-layer-research/continuum/finite-rb-growth-control-20261009/immutable-growth-helper-review-synthetic-20261009'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rec(path):
    return {'path': str(path.resolve()), 'sha256': sha(path), 'bytes': path.stat().st_size}


source = P/'analyze_growth.py'
scope = P/'LIMITED-MATRIX-SCOPE.md'
assert sha(source) == '7634bf135c20df9588f2e6cc33dd434d6e63925e6716586445443033dc7224e3'
assert sha(scope) == '96f489d8ee95dd7931499d19d2e6c8e4cb89712848852303cb951f245674ed4f'
assert sha(OLD/'analyze_growth.py') == '85e6b721ad143a73955811ce546afed78ac3ba88c3d8419348cd6e108f2aecda'
assert sha(OLD/'index.json') == '0d753253f577211bfab2c56dc6eed3bcaeada81810636a4fbb63d9d6e69839cb'
for filename in ('PLAN.md', 'FINAL-ADDENDUM.md'):
    assert (P/filename).read_bytes() == (OLD/filename).read_bytes()
preparation = json.loads((P/'source-preparation.json').read_text())
for filename, digest in preparation['files'].items():
    assert sha(P/filename) == digest
old_tree = ast.parse((OLD/'analyze_growth.py').read_text())
new_tree = ast.parse(source.read_text())
old_functions = {n.name: n for n in old_tree.body if isinstance(n, ast.FunctionDef)}
new_functions = {n.name: n for n in new_tree.body if isinstance(n, ast.FunctionDef)}
assert old_functions.keys() == new_functions.keys()
for name in old_functions:
    if name != 'analyze':
        assert ast.dump(old_functions[name], include_attributes=False) == ast.dump(new_functions[name], include_attributes=False), name
old_analyze = old_functions['analyze']
new_analyze = new_functions['analyze']
old_try = [n for n in old_analyze.body if isinstance(n, ast.Try)]
new_try = [n for n in new_analyze.body if isinstance(n, ast.Try)]
assert len(old_try) == len(new_try) == 1
assert ast.dump(old_try[0], include_attributes=False) == ast.dump(new_try[0], include_attributes=False)
old_tail = old_analyze.body[old_analyze.body.index(old_try[0])+1:]
new_tail = new_analyze.body[new_analyze.body.index(new_try[0])+1:]
assert [ast.dump(n, include_attributes=False) for n in old_tail] == [ast.dump(n, include_attributes=False) for n in new_tail]
diff = ''.join(difflib.unified_diff((OLD/'analyze_growth.py').read_text().splitlines(True),
                                  source.read_text().splitlines(True),
                                  fromfile=str(OLD/'analyze_growth.py'), tofile=str(source)))
(HERE/'source.diff').write_text(diff)
expected_before_try_changes = (
    'limited_finite_matrix_growth_admitted', 'analytic_projected_points',
    'unresolved_nongauge_continuum_comparator_recorded', 'LIMITED-MATRIX-SCOPE.md',
    'original_full_projection_defect_gate_passed',
    'general_nongauge_continuum_comparator_unresolved', 'both_original_FD_attempts_remain_failed')
for token in expected_before_try_changes:
    assert token in source.read_text()
metadata = next(n for n in new_analyze.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'receipt' for t in n.targets))
flags = {k.value: v.value for k, v in zip(metadata.value.keys, metadata.value.values)
         if isinstance(k, ast.Constant) and isinstance(v, ast.Constant)}
assert flags['original_full_projection_defect_gate_passed'] is False
assert flags['general_nongauge_continuum_comparator_unresolved'] is True
assert flags['both_original_FD_attempts_remain_failed'] is True
for token in ('complete nongauge comparator', 'remain failures', 'not admission',
              'complete', 'manufactured forcing-family', 'analytic projected',
              'J0 first at N8, rb=.98', 'wormhole-to-'):
    assert token in scope.read_text(), token
copies = HERE/'reviewed-source'
assert not copies.exists()
copies.mkdir()
for filename in ('analyze_growth.py', 'PLAN.md', 'FINAL-ADDENDUM.md', 'LIMITED-MATRIX-SCOPE.md', 'source-preparation.json'):
    shutil.copyfile(P/filename, copies/filename)
    assert sha(copies/filename) == sha(P/filename)
receipt = {
    'status': 'PASS_read_only_limited_source_scope_review_conditional_on_pending_admission',
    'reviewer': '/root/literature_gauge', 'review_source': rec(Path(__file__)),
    'input_pins': [rec(OLD/'index.json'), rec(OLD/'analyze_growth.py')]+[
        rec(P/name) for name in ('analyze_growth.py', 'PLAN.md', 'FINAL-ADDENDUM.md',
                                 'LIMITED-MATRIX-SCOPE.md', 'source-preparation.json')],
    'diff': rec(HERE/'source.diff'),
    'unchanged_numerical_functions': [name for name in old_functions if name != 'analyze'],
    'analyze_numerical_try_block_and_postprocessing_AST_identical': True,
    'source_changes': 'Only module description, preexecution admission keys and mandatory scope pin, and explicit unresolved/failure receipt flags differ from reviewed parent; numerical try block, exception/partial-payload handling and final checks are identical.',
    'findings': [
        'A finite source-bound matrix with positive E defines an algebraic ODE independently of whether its polynomial action converges to the PDE. Studying that named ODE is meaningful with the missing nongauge continuum comparator stated; it cannot validate a PDE or reclassify either failed FD gate.',
        'The narrowed scope preserves the original full-protocol hold and requires analytic projected-point evidence as a distinct gate, plus complete per-channel forcing-family coverage. The old PLAN/addendum are copied historical context, explicitly superseded only in admission scope.',
        'Correct Cholesky similarity z=L^T X, symmetric SAT+SAT.T energy work, complete finite spectrum residuals, finite exponential guards/consistency and bounded short SSPRK3 formulas are unchanged from the reviewed and synthetic-checked parent.',
        'No nonnormal forward eigenvalue certification, PDE energy transfer or uniform-in-degree/radius estimate follows. A positive finite growth result can reject the named numerical control under its declared criterion; a bounded finite result accepts none of the continuum/native/BH goals.',
        'Fixed physical envelopes, interpolation error and incoming boundary mismatch remain explicit. Separate analytic physical constraint readback of retained seeds/modes/Jv/time states is still required before interpreting constraint content.'
    ],
    'remaining_admission_conditions': [
        'Root must bind the successful analytic projected-point receipt and complete forcing-family review, together with every other specified source/mass/rule/angular/sector/rate gate in the new authorization.',
        'Both failed ordinary-FD attempts remain failed and must be retained; full nongauge C_ref[L_actual Phi] remains unresolved in every result.',
        'The first release, if separately authorized, is J0,N8,rb=.98 only. Other J, N and rb cases require separate authorization and evidence.'
    ],
    'corrections': [], 'actual_scientific_matrix_loaded': False,
    'analyzer_imported_or_executed': False, 'generator_eigensolve_or_propagation': False,
    'scope': 'Source/scope checkpoint only; this review itself does not authorize scientific execution or mark pending point controls passed.'}
path = HERE/'receipt.json'
assert not path.exists()
path.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
print(json.dumps({'status': receipt['status'], 'receipt': rec(path)}, indent=2))
