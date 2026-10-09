"""Source/metadata/AST only. Never imports the supplement or evaluates targets."""
from pathlib import Path
import ast
import difflib
import hashlib
import json

HERE = Path(__file__).resolve().parent
CAP = HERE / 'inputs'
BASE = Path('/Users/hz0693/research/hyperboloidal/build-layer-research')
OWNER = BASE / 'boundary/inner-field-difference-arithmetic-supplement-v2-held-20261009'
V1 = BASE / 'boundary/inner-field-difference-arithmetic-supplement-held-20261009'
MAIN = BASE / 'boundary/inner-joint-nonlinear-helper-source003-held-20261009'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


checks = []


def require(value, label):
    if not value:
        raise RuntimeError(label)
    checks.append(label)


def text(name):
    return (CAP / name).read_text()


index = json.loads(text('source-index.json'))
require(sha(CAP / 'source-index.json') == 'e7b31a31bef5ca4dd1745f9c68c0ba41d0c24718b38365016ae20f7e328d851e', 'exact v2 owner index')
require(index['source_only'] is True and index['execution_admitted'] is False, 'source-only held status')
for r in index['files']:
    original = Path(r['path'])
    captured = CAP / original.relative_to(OWNER)
    require(original.stat().st_size == r['bytes'] and sha(original) == r['sha256'], 'unchanged owner ' + str(original.relative_to(OWNER)))
    require(captured.stat().st_size == r['bytes'] and sha(captured) == r['sha256'], 'exact capture ' + str(original.relative_to(OWNER)))
protected = json.loads(text('input-pins.json'))
require(len(protected) == len({r['path'] for r in protected}) == 2780, '2780 unique protected inputs')
for r in protected:
    original = Path(r['path'])
    # Historical payloads are rehashed only, never decoded.
    require(original.stat().st_size == r['bytes'] and sha(original) == r['sha256'], 'protected ' + r['path'])
for name in ('probe.cpp', 'oracle.py', 'negative-expression-bindings.json'):
    require((CAP / name).read_bytes() == (V1 / name).read_bytes(), 'exact v1 scientific source ' + name)
require((CAP / 'inputs/inner_gauge.hpp').read_bytes() == (MAIN / 'inner_gauge.hpp').read_bytes(), 'exact main003 full helper')

binding = json.loads(text('negative-expression-bindings.json'))
old = Path(binding['original_source002_helper']['path'])
require(sha(old) == binding['original_source002_helper']['sha256'], 'original002 negative-expression source pin')
for line in binding['exact_original_expression_lines']:
    require(line in old.read_text() and line in text('probe.cpp'), 'original002 expression line byte exact ' + line.strip())
start = 'inline double ProductValue('
def product_block(source, finish):
    return source[source.index(start):source.index(finish)]
original_product = product_block(old.read_text(), 'template<class T> T Coefficient(')
extracted_product = product_block(text('inputs/inner_gauge.hpp'), '// Exact [1/2,2]')
require(original_product == extracted_product, 'Product/ProductValue exact original002 bodies')
require(hashlib.sha256(original_product.encode()).hexdigest() == binding['unchanged_Product_and_ProductValue_body_sha256'], 'Product body digest binds exact block')

recipe = json.loads(text('recipe.json'))
old_recipe = json.loads(text('history/recipe.json'))
for key in ('expected_counts', 'relative_seeds', 'near_ratios', 'near_reference_exponents',
            'nonzero_normal_target_relative_threshold', 'zero_target_absolute_threshold'):
    require(recipe[key] == old_recipe[key], 'unchanged arithmetic recipe ' + key)
require(recipe['expected_counts'] == {'witness': 18, 'negative-old-near': 3, 'near-bound': 108, 'total': 129}, 'exact fixed129 metadata counts')
require(recipe['actual_main_release_PASS_required_before_compile_or_arithmetic'] is False
        and recipe['actual_main003_overall_passed'] is False
        and recipe['original_v1_ineligible_unchanged'] is True, 'old main FAIL and v1 ineligibility retained explicitly')
for name in ('oracle.py', 'run_once.py', 'gate_context.py', 'prepare_source.py'):
    ast.parse(text(name), filename=name)
    require(True, 'standard-library AST parse ' + name)

actual_diff = ''.join(difflib.unified_diff(text('history/gate_context.py').splitlines(True), text('gate_context.py').splitlines(True),
                                         fromfile='held-v1-full-main-PASS', tofile='held-v2-exact-main-FAIL-W1-qualification'))
require(actual_diff == text('admission-only.diff'), 'exact guard-only retained diff')
require(text('run_once.py') == text('history/run_once.py').replace('after actual-main PASS.', 'actual main FAIL preserved; helper units only.'), 'runner body unchanged apart from scope docstring')

# Only explicit source/include names are inspected; no compiler is invoked.
require(not (OWNER / 'inner_gauge.hpp').exists(), 'no same-name root inner header shadows inputs')
require((OWNER / 'inputs/inner_gauge.hpp').is_file(), 'probe inner header resolves in first source include directory')
require((OWNER / 'inputs/reference_wave_map.hpp').is_file(), 'quoted reference wrapper resolves to exact frozen56d61 input')
require(recipe['compile_flags']['release'][0] == '-I' + str(OWNER / 'inputs'), 'first include directory binds exact helper/support bytes')
oracle_ast = ast.parse(text('oracle.py'))
guard_ast = ast.parse(text('gate_context.py'))
runner_ast = ast.parse(text('run_once.py'))
require(not any(isinstance(n, ast.Assert) for tree in (oracle_ast, guard_ast, runner_ast) for n in ast.walk(tree)), 'scientific checks and admission do not depend on assert')

report = {'passed': True, 'source_only': True, 'owner_files': len(index['files']),
          'protected_inputs_rehashed': len(protected), 'protected_bytes': sum(r['bytes'] for r in protected),
          'historical_jsonl_hash_only': sum(Path(r['path']).suffix == '.jsonl' for r in protected),
          'checks': checks, 'checks_count': len(checks),
          'no_candidate_import_compile_query_or_target_recomputation': True}
(HERE / 'source-metadata-checks.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
print(json.dumps({k: report[k] for k in ('passed', 'owner_files', 'protected_inputs_rehashed', 'historical_jsonl_hash_only', 'checks_count')}))
