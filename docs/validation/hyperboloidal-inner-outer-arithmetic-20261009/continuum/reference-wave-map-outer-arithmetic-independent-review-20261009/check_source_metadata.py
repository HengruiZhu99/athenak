"""Independent source/metadata/AST checks only; never import the candidate."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
CAP = HERE / 'inputs'
OWNER = Path('/Users/hz0693/research/hyperboloidal/build-layer-research/boundary/reference-wave-map-outer-arithmetic-source001-held-20261009')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def require(value, label):
    if not value:
        raise RuntimeError(label)
    checks.append(label)


def text(name):
    return (CAP / name).read_text()


def top_functions(source):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


checks = []
index = json.loads(text('source-index.json'))
require(sha(CAP / 'source-index.json') == '9da21a96d243441c95f276d1c076d3a4e561fc20921858be33f911cb3464f27e', 'exact owner source index')
require(index['source_only'] is True and index['execution_admitted'] is False, 'owner explicitly source-only and held')
for item in index['files']:
    original = Path(item['path'])
    captured = CAP / original.relative_to(OWNER)
    require(original.stat().st_size == item['bytes'] and sha(original) == item['sha256'], 'unchanged owner ' + str(original.relative_to(OWNER)))
    require(captured.stat().st_size == item['bytes'] and sha(captured) == item['sha256'], 'exact captured ' + str(original.relative_to(OWNER)))

protected = json.loads(text('input-pins.json'))
require(len(protected) == 2689 and len({x['path'] for x in protected}) == 2689, '2689 unique protected inputs')
for row in protected:
    path = Path(row['path'])
    # Rehash only: even historical JSONL is never parsed or copied here.
    require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'protected pin ' + row['path'])

legacy = text('inputs/reference_wave_map_legacy.hpp')
old_rwm = text('history/reference_wave_map.hpp')
require(legacy.replace('hyp::GaugeRHSParts<T> LegacyGauge(', 'hyp::GaugeRHSParts<T> Gauge(', 1) == old_rwm,
        'legacy exact old header after single name reversal')
old_inner, new_inner = text('history/inner_gauge.hpp'), text('inner_gauge.hpp')
start = old_inner.index('// A scalar adapter is explicit:')
end = old_inner.index('template<class T> T Coefficient(')
block = old_inner[start:end]
require(block in text('inputs/arithmetic_traits.hpp'), 'exact source003 complete arithmetic block extraction')
require(old_inner[:start] + old_inner[end:] == new_inner, 'only arithmetic extraction changes inner header')
require(old_inner[end:] == new_inner[new_inner.index('template<class T> T Coefficient('):], 'entire Coefficient and Gauge suffix exact source003')
require(text('source-template/reference_wave_map.hpp') == text('inputs/reference_wave_map.hpp'), 'wrapper exact source-template bytes')
require(text('run_once.py') == text('history/run_once.py').replace("auth.get('local_nonlinear_helper_execution_admitted')", "auth.get('outer_arithmetic_local_execution_admitted')"),
        'runner sole authorization key change')

recipe, old_recipe = json.loads(text('recipe.json')), json.loads(text('history/recipe.json'))
for key in ('expected_record_counts', 'thresholds', 'modes', 'raw22_order', 'gauge_raw22_indices',
            'geometry_raw22_indices', 'precision', 'high_contrast_families'):
    require(recipe[key] == old_recipe[key], 'unchanged recipe ' + key)
require(sum(recipe['expected_record_counts'].values()) == 15740, '15740 fixed records')

old_oracle, new_oracle = text('history/oracle.py'), text('oracle.py')
functions_old, functions_new = top_functions(old_oracle), top_functions(new_oracle)
for key in functions_old.keys() - {'main'}:
    require(functions_old[key] == functions_new[key], 'unchanged top-level oracle function ' + key)
old_main = next(n for n in ast.parse(old_oracle).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
new_main = next(n for n in ast.parse(new_oracle).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
old_nested = {n.name: ast.dump(n, include_attributes=False) for n in old_main.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
new_nested = {n.name: ast.dump(n, include_attributes=False) for n in new_main.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
require(old_nested == new_nested, 'all oracle MP algebra/AD/target/core/error nested functions exact AST')
for name, old, new in [('probe-baseline-contract.diff', text('history/probe.cpp'), text('probe.cpp')),
                       ('oracle-explicit-outer-contract.diff', old_oracle, new_oracle),
                       ('legacy-name-only.diff', old_rwm, legacy),
                       ('arithmetic-extraction.diff', old_inner, new_inner),
                       ('runner-admission-only.diff', text('history/run_once.py'), text('run_once.py'))]:
    actual = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='frozen-source003', tofile='fresh-outer-source001'))
    require(actual == text(name), 'exact retained diff ' + name)

old_probe, new_probe = text('history/probe.cpp'), text('probe.cpp')
for begin, finish in [('hyp::Z4cJet<double> State(', 'template<class T>void EmitGauge('),
                      ('void MainGrid(', 'void Duals('), ('void Duals(', 'void Coefficients('),
                      ('void Coefficients(', 'void CoreWitness('), ('void CoreWitness(', 'void Invalid('),
                      ('void Invalid(', 'void Nonrepresentable('), ('void Nonrepresentable(', '\n}\nint main(')]:
    require(old_probe[old_probe.index(begin):old_probe.index(finish)] == new_probe[new_probe.index(begin):new_probe.index(finish)],
            'unchanged probe registry/body ' + begin)

# Resolve the relevant quoted includes in exactly compiler search order.
repo = Path(recipe['repository'])
include_dirs = []
flags = recipe['compile_flags']['release']
i = 0
while i < len(flags):
    flag = flags[i]
    if flag.startswith('-I') and len(flag) > 2:
        d = Path(flag[2:]); include_dirs.append(d if d.is_absolute() else repo / d)
    elif flag == '-isystem':
        i += 1; d = Path(flags[i]); include_dirs.append(d if d.is_absolute() else repo / d)
    i += 1


def resolve(parent, name):
    for directory in [parent.parent] + include_dirs:
        proposed = directory / name
        if proposed.is_file():
            return proposed.resolve()
    raise RuntimeError('Unresolved quoted include ' + name)


closure = []
for parent, name, target in [
    (OWNER / 'probe.cpp', 'test_support.hpp', OWNER / 'inputs/test_support.hpp'),
    (OWNER / 'probe.cpp', 'inner_gauge.hpp', OWNER / 'inner_gauge.hpp'),
    (OWNER / 'inputs/test_support.hpp', 'reference_wave_map.hpp', OWNER / 'inputs/reference_wave_map.hpp'),
    (OWNER / 'inner_gauge.hpp', 'reference_wave_map.hpp', OWNER / 'inputs/reference_wave_map.hpp'),
    (OWNER / 'inputs/reference_wave_map.hpp', 'arithmetic_traits.hpp', OWNER / 'inputs/arithmetic_traits.hpp'),
    (OWNER / 'inputs/reference_wave_map.hpp', 'reference_wave_map_legacy.hpp', OWNER / 'inputs/reference_wave_map_legacy.hpp'),
    (OWNER / 'inputs/reference_wave_map_legacy.hpp', 'z4c/hyperboloidal/layer_gauge.hpp', repo / 'src/z4c/hyperboloidal/layer_gauge.hpp')]:
    actual = resolve(parent, name)
    require(actual == target.resolve(), 'quoted include closure ' + str(parent.relative_to(OWNER)) + ' -> ' + name)
    closure.append({'parent': str(parent), 'include': name, 'resolved': str(actual), 'sha256': sha(actual)})

for name in ('oracle.py', 'run_once.py', 'prepare_source.py'):
    ast.parse(text(name), filename=name)
    require(True, 'standard-library AST parse ' + name)

report = {
    'passed': True, 'source_only': True, 'source_index_sha256': sha(CAP / 'source-index.json'),
    'owner_indexed_files': len(index['files']), 'protected_inputs_rehashed': len(protected),
    'protected_total_bytes': sum(x['bytes'] for x in protected),
    'historical_jsonl_hash_only': sum(Path(x['path']).suffix == '.jsonl' for x in protected),
    'checks_count': len(checks), 'checks': checks, 'quoted_include_closure': closure,
    'candidate_import_compile_query_or_scientific_payload_decode': False,
    'reviewer_delegation': 'Attempted separate source-only admission reviewer; agent thread limit prevented delegation. All checks performed by this reviewer.'}
(HERE / 'source-metadata-checks.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
print(json.dumps({k: report[k] for k in ('passed', 'owner_indexed_files', 'protected_inputs_rehashed', 'checks_count', 'historical_jsonl_hash_only')}, sort_keys=True))
