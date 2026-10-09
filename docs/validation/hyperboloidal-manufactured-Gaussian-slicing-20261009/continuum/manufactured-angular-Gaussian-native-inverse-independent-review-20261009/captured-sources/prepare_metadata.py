"""Source/recipe preparation only; stdlib AST/hash/Fraction operations."""
from pathlib import Path
from fractions import Fraction
import ast
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    if (HERE/'source-index.json').exists() or (HERE/'recipe.json').exists():
        raise RuntimeError('one-shot metadata preparation refuses existing index/recipe')
    old_screen = ROOT/'build-layer-research/manufactured-angular-Gaussian-screen-v2-held-20261009'
    old_height = ROOT/'build-layer-research/continuum/native-angular-pulse-flat-derivatives-compact-full-held-20261009'
    compact = ROOT/'build-layer-research/continuum/manufactured-angular-time-wave-compact-ADM-pencil-20261009'
    prior_review = ROOT/'build-layer-research/continuum/manufactured-angular-Gaussian-screen-v2-independent-guard-review-20261009'
    source = json.loads((old_screen/'recipe.json').read_text())
    height = json.loads((old_height/'derivative-recipe.json').read_text())
    assert sha(old_screen/'source-index.json') == '287f96ced076617c877d4bfbd7d5d63e484dd8b2d2eb61a53c96e1e2f60fbf7b'
    assert sha(old_screen/'screen.py') == '6c9cae79f014d30a81717d05320c865ff406415afd2c0bf249b3571737636c8d'
    assert sha(old_height/'values_context.py') == '89c96ee729b8cc741757740534566c9557dd6fdce2e66213c2fe8ff1c7e8ffe7'
    assert (HERE/'values_context.py').read_bytes() == (old_height/'values_context.py').read_bytes()
    old_text = (old_screen/'screen.py').read_text()
    new_text = (HERE/'gaussian_radial.py').read_text()
    old_run = next(n for n in ast.parse(old_text).body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    new_factory = next(n for n in ast.parse(new_text).body if isinstance(n, ast.FunctionDef) and n.name == 'functions')
    old_lines, new_lines = old_text.splitlines(True), new_text.splitlines(True)
    for name in ('derivatives', 'radial'):
        old_def = next(n for n in old_run.body if isinstance(n, ast.FunctionDef) and n.name == name)
        new_def = next(n for n in new_factory.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ''.join(old_lines[old_def.lineno-1:old_def.end_lineno]) == ''.join(new_lines[new_def.lineno-1:new_def.end_lineno])
    pins = dict(source['pins'])
    for path in sorted(old_screen.iterdir()):
        if path.is_file():
            pins[str(path)] = sha(path)
    for name in ('values_context.py', 'derivative-recipe.json', 'source-index.json', 'PLAN.md'):
        pins[str(old_height/name)] = sha(old_height/name)
    for path in sorted(compact.iterdir()):
        if path.is_file():
            pins[str(path)] = sha(path)
    for name in ('index.json', 'receipt.json'):
        pins[str(prior_review/name)] = sha(prior_review/name)
    for path, expected in pins.items():
        assert sha(path) == expected
    labels = ['0', '.025', '.05', '.1', '.2', '.3', '.45', '.5', '.65', '.75',
              '.84', '.85', '.9', '.95', '.98', '.995',
              '1-1e-3', '1-1e-6', '1-1e-9', '1-1e-12', '1-1e-18']
    radii = []
    for label in labels:
        value = 1-Fraction(label[2:]) if label.startswith('1-') else Fraction(label)
        radii.append(dict(label=label, value=str(value)))
    assert len(radii) == len({row['value'] for row in radii}) == 21
    times = ['0', '.05', '.1', '.2', '.5', '1', '2', '4', '6']
    p_values = ['-1/2', '-3/8', '-1/4', '-1/8', '0', '1/8', '1/4', '3/8', '1/2']
    levels = [dict(digits=digits, height_order=order, series_terms=40 if digits == 80 else 60)
              for digits in (80, 110) for order in (128, 256)]
    layer = {name: height[name] for name in
             ('a', 'geometry_r0', 'geometry_r1', 'height_panels',
              'root_tolerance', 'maximum_root_iterations')}
    records = len(radii)*len(times)*len(p_values)*len(source['sigma'])*len(source['epsilon'])
    assert records == 13608
    recipe = dict(
        status='HELD source-only finite native-time Gaussian inverse/value screen',
        execution_admitted=False, a='1/2', S='1', sigma=source['sigma'], epsilon=source['epsilon'],
        native_radii=radii, native_times=times, p_values=p_values, angular_s='1',
        levels=levels, records_per_level=records, total_records=records*len(levels),
        comparison_pairs=[['precision', '80-128', '110-128'],
                          ['precision', '80-256', '110-256'],
                          ['height', '80-128', '80-256'],
                          ['height', '110-128', '110-256']],
        comparison_tolerance='1e-30', root_absolute_tolerance='1e-50',
        root_width_tolerance='1e-55', newton_iterations=16, bisection_iterations=512,
        layer=layer, pins=pins, python_runtime_path=source['python_runtime_path'],
        python_runtime_sha256=source['python_runtime_sha256'],
        mpmath_parent=source['mpmath_parent'], mpmath_init=source['mpmath_init'],
        environment=source['environment'],
        all_negative_D_events_preserved=True, positivity_required_for_consistency_pass=False,
        no_jets_kernel_queries_PDE_evolution_or_spectrum=True,
        source_extracts_byte_identical=True, runtime_cost_measured=False,
        scope='Finite declared native-time manufactured Gaussian inverse/ADM-value consistency screen; no global timelike, jets, source/kernel, evolution, Scri or BH adoption.',
        large_payload_policy='samples.jsonl and every NPZ/NPY/binary/object/executable/archive or file>1MiB are metadata-only for later compact collection.',
        expected_authorization_fields=['native_Gaussian_inverse_screen_authorized', 'source_index_sha256',
                                       'recipe_sha256', 'screen_source_sha256', 'outer_source_sha256'])
    save(HERE/'recipe.json', recipe)
    save(HERE/'AUTHORIZATION-SCHEMA.json', dict(
        native_Gaussian_inverse_screen_authorized=False,
        source_index_sha256='ROOT_MUST_BIND_EXACT_DIGEST',
        recipe_sha256='ROOT_MUST_BIND_EXACT_DIGEST',
        screen_source_sha256='ROOT_MUST_BIND_EXACT_DIGEST',
        outer_source_sha256='ROOT_MUST_BIND_EXACT_DIGEST'))
    syntax = []
    for path in sorted(HERE.glob('*.py')):
        ast.parse(path.read_text(), filename=str(path))
        syntax.append(path.name)
    save(HERE/'source-preparation.json', dict(
        source_only=True, scientific_execution=False, numerical_or_CAS_imports=False,
        preparation_command=[sys.executable, '-I', '-B', str(Path(__file__).resolve())],
        preparation_operations='Standard-library AST/hash/JSON/Fraction metadata only.',
        syntax_AST_only=syntax, height_copy_byte_equal=True,
        Gaussian_definitions_byte_equal=True, upstream_pins_rehashed=len(pins),
        fixed_events_per_level=records, fixed_total_events=records*len(levels),
        original_inputs_unchanged=True,
        root_solver_and_compact_ADM_math_not_numerically_tested=True))
    rows = [dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
            for path in sorted(HERE.iterdir()) if path.is_file() and path.name != 'source-index.json']
    save(HERE/'source-index.json', dict(source_only=True, execution_admitted=False,
                                      files=rows, file_count=len(rows), upstream_pin_count=len(pins),
                                      original_inputs_unchanged=True))
    print(json.dumps(dict(source_index_sha256=sha(HERE/'source-index.json'),
                          recipe_sha256=sha(HERE/'recipe.json'),
                          screen_sha256=sha(HERE/'screen.py'),
                          outer_sha256=sha(HERE/'launch_once.py'),
                          files=len(rows), upstream_pins=len(pins),
                          records_per_level=records, total_records=records*len(levels)), sort_keys=True))


if __name__ == '__main__':
    main()
