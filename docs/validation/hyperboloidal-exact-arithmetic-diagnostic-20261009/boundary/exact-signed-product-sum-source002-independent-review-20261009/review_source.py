"""Independent stdlib-only source/metadata review. No candidate imports/targets."""
import ast
import collections
import hashlib
import json
from pathlib import Path
import shutil
import time

P = Path(__file__).resolve().parent
OWNER = P.parents[1] / 'continuum/exact-signed-product-sum-source002-held-20261009'
OLD = OWNER.with_name('exact-signed-product-sum-source001-held-20261009')
PRIOR = P.with_name('exact-signed-product-sum-source001-independent-review-20261009')
EXPECTED = '36c72f687bad50fa61c3f1ac684cf5b9b28cfa99348e71e627ea60bad5489a09'
OLD_INDEX = '0f21b208093a5fe042e7e8f02b300380ac0a88694bf3afee5bbee0b214e72460'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    assert Path(path).stat().st_size <= 1 << 20
    return json.loads(Path(path).read_text())


def pin(path):
    p = Path(path).absolute()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def main():
    started = time.monotonic()
    assert sha(OWNER / 'source-index.json') == EXPECTED
    assert sha(OLD / 'source-index.json') == OLD_INDEX
    index = load(OWNER / 'source-index.json')
    external = load(OWNER / 'external-pins.json')
    assert len(external) == index['external_unique_pins'] == 2400
    rows = index['files'] + external + [pin(OWNER / 'source-index.json')]
    rows += [pin(PRIOR / name) for name in ['index.json', 'receipt.json', 'REVIEW.md', 'review_source.py']]
    rows += [pin(P / name) for name in ['PLAN.md', 'REVIEW.md', 'review_source.py']]
    protected = {}
    for row in rows:
        assert row['path'] not in protected or protected[row['path']] == row
        protected[row['path']] = row
        assert Path(row['path']).stat().st_size == row['bytes']
        assert sha(row['path']) == row['sha256']
    write(P / 'inputs-before.json', list(protected.values()))

    prior = load(PRIOR / 'receipt.json')
    assert prior['arithmetic_source_math_review_passed'] is True
    assert prior['passed'] is False and prior['compiler_or_arithmetic_executed'] is False
    assert prior['reviewed_source_index_sha256'] == OLD_INDEX
    context = load(OWNER / 'source001-context.json')
    assert context['not_actual_compile_failure'] is True
    for row in context['original_files']:
        assert row['path'] in protected and protected[row['path']] == row

    proof = load(OWNER / 'driver-reverse-proof.json')
    old_runner = (OLD / 'run_gate.py').read_text()
    new_runner = (OWNER / 'run_gate.py').read_text()
    guard = proof['added_guard']
    assert new_runner.count(guard) == 1
    reversed_runner = new_runner.replace(guard, '')
    assert reversed_runner == old_runner
    assert ast.dump(ast.parse(reversed_runner), include_attributes=False) == ast.dump(ast.parse(old_runner), include_attributes=False)
    assert "cmd = [recipe['compiler']['path']] + recipe[mode + '_flags']" in new_runner
    assert "require(str(Path(recipe['compiler']['path']).resolve()) == recipe['compiler']['resolved_path']" in guard
    assert "sha(recipe['compiler']['path']) == recipe['compiler']['sha256']" in guard
    assert "sha(recipe['compiler']['resolved_path']) == recipe['compiler']['resolved_sha256']" in guard
    assert new_runner.index('exact root release absent') < new_runner.index(guard)
    assert new_runner.index(guard) < new_runner.index("run([recipe['compiler']['path'], '--version']")
    assert "review.get('reviewed_source_index_sha256') == sha(P / 'source-index.json')" in new_runner
    assert "receipt['inputs_unchanged'] = not drift" in new_runner

    recipe = load(OWNER / 'recipe.json')
    old_recipe = load(OLD / 'recipe.json')
    expected_recipe = json.loads(json.dumps(old_recipe))
    expected_recipe['compiler'].update(path='/Library/Developer/CommandLineTools/usr/bin/clang++',
        resolved_path='/Library/Developer/CommandLineTools/usr/bin/clang',
        resolved_sha256=old_recipe['compiler']['sha256'], resolved_bytes=old_recipe['compiler']['bytes'])
    additions = ['DRIVER-CORRECTION.md', 'driver-guard-only.diff', 'recipe-driver-only.diff',
                 'source001-context.json', 'driver-reverse-proof.json']
    expected_recipe['local_sources'] += additions
    assert recipe == expected_recipe
    compiler = recipe['compiler']
    literal_driver = Path(compiler['path'])
    assert str(literal_driver.resolve()) == compiler['resolved_path']
    assert sha(literal_driver) == compiler['sha256']
    assert sha(compiler['resolved_path']) == compiler['resolved_sha256']
    assert compiler['path'] in protected and compiler['resolved_path'] in protected
    assert compiler['sha256'] == compiler['resolved_sha256']

    unchanged = []
    for entry in proof['unchanged_files']:
        name = entry['name']
        assert (OWNER / name).read_bytes() == (OLD / name).read_bytes()
        assert sha(OWNER / name) == entry['sha256']
        unchanged.append(dict(name=name, sha256=entry['sha256']))
    assert len(unchanged) == 10
    for name in ['fraction_oracle.py']:
        assert ast.dump(ast.parse((OWNER / name).read_text()), include_attributes=False) == ast.dump(ast.parse((OLD / name).read_text()), include_attributes=False)
    registry = load(OWNER / 'registry.json')
    cases = registry['cases']
    counts = collections.Counter(row['mode'] for row in cases)
    assert len(cases) == registry['case_count'] == 70
    assert len({row['id'] for row in cases}) == 70 and counts == {'scalar': 44, 'dual': 26}
    lines = []
    for case in cases:
        lines.append('CASE %s %s %d %d' % (case['id'], case['mode'], len(case['terms']), int(case['null_input'])))
        for term in case['terms']:
            assert len(term['atoms']) == 4 and all(len(pair) == 2 for pair in term['atoms'])
            lines.append('TERM %d %d %d %s' % (term['sign'], term['shift'], term['arity'],
                         ' '.join(word for pair in term['atoms'] for word in pair)))
    assert '\n'.join(lines) + '\n' == (OWNER / 'cases.txt').read_text()
    by_id = {row['id']: row for row in cases}
    assert len(by_id['dual_max_generated_128']['terms']) == 32
    assert all(t['arity'] == 4 for t in by_id['dual_max_generated_128']['terms'])

    copies = P / 'source-copies'
    copies.mkdir(exist_ok=False)
    for row in index['files']:
        original = Path(row['path'])
        assert original.stat().st_size <= 1 << 20
        shutil.copyfile(original, copies / original.name)
    shutil.copyfile(OWNER / 'source-index.json', copies / 'source-index.json')
    write(P / 'exact-equality.json', dict(unchanged_files=unchanged,
        runner_reverse_bytes_equal=True, runner_reverse_AST_equal=True,
        oracle_AST_equal=True, recipe_only_driver_and_history_changes=True,
        registry_text_equal=True, compiler_invocation_path=compiler['path'],
        compiler_resolved_target=compiler['resolved_path']))
    after = []
    for row in protected.values():
        assert Path(row['path']).stat().st_size == row['bytes'] and sha(row['path']) == row['sha256']
        after.append(row)
    assert str(literal_driver.resolve()) == compiler['resolved_path']
    write(P / 'inputs-after.json', after)
    receipt = dict(passed=True, passed_source_review=True, arithmetic_source_math_review_passed=True,
        reviewed_source_index_sha256=EXPECTED, source_inputs_unchanged=True,
        protected_pins=len(protected), registry_cases=70, scalar_cases=44, dual_cases=26,
        original_source001_finding_preserved=True, source001_was_not_executed=True,
        literal_CXX_driver_and_resolved_target_bound=True, runner_reverse_bytes_equal=True,
        runner_reverse_AST_equal=True, scientific_sources_byte_equal=True,
        compiler_or_arithmetic_executed=False, candidate_imported=False,
        scope='source/math/admission readiness only; standalone submitted-atom primitive, no units or RWM acceptance',
        execution_release_required=True, seconds=time.monotonic() - started)
    write(P / 'receipt.json', receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
