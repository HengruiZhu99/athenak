"""Stdlib saved-report association only; no arithmetic target recomputation."""
import collections
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import time

P = Path(__file__).resolve().parent
B = P.parents[1]
OWNER = B / 'continuum/exact-signed-product-sum-source002-held-20261009'
ATTEMPT = OWNER / 'attempts/units001'
ROOT = B / 'exact-signed-product-sum-root-release-20261009'
OUTER = ROOT / 'units-invocation001'
INDEX = '36c72f687bad50fa61c3f1ac684cf5b9b28cfa99348e71e627ea60bad5489a09'
CHILD = 'daf60646128b2a89d9f3abc7776fc73fc980924adcc13a8c7703ff0d5e864b2d'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def pin(path):
    path = Path(path).absolute()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def load(path):
    path = Path(path)
    assert path.stat().st_size <= 1 << 20
    assert path.name not in ['probe-release.stdout', 'probe-debug.stdout']
    return json.loads(path.read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def main():
    start = time.monotonic()
    assert sha(OWNER / 'source-index.json') == INDEX
    assert sha(ATTEMPT / 'receipt.json') == CHILD
    si = load(OWNER / 'source-index.json')
    ext = load(OWNER / 'external-pins.json')
    rows = si['files'] + ext + [pin(OWNER / 'source-index.json')]
    root_pins = load(OUTER / 'pins-before.json')
    assert isinstance(root_pins, dict)
    rows += [dict(path=path, bytes=Path(path).stat().st_size, sha256=digest) for path,digest in root_pins.items()]
    for folder in [ATTEMPT, ROOT]:
        rows += [pin(path) for path in folder.rglob('*') if path.is_file()]
    rows += [pin(P / name) for name in ['read_saved.py', 'PLAN.md']]
    protected = {}
    for row in rows:
        assert row['path'] not in protected or protected[row['path']] == row
        protected[row['path']] = row
        assert Path(row['path']).stat().st_size == row['bytes'] and sha(row['path']) == row['sha256']
    write(P / 'inputs-before.json', list(protected.values()))
    child = load(ATTEMPT / 'receipt.json')
    outer = load(OUTER / 'receipt.json')
    assert child['passed'] is child['completed'] is child['inputs_unchanged'] is True
    assert child['returncode'] == 0 and child['input_drift'] == []
    assert outer['passed'] is outer['completed'] is outer['inputs_unchanged'] is True
    assert outer['returncode'] == 0 and outer['child_receipt_sha256'] == CHILD
    assert child['no_RWM_adoption_or_saved_failure_upgrade'] is True
    assert outer['no_RWM_adoption_or_failure_upgrade'] is True
    assert child['python_flags'] == dict(optimize=0, isolated=1, dont_write_bytecode=True)
    assert child['source_index'] == pin(OWNER / 'source-index.json')
    recipe = load(OWNER / 'recipe.json')
    assert child['recipe'] == pin(OWNER / 'recipe.json')
    for name in recipe['local_sources'] + ['recipe.json', 'source-index.json', 'external-pins.json']:
        assert sha(ATTEMPT / name) == sha(OWNER / name)
    commands = {row['name']: row for row in child['commands']}
    assert len(commands) == 9 and len(child['commands']) == 9
    assert set(commands) == {'launch-HEAD', 'compiler-version', 'python-version',
        'compile-release', 'probe-release', 'oracle-release', 'compile-debug', 'probe-debug', 'oracle-debug'}
    for row in commands.values():
        assert row['returncode'] == 0
        for field in ['stdout', 'stderr']:
            assert pin(row[field]['path']) == row[field]
        assert row['stderr']['bytes'] == 0
        if row['input'] is not None:
            assert pin(row['input']['path']) == row['input']
    assert commands['compiler-version']['command'] == [recipe['compiler']['path'], '--version']
    assert recipe['compiler']['path'] == '/Library/Developer/CommandLineTools/usr/bin/clang++'
    assert str(Path(recipe['compiler']['path']).resolve()) == recipe['compiler']['resolved_path']
    registry = load(OWNER / 'registry.json')
    cases = registry['cases']
    assert len(cases) == 70 and collections.Counter(c['mode'] for c in cases) == {'scalar':44, 'dual':26}
    results = {}
    dependency_counts = {}
    omitted = []
    for mode in ['release', 'debug']:
        depfile = ATTEMPT / ('probe-' + mode + '.d')
        expected_compile = [recipe['compiler']['path']] + recipe[mode + '_flags'] + [
            '-MD', '-MF', str(depfile), str(ATTEMPT / 'probe.cpp'), '-o', str(ATTEMPT / ('probe-' + mode))]
        assert commands['compile-' + mode]['command'] == expected_compile
        assert commands['compile-' + mode]['stdout']['bytes'] == 0
        assert commands['probe-' + mode]['command'] == [str(ATTEMPT / ('probe-' + mode))]
        assert commands['probe-' + mode]['input'] == pin(ATTEMPT / 'cases.txt')
        expected_oracle = [recipe['python']['path'], '-I', '-B', str(ATTEMPT / 'fraction_oracle.py'),
            '--registry', str(ATTEMPT / 'registry.json'), '--output', str(ATTEMPT / ('probe-' + mode + '.stdout')),
            '--report', str(ATTEMPT / ('oracle-' + mode + '.json'))]
        assert commands['oracle-' + mode]['command'] == expected_oracle
        dependencies = load(ATTEMPT / ('dependencies-' + mode + '.json'))
        words = shlex.split(depfile.read_text().replace('\\\n', ' ').split(':',1)[1])
        dep_paths = [str((Path(word) if Path(word).is_absolute() else B.parent / word).resolve()) for word in words]
        assert dep_paths == [row['path'] for row in dependencies]
        for row in dependencies:
            assert pin(row['path']) == row
            assert row['path'] in protected
        assert str((ATTEMPT / 'signed_products.hpp').resolve()) in dep_paths
        dependency_counts[mode] = len(dependencies)
        assert child[mode + '_dependencies'] == pin(ATTEMPT / ('dependencies-' + mode + '.json'))
        assert child[mode + '_executable'] == pin(ATTEMPT / ('probe-' + mode))
        report = load(ATTEMPT / ('oracle-' + mode + '.json'))
        assert child[mode + '_oracle_report'] == pin(ATTEMPT / ('oracle-' + mode + '.json'))
        assert report['passed'] is report['hand_tie_controls'] is report['no_gauge_or_PDE_acceptance'] is True
        assert report['fixed_cases'] == 70 and report['scalar_cases'] == 44 and report['dual_cases'] == 26
        assert report['registry_sha256'] == sha(OWNER / 'registry.json')
        assert report['output_sha256'] == sha(ATTEMPT / ('probe-' + mode + '.stdout'))
        assert [row['id'] for row in report['cases']] == [case['id'] for case in cases]
        hand_bits = hand_statuses = invalid = generated = 0
        statuses = collections.Counter()
        for case, row in zip(cases, report['cases']):
            if row['validation'] != 'ok':
                assert case['hand_status'] == row['validation']
                invalid += 1
                statuses['validation:' + row['validation']] += 1
                continue
            fields = [row['result']] if case['mode'] == 'scalar' else [row['primal'], row['tangent']]
            statuses.update(field['status'] for field in fields)
            if 'hand_bits' in case:
                actual = fields[0].get('bits') if case['mode'] == 'scalar' else [field.get('bits') for field in fields]
                assert actual == case['hand_bits']
                hand_bits += 1
            if 'hand_status' in case:
                actual = fields[0]['status'] if case['mode'] == 'scalar' else [field['status'] for field in fields]
                assert actual == case['hand_status']
                hand_statuses += 1
            if case['mode'] == 'dual':
                assert row['generated_tangent_terms'] == sum(term['arity'] for term in case['terms'])
                generated += row['generated_tangent_terms']
        results[mode] = dict(cases=70, hand_bits_cases=hand_bits, valid_hand_status_cases=hand_statuses,
            invalid_status_cases=invalid, generated_terms_sum=generated, statuses=dict(statuses))
    assert results['release'] == results['debug']
    assert child['release_debug_probe_byte_equal'] is True
    assert sha(ATTEMPT / 'probe-release.stdout') == sha(ATTEMPT / 'probe-debug.stdout')
    assert (ATTEMPT / 'probe-release.stdout').stat().st_size == (ATTEMPT / 'probe-debug.stdout').stat().st_size
    assert sha(ATTEMPT / 'oracle-release.json') == sha(ATTEMPT / 'oracle-debug.json')
    for path in ATTEMPT.rglob('*'):
        if path.is_file() and (path.name in ['probe-release', 'probe-debug', 'probe-release.stdout', 'probe-debug.stdout']
            or '.dSYM' in str(path) or path.stat().st_size > 1 << 20):
            omitted.append(dict(**pin(path), reason='compiled or scientific JSONL payload; metadata only'))
    capture = P / 'compact-copies'
    capture.mkdir(exist_ok=False)
    for path, name in [(ATTEMPT / 'receipt.json', 'child-receipt.json'),
        (ATTEMPT / 'oracle-release.json', 'oracle-release-debug-identical.json'),
        (OUTER / 'receipt.json', 'root-receipt.json'), (OUTER / 'stdout.log','root-stdout.log'),
        (OUTER / 'stderr.log','root-stderr.log')]:
        shutil.copyfile(path, capture / name)
    write(P / 'metadata-only-payloads.json', omitted)
    for row in protected.values():
        assert pin(row['path']) == row
    write(P / 'inputs-after.json', list(protected.values()))
    summary = dict(actual_Release_Debug_passed=True, fixed_cases=70, scalar_cases=44, dual_cases=26,
        reports=results, dependency_counts=dependency_counts, all9_recorded_commands_successful=True,
        all_stderr_and_compile_stdout_empty=True, Release_Debug_stdout_equal_hash=True,
        probe_stdout_not_decoded_or_copied=True, oracle_reports_identical=True,
        release_executable=child['release_executable'], debug_executable=child['debug_executable'],
        actual_child_receipt=pin(ATTEMPT / 'receipt.json'), actual_root_receipt=pin(OUTER / 'receipt.json'),
        exact_source_index=pin(OWNER / 'source-index.json'), no_target_recomputed=True,
        scope='Bounded submitted-atom CPU arithmetic only; no metric inverse, RWM, gauge or PDE qualification')
    write(P / 'summary.json', summary)
    receipt = dict(passed_saved_readback=True, source_inputs_unchanged=True, protected_pins=len(protected),
        summary=pin(P / 'summary.json'), compiler_or_probe_executed=False, target_recomputed=False,
        seconds=time.monotonic() - start)
    write(P / 'receipt.json', receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
