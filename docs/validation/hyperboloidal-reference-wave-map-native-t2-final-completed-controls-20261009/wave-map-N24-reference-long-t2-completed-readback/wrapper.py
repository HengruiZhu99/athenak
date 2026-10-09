"""HELD single-use readback wrapper for nine completed t2 native controls.

All numerical gates reside in unchanged c1b948 analyze_snapshots.py and its
accepted native probe. A separate exact parent release is required to run.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NATIVE = ROOT / 'build-layer-research/reference-wave-map-native-held-20261009'
LONG = ROOT / 'build-layer-research/wave-map-native-t2-root-20261009'
PYTHON = '/Library/Developer/CommandLineTools/usr/bin/python3'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def checked(pins):
    for name, digest in pins.items(): assert sha(name) == digest, name

def main():
    release_path = Path(sys.argv[1]).resolve()
    name = sys.argv[2]
    release = load(release_path)
    assert release['completed_t2_snapshot_readback_authorized'] is True
    recipe_path = HERE / 'recipe.json'
    recipe = load(recipe_path)
    assert release['wrapper_sha256'] == sha(Path(__file__))
    assert release['recipe_sha256'] == sha(recipe_path)
    assert name in recipe['cases'] and name in release['cases']
    case_release = release['cases'][name]
    out = Path(case_release['attempt_directory']).resolve()
    assert out.parent == (HERE / 'attempts').resolve(), 'fresh attempt must be in this held wrapper tree'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'wrapper.py').write_bytes(Path(__file__).read_bytes())
    (out / 'release.json').write_bytes(release_path.read_bytes())
    (out / 'recipe.json').write_bytes(recipe_path.read_bytes())
    start = time.monotonic()
    protected = dict(recipe['fixed_pins'])
    protected[str(Path(__file__).resolve())] = release['wrapper_sha256']
    protected[str(recipe_path)] = release['recipe_sha256']
    protected[str(release_path)] = sha(release_path)
    receipt = {'case': name, 'wrapper_sha256': sha(Path(__file__)),
        'recipe_sha256': sha(recipe_path), 'root_release_sha256': sha(release_path),
        'launch_HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'scope': 'Completed t2 native binary64 readback only. No native step/operator or partial-result acceptance.'}
    command = []
    try:
        checked(protected)
        execution_release_path = LONG / 'release.json'
        execution_release = load(execution_release_path)
        assert len(execution_release['cases']) == 9
        specs = {x['name']: x for x in execution_release['cases']}
        spec = specs[name]
        assert spec == recipe['cases'][name]['execution_spec']
        assert execution_release['prerequisite_hashes'] == recipe['native_prerequisite_pins']
        checked(execution_release['prerequisite_hashes'])
        protected.update(execution_release['prerequisite_hashes'])
        launch_path = LONG / 'batch001' / name / 'launch-receipt.json'
        assert sha(launch_path) == case_release['launch_receipt_sha256']
        launch = load(launch_path)
        assert launch['passed_native_process_and_provenance'] is True
        assert launch['returncode'] == 0 and launch['error'] is None
        assert launch['mode'] == spec['mode'] and launch['input_path'] == spec['input_path']
        assert launch['native_execution_authorization_sha256'] == recipe['fixed_pins'][str(execution_release_path)]
        build_path = Path(spec['build_receipt'])
        build = load(build_path)
        assert build['passed_compile_link'] is True
        assert build['compiled_implementation'] == '27c19d20696ea6dd4704032c51dfd026218f64f2'
        assert launch['executable'] == build['executable'] and launch['executable_sha256'] == build['executable_sha256']
        assert launch['build_receipt'] == spec['build_receipt'] and launch['build_receipt_sha256'] == sha(build_path)
        assert launch['command'] == [build['executable'], '-i', spec['input_path']]
        assert Path(launch['cwd']).resolve() == launch_path.parent / 'output'
        assert Path(launch['output_directory']).resolve() == launch_path.parent / 'output'
        protected.update(build['source_before'])
        protected.update(build['all_compiler_dependency_sha256'])
        protected.update(spec['required_hashes'])
        protected[str(launch_path)] = case_release['launch_receipt_sha256']
        before = launch_path.with_name('protected-inputs-before.json')
        after = launch_path.with_name('protected-inputs-after.json')
        assert before.read_bytes() == after.read_bytes()
        protected.update(load(before))
        protected[str(before)] = sha(before)
        protected[str(after)] = sha(after)
        assert launch['outputs'], 'missing native output inventory'
        for relative, item in launch['outputs'].items():
            p = Path(launch['output_directory']) / relative
            assert p.stat().st_size == item['bytes']
            protected[str(p)] = item['sha256']
        protected[str(launch['run_log'])] = launch['run_log_sha256']
        protected[str(launch['stderr_path'])] = launch['stderr_sha256']
        checked(protected)
        auth = {'snapshot_readback_authorized': True,
            'analyzer_sha256': recipe['analyzer_sha256'],
            'launch_receipt': str(launch_path), 'launch_receipt_sha256': sha(launch_path),
            'output_directory': str(out / 'analysis'),
            'probe_executable': recipe['probe_executable'],
            'probe_executable_sha256': recipe['probe_executable_sha256'],
            'probe_recipe': recipe['probe_recipe'], 'probe_recipe_sha256': recipe['probe_recipe_sha256'],
            'reader_sha256': recipe['reader_sha256'], 'abi_sha256': recipe['abi_sha256'],
            'seam_receipt': recipe['seam_receipt'], 'seam_receipt_sha256': recipe['seam_receipt_sha256'],
            'completed_t2_wrapper_release_sha256': sha(release_path),
            'scope': 'Unchanged completed-process and exact-target acceptance gates; no partial diagnostic substitution.'}
        auth_path = out / 'authorization.json'
        dump(auth_path, auth)
        protected[str(auth_path)] = sha(auth_path)
        dump(out / 'protected-inputs-before.json', protected)
        command = [PYTHON, '-B', recipe['analyzer'], str(auth_path)]
        env = dict(os.environ)
        env['PYTHONPATH'] = str(ROOT / 'build-layer-research/boundary/python-deps')
        env['OPENBLAS_NUM_THREADS'] = '1'
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        receipt.update(command=command, cwd=str(ROOT), authorization_sha256=sha(auth_path),
            launch_receipt_sha256=sha(launch_path),
            environment_overrides={x: env[x] for x in ['PYTHONPATH', 'OPENBLAS_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE']})
        dump(out / 'launch-before.json', receipt)
        with (out / 'stdout').open('wb') as so, (out / 'stderr').open('wb') as se:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=so, stderr=se)
        receipt['returncode'] = result.returncode
        checked(protected)
        dump(out / 'protected-inputs-after.json', protected)
        assert result.returncode == 0, 'analyzer failure retained without gate relaxation'
        assert (out / 'stderr').read_bytes() == b''
        science = load(out / 'analysis/receipt.json')
        assert science['passed_saved_snapshot_finite_and_diagnostic_gates'] is True
        assert science['target_time'] == 2.0 and science['mode'] == spec['mode']
        receipt.update(passed_completed_t2_snapshot_gates=True,
            analyzer_receipt_sha256=sha(out / 'analysis/receipt.json'),
            protected_before_after_equal=True)
    except Exception as exc:
        receipt.update(passed_completed_t2_snapshot_gates=False, error=repr(exc), command=command)
    for log in ['stdout', 'stderr']:
        p = out / log
        if not p.exists(): p.write_bytes(b'')
        receipt[log + '_sha256'] = sha(p)
    receipt['seconds'] = time.monotonic() - start
    dump(out / 'receipt.json', receipt)
    print(json.dumps({'case': name, 'passed': receipt['passed_completed_t2_snapshot_gates'],
        'error': receipt.get('error'), 'receipt': str(out / 'receipt.json'),
        'seconds': receipt['seconds']}), flush=True)
    assert receipt['passed_completed_t2_snapshot_gates'] is True

if __name__ == '__main__':
    main()
