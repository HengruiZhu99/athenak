"""Single-use outer provenance/log wrapper for the eight released preflights.

The fixed analyzer owns all numerical formulas and thresholds. This wrapper
adds no scientific calculation and preserves failures before its inner try.
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
PREFLIGHT = ROOT / 'build-layer-research/wave-map-native-preflight-root-20261009'
REVIEW = ROOT / 'build-layer-research/boundary/reference-wave-map-native-seam-snapshot-independent-review-20261009/guard-revision002/receipt.json'
PYTHON = '/Library/Developer/CommandLineTools/usr/bin/python3'
PINS = {
    HERE / 'analyze_snapshots.py': 'c1b9487717e85e920b274b7dcb429290eed6b0d96b3b6cbd8e00d6ea352f7c72',
    HERE / 'analyzer-guard-review-index.json': 'defa29818d8f8c2ce95d32018ce7bc59989083c9f1429734548d10b8b93fb882',
    HERE / 'snapshot-authorization-schema-v2.json': '932165d3ef0e61d82b421956bb2d75fff61310099c03c2c5a3eb1e3a4ae45a04',
    HERE / 'probe-build-held/native-array-probe': '584e74bc257e7661310fff684af6d5ccf12c18dd24886a7e0ae9941c87161a54',
    HERE / 'probe-build-held/recipe.json': '83640cb37b7132eb32b026801e96331768aade94fce9f89ddfade39768f44932',
    HERE / 'probe-attempts/compile-and-seam-001/receipt.json': '54a7f3f955c294bcc45307986ec33f4d7a2fb5e10d6626c89df94a68061a2b73',
    ROOT / 'build-layer-research/time-projection-controls/rst-reader-gate/restart_reader.py': '74c62c97afd113819438d8be42be84cc605902983bd974022ebb7e0747f3cb13',
    ROOT / 'build-layer-research/time-projection-controls/rst-reader-gate/abi.json': '4e7599223e62ea4aa020b0efc691168a7ec2fcd6d83c38f6f45c4c6945d04cec',
    REVIEW: '8b237dab197852bcb4e4b057c69cbe92a76c5b012ad827b9aa1f3a6dd0202e78',
    PREFLIGHT / 'run_preflights.py': '253dc3840381f2bd413d562020f6640238dff9d0f39e33cc88ab9e9e298ac131',
    PREFLIGHT / 'release.json': 'dc5787c5b579b0137cb96cea3953cdb706f22ebbd94f5e2fd12c03c1fc54ff9f',
}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(p):
    return json.loads(Path(p).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')

def checked(pins):
    for name, digest in pins.items():
        assert sha(name) == digest, name

def main():
    name = sys.argv[1]
    release = load(PREFLIGHT / 'release.json')
    cases = {q['name']: q for q in release['cases']}
    assert name in cases, 'only the eight fixed parent-released preflights'
    out = HERE / 'snapshot-attempts' / (name + '-001')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'runner.py').write_bytes(Path(__file__).read_bytes())
    start = time.monotonic()
    protected = {str(p): digest for p, digest in PINS.items()}
    protected[str(Path(__file__).resolve())] = sha(Path(__file__))
    receipt = {'case': name, 'runner_sha256': sha(Path(__file__)),
        'launch_HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'scope': 'Released short/reference native-array readback only; no time advance, operators or long continuation.'}
    command = []
    try:
        checked(protected)
        launch_path = PREFLIGHT / 'batch001' / name / 'launch-receipt.json'
        launch = load(launch_path)
        assert launch['passed_native_process_and_provenance'] is True
        assert launch['returncode'] == 0 and launch['error'] is None
        assert launch['mode'] == cases[name]['mode']
        assert launch['input_path'] == cases[name]['input_path']
        assert launch['native_execution_authorization_sha256'] == PINS[PREFLIGHT / 'release.json']
        protected[str(launch_path)] = sha(launch_path)
        native_before = launch_path.with_name('protected-inputs-before.json')
        native_after = launch_path.with_name('protected-inputs-after.json')
        assert native_before.read_bytes() == native_after.read_bytes()
        protected.update(load(native_before))
        protected[str(native_before)] = sha(native_before)
        protected[str(native_after)] = sha(native_after)
        for relative, item in launch['outputs'].items():
            path = Path(launch['output_directory']) / relative
            assert path.stat().st_size == item['bytes']
            protected[str(path)] = item['sha256']
        protected[str(launch['run_log'])] = launch['run_log_sha256']
        protected[str(launch['stderr_path'])] = launch['stderr_sha256']
        checked(protected)
        auth = {'snapshot_readback_authorized': True,
            'analyzer_sha256': PINS[HERE / 'analyze_snapshots.py'],
            'launch_receipt': str(launch_path), 'launch_receipt_sha256': sha(launch_path),
            'output_directory': str(out / 'analysis'),
            'probe_executable': str(HERE / 'probe-build-held/native-array-probe'),
            'probe_executable_sha256': PINS[HERE / 'probe-build-held/native-array-probe'],
            'probe_recipe': str(HERE / 'probe-build-held/recipe.json'),
            'probe_recipe_sha256': PINS[HERE / 'probe-build-held/recipe.json'],
            'reader_sha256': PINS[ROOT / 'build-layer-research/time-projection-controls/rst-reader-gate/restart_reader.py'],
            'abi_sha256': PINS[ROOT / 'build-layer-research/time-projection-controls/rst-reader-gate/abi.json'],
            'seam_receipt': str(HERE / 'probe-attempts/compile-and-seam-001/receipt.json'),
            'seam_receipt_sha256': PINS[HERE / 'probe-attempts/compile-and-seam-001/receipt.json'],
            'parent_release': 'Root NEW_TASK/MESSAGE releases exact c1b948 analyzer, same probe and the eight completed short/reference cases; no long continuation.',
            'independent_guard_review_sha256': PINS[REVIEW]}
        auth_path = out / 'authorization.json'
        dump(auth_path, auth)
        protected[str(auth_path)] = sha(auth_path)
        dump(out / 'protected-inputs-before.json', protected)
        command = [PYTHON, str(HERE / 'analyze_snapshots.py'), str(auth_path)]
        env = dict(os.environ)
        env['PYTHONPATH'] = str(ROOT / 'build-layer-research/boundary/python-deps')
        env['OPENBLAS_NUM_THREADS'] = '1'
        receipt.update(command=command, cwd=str(ROOT),
            environment_overrides={x: env[x] for x in ['PYTHONPATH', 'OPENBLAS_NUM_THREADS']},
            authorization_sha256=sha(auth_path), launch_receipt_sha256=sha(launch_path))
        dump(out / 'launch-before.json', receipt)
        with (out / 'stdout').open('wb') as so, (out / 'stderr').open('wb') as se:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=so, stderr=se)
        receipt['returncode'] = result.returncode
        checked(protected)
        dump(out / 'protected-inputs-after.json', protected)
        assert result.returncode == 0, 'analyzer stopped; retained stdout/stderr and inner failure if available'
        assert (out / 'stderr').read_bytes() == b''
        science = load(out / 'analysis/receipt.json')
        assert science['passed_saved_snapshot_finite_and_diagnostic_gates'] is True
        receipt.update(passed_fixed_snapshot_readback=True,
            analyzer_receipt_sha256=sha(out / 'analysis/receipt.json'),
            protected_before_after_equal=True)
    except Exception as exc:
        receipt.update(passed_fixed_snapshot_readback=False, error=repr(exc), command=command)
    for log in ['stdout', 'stderr']:
        path = out / log
        if not path.exists():
            path.write_bytes(b'')
        receipt[log + '_sha256'] = sha(path)
    receipt['seconds'] = time.monotonic() - start
    dump(out / 'receipt.json', receipt)
    print(json.dumps({'case': name, 'passed': receipt['passed_fixed_snapshot_readback'],
        'error': receipt.get('error'), 'receipt': str(out / 'receipt.json'),
        'seconds': receipt['seconds']}), flush=True)
    assert receipt['passed_fixed_snapshot_readback'] is True

if __name__ == '__main__':
    main()
