"""Single-use fixed native preflights; no scientific acceptance is inferred."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NATIVE = ROOT / 'build-layer-research/reference-wave-map-native-held-20261009'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(p):
    return json.loads(Path(p).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def dump(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def checked(pins):
    for name, digest in pins.items():
        assert sha(name) == digest, name


def run_case(spec, release, parent):
    case = parent / spec['name']
    case.mkdir(exist_ok=False)
    out = case / 'output'
    out.mkdir(exist_ok=False)
    build_path = Path(spec['build_receipt'])
    build = load(build_path)
    assert build['passed_compile_link'] and not build['native_executed']
    assert build['mode'] == spec['mode']
    assert build['compiled_implementation'] == '27c19d20696ea6dd4704032c51dfd026218f64f2'
    pins = dict(build['source_before'])
    pins.update(build['all_compiler_dependency_sha256'])
    pins.update(spec['required_hashes'])
    pins.update(release['prerequisite_hashes'])
    pins[str(HERE / 'release.json')] = sha(HERE / 'release.json')
    pins[str(Path(__file__).resolve())] = release['launcher_sha256']
    dump(case / 'protected-inputs-before.json', pins)
    command = [build['executable'], '-i', spec['input_path']]
    launch = {'mode': spec['mode'], 'input_path': spec['input_path'],
              'input_sha256': spec['required_hashes'][spec['input_path']],
              'executable': build['executable'],
              'executable_sha256': build['executable_sha256'],
              'build_receipt': str(build_path), 'build_receipt_sha256': sha(build_path),
              'command': command, 'cwd': str(out), 'output_directory': str(out),
              'launch_HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'native_execution_authorization_sha256': sha(HERE / 'release.json'),
              'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'Native output generation only; snapshot admission and stability verdict remain separate.'}
    dump(case / 'launch-before.json', launch)
    start = time.monotonic()
    error = None
    code = None
    try:
        checked(pins)
        with (case / 'native.stdout').open('wb') as so, (case / 'native.stderr').open('wb') as se:
            process = subprocess.run(command, cwd=out, stdout=so, stderr=se, timeout=600)
            code = process.returncode
        checked(pins)
        dump(case / 'protected-inputs-after.json', pins)
    except Exception as exc:
        error = repr(exc)
    for name in ['native.stdout', 'native.stderr']:
        if not (case / name).exists():
            (case / name).write_bytes(b'')
    outputs = {str(p.relative_to(out)): {'sha256': sha(p), 'bytes': p.stat().st_size}
               for p in out.rglob('*') if p.is_file()}
    launch.update(returncode=code, error=error, seconds=time.monotonic() - start,
                  run_log=str(case / 'native.stdout'), run_log_sha256=sha(case / 'native.stdout'),
                  stderr_path=str(case / 'native.stderr'), stderr_sha256=sha(case / 'native.stderr'),
                  sources_before_after_equal=error is None, outputs=outputs,
                  passed_native_process_and_provenance=code == 0 and error is None)
    dump(case / 'launch-receipt.json', launch)
    print(json.dumps({'case': spec['name'], 'returncode': code, 'error': error,
                      'seconds': launch['seconds'], 'outputs': len(outputs)}), flush=True)
    return launch


def main():
    release_path = HERE / 'release.json'
    release = load(release_path)
    assert release['native_preflight_output_generation_authorized'] is True
    assert sha(Path(__file__)) == release['launcher_sha256']
    checked(release['prerequisite_hashes'])
    seam = load(NATIVE / 'probe-attempts/compile-and-seam-001/receipt.json')
    assert seam['passed_compile_and_fixed_t0_seam'] and seam['row_count'] == 18
    assert seam['max_scaled_error'] <= 2e-12 and seam['reference_rhs_abs'] <= 1e-10
    parent = HERE / 'batch001'
    parent.mkdir(exist_ok=False)
    dump(parent / 'release.json', release)
    (parent / 'run_preflights.py').write_bytes(Path(__file__).read_bytes())
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_case, spec, release, parent) for spec in release['cases']]
        results = [f.result() for f in futures]
    passed = all(x['passed_native_process_and_provenance'] for x in results)
    dump(parent / 'receipt.json', {'passed_native_processes_and_provenance': passed,
        'native_cases': len(results), 'launcher_sha256': release['launcher_sha256'],
        'release_sha256': sha(release_path),
        'cases': {spec['name']: sha(parent/spec['name']/'launch-receipt.json') for spec in release['cases']},
        'scope': 'Completion/provenance only. No snapshot diagnostic, convergence or stability acceptance.'})
    assert passed


if __name__ == '__main__':
    main()
