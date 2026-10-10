"""UNEXECUTED: three disjoint v10 cases, each with NEW900-second group cap."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


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
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify(pins):
    for path, digest in pins.items():
        if sha(path) != digest:
            raise RuntimeError('changed root source/runtime/review pin '+path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('case', choices=['primary','radial_pair','angular_pair'])
    ap.add_argument('--release-sha256', required=True)
    args = ap.parse_args()
    inv = HERE/(args.case+'-invocation001')
    inv.mkdir(exist_ok=False)
    started = time.monotonic()
    record = {'completed': False, 'returncode': None, 'case': args.case,
              'process_group_cap_seconds': 900, 'cap_is_new_not_historical': True,
              'cap_reached': False, 'generator_spectrum_or_propagation_admitted': False}
    pins = {str(Path(__file__).resolve()): sha(__file__)}
    try:
        if not (sys.flags.isolated and sys.dont_write_bytecode and sys.flags.optimize == 0):
            raise RuntimeError('root launcher requires -I -B and optimize0')
        if sha(HERE/'release.json') != args.release_sha256:
            raise RuntimeError('exact separately prepared release required')
        release = load(HERE/'release.json')
        suite = Path(release['suite'])
        recipe = load(suite/'recipe.json')
        auth = HERE/'authorization.json'
        pins.update(release['pins'])
        pins[str(HERE/'release.json')] = args.release_sha256
        pins[str(auth)] = release['authorization_sha256']
        pins[str(HERE/'review.json')] = release['review_sha256']
        preparation_path = HERE/'preparation001/receipt.json'
        pins[str(preparation_path)] = sha(preparation_path)
        if not (release['process_group_cap_seconds'] == 900 and release['cap_is_new_not_historical'] is True
                and args.case in release['cases']
                and sha(HERE/'launcher-source-index.json') == release['launcher_source_index_sha256']):
            raise RuntimeError('exact fresh900-second/case/source contract required')
        verify(pins)
        preparation = load(preparation_path)
        if not (preparation.get('completed') is True
                and preparation.get('inputs_unchanged') is True
                and preparation.get('returncode') == 0
                and preparation.get('release_sha256') == args.release_sha256
                and preparation.get('authorization_sha256') == release['authorization_sha256']):
            raise RuntimeError('successful one-shot release preparation required')
        output = suite/'attempts'/('independent-'+args.case+'001')
        if output.exists():
            raise RuntimeError('fresh disjoint fixed child attempt required')
        env = dict(os.environ)
        for key in ('PYTHONHOME','PYTHONWARNINGS','PYTHONSTARTUP'):
            env.pop(key, None)
        env.update(recipe['environment'])
        env.update(PYTHONOPTIMIZE='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
        command = [recipe['python'],'-I','-B',str(HERE/'child_review_gate.py'),args.case,
                   '--release-sha256',args.release_sha256]
        actual = [recipe['python'],'-B','-s',str(suite/'verify_retained.py'),
                  '--authorization',str(auth),'--authorization-sha256',release['authorization_sha256'],
                  '--case',args.case,'--output',str(output)]
        write(inv/'command.json', {'command': command, 'actual_after_stdlib_review_gate': actual,
            'environment': {**recipe['environment'],'PYTHONOPTIMIZE':'0','OMP_NUM_THREADS':'1',
                            'OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'},
            'cwd': str(REPO), 'new_process_group': True, 'process_group_cap_seconds': 900,
            'cap_is_new_not_historical': True, 'source_and_runtime_inputs_readonly': True})
        write(inv/'pins-before.json', pins)
        with (inv/'stdout.log').open('xb') as stdout, (inv/'stderr.log').open('xb') as stderr:
            proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=stdout, stderr=stderr,
                                    start_new_session=True)
            record['process_group_id'] = proc.pid
            child_started = time.monotonic()
            try:
                proc.wait(timeout=900)
            except subprocess.TimeoutExpired:
                record['cap_reached'] = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            record['child_wall_seconds'] = time.monotonic()-child_started
            record['returncode'] = proc.returncode
        child_receipt = output/'receipt.json'
        if child_receipt.exists():
            child = load(child_receipt)
            record['child_receipt_sha256'] = sha(child_receipt)
            record['child_completed'] = child.get('completed')
            record['child_inputs_unchanged'] = child.get('inputs_unchanged')
            record['child_passed'] = child.get('passed')
            record['completed'] = bool(not record['cap_reached'] and proc.returncode == 0
                and child.get('completed') is True and child.get('returncode') == 0
                and child.get('inputs_unchanged') is True and child.get('passed') is True)
        if (output/'result.json').exists():
            record['result_sha256'] = sha(output/'result.json')
        record['five_outer_measure_audit_files'] = {
            name: sha(output/('tiny-outer-measure-'+name+'-rounding.json'))
            if (output/('tiny-outer-measure-'+name+'-rounding.json')).exists() else None
            for name in ('E','Ks','Kw','G','loads')}
        if record['completed'] and not all(record['five_outer_measure_audit_files'].values()):
            raise RuntimeError('completed child must persist all five separate audits')
    except BaseException as exc:
        record.update(completed=False, error=type(exc).__name__+': '+str(exc))
        (inv/'failure.txt').write_text(traceback.format_exc())
    finally:
        try:
            verify(pins)
            record['source_runtime_review_pins_unchanged'] = True
        except BaseException as exc:
            record.update(completed=False, source_runtime_review_pins_unchanged=False, post_pin_failure=str(exc))
        write(inv/'pins-after.json', pins)
        record['seconds'] = time.monotonic()-started
        record['stdout_sha256'] = sha(inv/'stdout.log') if (inv/'stdout.log').exists() else None
        record['stderr_sha256'] = sha(inv/'stderr.log') if (inv/'stderr.log').exists() else None
        write(inv/'receipt.json', record)
    print(json.dumps(record))
    if not (record['completed'] and record['source_runtime_review_pins_unchanged']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
