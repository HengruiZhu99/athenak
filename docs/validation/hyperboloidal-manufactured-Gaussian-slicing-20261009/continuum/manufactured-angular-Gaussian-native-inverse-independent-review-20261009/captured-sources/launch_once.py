"""HELD one-shot stdlib lifecycle; no scientific imports."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda s:
                      (_ for _ in ()).throw(ValueError(s)))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--authorization', type=Path, required=True)
    ap.add_argument('--authorization-sha256', required=True)
    args = ap.parse_args()
    out = HERE/'outer-invocation001'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    receipt = dict(completed=False, returncode=None, accepted_consistency_execution=False,
                   sampled_positivity_accepted=False, scope='Finite native-time manufactured inverse/value screen only')
    pins = {str(Path(__file__).resolve()): sha(__file__)}
    before = None
    try:
        if sys.flags.optimize != 0:
            raise RuntimeError('optimized outer runtime rejected')
        if sha(args.authorization) != args.authorization_sha256:
            raise RuntimeError('authorization digest differs')
        auth = load(args.authorization)
        recipe, index = load(HERE/'recipe.json'), load(HERE/'source-index.json')
        if not (auth.get('native_Gaussian_inverse_screen_authorized') is True
                and auth.get('source_index_sha256') == sha(HERE/'source-index.json')
                and auth.get('recipe_sha256') == sha(HERE/'recipe.json')
                and auth.get('screen_source_sha256') == sha(HERE/'screen.py')
                and auth.get('outer_source_sha256') == sha(__file__)):
            raise RuntimeError('exact source/math/index/outer authorization required')
        pins.update(recipe['pins'])
        for row in index['files']:
            pins[row['path']] = row['sha256']
        pins[str(HERE/'source-index.json')] = sha(HERE/'source-index.json')
        pins[str(args.authorization.resolve())] = args.authorization_sha256
        before = {path: sha(path) for path in pins}
        save(out/'pins-before.json', before)
        if before != pins:
            raise RuntimeError('pre-launch pinned input differs')
        if (HERE/'attempt001').exists():
            raise RuntimeError('one-shot child destination already exists')
        environment = os.environ.copy()
        environment.update(recipe['environment'])
        command = [recipe['python_runtime_path'], '-I', '-B', str(HERE/'screen.py'),
                   '--authorization', str(args.authorization.resolve()),
                   '--authorization-sha256', args.authorization_sha256]
        receipt.update(command=command, environment=recipe['environment'], cwd=str(HERE))
        save(out/'invocation.json', receipt)
        with (out/'stdout.log').open('x') as stdout, (out/'stderr.log').open('x') as stderr:
            child = subprocess.run(command, cwd=HERE, env=environment,
                                   stdout=stdout, stderr=stderr, check=False)
        receipt['returncode'] = child.returncode
        child_receipt = load(HERE/'attempt001/receipt.json')
        receipt['child_receipt_sha256'] = sha(HERE/'attempt001/receipt.json')
        accepted = (child.returncode == 0 and child_receipt.get('completed') is True
                    and child_receipt.get('returncode') == 0
                    and child_receipt.get('inputs_unchanged') is True
                    and child_receipt.get('checks_passed') is True)
        receipt.update(completed=accepted, accepted_consistency_execution=accepted,
                       all_sampled_D_positive=child_receipt.get('all_sampled_D_positive'),
                       sampled_positivity_accepted=False)
    except BaseException as exc:
        receipt['failure'] = repr(exc)
        (out/'failure.txt').write_text(traceback.format_exc())
    finally:
        try:
            after = {path: sha(path) for path in pins}
            save(out/'pins-after.json', after)
            receipt['inputs_unchanged'] = bool(before and before == after)
        except BaseException as exc:
            receipt.update(inputs_unchanged=False, post_pin_failure=repr(exc))
        if not receipt.get('inputs_unchanged'):
            receipt.update(completed=False, accepted_consistency_execution=False)
        receipt['seconds'] = time.monotonic()-started
        outputs = []
        for directory in (out, HERE/'attempt001'):
            if directory.exists():
                for path in sorted(directory.rglob('*')):
                    if path.is_file() and path != out/'receipt.json':
                        outputs.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path),
                                            large_payload=path.suffix == '.jsonl' or path.stat().st_size > 1048576))
        receipt['output_pins'] = outputs
        save(out/'receipt.json', receipt)
    print(json.dumps(receipt), flush=True)
    return 0 if receipt['accepted_consistency_execution'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
