"""Stdlib one-shot outer logger; no science before exact independent/root release."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1 << 20),b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--authorization',type=Path,required=True)
    parser.add_argument('--authorization-sha256',required=True)
    args = parser.parse_args()
    recipe = load(HERE/'recipe.json')
    dest = Path(recipe['outer_attempt'])
    dest.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    command = [recipe['python'],'-B','-s',str(HERE/'diagnose_mass.py'),
        '--authorization',str(args.authorization.resolve()),'--authorization-sha256',args.authorization_sha256]
    env = dict(os.environ)
    for key in ['PYTHONHOME','PYTHONWARNINGS']:
        env.pop(key,None)
    env.update(recipe['environment'])
    record = dict(completed=False,returncode=1,child_started=False,command=command,
        environment=recipe['environment'],unset_environment=['PYTHONHOME','PYTHONWARNINGS'],
        original_v9_radial_readback_passed=False,scope=recipe['scope'])
    pins = {str(HERE/'recipe.json'):sha(HERE/'recipe.json'),str(HERE/'run_once.py'):sha(__file__)}
    try:
        if not (sys.flags.no_user_site and sys.dont_write_bytecode and not sys.flags.optimize):
            raise RuntimeError('require unoptimized -B -s outer runtime')
        if str(Path(sys.executable).resolve()) != recipe['resolved_python']:
            raise RuntimeError('outer interpreter differs')
        if sha(args.authorization) != args.authorization_sha256:
            raise RuntimeError('root authorization identity differs')
        auth = load(args.authorization)
        if not (auth.get('bounded_saved_mass_diagnostic_authorized') is True and
                auth.get('wrapper_source_sha256') == sha(__file__) and
                auth.get('recipe_sha256') == sha(HERE/'recipe.json') and
                auth.get('source_index_sha256') == sha(HERE/'source-index.json') and
                auth.get('only_selected_operands_no_E_accumulation_no_SVD_no_queries') is True and
                isinstance(auth.get('independent_review_receipt'),dict)):
            raise RuntimeError('exact one-shot bounded mass-product release absent')
        review_pin = auth['independent_review_receipt']
        if sha(review_pin['path']) != review_pin['sha256']:
            raise RuntimeError('independent review pin differs')
        review = load(review_pin['path'])
        if not (review.get('passed') is True and
                review.get('reviewed_source_index_sha256') == auth['source_index_sha256']):
            raise RuntimeError('exact independent source review absent')
        pins.update(load(HERE/'input-pins.json'))
        pins[str(args.authorization.resolve())] = args.authorization_sha256
        pins[review_pin['path']] = review_pin['sha256']
        pins[str(HERE/'source-index.json')] = auth['source_index_sha256']
        for row in load(HERE/'source-index.json')['files']:
            if row['path'] in pins and pins[row['path']] != row['sha256']:
                raise RuntimeError('conflicting source pin')
            pins[row['path']] = row['sha256']
        for path,digest in pins.items():
            if sha(path) != digest:
                raise RuntimeError('changed input before child: ' + path)
        write(dest/'pins-before.json',pins)
        record['child_started'] = True
        with (dest/'stdout.log').open('wb') as stdout,(dest/'stderr.log').open('wb') as stderr:
            child = subprocess.run(command,env=env,stdout=stdout,stderr=stderr,check=False,cwd=str(HERE))
        record['actual_child_returncode'] = child.returncode
        receipt = load(Path(recipe['attempt'])/'receipt.json')
        record.update(completed=(child.returncode == 0 and receipt.get('completed') is True and
            receipt.get('inputs_unchanged') is True and receipt.get('returncode') == 0),returncode=child.returncode)
        record['child_receipt_sha256'] = sha(Path(recipe['attempt'])/'receipt.json')
        record['stdout_sha256'] = sha(dest/'stdout.log')
        record['stderr_sha256'] = sha(dest/'stderr.log')
        if not record['completed']:
            raise RuntimeError('child failed or incomplete; preserve all outputs')
    except BaseException as error:
        record.update(returncode=1,failure=type(error).__name__ + ': ' + str(error))
        (dest/'failure.txt').write_text(traceback.format_exc())
    finally:
        try:
            for path,digest in pins.items():
                if sha(path) != digest:
                    raise RuntimeError('changed protected input: ' + path)
            record['inputs_unchanged'] = True
        except BaseException as error:
            record.update(inputs_unchanged=False,post_pin_failure=str(error),returncode=1)
        record['seconds'] = time.monotonic()-started
        write(dest/'receipt.json',record)
    print(json.dumps(record),flush=True)
    if not (record['completed'] and record['inputs_unchanged'] and record['returncode'] == 0):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
