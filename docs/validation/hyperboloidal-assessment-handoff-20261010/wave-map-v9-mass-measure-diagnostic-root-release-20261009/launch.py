"""HELD root one-shot180-second process-group launcher; no scientific imports."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SRC = BASE/'boundary/reference-wave-map-v9-mass-measure-diagnostic-held-20261009'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1 << 20),b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    path = Path(path)
    # Larger hash-only runtime manifests are not scientific result payloads.
    limit = 16 << 20 if path.name in {'pins-prepared001.json','scipy-runtime-pins001.json'} else 1 << 20
    if path.suffix=='.jsonl' or path.stat().st_size>limit:
        raise RuntimeError('compact metadata JSON only')
    return json.loads(path.read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))


def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')


def verify(pins):
    for path,digest in pins.items():
        if sha(path) != digest:
            raise RuntimeError('changed protected input: '+path)


def check_runtime_membership(runtime):
    actual = {str(p.absolute()) for root in runtime['roots'] for p in Path(root).rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.pyo'}}
    if actual != set(runtime['files']):
        raise RuntimeError('SciPy runtime inventory membership changed')


def terminate_group(child,record):
    # This process owns the child's new session/group; no unrelated PID is targeted.
    actions = record.setdefault('process_group_cleanup',[])
    for sig in [signal.SIGTERM,signal.SIGKILL]:
        try:
            os.killpg(child.pid,sig)
            actions.append(signal.Signals(sig).name)
        except ProcessLookupError:
            actions.append(signal.Signals(sig).name+': group absent')
        if sig == signal.SIGTERM:
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                actions.append('SIGTERM grace expired')
    child.wait()


def main():
    out = HERE/'outer-invocation001'
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    record = dict(completed=False,returncode=1,child_started=False,root_cap_seconds=180,
        all_prepared_inputs_verified=False,
        original_v9_radial_readback_passed=False,scope='Bounded saved E product classification only')
    pins = {}
    child = None
    runtime = None
    try:
        if not (sys.flags.isolated and sys.dont_write_bytecode and sys.flags.optimize==0):
            raise RuntimeError('root launcher requires unoptimized -I -B')
        auth = load(HERE/'release.json')
        if auth.get('root_process_group_cap_seconds') != 180 or auth.get('bounded_saved_mass_diagnostic_authorized') is not True:
            raise RuntimeError('exact bounded release absent')
        for name,key in [('launcher-source-index.json','launcher_source_index_sha256'),
                         ('scipy-runtime-pins001.json','scipy_runtime_pins_sha256'),
                         ('pins-prepared001.json','pins_prepared_sha256')]:
            if sha(HERE/name) != auth[key]:
                raise RuntimeError('prepared root identity differs: '+name)
        pins = load(HERE/'pins-prepared001.json')
        for name in ['release.json','preparation-receipt001.json','pins-prepared001.json']:
            pins[str(HERE/name)] = sha(HERE/name)
        runtime = load(HERE/'scipy-runtime-pins001.json')
        check_runtime_membership(runtime)
        verify(pins)
        record['all_prepared_inputs_verified'] = True
        recipe = load(SRC/'recipe.json')
        for key in ['attempt','outer_attempt']:
            if Path(recipe[key]).exists():
                raise RuntimeError('fresh owner output required: '+recipe[key])
        preparation = load(HERE/'preparation-receipt001.json')
        if not (preparation['prepared'] is preparation['inputs_unchanged'] is True and
                preparation['science_executed'] is False and preparation['original_root_source_pins']==3283):
            raise RuntimeError('successful exact root preparation required')
        command = [recipe['python'],'-B','-s',str(SRC/'run_once.py'),
            '--authorization',str(HERE/'release.json'),'--authorization-sha256',sha(HERE/'release.json')]
        env = dict(os.environ)
        for key in ['PYTHONHOME','PYTHONWARNINGS']:
            env.pop(key,None)
        env.update(recipe['environment'])
        record.update(command=command,environment=recipe['environment'],
            unset_environment=['PYTHONHOME','PYTHONWARNINGS'],cwd=str(BASE.parent),
            start_new_session=True,protected_pins=len(pins),scipy_metadata_files=len(runtime['files']))
        write(out/'pins-before.json',pins)
        write(out/'command.json',{k:record[k] for k in ['command','environment','unset_environment','cwd','start_new_session','root_cap_seconds']})
        with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
            launched = time.monotonic()
            child = subprocess.Popen(command,cwd=BASE.parent,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            record.update(child_started=True,process_group_id=child.pid)
            try:
                child.wait(timeout=180)
            except subprocess.TimeoutExpired:
                record['root_cap_reached'] = True
                terminate_group(child,record)
                raise RuntimeError('root180-second process-group cap reached; preserve partial outputs')
            record['child_wall_seconds'] = time.monotonic()-launched
            record['actual_child_returncode'] = child.returncode
        if child.returncode != 0:
            raise RuntimeError('owner wrapper returned nonzero')
        child_receipt = load(Path(recipe['attempt'])/'receipt.json')
        wrapper_receipt = load(Path(recipe['outer_attempt'])/'receipt.json')
        for receipt in [child_receipt,wrapper_receipt]:
            if not (receipt.get('completed') is True and receipt.get('inputs_unchanged') is True and
                    type(receipt.get('returncode')) is int and receipt['returncode']==0):
                raise RuntimeError('owner diagnostic/wrapper incomplete or failed')
        result_path = Path(recipe['attempt'])/'result.json'
        result = load(result_path)
        if not (result['diagnostic_completed'] is True and result['counts']['multiply_components']==131072
                and len(result['radii'])==32 and result['E_accumulated'] is False
                and result['operator_matrices_loaded'] is False and result['source_tables_loaded'] is False
                and result['SVD_executed'] is False and result['query_or_generator_executed'] is False
                and result['original_v9_radial_readback_passed'] is False):
            raise RuntimeError('bounded diagnostic output scope/count differs')
        record.update(completed=True,returncode=0,
            child_receipt_sha256=sha(Path(recipe['attempt'])/'receipt.json'),
            wrapper_receipt_sha256=sha(Path(recipe['outer_attempt'])/'receipt.json'),
            result_sha256=sha(result_path),counts=result['counts'],
            first_strict_array_exception=result['first_strict_array_exception'],
            last_strict_array_exception=result['last_strict_array_exception'])
    except BaseException as error:
        if child is not None:
            terminate_group(child,record)
            record['actual_child_returncode'] = child.returncode
        record.update(completed=False,returncode=1,failure=type(error).__name__+': '+str(error))
        (out/'failure.txt').write_text(traceback.format_exc())
    finally:
        try:
            verify(pins)
            if runtime is not None:
                check_runtime_membership(runtime)
            after = {path:sha(path) for path in pins}
            write(out/'pins-after.json',after)
            record['inputs_unchanged'] = bool(pins) and after==pins
        except BaseException as error:
            record.update(inputs_unchanged=False,post_pin_failure=str(error),completed=False,returncode=1)
        record['seconds'] = time.monotonic()-start
        for name in ['stdout.log','stderr.log']:
            if (out/name).exists():
                record[name+'_sha256'] = sha(out/name)
                record[name+'_bytes'] = (out/name).stat().st_size
        write(out/'receipt.json',record)
    print(json.dumps(record,sort_keys=True),flush=True)
    if not (record['completed'] and record['inputs_unchanged'] and record['returncode']==0):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
