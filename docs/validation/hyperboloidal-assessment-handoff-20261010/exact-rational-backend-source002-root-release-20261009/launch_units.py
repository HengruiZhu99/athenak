"""UNEXECUTED one-shot full120s process-group lifecycle for exact82 CPU units."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent


def require(value,message):
    if not value:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def pin(path):
    path=Path(path).resolve();return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def load(path):
    def reject(value):raise ValueError('nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(),parse_constant=reject)


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def check(entry):
    require(pin(entry['path'])=={key:entry[key] for key in ('path','bytes','sha256')},'pin drift: '+entry['path'])


def main():
    output=HERE/'units-invocation001';output.mkdir(exist_ok=False)
    record=dict(passed=False,completed=False,returncode=None,inputs_unchanged=False,process_group_cap_seconds=120,
                no_native_RWM_or_gauge_adoption=True)
    protected=[];process=None;owner=None;started=time.monotonic()
    try:
        ap=argparse.ArgumentParser();ap.add_argument('--release-sha256',required=True);args=ap.parse_args()
        require(sys.flags.isolated==1 and sys.dont_write_bytecode and sys.flags.optimize==0
                and os.environ.get('PYTHONOPTIMIZE')=='0','root launch requires -I -B optimize0')
        require(sha(HERE/'units-release.json')==args.release_sha256,'actual exact release digest required')
        release=load(HERE/'units-release.json');owner=Path(release['owner']).resolve()
        protected=release['protected_pins']+[pin(HERE/'units-release.json')]
        for item in protected:check(item)
        recipe=load(owner/'recipe.json');auth=load(release['authorization']['path'])
        require(sha(owner/'source-index.json')==release['source_index_sha256']
                and sha(owner/'recipe.json')==release['recipe_sha256'],'candidate source/recipe binding')
        require(release['fixed_cases']==82 and release['root_process_group_cap_seconds']==120
                and release['per_command_cap_seconds']==60,'fixed execution scope/cap')
        require(not Path(release['child_attempt']).exists(),'child attempt already exists')
        root_review=load(release['root_source_review']['path'])
        require(root_review.get('passed') is True and root_review.get('root_source_math_reviewed') is True,
                'actual root source/math review required')
        env=dict(os.environ)
        for key in recipe['sanitized_environment']:env.pop(key,None)
        env.update(recipe['environment'])
        command=[recipe['python']['path'],'-I','-B',str(owner/'run_gate.py'),
                 '--recipe',str(owner/'recipe.json'),'--authorization',release['authorization']['path']]
        write(output/'pins-before.json',protected)
        write(output/'invocation.json',dict(argv=command,environment=recipe['environment'],
            sanitized_environment=recipe['sanitized_environment'],cwd=str(HERE.parents[1]),
            exact_release=pin(HERE/'units-release.json'),process_group_cap_seconds=120))
        with (output/'stdout.log').open('xb') as stdout,(output/'stderr.log').open('xb') as stderr:
            process=subprocess.Popen(command,cwd=HERE.parents[1],env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            record['child_pid']=process.pid
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                record['process_group_cap_exceeded']=True
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            record['returncode']=process.returncode
        child=Path(release['child_attempt'])/'receipt.json'
        require(child.is_file(),'child has no actual receipt')
        receipt=load(child);record['child_receipt']=pin(child)
        require(record['returncode']==0 and not record.get('process_group_cap_exceeded',False)
                and receipt.get('completed') is True and receipt.get('passed') is True and receipt.get('returncode')==0
                and receipt.get('inputs_unchanged') is True and receipt.get('fixed_cases')==82
                and receipt.get('both_dependency_closures_passed') is True
                and receipt.get('release_debug_probe_byte_equal') is True,
                'actual child82-unit acceptance failed')
        require(receipt['source_index']['sha256']==release['source_index_sha256']
                and receipt['recipe']['sha256']==release['recipe_sha256']
                and receipt['authorization']['sha256']==release['authorization']['sha256'],'child source/auth binding')
        for item in receipt['outputs']:check(item)
        for item in protected:check(item)
        record.update(passed=True,completed=True)
    except BaseException as exc:
        record.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        # Always clear this unique child session's process group, including
        # any orphan compiler children after a failed60s individual command.
        if process is not None:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            if process.poll() is None:process.wait()
            record['actual_child_returncode']=process.returncode
        drift=[]
        for item in protected:
            try:check(item)
            except BaseException as exc:drift.append(dict(path=item['path'],error=repr(exc)))
        record['inputs_unchanged']=bool(protected) and not drift;record['post_pin_failures']=drift
        if not record['inputs_unchanged']:record.update(passed=False,completed=False)
        record['seconds']=time.monotonic()-started
        folders=[output]+([owner/'attempts/units001'] if owner is not None else [])
        record['output_inventory']=[pin(path) for folder in folders if folder.exists() for path in sorted(folder.rglob('*'))
                                    if path.is_file() and path!=output/'receipt.json']
        write(output/'receipt.json',record)
    print(json.dumps({key:value for key,value in record.items() if key!='output_inventory'},sort_keys=True))
    return 0 if record['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
