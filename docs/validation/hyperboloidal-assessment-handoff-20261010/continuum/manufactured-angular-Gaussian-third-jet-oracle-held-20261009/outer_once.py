"""Held standard-library one-shot outer; no candidate import in this file."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            h.update(block)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))

def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def snapshot(pins):
    result={}
    for p in pins:
        try:result[p]=sha(p)
        except OSError as e:result[p]={'error':str(e)}
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--authorization',required=True)
    parser.add_argument('--authorization-sha256',required=True)
    parser.add_argument('--invocation',required=True)
    args=parser.parse_args()
    invocation=Path(args.invocation).resolve()
    invocation.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();pins={};before={};child_output=None;returncode=None
    result={'completed':False,'accepted_stage':False,'scientific_imports_in_outer':False}
    try:
        if sys.flags.optimize!=0 or sys.flags.isolated!=1 or sys.flags.dont_write_bytecode!=1:
            raise RuntimeError('exact unoptimized -I -B outer route required')
        if sha(args.authorization)!=args.authorization_sha256:
            raise RuntimeError('exact authorization hash mismatch')
        auth=load(args.authorization);recipe=load(HERE/'recipe.json');index=load(HERE/'source-index.json')
        stage=auth.get('Gaussian_third_jet_oracle_stage_authorized')
        if stage not in ('units','timing','full'):raise RuntimeError('root stage release missing')
        if Path(auth['outer_output']).resolve()!=invocation:raise RuntimeError('exact outer path not admitted')
        child_output=Path(auth['output']).resolve()
        if child_output.exists() or child_output==invocation:raise RuntimeError('child output must be fresh and separate')
        expected={'source_index_sha256':sha(HERE/'source-index.json'),'recipe_sha256':sha(HERE/'recipe.json'),'driver_sha256':sha(HERE/'run_oracle.py')}
        if any(auth.get(k)!=v for k,v in expected.items()):raise RuntimeError('source/index/recipe release binding')
        pins={**recipe['protected_inputs'],**index['files'],**auth['review_pins']}
        pins[str(HERE/'source-index.json')]=expected['source_index_sha256']
        pins[str(Path(args.authorization).resolve())]=args.authorization_sha256
        for name in ('unit_receipt','timing_receipt'):
            if name in auth:pins[auth[name]['path']]=auth[name]['sha256']
        before=snapshot(pins)
        if before!=pins:raise RuntimeError('before pin mismatch')
        save(invocation/'pins-before.json',before)
        env=os.environ.copy()
        for name in ('PYTHONHOME','PYTHONPATH','PYTHONWARNINGS','PYTHONSTARTUP','PYTHONUSERBASE'):
            env.pop(name,None)
        env.update(recipe['environment'])
        command=[recipe['python_runtime_path'],'-I','-B',str(HERE/'run_oracle.py'),'--stage',stage,
                 '--recipe',str(HERE/'recipe.json'),'--authorization',str(Path(args.authorization).resolve()),
                 '--authorization-sha256',args.authorization_sha256,'--output',str(child_output)]
        save(invocation/'invocation.json',{'command':command,'cwd':str(HERE),'environment':recipe['environment'],
             'removed_environment':['PYTHONHOME','PYTHONPATH','PYTHONWARNINGS','PYTHONSTARTUP','PYTHONUSERBASE'],
             'timeout_seconds':recipe['outer_seconds'][stage]})
        with (invocation/'stdout.log').open('wb') as stdout,(invocation/'stderr.log').open('wb') as stderr:
            process=subprocess.Popen(command,cwd=str(HERE),env=env,stdout=stdout,stderr=stderr)
            try:returncode=process.wait(timeout=recipe['outer_seconds'][stage])
            except subprocess.TimeoutExpired:
                process.kill();returncode=process.wait()
                raise TimeoutError('fixed outer resource cap; partial child evidence preserved')
        result['returncode']=returncode
        child=load(child_output/'receipt.json');report=load(child_output/'result.json')
        if returncode!=0 or not(child.get('completed') and child.get('passed') and child.get('sources_unchanged') and report.get('passed')):
            raise RuntimeError('actual child completion/pass/unchanged and result pass required')
        if child.get('stage')!=stage or any(child.get(k)!=v for k,v in expected.items()):
            raise RuntimeError('actual child source/stage binding')
        if stage=='units' and report.get('checks')!=318:raise RuntimeError('fixed unit count')
        if stage=='timing' and report.get('records')!=20:raise RuntimeError('fixed timing count')
        if stage=='full' and (report.get('records'),report.get('identity_checks'),report.get('precision_checks'))!=(5010,3330320,470752):
            raise RuntimeError('fixed full count')
        result.update(completed=True,accepted_stage=True,stage=stage,child_receipt_sha256=sha(child_output/'receipt.json'),result_sha256=sha(child_output/'result.json'))
    except BaseException as e:
        result.update(error_type=type(e).__name__,error=str(e))
        (invocation/'failure.txt').write_text(traceback.format_exc())
    finally:
        after=snapshot(pins)
        result['inputs_unchanged']=bool(before) and before==after
        if not result['inputs_unchanged']:result['accepted_stage']=False
        result['returncode']=returncode
        result['elapsed_seconds']=time.monotonic()-start
        save(invocation/'pins-after.json',after)
        files=list(invocation.rglob('*'))
        if child_output is not None and child_output.exists():files+=list(child_output.rglob('*'))
        result['outputs']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),
             'policy':'large_payload' if p.suffix in ('.npz','.npy','.jsonl') or p.stat().st_size>1048576 else 'source_or_receipt'}
             for p in sorted(files) if p.is_file() and p!=invocation/'receipt.json']
        save(invocation/'receipt.json',result)
    return 0 if result['accepted_stage'] else 1

if __name__=='__main__':
    raise SystemExit(main())
