#!/usr/bin/env python3
"""HELD one-shot local build/query runner. Standard-library guards only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def pin(path):
    path=Path(path).resolve();data=path.read_bytes()
    return {'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
def read(path):return json.loads(Path(path).read_text())
def guard(rows):
    for row in rows:
        if pin(row['path'])['sha256']!=row['sha256']:
            raise RuntimeError('Input drift: '+row['path'])
def write_new(path,data):
    path=Path(path)
    if path.exists():raise RuntimeError('Refuse overwrite '+str(path))
    path.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--authorization',required=True)
    p.add_argument('--build',required=True,choices=['release','debug']);args=p.parse_args()
    if sys.flags.optimize or not sys.flags.isolated or not sys.dont_write_bytecode:
        raise RuntimeError('Require isolated unoptimized -I -B Python')
    here=Path(__file__).resolve().parent;recipe_path=here/'recipe.json'
    recipe=read(recipe_path);auth=read(args.authorization)
    if not auth.get('far_complete_dual_local_execution_admitted'):
        raise RuntimeError('Exact root local authorization required')
    if auth['recipe_sha256']!=pin(recipe_path)['sha256'] or auth['source_index_sha256']!=pin(here/'source-index.json')['sha256']:
        raise RuntimeError('Root authorization source/recipe mismatch')
    if args.build not in auth['allowed_builds']:
        raise RuntimeError('Build not admitted')
    index=read(here/'source-index.json');inputs=read(recipe['input_pins'])
    guard(index['files']);guard(inputs)
    # Exact source/context pins and fresh root authorization govern this separate supplement.
    for key,value in recipe['environment'].items():
        if os.environ.get(key)!=value:raise RuntimeError('Environment guard '+key)
    attempt=here/'attempts'/recipe['attempt_names'][args.build]
    if attempt.exists():raise RuntimeError('One-shot attempt already exists')
    attempt.mkdir(parents=True)
    commands=[];receipt={'completed':False,'passed':False,'returncode':1,
      'scientific_scope':'direct finite-Omega RWM complete field-dual arithmetic supplement only',
      'no_matrix_evolution_native_or_BH_admission':True,'build':args.build,
      'source_index_sha256':pin(here/'source-index.json')['sha256'],
      'recipe_sha256':pin(recipe_path)['sha256'],
      'authorization':pin(args.authorization),'input_count':len(inputs),
      'source_inputs_unchanged':False,'commands':commands}
    def command(name,cmd):
        started=time.monotonic();stdout=attempt/(name+'.stdout');stderr=attempt/(name+'.stderr')
        # Exact bytes/whitespace are retained, including failures.
        with stdout.open('wb') as out,stderr.open('wb') as err:
            done=subprocess.run(cmd,cwd=recipe['repository'],env=os.environ.copy(),stdout=out,stderr=err)
        commands.append({'name':name,'command':cmd,'returncode':done.returncode,
                         'seconds':time.monotonic()-started,'stdout':pin(stdout),'stderr':pin(stderr)})
        if done.returncode:raise RuntimeError('Command failed: '+name)
        return stdout
    try:
        command('compiler-version',[recipe['compiler'],'--version'])
        command('launch-HEAD',['git','rev-parse','HEAD'])
        command('python-version',[recipe['python'],'-I','-B','--version'])
        exe=attempt/('far-dual-probe-'+args.build);dep=attempt/'dependencies.d'
        cmd=[recipe['compiler']]+recipe['compile_flags'][args.build]+['-MD','-MF',str(dep),str(here/'probe.cpp'),'-o',str(exe)]
        command('compile',cmd)
        tokens=shlex.split(dep.read_text().replace('\\\n',' '))
        dependencies=[];known={row['path']:row['sha256'] for row in inputs+index['files']}
        for token in tokens[1:]:
            path=Path(token)
            if not path.is_absolute():path=Path(recipe['repository'])/path
            row=pin(path)
            if row['path'] not in known or known[row['path']]!=row['sha256']:
                raise RuntimeError('New/unpinned compile dependency: '+row['path'])
            dependencies.append(row)
        write_new(attempt/'dependencies.json',dependencies)
        receipt['executable_before']=pin(exe)
        for mode in recipe['modes']:
            output=command(mode,[str(exe),mode])
            target=attempt/(mode+'.jsonl')
            # Preserve stdout in place; the scientific payload copy is exact.
            target.write_bytes(output.read_bytes())
        command('oracle',[recipe['python'],'-I','-B',str(here/'oracle.py'),'--authorization',str(Path(args.authorization).resolve()),'--recipe',str(recipe_path),'--attempt',str(attempt)])
        report=read(attempt/'oracle-report.json')
        if not report['passed']:raise RuntimeError('Oracle did not pass')
        guard(inputs);guard(index['files']);guard(dependencies)
        receipt['executable_after']=pin(exe)
        if receipt['executable_after']!=receipt['executable_before']:
            raise RuntimeError('Executable changed')
        receipt.update(completed=True,passed=True,returncode=0,source_inputs_unchanged=True,
                       oracle_report=pin(attempt/'oracle-report.json'))
    except BaseException as error:
        receipt['failure']={'type':type(error).__name__,'message':str(error)}
        try:
            guard(inputs);guard(index['files']);receipt['source_inputs_unchanged']=True
        except BaseException as drift:receipt['input_guard_failure']=str(drift)
    finally:
        receipt['output_inventory']=[pin(path) for path in sorted(attempt.rglob('*')) if path.is_file()]
        write_new(attempt/'receipt.json',receipt)
    print(json.dumps({'attempt':str(attempt),'passed':receipt['passed'],
                      'receipt':pin(attempt/'receipt.json')},sort_keys=True))
    if not receipt['passed']:raise SystemExit(1)


if __name__=='__main__':main()
