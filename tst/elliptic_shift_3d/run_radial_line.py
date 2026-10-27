#!/usr/bin/env python3
"""Bounded standalone scalar radial-line preconditioner comparison."""
import argparse
import hashlib
import json
import shlex
import subprocess
import tarfile
import time
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=Path.cwd();build=root/'build-shift-3d';out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    flags=(build/'CMakeFiles/elliptic_shift_3d_coupled_pilot.dir/flags.make').read_text();options=[]
    for name in ['CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS']:
        options+=shlex.split(next(line.split('=',1)[1] for line in flags.splitlines() if line.startswith(name+' =')))
    link=shlex.split((build/'CMakeFiles/elliptic_shift_3d_coupled_pilot.dir/link.txt').read_text());libs=[];i=1
    while i<len(link):
        if link[i]=='-o':i+=2;continue
        if not link[i].endswith('.o'):libs.append(link[i])
        i+=1
    binary=build/'test_radial_line_standalone';dep=out/'compiler-dependencies.d'
    command=[link[0],*options,'-MMD','-MF',str(dep),str(root/'tst/elliptic_shift_3d/test_radial_line.cpp'),'-o',str(binary),*libs]
    manifest=dict(scope='Scalar CPU preconditioner prototype; unchanged full operator; no vector/native/evolution claim',compile_command=command,cases=[])
    def save():(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save();before=time.monotonic();run=subprocess.run(command,cwd=build,capture_output=True,text=True,timeout=90)
    (out/'build.stdout').write_text(run.stdout);(out/'build.stderr').write_text(run.stderr);manifest['build']=dict(exit_code=run.returncode,seconds=time.monotonic()-before);save()
    if run.returncode:print(run.stderr,flush=True);raise SystemExit(run.returncode)
    dependencies=shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]);files=[Path(name).resolve() for name in dependencies]
    files.extend([Path(__file__).resolve(),build/'config.hpp'])
    hashes={str(path):sha(path) for path in files};manifest['dependency_hashes']=hashes;manifest['binary_sha256']=sha(binary)
    with tarfile.open(out/'compiled-sources.tar.gz','w:gz') as archive:
        for path in sorted(set(files)):archive.add(path,arcname=str(path.relative_to(root)) if path.is_relative_to(root) else str(path).lstrip('/'))
    cases=[('point-discrete',['--mode','point','--source','discrete']),('line-discrete',['--mode','line','--source','discrete']),('line-continuum',['--mode','line','--source','continuum'])]
    for name,options in cases:
        case=dict(name=name,command=[str(binary),*options,'--n','8','--maxiter','120']);manifest['cases'].append(case);save();before=time.monotonic();print(name,'started',flush=True)
        try:
            run=subprocess.run(case['command'],capture_output=True,text=True,timeout=180)
            (out/(name+'.stdout')).write_text(run.stdout);(out/(name+'.stderr')).write_text(run.stderr)
            case.update(exit_code=run.returncode,seconds=time.monotonic()-before)
            if run.stdout:
                try:case['result']=json.loads(run.stdout)
                except ValueError:case['unparsed_stdout']=True
        except subprocess.TimeoutExpired as error:
            case.update(exit_code=124,seconds=time.monotonic()-before,status='timeout')
            for suffix,value in [('stdout',error.stdout),('stderr',error.stderr)]:
                (out/(name+'.'+suffix)).write_bytes(value.encode() if isinstance(value,str) else value or b'')
        case['dependencies_unchanged']=all(sha(Path(path))==value for path,value in hashes.items());case['binary_unchanged']=sha(binary)==manifest['binary_sha256'];save()
        result=case.get('result',{});print(name,case['exit_code'],result.get('iterations'),result.get('original_residual_rms'),case['seconds'],flush=True)
    print(json.dumps(dict(exit_codes=[c['exit_code'] for c in manifest['cases']])),flush=True)


if __name__=='__main__':main()
