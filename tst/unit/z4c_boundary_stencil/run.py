#!/usr/bin/env python3
"""Compile the helper point test against an existing host-accessible CPU build.

This is a local CPU unit test, not a GPU portability or evolution stability test.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--build-dir',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args();b=a.build_dir.resolve();out=a.output_dir.resolve()
out.mkdir(parents=True,exist_ok=True)
flags={}
for line in (b/'src/CMakeFiles/athena.dir/flags.make').read_text().splitlines():
    if ' = ' in line:
        key,value=line.split(' = ',1);flags[key]=shlex.split(value)
compiler='/usr/bin/c++'
cache=(b/'CMakeCache.txt').read_text().splitlines()
for line in cache:
    if line.startswith(('CMAKE_CXX_COMPILER:FILEPATH=', 'CMAKE_CXX_COMPILER:STRING=')):
        compiler=line.split('=',1)[1]
source=Path(__file__).with_name('point_tests.cpp').resolve()
# Reuse the build's libraries instead of assuming a particular OpenMP runtime.
libraries=[]
for token in shlex.split((b/'src/CMakeFiles/athena.dir/link.txt').read_text()):
    if token.startswith(('-l','-L','-Wl,-rpath')):
        libraries.append(token)
    elif token.endswith(('.a','.dylib','.so')) or '.so.' in token:
        lib=Path(token)
        libraries.append(str(lib if lib.is_absolute() else (b/'src'/lib).resolve()))
command=[compiler,*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],str(source),
         *libraries,'-o',str(out/'point_tests')]
(out/'compile-command.json').write_text(json.dumps(command,indent=2)+'\n')
with (out/'build.log').open('w') as log:
    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
env=dict(os.environ,OMP_NUM_THREADS='1',OMP_PROC_BIND='false')
result=subprocess.run([str(out/'point_tests')],text=True,capture_output=True,env=env)
(out/'run.stdout').write_text(result.stdout);(out/'run.stderr').write_text(result.stderr)
data=json.loads(result.stdout.strip().splitlines()[-1])
(out/'result.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data))
raise SystemExit(result.returncode or (not data['passed']))
