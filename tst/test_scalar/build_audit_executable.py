#!/usr/bin/env python3
"""Compile a read-only audit using exactly the scalar build's flags and libraries."""
import argparse,hashlib,json,shlex,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise RuntimeError('Existing audit binary is immutable')
values={}
for line in (a.build/'CMakeFiles/test_scalar.dir/flags.make').read_text().splitlines():
 if ' = ' in line:key,value=line.split(' = ',1);values[key]=value
link=shlex.split((a.build/'CMakeFiles/test_scalar.dir/link.txt').read_text());index=next(i for i,v in enumerate(link) if v.endswith('/main.cpp.o'));obj=a.output.with_suffix('.o').resolve();link[index]=str(obj);link[link.index('-o')+1]=str(a.output.resolve());compile_command=[link[0],*shlex.split(values['CXX_DEFINES']),*shlex.split(values['CXX_INCLUDES']),*shlex.split(values['CXX_FLAGS']),'-c',str(a.source.resolve()),'-o',str(obj)]
with a.output.with_suffix('.build.log').open('w') as log:
 subprocess.run(compile_command,cwd=a.build,stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run(link,cwd=a.build,stdout=log,stderr=subprocess.STDOUT,check=True)
a.output.with_suffix('.build.json').write_text(json.dumps(dict(commands=[compile_command,link],source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest()),indent=2)+'\n')
