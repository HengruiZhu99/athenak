#!/usr/bin/env python3
"""Bounded continuum MMS for an unadopted composed metric derivative."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--build',type=Path,default=Path('build-shift-3d'));a=p.parse_args()
root=Path.cwd();a.output.mkdir(parents=True,exist_ok=False);out=a.output.resolve();build=a.build.resolve()
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
flags=(build/'CMakeFiles/elliptic_shift_3d_composite_regrid_tests.dir/flags.make').read_text().splitlines();args=[]
for key in ('CXX_DEFINES =','CXX_INCLUDES ='):args+=shlex.split(next(x.split('=',1)[1]for x in flags if x.startswith(key)))
source=root/'tst/elliptic_shift_3d/probe_composed_spectrum.cpp';binary=out/'probe'
libraries=[build/f'kokkos/{name}/src/libkokkos{name}.a'for name in ('containers','algorithms','core','simd')]
command=['/usr/bin/c++','-O3','-std=c++17',*args,str(source),'-o',str(binary),*map(str,libraries)]
manifest=dict(scope='No-solve restricted low/high-frequency singular controls; no full spectrum or gap claim',source_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),runs={})
def run(name,command,limit):
 start=time.monotonic()
 with (out/(name+'.stdout')).open('w')as stdout,(out/(name+'.stderr')).open('w')as stderr:
  code=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=limit).returncode
 row=dict(command=command,exit_code=code,maximum_seconds=limit,seconds=time.monotonic()-start);manifest['runs'][name]=row
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 return code
if run('dependencies',['/usr/bin/c++','-std=c++17',*args,'-MM',str(source)],60):raise SystemExit(1)
dependencies=(out/'dependencies.stdout').read_text().replace('\\\n',' ').split(':',1)[1].split();files={Path(x).resolve()for x in dependencies}|set(libraries)|{Path(__file__).resolve()}
project={str(f.relative_to(root)):sha(f)for f in files if f.is_file()and root in f.parents}
with tarfile.open(out/'compiled-source.tar.gz','w:gz')as archive:
 for name in sorted(project):archive.add(root/name,arcname=name)
manifest['project_dependencies']=project
if run('build',command,60):raise SystemExit(1)
(out/'.gitignore').write_text('/probe\n')
manifest['binary_sha256']=sha(binary)
for n,shells,refine in [(6,7,0),(8,7,0),(10,7,0),(6,7,1),(8,7,1),(6,1,0),(6,2,0)]:
 run(f'n{n}-s{shells}-r{refine}',[str(binary),str(n),str(shells),str(refine)],180)
manifest['binary_unchanged']=sha(binary)==manifest['binary_sha256']
manifest['project_inputs_unchanged']=all(sha(root/name)==value for name,value in project.items())
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items()if k!='project_dependencies'},indent=2))

if any(row["exit_code"] != 0 for row in manifest["runs"].values()):
 raise SystemExit(1)
