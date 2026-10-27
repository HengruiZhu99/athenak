#!/usr/bin/env python3
"""Frozen bounded host sampler build; no shared CMake/engine mutation."""
import argparse,hashlib,json,shlex,subprocess,tarfile,time,os,signal,resource
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0));r=Path.cwd();b=r/'build-shift-3d';o=a.output.resolve();o.mkdir(parents=True,exist_ok=False);frozen=o/'source';frozen.mkdir();sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();archive=r/'docs/elliptic-shift-3d/evidence/scri-hermite-boundary-qualified-20261027/compiled-source.tar.gz'
with tarfile.open(archive) as t:
 for member in t.getmembers():
  target=(frozen/member.name).resolve()
  if not target.is_relative_to(frozen) or not member.isfile():
   if member.isdir():continue
   raise RuntimeError('Unexpected archive path/type')
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(t.extractfile(member).read())
files=['src/mesh/cubed_sphere/surface_polynomial_weights.hpp','src/mesh/cubed_sphere/hermite_boundary_sampler.hpp','tst/elliptic_shift_3d/test_hermite_sampler_contract.cpp','src/mesh/cubed_sphere/scri_surface.hpp','src/mesh/cubed_sphere/composite_difference.hpp','src/mesh/cubed_sphere/composite.hpp','src/mesh/cubed_sphere/map.hpp','src/athena.hpp','build-shift-3d/config.hpp','LICENSE']
for name in files:
 target=frozen/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((r/name).read_bytes())
flags={};flag=b/'CMakeFiles/elliptic_shift_3d_metric_divergence_tests.dir/flags.make'
for line in flag.read_text().splitlines():
 if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
  k,v=line.split('=',1);flags[k.strip()]=shlex.split(v)
flags['CXX_INCLUDES']=[x for x in flags['CXX_INCLUDES'] if x!='-I'+str(r/'src')];link=shlex.split((flag.parent/'link.txt').read_text());libs=[(b/x).resolve() for x in link if x.endswith('.a')];binary=o/'sampler-contract';commands={'build':[link[0],*flags['CXX_DEFINES'],'-I'+str(frozen/'src'),'-I'+str(frozen/'build-shift-3d'),*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],str(frozen/'tst/elliptic_shift_3d/test_hermite_sampler_contract.cpp'),'-o',str(binary),*map(str,libs)],'contract':[str(binary)]}
prior=json.loads((r/'docs/elliptic-shift-3d/evidence/scri-hermite-boundary-qualified-20261027/manifest.json').read_text());m=dict(scope='Host immutable-stage Hermite sampler contract; no evolution integration',inputs=[dict(path=str(f),sha256=sha(f)) for f in [archive,*[r/n for n in files],Path(__file__).resolve()]],runtime_hashes_before={f:sha(Path(f)) for f in prior['runtime_hashes_before']},runs={})
for name,cmd in commands.items():
 start=time.monotonic()
 with (o/(name+'.stdout')).open('w') as so,(o/(name+'.stderr')).open('w') as se:
  child=subprocess.Popen(cmd,stdout=so,stderr=se,start_new_session=True)
  try:code=child.wait(timeout=180)
  except subprocess.TimeoutExpired:
   code=124
   if os.getpgid(child.pid)==child.pid:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
 m['runs'][name]=dict(command=cmd,exit_code=code,seconds=time.monotonic()-start);(o/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(name,code,m['runs'][name]['seconds'],flush=True)
 if code:raise SystemExit(code)
m['inputs_unchanged']=all(sha(Path(x['path']))==x['sha256'] for x in m['inputs']);m['runtime_unchanged']=all(sha(Path(f))==h for f,h in m['runtime_hashes_before'].items());(o/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
