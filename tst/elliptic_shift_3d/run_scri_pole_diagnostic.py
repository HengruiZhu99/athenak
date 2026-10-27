#!/usr/bin/env python3
"""Frozen-source read-only boundary diagnostic; bounded processes, no solve/run."""
import argparse,hashlib,json,shlex,subprocess,tarfile,time,os,signal,resource
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0));r=Path.cwd();b=r/'build-shift-3d';o=a.output.resolve();o.mkdir(parents=True,exist_ok=False)
archive=r/'docs/elliptic-shift-3d/evidence/amr-log-halo-coupled-20261027/source.tar.gz';frozen=o/'source';frozen.mkdir()
with tarfile.open(archive) as t:
 for member in t.getmembers():
  target=(frozen/member.name).resolve()
  if not target.is_relative_to(frozen) or not member.isfile():
   if member.isdir():continue
   raise RuntimeError('Unexpected archive path/type')
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(t.extractfile(member).read())
src=r/'tst/elliptic_shift_3d/diagnose_scri_pole.cpp';dest=frozen/'tst/elliptic_shift_3d/diagnose_scri_pole.cpp';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(src.read_bytes())
flags={};flag=b/'CMakeFiles/elliptic_shift_3d_metric_divergence_tests.dir/flags.make'
for line in flag.read_text().splitlines():
 if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
  k,v=line.split('=',1);flags[k.strip()]=shlex.split(v)
link=shlex.split((flag.parent/'link.txt').read_text());libs=[(b/x).resolve() for x in link if x.endswith('.a')]
checkpoint=r/'docs/elliptic-shift-3d/evidence/amr-log-halo-coupled-20261027/coupled.bin';initial=r/'docs/elliptic-shift-3d/evidence/amr-core-pilot-20261027/initial.dat';tag='b79a95d2b4452b4ec64e32be753c54f78a4551b115f46ff4a133fb7bdbdb69f1';binary=o/'diagnose-scri-pole';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
commands={'build':[link[0],*flags['CXX_DEFINES'],'-I'+str(frozen/'src'),'-I'+str(frozen/'build-shift-3d'),*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],str(dest),'-o',str(binary),*map(str,libs)],'diagnostic':[str(binary),str(checkpoint),tag,str(initial),str(o/'surface-poles.csv')]}
m=dict(scope='initial and frozen midpoint boundary jets; diagnostic ablations only; no solve/evolution/PDE change',inputs=[dict(path=str(p),sha256=sha(p)) for p in [archive,checkpoint,initial,src]],libraries=[dict(path=str(p),sha256=sha(p)) for p in libs],runs={})
for name,cmd in commands.items():
 t=time.monotonic()
 with (o/(name+'.stdout')).open('w') as so,(o/(name+'.stderr')).open('w') as se:
  child=subprocess.Popen(cmd,stdout=so,stderr=se,start_new_session=True)
  try:code=child.wait(timeout=180)
  except subprocess.TimeoutExpired:
   code=124
   if os.getpgid(child.pid)==child.pid:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
 m['runs'][name]=dict(command=cmd,exit_code=code,seconds=time.monotonic()-t);(o/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(name,code,m['runs'][name]['seconds'],flush=True)
 if code:raise SystemExit(code)
m['inputs_unchanged']=all(sha(Path(x['path']))==x['sha256'] for x in m['inputs']);m['binary_sha256']=sha(binary);(o/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
