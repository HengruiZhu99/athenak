#!/usr/bin/env python3
"""Run four complete local Minkowski source controls from frozen radial source."""
from pathlib import Path
import argparse,json,tarfile,subprocess,time,hashlib,resource,os,signal
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0));r=Path.cwd();p=r/'docs/elliptic-shift-3d/evidence/radial-scri-jet-final-20261027';o=a.output.resolve();o.mkdir(parents=True,exist_ok=False);s=o/'source';s.mkdir()
with tarfile.open(p/'compiled-source.tar.gz') as t:
 for member in t.getmembers():
  target=(s/member.name).resolve()
  if member.isdir():continue
  if not target.is_relative_to(s) or not member.isfile():raise RuntimeError('unsafe archive member')
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(t.extractfile(member).read())
f=r/'tst/elliptic_shift_3d/test_radial_scri_minkowski.cpp';target=s/f.relative_to(r);target.write_bytes(f.read_bytes());m=json.loads((p/'manifest.json').read_text());cmd=m['runs']['pole-build']['command'];cmd=[x.replace(str(p/'source'),str(s)).replace(str(p/'pole'),str(o/'minkowski')) for x in cmd];cmd=[str(target) if x.endswith('/diagnose_radial_scri_jet.cpp') else x for x in cmd];runs={}
for name,c in [('build',cmd),('deps',cmd[:cmd.index(str(target))]+['-M',str(target)]),('minkowski',[str(o/'minkowski'),str(r/'docs/elliptic-shift-3d/evidence/hermite-supported-coupled-20261027/coupled.checkpoint'),'ef2196506b829ccabb6d072909db40e8dd8891f394a4baa1429eced0858fc58b'])]:
 t=time.monotonic()
 with (o/(name+'.stdout')).open('w') as so,(o/(name+'.stderr')).open('w') as se:
  ch=subprocess.Popen(c,stdout=so,stderr=se,start_new_session=True)
  try:code=ch.wait(timeout=180)
  except subprocess.TimeoutExpired:
   os.killpg(ch.pid,signal.SIGTERM)
   try:ch.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(ch.pid,signal.SIGKILL);ch.wait(timeout=5)
   code=124
 runs[name]=dict(command=c,exit_code=code,seconds=time.monotonic()-t);print(name,code,flush=True)
 if code:break
with tarfile.open(o/'compiled-source.tar.gz','w:gz') as tar:
 for f in sorted(s.rglob('*')):
  if f.is_file():tar.add(f,arcname=str(f.relative_to(s)))
(o/'manifest.json').write_text(json.dumps(dict(runs=runs,parent_source_sha256=hashlib.sha256((p/'compiled-source.tar.gz').read_bytes()).hexdigest(),source_sha256=hashlib.sha256((o/'compiled-source.tar.gz').read_bytes()).hexdigest()),indent=2)+'\n')
