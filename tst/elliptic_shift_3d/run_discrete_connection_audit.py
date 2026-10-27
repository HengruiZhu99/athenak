#!/usr/bin/env python3
"""Freeze actual affine accepted-state headers, then one bounded read-only audit."""
import argparse,gzip,hashlib,json,os,resource,shlex,shutil,signal,subprocess,tarfile,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0));root=Path.cwd();cap=root/'docs/elliptic-shift-3d/evidence/affine-coupled-n6-accepted-20261027';out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((cap/'manifest.json').read_text());(out/'.gitignore').write_text('/source/\n/audit\n/accepted.bin\n/*.csv\n')
with tarfile.open(cap/'compiled-source.tar.gz') as archive:
 for f in archive.getmembers():
  target=(source/f.name).resolve()
  if f.isdir():continue
  if not f.isfile() or not target.is_relative_to(source):raise RuntimeError('unsafe archive member')
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.extractfile(f).read())
for name,h in m['sources_sha256'].items():
 if (source/name).exists():assert sha(source/name)==h,name
flag=root/'build-shift-3d/CMakeFiles/elliptic_shift_3d_coupled_pilot.dir/flags.make';flags=[]
for line in flag.read_text().splitlines():
 if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):flags+=shlex.split(line.split('=',1)[1].replace(str(root),str(source)))
link=shlex.split((flag.parent/'link.txt').read_text());libs=[(root/'build-shift-3d'/x).resolve() for x in link if x.endswith('.a')];library_hashes={str(f):sha(f) for f in libs}
for f in libs:
 target=source/f.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
for name in ['LICENSE','kokkos/LICENSE','tst/elliptic_shift_3d/audit_discrete_connection.cpp']:
 target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,target)
snapshot=cap/'coupled.checkpoint.accepted.checkpoint';shutil.copyfile(snapshot,out/'accepted.bin');driver=source/'tst/elliptic_shift_3d/audit_discrete_connection.cpp';report=dict(scope='Actual failed affine finalaccepted N6; no evolution; actual affine state plus fresh old-operator solve on identical geometry; no operator/gauge modification',accepted_manifest_sha256=sha(cap/'manifest.json'),accepted_source_sha256=sha(cap/'compiled-source.tar.gz'),snapshot_sha256=sha(snapshot),snapshot_source_tag=m['binary_sha256'],driver_sha256=sha(driver),copied_libraries=library_hashes,maximum_seconds_per_process=180,runs={});commands={'build':['/usr/bin/c++',*flags,str(driver),'-o',str(out/'audit'),*[str(source/f.relative_to(root)) for f in libs]],'dependencies':['/usr/bin/c++',*flags,'-M',str(driver)],'audit':[str(out/'audit'),str(out/'accepted.bin'),m['binary_sha256'],str(out/'actual')],'old_operator_resolve':[str(out/'audit'),str(out/'accepted.bin'),m['binary_sha256'],str(out/'old_operator'),'old_operator_resolve']}
for name,cmd in commands.items():
 start=time.monotonic()
 with (out/(name+'.stdout')).open('w') as so,(out/(name+'.stderr')).open('w') as se:
  child=subprocess.Popen(cmd,stdout=so,stderr=se,start_new_session=True)
  try:code=child.wait(timeout=180)
  except subprocess.TimeoutExpired:
   assert os.getpgid(child.pid)==child.pid;os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
   code=124
 report['runs'][name]=dict(command=cmd,exit_code=code,seconds=time.monotonic()-start,maximum_seconds=180);(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(name,code,round(report['runs'][name]['seconds'],3),flush=True)
 if code:raise SystemExit(code)
report['result']={name:json.loads((out/(name+'.stdout')).read_text()) for name in ['audit','old_operator_resolve']};paths=(out/'dependencies.stdout').read_text().replace('\\\n',' ').split(':',1)[1].split();report['live_project_dependencies']=[x for x in paths if x.startswith(str(root)+'/') and not x.startswith(str(source)+'/')];assert not report['live_project_dependencies'];report['external_dependency_hashes']={f:sha(Path(f)) for f in paths if not f.startswith(str(source)+'/')};report['copied_libraries_stable']=all(sha(Path(f))==v for f,v in library_hashes.items())
for f in out.glob('*.csv'):
 target=f.with_name(f.name+'.gz')
 with gzip.open(target,'wb') as z:z.write(f.read_bytes())
 assert gzip.open(target,'rb').read()==f.read_bytes()
with tarfile.open(out/'compiled-source.tar.gz','w:gz') as archive:
 for f in sorted(source.rglob('*')):
  if f.is_file():archive.add(f,arcname=str(f.relative_to(source)))
report['compiled_source_sha256']=sha(out/'compiled-source.tar.gz');(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
