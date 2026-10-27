#!/usr/bin/env python3
"""Freeze original Arnoldi dependencies; fresh energy action, no Arnoldi rerun."""
import argparse,gzip,hashlib,json,os,resource,shlex,shutil,signal,subprocess,tarfile,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0));root=Path.cwd();cap=root/'docs/elliptic-shift-3d/evidence/composed-arnoldi-20261027';m=json.loads((cap/'manifest.json').read_text());out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();report={'scope':'Flat physical integration-by-parts defect of negative composed witness, smooth zeroBC control; no Arnoldi/evolution/operator adoption','runs':{},'input_manifest_sha256':sha(cap/'manifest.json'),'input_source_archive_sha256':sha(cap/'compiled-source.tar.gz'),'source_hashes':{}}
with tarfile.open(cap/'compiled-source.tar.gz') as archive:
 for f in archive.getmembers():
  target=(source/f.name).resolve()
  if f.isdir():continue
  if not f.isfile() or not target.is_relative_to(source):raise RuntimeError('unsafe archive member')
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.extractfile(f).read())
for relative,digest in m['project_source_hashes'].items():assert sha(source/relative)==digest,relative
for name in ['LICENSE','tst/elliptic_shift_3d/audit_composed_energy.cpp','tst/elliptic_shift_3d/run_composed_energy_audit.py']:
 target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,target);report['source_hashes'][name]=sha(target)
libs=[]
for r in m['libraries']:
 f=Path(r['path']);assert sha(f)==r['sha256'];target=source/f.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target);libs.append(str(target))
raw=gzip.decompress((cap/'composed-negative.witness.csv.gz').read_bytes());(out/'witness.csv').write_bytes(raw);report['witness_uncompressed_sha256']=sha(out/'witness.csv');report['original_library_hashes']=m['libraries']
oldcmd=m['runs']['build']['command'];flags=oldcmd[1:oldcmd.index(str(cap/'source/tst/elliptic_shift_3d/probe_composed_arnoldi.cpp'))];flags=[x.replace(str(cap/'source'),str(source)) for x in flags];driver=str(source/'tst/elliptic_shift_3d/audit_composed_energy.cpp');commands={'dependencies':['/usr/bin/c++',*flags,'-M',driver],'build':['/usr/bin/c++',*flags,driver,'-o',str(out/'audit'),*libs],'energy':[str(out/'audit'),str(out/'witness.csv'),str(out/'energy')]}
(out/'.gitignore').write_text('/source/\n/audit\n/*.csv\n')
for name,cmd in commands.items():
 start=time.monotonic();env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1')
 with (out/(name+'.stdout')).open('w') as so,(out/(name+'.stderr')).open('w') as se:
  child=subprocess.Popen(cmd,stdout=so,stderr=se,env=env,start_new_session=True)
  try:code=child.wait(timeout=60)
  except subprocess.TimeoutExpired:
   assert os.getpgid(child.pid)==child.pid;os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
   code=124
 report['runs'][name]={'command':cmd,'exit_code':code,'seconds':time.monotonic()-start,'maximum_seconds':60};(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(name,code,round(report['runs'][name]['seconds'],3),flush=True)
 if code:raise SystemExit(code)
paths=[Path(x).resolve() for x in shlex.split((out/'dependencies.stdout').read_text().replace('\\\n',' '))[1:]];report['live_project_dependency_leaks']=[str(x) for x in paths if x.is_relative_to(root) and not x.is_relative_to(source)];assert not report['live_project_dependency_leaks'];report['actual_compiler_dependency_hashes']={str(f):sha(f) for f in paths};report['binary_sha256']=sha(out/'audit');report['result']=json.loads((out/'energy.stdout').read_text());report['original_libraries_unchanged']=all(sha(Path(r['path']))==r['sha256'] for r in m['libraries']);assert report['original_libraries_unchanged']
for f in out.glob('energy.*.csv'):
 compressed=f.with_name(f.name+'.gz');compressed.write_bytes(gzip.compress(f.read_bytes(),mtime=0));assert gzip.decompress(compressed.read_bytes())==f.read_bytes()
with tarfile.open(out/'compiled-source.tar.gz','w:gz') as archive:
 for f in sorted(source.rglob('*')):
  if f.is_file():archive.add(f,arcname=f.relative_to(source))
report['compiled_source_sha256']=sha(out/'compiled-source.tar.gz');(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({name:case['regions']['all'] for name,case in report['result']['cases'].items()},indent=2))

if any(row["exit_code"] != 0 for row in manifest["runs"].values()):
 raise SystemExit(1)
