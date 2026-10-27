#!/usr/bin/env python3
"""Read-only accepted-state Khat/Theta pole split, no second coupled run."""
import argparse,gzip,hashlib,json,os,resource,signal,subprocess,tarfile,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0));root=Path.cwd();parent=a.capture.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((parent/'manifest.json').read_text());assert sha(parent/'compiled-source.tar.gz')==m['compiled_source_sha256'];(out/'.gitignore').write_text('/source/\n/diagnostic\n/*.csv\n')
with tarfile.open(parent/'compiled-source.tar.gz') as archive:
 for f in archive.getmembers():
  target=(source/f.name).resolve()
  if f.isdir():continue
  if not f.isfile() or not target.is_relative_to(source):raise RuntimeError('unsafe archive member')
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.extractfile(f).read())
driver=root/'tst/elliptic_shift_3d/diagnose_final_scri_state.cpp';target=source/driver.relative_to(root);target.write_bytes(driver.read_bytes());cmd=[x.replace(str(parent/'source'),str(source)).replace(str(parent/'diagnostic'),str(out/'diagnostic')) for x in m['runs']['diagnostic-build']['command']];idx=cmd.index(str(target));report=dict(scope='Read-only final accepted scalar trace split; exactly the IO-captured state; no evolution/solve',capture_manifest_sha256=sha(parent/'manifest.json'),capture_source_sha256=m['compiled_source_sha256'],driver_sha256=sha(driver),runs={})
for name,c in [('build',cmd),('dependencies',cmd[:idx]+['-M',str(target)]),('mink',[str(out/'diagnostic'),str(parent/'mink-io/coupled.checkpoint.final'),m['io_binary_sha256'],str(out/'mink.csv')]),('n10',[str(out/'diagnostic'),str(parent/'n10/coupled.checkpoint.final'),m['io_binary_sha256'],str(out/'n10.csv')])]:
 start=time.monotonic()
 with (out/(name+'.stdout')).open('w') as so,(out/(name+'.stderr')).open('w') as se:
  child=subprocess.Popen(c,stdout=so,stderr=se,start_new_session=True)
  try:code=child.wait(timeout=180)
  except subprocess.TimeoutExpired:
   assert os.getpgid(child.pid)==child.pid;os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
   code=124
 report['runs'][name]=dict(command=c,exit_code=code,seconds=time.monotonic()-start,maximum_seconds=180);(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(name,code,round(report['runs'][name]['seconds'],3),flush=True)
 if code:raise SystemExit(code)
paths=(out/'dependencies.stdout').read_text().replace('\\\n',' ').split(':',1)[1].split();report['live_project_dependencies']=[x for x in paths if x.startswith(str(root)+'/') and not x.startswith(str(source)+'/')];assert not report['live_project_dependencies'];report['results']={name:json.loads((out/(name+'.stdout')).read_text()) for name in ['mink','n10']};report['prior_n10_pole_bitwise_same']=report['results']['n10']['methods'][0]['pole_max']==m['N10_certificate']['methods'][0]['pole_max'];assert report['prior_n10_pole_bitwise_same']
for name in ['mink','n10']:
 f=out/(name+'.csv')
 with gzip.open(out/(name+'.csv.gz'),'wb') as z:z.write(f.read_bytes())
 assert gzip.open(out/(name+'.csv.gz'),'rb').read()==f.read_bytes()
with tarfile.open(out/'compiled-source.tar.gz','w:gz') as archive:
 for f in sorted(source.rglob('*')):
  if f.is_file():archive.add(f,arcname=str(f.relative_to(source)))
report['compiled_source_sha256']=sha(out/'compiled-source.tar.gz');(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
