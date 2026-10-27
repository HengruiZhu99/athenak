#!/usr/bin/env python3
"""Same-frozen-state Hermite7/8 probe; no solve, coupled rerun or shared edit."""
import argparse,gzip,hashlib,json,os,resource,signal,subprocess,tarfile,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--family',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0));root=Path.cwd();parent=a.family.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((parent/'manifest.json').read_text());assert sha(parent/'compiled-source.tar.gz')==m['source_archive_sha256'];(out/'.gitignore').write_text('/source/\n/diagnostic\n/frozen-midpoint.bin\n/poles.csv\n')
with tarfile.open(parent/'compiled-source.tar.gz') as archive:
 for f in archive.getmembers():
  target=(source/f.name).resolve()
  if f.isdir():continue
  if not f.isfile() or not target.is_relative_to(source):raise RuntimeError('unsafe archive member')
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.extractfile(f).read())
driver=root/'tst/elliptic_shift_3d/diagnose_related_scri_nodes.cpp';target=source/driver.relative_to(root);target.write_bytes(driver.read_bytes());checkpoint=out/'frozen-midpoint.bin';checkpoint.write_bytes(gzip.open(parent/'n10/coupled.checkpoint.gz','rb').read());tag=m['binary_sha256'];command=[x.replace(str(parent/'source'),str(source)).replace(str(parent/'pilot'),str(out/'diagnostic')) for x in m['build']['frozen-build']['command']];command=[str(target) if x.endswith('/tst/elliptic_shift_3d/coupled.cpp') else x for x in command];report=dict(parent_manifest_sha256=sha(parent/'manifest.json'),parent_compiled_source_sha256=sha(parent/'compiled-source.tar.gz'),checkpoint_sha256=sha(checkpoint),checkpoint_compressed_sha256=sha(parent/'n10/coupled.checkpoint.gz'),checkpoint_source_tag=tag,driver_sha256=sha(driver),scope='Same saved N10midpoint fields and warmstart shift, Hermite7/8 sampling only; no shift solve or evolution',runs={})
for name,cmd in [('build',command),('dependencies',command[:command.index(str(target))]+['-M',str(target)]),('diagnostic',[str(out/'diagnostic'),str(checkpoint),tag,str(out/'poles.csv')])]:
 start=time.monotonic()
 with (out/(name+'.stdout')).open('w') as so,(out/(name+'.stderr')).open('w') as se:
  child=subprocess.Popen(cmd,stdout=so,stderr=se,start_new_session=True);row=dict(command=cmd,maximum_seconds=180,owned_process_group=child.pid)
  try:row['exit_code']=child.wait(timeout=180)
  except subprocess.TimeoutExpired:
   assert os.getpgid(child.pid)==child.pid;os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
   row.update(exit_code=124,owned_process_reaped=True)
 row['seconds']=time.monotonic()-start;report['runs'][name]=row;(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(name,row['exit_code'],row['seconds'],flush=True)
 if row['exit_code']:raise SystemExit(row['exit_code'])
paths=(out/'dependencies.stdout').read_text().replace('\\\n',' ').split(':',1)[1].split();report['live_project_dependencies']=[x for x in paths if x.startswith(str(root)+'/') and not x.startswith(str(source)+'/')];assert not report['live_project_dependencies'];report['result']=json.loads((out/'diagnostic.stdout').read_text());assert report['result']['frozen_data_unchanged'];f=out/'poles.csv'
with gzip.open(out/'poles.csv.gz','wb') as z:z.write(f.read_bytes())
report['lossless_csv']=gzip.open(out/'poles.csv.gz','rb').read()==f.read_bytes();report['poles_csv_sha256']=sha(f)
with tarfile.open(out/'compiled-source.tar.gz','w:gz') as archive:
 for f in sorted(source.rglob('*')):
  if f.is_file():archive.add(f,arcname=str(f.relative_to(source)))
report['compiled_source_sha256']=sha(out/'compiled-source.tar.gz');report['binary_sha256']=sha(out/'diagnostic');(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
