#!/usr/bin/env python3
"""Frozen bounded AD-global-radial-interpolant tests; no production mutation."""
import argparse,hashlib,json,os,resource,signal,subprocess,tarfile,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--matrix',choices=['first','all'],default='first');a=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0));root=Path.cwd();parent=root/'docs/elliptic-shift-3d/evidence/coupled-surface-mass-20261027';out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=json.loads((parent/'manifest.json').read_text());assert sha(parent/'compiled-source.tar.gz')==m['compiled_source_sha256'];(out/'.gitignore').write_text('/source/\n/mms\n')
with tarfile.open(parent/'compiled-source.tar.gz') as archive:
 for f in archive.getmembers():
  target=(source/f.name).resolve()
  if f.isdir():continue
  if not f.isfile() or not target.is_relative_to(source):raise RuntimeError('unsafe archive member')
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.extractfile(f).read())
for name in ['src/mesh/cubed_sphere/radial_interface_jet_probe.hpp','tst/elliptic_shift_3d/test_radial_interface_jet_probe.cpp']:
 target=source/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((root/name).read_bytes())
command=[x.replace(str(parent/'source'),str(source)).replace(str(parent/'extract'),str(out/'mms')) for x in m['runs']['build']['command']];command=[str(source/'tst/elliptic_shift_3d/test_radial_interface_jet_probe.cpp') if x.endswith('/tst/elliptic_shift_3d/extract_surface_mass.cpp') else x for x in command];target=source/'tst/elliptic_shift_3d/test_radial_interface_jet_probe.cpp';idx=command.index(str(target));report=dict(scope='ADjetprototypeofexistingglobalradial6pointvalueinterpolant; no productionSampler/PDE/evolution edits',parent_source_sha256=m['compiled_source_sha256'],predeclared_guards=dict(radial_condition=1e8,value_L1=8192,scaled_gradient_L1=1e5,scaled_hessian_L1=1e6,partition_reproduction=1e-10),runs={})
commands={'build':command,'dependencies':command[:idx]+['-M',str(target)]};cases=[(7,1,[6])] if a.matrix=='first' else [(1,1,[6,12,18]),(2,1,[6,10,14]),(7,1,[6,8,10]),(7,2,[6])]
for layers,sub,ns in cases:
 for n in ns:commands[f'mms-{layers}-{sub}-{n}']=[str(out/'mms'),str(n),str(layers),str(sub)]
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
 report['runs'][name]=dict(command=cmd,exit_code=code,seconds=time.monotonic()-start,maximum_seconds=180);(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(name,code,report['runs'][name]['seconds'],flush=True)
 if code and name in ['build','dependencies']:raise SystemExit(code)
paths=(out/'dependencies.stdout').read_text().replace('\\\n',' ').split(':',1)[1].split();report['live_project_paths']=[x for x in paths if x.startswith(str(root)+'/') and not x.startswith(str(source)+'/')];assert not report['live_project_paths']
with tarfile.open(out/'compiled-source.tar.gz','w:gz') as archive:
 for f in sorted(source.rglob('*')):
  if f.is_file():archive.add(f,arcname=str(f.relative_to(source)))
report['compiled_source_sha256']=sha(out/'compiled-source.tar.gz');(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
