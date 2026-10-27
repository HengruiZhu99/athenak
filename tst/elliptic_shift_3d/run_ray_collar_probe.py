#!/usr/bin/env python3
"""Bounded frozen-snapshot atlas test; no shared build/configuration mutation."""
import argparse,hashlib,json,shlex,subprocess,tarfile,time,os,signal,resource
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--donors',type=int,default=48);p.add_argument('--owner-scaling',action='store_true');p.add_argument('--outer-only',action='store_true');p.add_argument('--unified',action='store_true');p.add_argument('--matrix',choices=['first','transition','all'],default='first');a=p.parse_args();resource.setrlimit(resource.RLIMIT_CORE,(0,0));r=Path.cwd();b=r/'build-shift-3d';o=a.output.resolve();o.mkdir(parents=True,exist_ok=False);frozen=o/'source';frozen.mkdir();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
archive=r/'docs/elliptic-shift-3d/evidence/amr-log-halo-coupled-20261027/source.tar.gz'
with tarfile.open(archive) as t:
 for member in t.getmembers():
  target=(frozen/member.name).resolve()
  if not target.is_relative_to(frozen) or not member.isfile():
   if member.isdir():continue
   raise RuntimeError('Unexpected archive path/type')
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(t.extractfile(member).read())
files=['src/mesh/cubed_sphere/centered_surface_atlas.hpp','tst/elliptic_shift_3d/test_centered_surface_atlas.cpp','tst/elliptic_shift_3d/diagnose_scri_atlas.cpp','tst/elliptic_shift_3d/diagnose_scri_pole.cpp','src/mesh/cubed_sphere/cross_face_boundary_probe.hpp','tst/elliptic_shift_3d/test_cross_face_boundary_probe.cpp','src/mesh/cubed_sphere/volume_mls_probe.hpp','src/mesh/cubed_sphere/surface_mls_probe.hpp','tst/elliptic_shift_3d/test_volume_mls_probe.cpp','src/mesh/cubed_sphere/balanced_volume_mls_probe.hpp','tst/elliptic_shift_3d/test_balanced_volume_mls_probe.cpp','src/mesh/cubed_sphere/ray_collar_probe.hpp','tst/elliptic_shift_3d/test_ray_collar_probe.cpp']
for name in files:
 target=frozen/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((r/name).read_bytes())
flags={};flag=b/'CMakeFiles/elliptic_shift_3d_metric_divergence_tests.dir/flags.make'
for line in flag.read_text().splitlines():
 if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):
  k,v=line.split('=',1);flags[k.strip()]=shlex.split(v)
link=shlex.split((flag.parent/'link.txt').read_text());libs=[(b/x).resolve() for x in link if x.endswith('.a')];checkpoint=r/'docs/elliptic-shift-3d/evidence/amr-log-halo-coupled-20261027/coupled.bin';tag='b79a95d2b4452b4ec64e32be753c54f78a4551b115f46ff4a133fb7bdbdb69f1';commands={}
for name,source in [('analytic','test_centered_surface_atlas.cpp'),('snapshot','diagnose_scri_atlas.cpp')]:
 binary=o/name;commands[name+'-build']=[link[0],*flags['CXX_DEFINES'],'-I'+str(frozen/'src'),'-I'+str(frozen/'build-shift-3d'),*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],str(frozen/'tst/elliptic_shift_3d'/source),'-o',str(binary),*map(str,libs)];commands[name]=[str(binary)] if name=='analytic' else [str(binary),str(checkpoint),tag,str(o/'atlas-poles.csv')]
m=dict(scope='Centered cross-face angular derivative diagnostic, no PDE/operator adoption',inputs=[dict(path=str(p),sha256=sha(p)) for p in [archive,checkpoint,*[r/f for f in files]]],runs={})
commands={}
binary=o/'full-jet-mms';commands['build']=[link[0],*flags['CXX_DEFINES'],'-I'+str(frozen/'src'),'-I'+str(frozen/'build-shift-3d'),*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],str(frozen/'tst/elliptic_shift_3d/test_ray_collar_probe.cpp'),'-o',str(binary),*map(str,libs)]
cases={'first':[(1,1,[6])],'transition':[(1,1,[6,12,18])],'all':[(1,1,[6,12,18]),(2,1,[6,10,14]),(7,1,[6,8,10]),(7,2,[6])]}[a.matrix]
for layers,sub,ns in cases:
 for n in ns:
  for radial in [6]:commands[f'mms-{layers}-{sub}-{n}-{radial}']=[str(binary),str(n),str(layers),str(sub),str(radial),str(a.donors),str(int(a.unified)),str(int(a.outer_only)),str(int(a.owner_scaling))]
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
m['inputs_unchanged']=all(sha(Path(x['path']))==x['sha256'] for x in m['inputs']);(o/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
