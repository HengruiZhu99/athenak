#!/usr/bin/env python3
"""Bounded, frozen-source initial transition derivative/source localization."""
import argparse,hashlib,json,os,resource,shlex,shutil,signal,subprocess,sys,tarfile,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path.cwd();out=a.output.resolve();out.mkdir(exist_ok=False);resource.setrlimit(resource.RLIMIT_CORE,(0,0));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();source=root/'docs/elliptic-shift-3d/evidence/supported-puncture-coupled-20261027';capsule=out/'source';capsule.mkdir();archive=source/'compiled-source.tar.gz'
with tarfile.open(archive) as t:
 for member in t.getmembers():assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
 t.extractall(capsule)
for name in ['diagnose_transition_jets.cpp','test_amr_core_derivatives.cpp','wormhole_exact_jets.py','radial_reference.py']:
 target=capsule/'tst/elliptic_shift_3d'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/'tst/elliptic_shift_3d'/name,target)
for f in (root/'tst/elliptic_shift_1d').glob('*.py'):
 target=capsule/f.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
for f in (root/'build-shift-3d/kokkos').glob('*/src/libkokkos*.a'):
 target=capsule/f.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
files={str(f.relative_to(capsule)):sha(f) for f in capsule.rglob('*') if f.is_file()};m={'scope':'Frozen supported-puncture-v15 initial exact-data derivative/source diagnostic; no solve/evolution or production adoption','source_archive':str(archive),'source_archive_sha256':sha(archive),'source_sha256':files,'maximum_seconds_per_command':180,'commands':[]}
def save():(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
def run(name,command):
 row={'name':name,'argv':list(map(str,command)),'maximum_seconds':180};m['commands'].append(row);save();start=time.monotonic()
 with (out/(name+'.stdout')).open('w') as so,(out/(name+'.stderr')).open('w') as se:
  process=subprocess.Popen(list(map(str,command)),cwd=root,stdout=so,stderr=se,start_new_session=True);row['owned_process_group']=process.pid;save()
  try:row['exit_code']=process.wait(timeout=180)
  except subprocess.TimeoutExpired:
   assert os.getpgid(process.pid)==process.pid;os.killpg(process.pid,signal.SIGTERM)
   try:process.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
   row['exit_code']=124;row['owned_process_reaped']=True
 row['seconds']=time.monotonic()-start;save();print(name,row['exit_code'],round(row['seconds'],3),flush=True);assert row['exit_code']==0
flags=[]
for line in (root/'build-shift-3d/CMakeFiles/elliptic_shift_3d_native_fields_tests.dir/flags.make').read_text().splitlines():
 if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):flags+=shlex.split(line.split('=',1)[1].replace(str(root),str(capsule)))
binary=out/'diagnostic';run('compile',['/usr/bin/c++',*flags,'-MMD','-MF',out/'diagnostic.d',capsule/'tst/elliptic_shift_3d/diagnose_transition_jets.cpp','-o',binary,*map(str,(capsule/'build-shift-3d/kokkos').glob('*/src/libkokkos*.a'))]);initial=root/'docs/elliptic-shift-3d/evidence/amr-core-pilot-20261027/initial.dat';m['initial_sha256']=sha(initial);m['binary_sha256']=sha(binary);save();run('catalog',[binary,'catalog',initial,out/'catalog.csv']);run('analytic-oracle',[sys.executable,capsule/'tst/elliptic_shift_3d/wormhole_exact_jets.py',out/'catalog.csv',out/'exact.dat']);run('diagnose',[binary,'diagnose',initial,out/'exact.dat',out/'diagnostic.csv']);m['source_unchanged']=all(sha(capsule/f)==h for f,h in files.items());m['initial_unchanged']=sha(initial)==m['initial_sha256'];assert m['source_unchanged'] and m['initial_unchanged'];save()
