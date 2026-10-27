#!/usr/bin/env python3
"""Freeze already-qualified native binary/physics headers and control dependencies."""
import argparse,gzip,hashlib,json,os,shlex,shutil,signal,subprocess,tarfile,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path.cwd();archive=a.archive.resolve();m=json.loads((archive/'manifest.json').read_text());out=a.output.resolve();out.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();source=archive/m['compiled_dependency_archive']['path'];assert sha(source)==m['compiled_dependency_archive']['sha256']
with tarfile.open(source,'r:gz') as t:
 for f in t.getmembers():assert f.isfile() and f.name.startswith('src/') and '..' not in Path(f.name).parts
 t.extractall(out)
assert all(sha(out/f)==h for f,h in m['compiled_dependency_archive']['source_sha256'].items())
binary=out/'athena-qualified';binary.write_bytes(gzip.decompress((archive/'athena-qualified.gz').read_bytes()));binary.chmod(0o755);assert sha(binary)==m['binary_sha256']
for f in root.joinpath('kokkos').rglob('*'):
 if f.is_file() and f.suffix in ['.hpp','.h','.inc']:
  target=out/f.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
for f in (root/'build-shift-3d').rglob('*'):
 if f.is_file() and (f.suffix in ['.hpp','.h','.inc'] or f.name.startswith('libkokkos') and f.suffix=='.a'):
  target=out/f.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
for name in ['tst/elliptic_shift_3d/native_wormhole_reference.cpp','tst/elliptic_shift_3d/wormhole_initial.py','tst/elliptic_shift_1d/wormhole_isotropic.py','tst/elliptic_shift_1d/wormhole_data.py','inputs/z4c/composite_wormhole_static8.athinput']:
 target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,target)
flags=[]
for line in (root/'build-shift-3d/CMakeFiles/elliptic_shift_3d_native_fields_tests.dir/flags.make').read_text().splitlines():
 if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):flags+=shlex.split(line.split('=',1)[1].replace(str(root),str(out)))
command=['/usr/bin/c++',*flags,str(out/'tst/elliptic_shift_3d/native_wormhole_reference.cpp'),'-o',str(out/'native_wormhole_reference'),*map(str,(out/'build-shift-3d/kokkos').glob('*/src/libkokkos*.a'))]
(out/'compile-command.json').write_text(json.dumps(command,indent=2)+'\n')
files={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()};(out/'manifest.json').write_text(json.dumps({'source_commit':'dcd6bac9df1a2060b656201fd1ec367eccfa4d17','production_archive_sha256':sha(source),'production_binary_sha256':sha(binary),'compiled_headers_match_qualified_archive':True,'capsule_source_sha256':files,'independent_from_live_PDE_headers':True,'scope':'Fixed v14 native physics and headers; new control harness arguments only','compile_command':command},indent=2)+'\n');print('capsule files',len(files))

# Compile only the matched harness against this fixed source capsule.
start=time.monotonic()
with (out/'compile.stdout').open('w') as so,(out/'compile.stderr').open('w') as se:
 process=subprocess.Popen(command,stdout=so,stderr=se,start_new_session=True)
 try:code=process.wait(timeout=90)
 except subprocess.TimeoutExpired:
  assert os.getpgid(process.pid)==process.pid;os.killpg(process.pid,signal.SIGTERM)
  try:process.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
  code=124
cm=json.loads((out/'manifest.json').read_text());cm['compile']={'exit_code':code,'maximum_seconds':90,'seconds':time.monotonic()-start,'owned_process_group':process.pid}
if code==0:cm['matched_reference_binary_sha256']=sha(out/'native_wormhole_reference')
(out/'manifest.json').write_text(json.dumps(cm,indent=2)+'\n');assert code==0, 'capsule reference compile failed; see compile.stderr'
