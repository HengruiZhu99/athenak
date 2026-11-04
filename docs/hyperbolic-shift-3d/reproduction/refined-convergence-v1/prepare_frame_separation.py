"""Package an immutable committed source plus explicit new launcher/build records.

No remote execution or submission. Kokkos archive/manifest reused byte-for-byte
from the already frozen RF2 package. Python tarfile avoids AppleDouble entries.
"""
import argparse,hashlib,io,json,pathlib,shutil,subprocess,tarfile
p=argparse.ArgumentParser();p.add_argument('destination');p.add_argument('--source-commit',required=True);a=p.parse_args()
HERE=pathlib.Path(__file__).resolve().parent;ROOT=next(p for p in HERE.parents if (p/'CMakeLists.txt').exists());out=pathlib.Path(a.destination).resolve();out.mkdir(parents=True,exist_ok=False)
commit=subprocess.check_output(['git','rev-parse',a.source_commit],cwd=ROOT,text=True).strip()
sha=lambda x:hashlib.sha256(x).hexdigest();record=lambda p:dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes()))
files={};names=subprocess.check_output(['git','ls-tree','-r','--name-only',commit],cwd=ROOT,text=True).splitlines()
names=[n for n in names if n in ('CMakeLists.txt','config.hpp.in') or n.startswith(('src/','tst/','inputs/hyperbolic_shift_3d/refined_convergence/'))]
with tarfile.open(out/'source.tar.gz','w:gz') as archive:
 for name in names:
  b=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT);entry=tarfile.TarInfo(name);entry.size=len(b);entry.mode=0o644;archive.addfile(entry,io.BytesIO(b));files[name]=dict(bytes=len(b),sha256=sha(b))
old=ROOT/'output/refined-convergence/aurora-package-v1';previous=json.loads((old/'source-manifest.json').read_text());shutil.copy2(old/'kokkos.tar.gz',out/'kokkos.tar.gz')
assert record(out/'kokkos.tar.gz')==previous['archives']['kokkos.tar.gz']
files.update({f:v for f,v in previous['files'].items() if f.startswith('kokkos/')})
manifest=dict(source_commit=commit,kokkos_commit=previous['kokkos_commit'],files=files,archives={n:record(out/n) for n in ['source.tar.gz','kokkos.tar.gz']})
(out/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
shutil.copytree(HERE/'launcher-v3',out/'launcher-v3',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(HERE/'launcher-v3/storage.py',out/'storage.py');shutil.copy2(HERE/'aurora-build/build.sh',out/'build.sh')
s=(HERE/'aurora-build/build.py').read_text();needle=" write('science.json',science);state.update"
extra=''' verifier_cmd=['c++','-O2','-std=c++17','launcher-v3/verify_checkpoint.cpp','-o','launcher-v3/verify-checkpoint']
 subprocess.run(verifier_cmd,check=True);os.chmod('launcher-v3/verify-checkpoint',0o755)
 assert os.access('launcher-v3/verify-checkpoint',os.X_OK)
 write('launcher-v3/verifier-build.json',dict(command=verifier_cmd,sha256=digest('launcher-v3/verify-checkpoint'),mode=oct(pathlib.Path('launcher-v3/verify-checkpoint').stat().st_mode&0o777)))
 launch={str(p.relative_to('launcher-v3')):dict(bytes=p.stat().st_size,sha256=digest(p),mode=oct(p.stat().st_mode&0o777)) for p in sorted(pathlib.Path('launcher-v3').iterdir()) if p.is_file() and p.name!='manifest.json'}
 write('launcher-v3/manifest.json',launch)
'''
assert needle in s;s=s.replace(needle,extra+needle);(out/'build.py').write_text(s)
records={str(p.relative_to(out)):record(p) for p in out.rglob('*') if p.is_file()};(out/'package-records.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(dict(destination=str(out),source_commit=commit,source_manifest_sha256=record(out/'source-manifest.json')['sha256'],files=len(files)),indent=2))
