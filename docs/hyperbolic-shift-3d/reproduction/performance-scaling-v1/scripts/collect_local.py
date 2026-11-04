#!/usr/bin/env python3
import pathlib,hashlib,json,tarfile
root=pathlib.Path('build-performance').resolve();dest=pathlib.Path('docs/hyperbolic-shift-3d/reproduction/performance-scaling-v1/evidence');dest.mkdir(exist_ok=True)
folders=['cpu-bench','cpu-m4-bench','cpu-m4-replacement','cpu-m4-clean','gpu-rejection','baseline-matrix','m0-restart','qualification','final-check','evidence','m4-bin']
large=[];small=[]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
for name in folders:
 d=root/name
 if not d.exists():continue
 for p in sorted(d.rglob('*')):
  if not p.is_file():continue
  if p.suffix in ['.checkpoint','.bin','.o'] or p.stat().st_size>5*1024**2 or 'bin' in p.relative_to(root).parts or p.name in ['athena','evolution.bin']:
   large.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p)))
  else:small.append(p)
for p in sorted(root.glob('*.tar.gz')):
 large.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p)))
for p in sorted((root/'frozen').glob('*')):
 if p.is_file():large.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p)))
(dest/'local-large-artifacts.json').write_text(json.dumps(large,indent=2)+'\n')
with tarfile.open(dest/'cpu-evidence.tar.gz','w:gz') as tar:
 for p in small:tar.add(p,arcname=str(p.relative_to(root)))
print(len(small),'small files;',len(large),'large artifact receipts')
