import pathlib,tarfile,hashlib,json,sys
root=pathlib.Path('/home/hzhu/hyperboloidal-performance-scaling-20261103');job=int(sys.argv[1]);paths=[];manifest=[]
for dirname in ['gpu-job'+str(job),'job'+str(job)+'-components','evidence']:
 for p in sorted((root/dirname).rglob('*')):
  if not p.is_file():continue
  if 'bin' in p.relative_to(root).parts:continue
  if p.suffix in ['.checkpoint','.bin'] or p.stat().st_size>10*1024**2:
   h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(1024**2),b''):h.update(b)
   manifest.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=h.hexdigest()))
  else:paths.append(p)
for p in root.glob('hyp-perf-v1.[oe]*'):paths.append(p)
m=root/('job'+str(job)+'-large-artifacts.json');m.write_text(json.dumps(manifest,indent=2)+'\n');paths.append(m)
with tarfile.open(root/('job'+str(job)+'-evidence.tar.gz'),'w:gz') as t:
 for p in paths:t.add(p,arcname=str(p.relative_to(root)))
