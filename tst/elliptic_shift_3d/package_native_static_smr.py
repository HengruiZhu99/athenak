#!/usr/bin/env python3
"""Lossless static-SMR evidence packaging; originals retained, executables omitted."""
import argparse,gzip,hashlib,io,json,tarfile,time
from pathlib import Path
MAGIC={b'\xcf\xfa\xed\xfe',b'\xfe\xed\xfa\xcf',b'\xce\xfa\xed\xfe',b'\xfe\xed\xfa\xce',b'\xca\xfe\xba\xbe',b'\xbe\xba\xfe\xca'}
def sha(data):return hashlib.sha256(data).hexdigest()
def binary(path):
 with path.open('rb') as f:first=f.read(8)
 return first[:4] in MAGIC or first==b'!<arch>\n' or path.suffix in ['.o','.a','.dylib']
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path.cwd();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);base=root/'docs/elliptic-shift-3d/evidence';dirs=sorted([p for p in base.glob('native-static-smr-*') if p.is_dir() and p!=out]+[p for p in base.glob('native-static-leaves-*') if p.is_dir()]);summary={'scope':'All new stable native static-SMR/leaf evidence including failed attempts; all local originals retained; no MachO/object or private source/build tree staged','directories':[],'source_ownership':'No changes outside new native phase evidence/owned packaging script','started_unix':time.time()}
 for folder in dirs:
  patterns=set((folder/'.gitignore').read_text().splitlines()) if (folder/'.gitignore').exists() else set();archives=[];compressed=[];excluded=[]
  source_dirs=[folder/name for name in ['source','compiled-source','project-headers','control-source'] if (folder/name).is_dir()]
  for tree in source_dirs:
   members=[];raw=io.BytesIO()
   with tarfile.open(fileobj=raw,mode='w') as tar:
    for path in sorted(tree.rglob('*')):
     if not path.is_file():continue
     rel=path.relative_to(tree)
     if any(part in ['CMakeFiles','__pycache__','.git'] for part in rel.parts) or binary(path):
      excluded.append({'path':str(path.relative_to(folder)),'sha256':sha(path.read_bytes()),'reason':'private build/object/binary inside source tree'});continue
     data=path.read_bytes();info=tarfile.TarInfo(str(rel));info.size=len(data);info.mode=0o644;info.mtime=0;tar.addfile(info,io.BytesIO(data));members.append({'path':str(rel),'bytes':len(data),'sha256':sha(data)})
   packed=gzip.compress(raw.getvalue(),compresslevel=6,mtime=0);target=folder/(tree.name+'.tar.gz')
   if target.exists():assert target.read_bytes()==packed,'refusing different source archive'
   else:target.write_bytes(packed)
   with tarfile.open(fileobj=io.BytesIO(gzip.decompress(packed)),mode='r:') as tar:
    for row in members:assert sha(tar.extractfile(row['path']).read())==row['sha256']
   archives.append({'original_directory':tree.name,'archive':target.name,'sha256':sha(packed),'members':members,'roundtrip_verified':True});patterns.add('/'+tree.name+'/')
  for path in sorted(folder.rglob('*')):
   if not path.is_file() or any(path.is_relative_to(tree) for tree in source_dirs):continue
   rel=str(path.relative_to(folder))
   if binary(path):excluded.append({'path':rel,'bytes':path.stat().st_size,'sha256':sha(path.read_bytes()),'reason':'MachO/object/static-library; rebuild from source'});patterns.add('/'+rel);continue
   if any(part in ['__pycache__','CMakeFiles','.git'] for part in path.relative_to(folder).parts):patterns.add('/'+rel);continue
   scientific=path.suffix in ['.csv','.rst','.bin','.composite','.checkpoint'] or (path.suffix=='.dat' and path.stat().st_size>65536)
   large_log=path.suffix in ['.stdout','.stderr'] and path.stat().st_size>131072
   if not(scientific or large_log):continue
   data=path.read_bytes();target=path.with_name(path.name+'.gz');packed=gzip.compress(data,compresslevel=6,mtime=0)
   if target.exists():assert gzip.decompress(target.read_bytes())==data,'different pre-existing compressed evidence'
   else:target.write_bytes(packed)
   assert gzip.decompress(target.read_bytes())==data
   compressed.append({'original':rel,'compressed':str(target.relative_to(folder)),'bytes':len(data),'sha256':sha(data),'compressed_bytes':target.stat().st_size,'compressed_sha256':sha(target.read_bytes()),'lossless':True});patterns.add('/'+rel)
  (folder/'compressed-files.json').write_text(json.dumps(compressed,indent=2)+'\n');(folder/'source-archives.json').write_text(json.dumps(archives,indent=2)+'\n');(folder/'excluded-local-build-files.json').write_text(json.dumps(excluded,indent=2)+'\n');(folder/'.gitignore').write_text('\n'.join(sorted(patterns))+'\n')
  # Enumerate exactly the intended Git-facing package. Files under ignored raw
  # trees/binaries have verified gzip or source-archive counterparts above.
  kept=[];ignored={r['original'] for r in compressed}|{r['path'] for r in excluded}
  for path in sorted(folder.rglob('*')):
   if not path.is_file() or any(path.is_relative_to(tree) for tree in source_dirs):continue
   rel=str(path.relative_to(folder))
   if rel in ignored or any(part in ['__pycache__','CMakeFiles','.git'] for part in path.relative_to(folder).parts):continue
   assert not binary(path),'binary leaked into package';kept.append({'path':rel,'bytes':path.stat().st_size,'sha256':sha(path.read_bytes())})
  row={'directory':str(folder.relative_to(root)),'compressed_files':len(compressed),'source_archives':len(archives),'excluded_build_files':len(excluded),'raw_compressed_bytes':sum(r['bytes'] for r in compressed),'gzip_bytes':sum(r['compressed_bytes'] for r in compressed),'package_files':kept};summary['directories'].append(row);(out/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n');print(folder.name,len(compressed),'gzip',len(archives),'source archives',len(excluded),'excluded',flush=True)
 summary['total_package_files']=sum(len(r['package_files']) for r in summary['directories']);summary['total_package_bytes']=sum(f['bytes'] for r in summary['directories'] for f in r['package_files']);summary['finished_unix']=time.time();(out/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'directories':len(dirs),'package_files':summary['total_package_files'],'package_bytes':summary['total_package_bytes'],'manifest_sha256':sha((out/'manifest.json').read_bytes())}))
if __name__=='__main__':main()
