#!/usr/bin/env python3
"""Freeze actual Athena compiler dependencies without mutating runtime receipts."""
import argparse,hashlib,json,shlex
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('build_evidence',type=Path);a=p.parse_args();root=Path.cwd();out=a.build_evidence.resolve();manifest=json.loads((out/'manifest.json').read_text());deps=set();dependency_files=[];sha=lambda data:hashlib.sha256(data).hexdigest()
for f in sorted((root/'build-shift-3d/src/CMakeFiles/athena.dir').rglob('*.o.d')):
 data=f.read_bytes();dependency_files.append({'path':str(f.relative_to(root)),'sha256':sha(data)})
 for value in shlex.split(data.decode().replace('\\\n',''))[1:]:
  file=Path(value);file=file if file.is_absolute() else root/'build-shift-3d/src'/file;file=file.resolve()
  if file.is_relative_to(root) and file.is_file():deps.add(file)
for f in [root/'CMakeLists.txt',root/'src/CMakeLists.txt',root/'build-shift-3d/config.hpp']:deps.add(f)
records=[]
for f in sorted(deps):
 rel=f.relative_to(root);frozen=out/'source'/rel;known=str(rel) in manifest['source_hashes_before'];data=frozen.read_bytes() if frozen.exists() else f.read_bytes()
 if known:assert sha(data)==manifest['source_hashes_before'][str(rel)],str(rel)
 target=out/'compiled-source'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);records.append({'path':str(rel),'sha256':sha(data),'recorded_prepost_source':known,'source':'verified build snapshot' if known else 'exact vendor/generated dependency content'})
result={'build_manifest_sha256':sha((out/'manifest.json').read_bytes()),'binary_sha256':manifest['binary_sha256'],'actual_dependency_files':dependency_files,'compiled_sources':records,'source_before_after_verified':sum(r['recorded_prepost_source'] for r in records),'main_runtime_manifest_unmodified':True};(out/'actual-compiled-source-manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'dependency_files':len(dependency_files),'source_files':len(records),'prepost_verified':result['source_before_after_verified']}))
