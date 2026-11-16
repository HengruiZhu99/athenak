#!/usr/bin/env python3
"""Rebuild the three host-reference consumers; reuse all bulk evolution objects."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
root,baseline=map(lambda x:Path(x).resolve(),sys.argv[1:3]);source=root/'source';out=root/'build';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=json.loads((baseline/'build/receipt.json').read_text())
manifest={str(p.relative_to(source)):sha(p) for p in sorted((source/'src').rglob('*')) if p.is_file()}
changed=[p for p,h in old['source_sha256'].items() if manifest[p]!=h]
assert changed==['src/z4c_hyperboloidal/spherical_reference_3d.hpp'],changed
(out/'scri_provenance.hpp').write_text('#define SCRI_SOURCE_MANIFEST R"scri('+json.dumps(manifest,sort_keys=True,separators=(',',':'))+')scri"\n')
template=[x.replace(str(baseline/'source'),str(source)).replace(str(baseline/'build'),str(out)) for x in old['commands'][0]]
# The numerical Kokkos/MPI library configuration stays pinned to the old build.
template=[x.replace(str(source/'kokkos'),str(baseline/'source/kokkos')) for x in template]
objects=[x for x in old['commands'][1] if x.endswith('.o')]
before={x:sha(x) for x in objects};commands=[];replacement={};start=time.monotonic()
for name in ['z4c_hyperboloidal/z4c_hyperboloidal.cpp','z4c_hyperboloidal/horizon_sampling.cpp','outputs/scri.cpp']:
    obj=out/'objects'/(name+'.o');obj.parent.mkdir(parents=True,exist_ok=True)
    cmd=template.copy();cmd[cmd.index('-c')+1]=str(source/'src'/name);cmd[cmd.index('-o')+1]=str(obj)
    subprocess.run(cmd,check=True);commands.append(cmd)
    matches=[x for x in objects if x.endswith('/'+name+'.o')];assert len(matches)==1,matches
    replacement[matches[0]]=str(obj)
link=[replacement.get(x,x) for x in old['commands'][1]];link[link.index('-o')+1]=str(out/'athena')
cwd=baseline.parent/'hybrid-overlap-20261111/build-perf-pad32-20261114/src'
subprocess.run(link,cwd=cwd,check=True);commands.append(link)
assert before=={x:sha(x) for x in objects}
receipt={'changed_sources':changed,'source_sha256':manifest,'commands':commands,
    'replaced_objects':replacement,'original_objects_unchanged':before,
    'executables':{'athena':sha(out/'athena')},'seconds':time.monotonic()-start}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'seconds':receipt['seconds'],'executable':receipt['executables']}))
