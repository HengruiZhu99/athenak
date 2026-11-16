#!/usr/bin/env python3
"""Dependency-audited passive overlay on the existing validated SYCL build."""
import hashlib,json,pathlib,subprocess,sys,time
root=pathlib.Path(sys.argv[1]).resolve();base=pathlib.Path(sys.argv[2]).resolve()
out=root/'build';out.mkdir(exist_ok=False);source=root/'source'
plan=json.loads((base/'build-final/build-plan.json').read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
changed=[]
for name,value in plan['source_sha256'].items():
 p=source/name
 assert p.is_file(),name
 if sha(p)!=value:changed.append(name)
assert changed==['src/outputs/scri.cpp'],changed
# Only the recorder includes the new diagnostic header. Everything numerical
# remains the exact existing object, checked by content at both ends of build.
link=plan['link'];objects=[pathlib.Path(x) for x in link if x.endswith('.o')]
receipt={'baseline':str(base),'changed_sources':changed,'objects_before':{str(p):sha(p) for p in objects},'commands':[]}
manifest={str(p.relative_to(source)):sha(p) for p in sorted((source/'src').rglob('*')) if p.is_file()}
(out/'scri_provenance.hpp').write_text('#define SCRI_SOURCE_MANIFEST R"scri('+json.dumps(manifest,sort_keys=True,separators=(',',':'))+')scri"\n')
command=next(c for c in plan['commands'] if any(x.endswith('/outputs/scri.cpp') for x in c))
# Keep compiler/options/dependency library paths exactly; replace source/build.
command=[x.replace(str(base/'source'),str(source)).replace(str(base/'build-final'),str(out)) for x in command]
output=out/'objects/outputs/scri.cpp.o';output.parent.mkdir(parents=True)
command=[x.replace(str(source/'kokkos'),str(base/'source/kokkos')) for x in command]
command[command.index('-o')+1]=str(output)
start=time.monotonic()
for name,src in [('athena',source/'src/outputs/scri.cpp'),('nonradiative_seed',source/'tst/z4c_hyperboloidal/nonradiative_seed.cpp')]:
 obj=output if name=='athena' else out/'seed.o'
 compile_cmd=command.copy();compile_cmd[compile_cmd.index('-c')+1]=str(src);compile_cmd[compile_cmd.index('-o')+1]=str(obj)
 receipt['commands'].append(compile_cmd);subprocess.run(compile_cmd,check=True)
 actual=[str(output) if x.endswith('/outputs/scri.cpp.o') else x for x in link]
 if name!='athena':actual=[str(obj) if x.endswith('/main.cpp.o') else x for x in actual]
 actual[actual.index('-o')+1]=str(out/name)
 # Original link has relative Kokkos archive paths, interpreted in its base build.
 cwd=next(p.parent.parent.parent for p in objects if 'build-perf-pad32' in str(p) and p.name=='main.cpp.o')
 receipt['commands'].append(actual);subprocess.run(actual,cwd=cwd,check=True)
receipt['seconds']=time.monotonic()-start
receipt['objects_after']={str(p):sha(p) for p in objects};assert receipt['objects_before']==receipt['objects_after']
receipt['source_sha256']=manifest
receipt['executables']={n:sha(out/n) for n in ['athena','nonradiative_seed']}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'seconds':receipt['seconds'],'executables':receipt['executables']}))
