#!/usr/bin/env python3
"""Guard-only native TU rebuild against frozen5c8916 headers; shared build untouched."""
import argparse,hashlib,json,shlex,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path.cwd();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);base=root/'docs/elliptic-shift-3d/evidence/native-adaptive-build-preflight-20261027';frozen=base/'source';source=root/'src/z4c/hyperboloidal/native_composite_evolution.cpp';own=out/'native_composite_evolution.cpp';own.write_bytes(source.read_bytes());build=root/'build-shift-3d/src';flags=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for line in (build/'CMakeFiles/athena.dir/flags.make').read_text().splitlines():
 if line.startswith(('CXX_DEFINES =','CXX_INCLUDES =','CXX_FLAGS =')):flags.extend(shlex.split(line.split('=',1)[1]))
obj=out/'native_composite_evolution.cpp.o';compile=['/usr/bin/c++','-I'+str(frozen/'src'),'-I'+str(frozen/'build-shift-3d'),*flags,'-c',str(own),'-o',str(obj)]
link=shlex.split((build/'CMakeFiles/athena.dir/link.txt').read_text());inputs=[]
for i,t in enumerate(link):
 if t.endswith(('native_composite_evolution.cpp.o',)):link[i]=str(obj)
 elif t.endswith(('.o','.a')):
  original=(build/t).resolve();link[i]=str(original);inputs.append({'path':str(original),'sha256':sha(original)})
 if t=='-o':link[i+1]=str(out/'athena-frozen')
# Probe all transitive includes, then archive any remaining project/vendor
# headers and replace EVERY live project include directory before compilation.
probe=['/usr/bin/c++','-I'+str(frozen/'src'),'-I'+str(frozen/'build-shift-3d'),*flags,'-M',str(own)]
first=subprocess.run(probe,cwd=build,capture_output=True,text=True,timeout=180);(out/'dependency-first.stdout').write_text(first.stdout);(out/'dependency-first.stderr').write_text(first.stderr);first.check_returncode()
def dependencies(text):return [Path(x).resolve() for x in shlex.split(text.replace('\\\n',''))[1:]]
archive=out/'project-headers';header_records=[]
for dep in dependencies(first.stdout):
 if dep.is_relative_to(root) and not dep.is_relative_to(frozen) and dep!=own:
  data=dep.read_bytes();target=archive/dep.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);header_records.append({'original':str(dep),'frozen':str(target),'sha256':sha(target)})
newflags=[]
for token in flags:
 if token.startswith('-I'+str(root)):token='-I'+str(archive/Path(token[2:]).relative_to(root))
 elif token.startswith(str(root)):token=str(archive/Path(token).relative_to(root))
 newflags.append(token)
compile_flags=['-I'+str(frozen/'src'),'-I'+str(frozen/'build-shift-3d'),*newflags]
second=['/usr/bin/c++',*compile_flags,'-M',str(own)];r=subprocess.run(second,cwd=build,capture_output=True,text=True,timeout=180);(out/'dependency-frozen.stdout').write_text(r.stdout);(out/'dependency-frozen.stderr').write_text(r.stderr);r.check_returncode()
live=[str(p) for p in dependencies(r.stdout) if p.is_relative_to(root) and not p.is_relative_to(frozen) and not p.is_relative_to(archive) and p!=own]
if live:raise RuntimeError('live project header leaked: '+str(live))
compile=['/usr/bin/c++',*compile_flags,'-c',str(own),'-o',str(obj)]
m={'scope':'Cyclic-zoom constructor guard only; original5c8916 physics, own native TU against immutable base engine and hashed Athena object/link inputs; no shared build mutation','base_build_manifest_sha256':sha(base/'manifest.json'),'base_frozen_source':str(frozen),'patched_source_sha256':sha(own),'shared_source_sha256':sha(source),'inputs':inputs,'maximum_seconds_per_process':180,'runs':{},'header_archive':header_records,'frozen_compile_flags':compile_flags,'dependency_probe_commands':[probe,second],'live_project_headers_after_freeze':live}
for label,cmd in [('compile',compile),('link',link)]:
 t=time.monotonic();r=subprocess.run(cmd,cwd=build,capture_output=True,text=True,timeout=180);(out/(label+'.stdout')).write_text(r.stdout);(out/(label+'.stderr')).write_text(r.stderr);m['runs'][label]={'command':cmd,'seconds':time.monotonic()-t,'exit_code':r.returncode};(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(label,r.returncode,flush=True)
 if r.returncode:raise RuntimeError(label+' failed')
m['vendor_headers_unchanged']=all(sha(Path(r['original']))==r['sha256'] and sha(Path(r['frozen']))==r['sha256'] for r in header_records);m['inputs_unchanged']=all(sha(Path(r['path']))==r['sha256'] for r in inputs);m['shared_source_unchanged']=sha(source)==m['shared_source_sha256'];m['binary_sha256']=sha(out/'athena-frozen');(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');assert m['inputs_unchanged'] and m['shared_source_unchanged'] and m['vendor_headers_unchanged'];print(m['binary_sha256'])
