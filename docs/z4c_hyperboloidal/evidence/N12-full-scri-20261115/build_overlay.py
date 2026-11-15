#!/usr/bin/env python3
"""Build an isolated SYCL overlay; reuse only objects with unchanged dependencies."""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, re, shlex, subprocess
p=argparse.ArgumentParser()
for key in ['source','baseline-source','baseline-build','output']:p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--jobs',type=int,default=4)
a=p.parse_args();src=a.source.resolve();oldsrc=a.baseline_source.resolve();old=a.baseline_build.resolve();out=a.output.resolve();out.mkdir(exist_ok=False)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
flags=(old/'src/CMakeFiles/athena.dir/flags.make').read_text()
def opts(key):return shlex.split(re.search(r'^'+key+r' = (.*)$',flags,re.M)[1])
link=shlex.split((old/'src/CMakeFiles/athena.dir/link.txt').read_text())
compiler=link[0];cxx=opts('CXX_FLAGS');defines=opts('CXX_DEFINES')
includes=[x.replace(str(oldsrc),str(src)) for x in opts('CXX_INCLUDES')]
assert (src/'kokkos').resolve()==(oldsrc/'kokkos').resolve()
# Use precisely the selected build's generated platform/configuration headers.
manifest_files=sorted(set(list((src/'src/z4c_hyperboloidal').glob('*.hpp'))+list((src/'src/z4c_hyperboloidal').glob('*.cpp'))+list((src/'src/mesh').glob('chart_atlas.*'))+list((src/'src/outputs').glob('scri.*'))))
manifest=''.join(sha(f)+'  '+str(f.relative_to(src))+'\n' for f in manifest_files)
manifest+=(old/'config.hpp').read_text()+'\nCXX_FLAGS='+shlex.join(cxx)+'\n'
(out/'scri_provenance.hpp').write_text('#define SCRI_SOURCE_MANIFEST R"scri('+manifest+')scri"\n')
source_hash={str(f.relative_to(src)):sha(f) for f in (src/'src').rglob('*') if f.is_file()}
changes=[];reused={};commands=[];newlink=[];jobs=[]
for token in link:
 if not token.endswith('.o'):newlink.append(token);continue
 obj=old/'src'/token
 dep=Path(str(obj)+'.d')
 if not dep.exists():raise RuntimeError('missing dependency evidence '+str(dep))
 dependencies=shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1])
 changed=[]
 for name in dependencies:
  f=Path(name)
  if f.is_relative_to(oldsrc):
   rel=f.relative_to(oldsrc)
   if str(rel).startswith('kokkos/'):continue
   candidate=src/rel
   if not candidate.is_file() or sha(f)!=sha(candidate):changed.append(str(rel))
 source=token.removeprefix('CMakeFiles/athena.dir/').removesuffix('.o')
 if changed:
  dest=out/'objects'/Path(source+'.o');dest.parent.mkdir(parents=True,exist_ok=True)
  cmd=[compiler,*cxx,*defines,'-I'+str(out),*includes,'-c',str(src/'src'/source),'-o',str(dest)]
  commands.append(cmd);jobs.append((cmd,out/'objects'/Path(source+'.log')))
  changes.append({'object':str(obj),'source':source,'changed_dependencies':sorted(set(changed))});newlink.append(str(dest))
 else:reused[str(obj)]=sha(obj);newlink.append(str(obj))
source='outputs/scri.cpp';dest=out/'objects'/Path(source+'.o');dest.parent.mkdir(parents=True,exist_ok=True)
cmd=[compiler,*cxx,*defines,'-I'+str(out),*includes,'-c',str(src/'src'/source),'-o',str(dest)]
commands.append(cmd);jobs.append((cmd,out/'objects'/Path(source+'.log')));newlink.insert(newlink.index('-o'),str(dest))
newlink[newlink.index('-o')+1]=str(out/'athena')
receipt={'source_sha256':source_hash,'rebuilt':changes,'reused_objects':reused,'commands':commands,'link':newlink}
(out/'build-plan.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('Rebuilding',len(jobs),'translation units; reusing',len(reused),'unchanged objects',flush=True)
def compile(job):
 cmd,log=job
 with log.open('w') as f:result=subprocess.run(cmd,cwd=old/'src',stdout=f,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError('compile failed: '+str(log))
 print('Built',cmd[-1],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:list(pool.map(compile,jobs))
with (out/'link.log').open('w') as f:subprocess.run(newlink,cwd=old/'src',stdout=f,stderr=subprocess.STDOUT,check=True)
assert all(sha(Path(f))==h for f,h in reused.items())
assert all(sha(src/f)==h for f,h in source_hash.items())
receipt.update(executable_sha256=sha(out/'athena'),baseline_unchanged=True,source_unchanged=True)
(out/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('OVERLAY_BUILD_COMPLETE',receipt['executable_sha256'],flush=True)
