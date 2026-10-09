"""Compile local wide supplement, no matrix/global/native evolution."""
from pathlib import Path
import hashlib,json,os,re,shlex,subprocess,time
root=Path(__file__).resolve().parents[3];w=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
flags=(root/'build-layer-release/CMakeFiles/hyperboloidal_layer_constraint_tangent.dir/flags.make').read_text()
makevar=lambda k:shlex.split(re.search(r'^'+k+r' = (.*)$',flags,re.M).group(1))
common=['/usr/bin/c++','-I'+str(w/'overlay'),'-I'+str(w)]+makevar('CXX_DEFINES')+makevar('CXX_INCLUDES')+makevar('CXX_FLAGS')
libs=[str(root/f'build-layer-release/kokkos/{p}/src/libkokkos{p}.a') for p in ['containers','algorithms','core','simd']]
receipt={'compiler_version':subprocess.check_output(['/usr/bin/c++','--version'],text=True),'builds':[],'tests':[]}
def run(cmd,log):
 t=time.monotonic();p=subprocess.run(cmd,cwd=root,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0'},capture_output=True,text=True)
 (w/log).write_text(p.stdout+p.stderr);r={'command':cmd,'seconds':time.monotonic()-t,'exit_status':p.returncode}
 if p.returncode:raise RuntimeError(log+': '+p.stdout+p.stderr)
 return r
for mode in ['release','asan']:
 cmd=common.copy()
 if mode=='asan':cmd+=[ '-O0','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer']
 exe=w/f'local-gate-{mode}';source=w/'local_gate.cpp'
 b=run(cmd+[str(source),'-o',str(exe)]+libs,f'build-local-{mode}.log')
 b.update({'source_sha256':sha(source),'executable_sha256':sha(exe),'mode':mode});receipt['builds'].append(b)
 dep=run(cmd+[str(source),'-M','-MT','audit'],f'dependencies-{mode}.make')
 text=(w/f'dependencies-{mode}.make').read_text().replace('\\\n',' ')
 paths=shlex.split(text.split(':',1)[1]);b['compiler_dependency_hashes']={str(Path(p).resolve()):sha(Path(p)) for p in paths};b['link_archives']={p:sha(Path(p)) for p in libs}
 test=run([str(exe)],f'test-local-{mode}.log');receipt['tests'].append(test);print(mode,(w/f'test-local-{mode}.log').read_text().strip(),flush=True)
 (w/'local-build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
exe=w/'export-radial';source=w/'export_radial.cpp';b=run(common+[str(source),'-o',str(exe)]+libs,'build-export.log');b.update({'source_sha256':sha(source),'executable_sha256':sha(exe)});receipt['builds'].append(b)
(w/'local-build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
