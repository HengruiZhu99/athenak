from pathlib import Path
import hashlib,json,subprocess,time
p=Path(__file__).resolve().parent;repo=p.parents[2]
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
inputs=[p/'constraint_tangent.cpp',p/'subsidiary.hpp',p/'DERIVATION.md']
inputs+=list((repo/'src/z4c/hyperboloidal').glob('*.hpp'))
before={str(f.relative_to(repo)):sha(f) for f in inputs}
cmd=['/usr/bin/c++','-std=c++17','-O3','-DNDEBUG','-DKOKKOS_DEPENDENCE','-Isrc','-Ibuild-layer-release','-Ibuild-layer-release/kokkos','-Ibuild-layer-release/kokkos/core/src','-Ikokkos/core/src','-Ibuild-layer-release/kokkos/containers/src','-Ikokkos/containers/src','-Ibuild-layer-release/kokkos/algorithms/src','-Ikokkos/algorithms/src','-Ibuild-layer-release/kokkos/simd/src','-Ikokkos/simd/src','-isystem','kokkos/tpls/desul/include','-isystem','kokkos/tpls/mdspan/include',str(p/'constraint_tangent.cpp'),'-o',str(p/'constraint_tangent')]
results=[];t=time.monotonic()
r=subprocess.run(cmd,cwd=repo,text=True,capture_output=True);results.append({'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr});r.check_returncode()
for kappa in [5,10]:
 f=p/f'tangent-kappa{kappa}.json';command=[str(p/'constraint_tangent'),str(kappa)]
 with f.open('w') as stream:r=subprocess.run(command,cwd=repo,text=True,stdout=stream,stderr=subprocess.PIPE)
 results.append({'command':command,'returncode':r.returncode,'stdout_file':str(f.relative_to(repo)),'stdout_sha256':sha(f),'stderr':r.stderr});r.check_returncode()
for kappa in [5,10]:
 f=p/f'generator-kappa{kappa}.json';command=[str(p/'constraint_tangent'),str(kappa),'--generator-only']
 with f.open('w') as stream:r=subprocess.run(command,cwd=repo,text=True,stdout=stream,stderr=subprocess.PIPE)
 results.append({'command':command,'returncode':r.returncode,'stdout_file':str(f.relative_to(repo)),'stdout_sha256':sha(f),'stderr':r.stderr});r.check_returncode()
receipt={'scope':'Exploratory continuum C_Z4c0 subsidiary identity and coefficient-gradient audit; no production changes/global stability/nonlinear scri closure.','launch_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'compiled_runtime_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','source_before':before,'source_after':{str(f.relative_to(repo)):sha(f) for f in inputs},'commands':results,'compiler':subprocess.check_output(['/usr/bin/c++','--version'],text=True),'binary_sha256':sha(p/'constraint_tangent'),'seconds':time.monotonic()-t}
assert receipt['source_before']==receipt['source_after'];(p/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS actual dual kernel/source hashes',receipt['seconds'])
