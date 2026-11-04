"""Separate local candidate build; frozen production executable is never overwritten."""
from pathlib import Path
import hashlib,json,shlex,subprocess
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists())
O=R/'output/refined-convergence/sealed-topology-candidate-v1';O.mkdir(parents=True,exist_ok=False)
T=R/'build-refined-convergence/tst/hyperbolic_shift_3d';target=T/'CMakeFiles/hyperbolic_shift_3d_refined_amr.dir';flags={}
for line in (target/'flags.make').read_text().splitlines():
 if ' = ' in line:k,v=line.split(' = ',1);flags[k]=shlex.split(v)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
production=T/'hyperbolic_shift_3d_refined_amr';before=sha(production);jobs=[];records=[]
for name,source in [('support','test_amr_sealed_topology.cpp'),('evolve','amr_evolution.cpp')]:
 src=R/'tst/hyperbolic_shift_3d'/source;obj=O/(name+'.o');binary=O/name
 command=['/usr/bin/c++']+flags['CXX_DEFINES']+flags['CXX_INCLUDES']+flags['CXX_FLAGS']+['-I'+str(R/'tst/hyperbolic_shift_3d'),'-c',str(src),'-o',str(obj)]
 log=(O/(name+'-compile.log')).open('w');process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
 jobs.append((name,obj,binary,process,log));records.append(dict(name=name,compile=command,source_sha256=sha(src)))
for (name,obj,binary,process,log),record in zip(jobs,records):
 assert process.wait()==0,name+' compile failed';log.close()
 link=shlex.split((target/'link.txt').read_text());link[link.index('CMakeFiles/hyperbolic_shift_3d_refined_amr.dir/amr_evolution.cpp.o')]=str(obj);link[link.index('-o')+1]=str(binary)
 with (O/(name+'-link.log')).open('w') as f:subprocess.run(link,cwd=T,stdout=f,stderr=subprocess.STDOUT,check=True)
 record.update(link=link,binary_sha256=sha(binary))
assert sha(production)==before
headers=['tst/hyperbolic_shift_3d/refined_configuration.hpp','src/z4c/hyperboloidal/cartesian_amr.hpp','tst/hyperbolic_shift_3d/coupled_mpi.cpp','src/mesh/cubed_sphere/composite.hpp','src/mesh/cubed_sphere/composite_difference.hpp']
result=dict(targets=records,production_binary_sha256_unchanged=before,headers={p:sha(R/p) for p in headers},qualification='Separate candidate binaries; reused unchanged native static libraries, no remote production update')
libs=[(T/'libhyperbolic_athenak_amr.a').resolve()]+list((R/'build-refined-convergence/kokkos').rglob('libkokkos*.a'))
result['reused_static_libraries']={str(p.resolve()):sha(p) for p in libs}
(O/'build.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(completed=True,targets={x['name']:x['binary_sha256'] for x in records},production_binary_unchanged=True)))
