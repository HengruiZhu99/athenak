#!/usr/bin/env python3
"""Run the existing bounded MPI fidelity matrix against a separately frozen control."""
from pathlib import Path
import argparse,subprocess,hashlib,json,time,os
root=next(p for p in Path(__file__).resolve().parents if (p/'CMakeLists.txt').is_file())
p=argparse.ArgumentParser()
for name in ['binary','control','initial','mpiexec','output-dir']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources=[root/'tst/hyperbolic_shift_3d'/name for name in ['CMakeLists.txt','coupled_mpi.cpp','fixed_composite_mesh.hpp','test_sparse_stage_mpi.py']]
for folder in ['src/z4c/hyperboloidal','src/mesh/cubed_sphere']:sources+=sorted((root/folder).glob('*.hpp'))
pins={str(q.relative_to(root)):sha(q) for q in sources};bins={str(q.resolve()):sha(q) for q in [a.binary,a.control]}
cmd=['python3',str(root/'tst/hyperbolic_shift_3d/test_sparse_stage_mpi.py')]
for name in ['binary','control','initial','mpiexec']:cmd+=['--'+name,str(getattr(a,name).resolve())]
cmd+=['--output-dir',str(out/'cases')]
start=time.monotonic()
with (out/'run.log').open('w') as log:
 try:r=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=420,env=dict(os.environ,HYPERBOLIC_PLAN_REBUILD_INTERVAL='0'));code=r.returncode
 except subprocess.TimeoutExpired:code=None
receipt=dict(command=cmd,returncode=code,seconds=time.monotonic()-start,source_sha256=pins,binary_sha256=bins,log_sha256=sha(out/'run.log'))
(out/'run.json').write_text(json.dumps(receipt,indent=2)+'\n')
assert pins=={str(q.relative_to(root)):sha(q) for q in sources},'source changed during run'
assert bins=={str(q.resolve()):sha(q) for q in [a.binary,a.control]},'binary changed during run'
assert code==0,(code,(out/'run.log').read_text())
for name in ['one','two','part','restart','minkowski-fast','puncture-fast','failure-sparse']:
 report=json.loads((out/'cases'/f'{name}.report.json').read_text())
 assert report['mesh_plan_lifetime']=='coordinated_generation'
 assert report['mesh_plan_generations']==1 and report['forced_plan_rebuilds']==0
print('PASS: frozen-control MPI/restart/failure fidelity and one persistent generation in every candidate case',flush=True)
