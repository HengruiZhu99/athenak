#!/usr/bin/env python3
"""Build and run bounded one/two-rank generation fixtures; no evolution or submission."""
from pathlib import Path
import argparse,subprocess,time,json,hashlib
root=next(p for p in Path(__file__).resolve().parents if (p/'CMakeLists.txt').is_file())
parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=root/'output/hyperbolic-core-shell/mesh-plan-bundle-qualified-v1');args=parser.parse_args()
out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
binary=root/'build-hyperbolic-mpi/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mesh_plan_bundle'
sources=['src/athena.hpp','src/globals.hpp','tst/hyperbolic_shift_3d/test_mesh_plan_bundle.cpp','tst/hyperbolic_shift_3d/CMakeLists.txt','tst/hyperbolic_shift_3d/fixed_composite_mesh.hpp','tst/hyperbolic_shift_3d/test_sparse_stage_exchange.cpp']
for folder in ['src/z4c/hyperboloidal','src/mesh/cubed_sphere']:
 sources += [str(p.relative_to(root)) for p in sorted((root/folder).glob('*.hpp'))]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source_hashes={p:digest(root/p) for p in sources}
def run(name,cmd):
 start=time.monotonic();log=out/(name+'.log')
 with log.open('w') as stream:
  try:r=subprocess.run(cmd,cwd=root,stdout=stream,stderr=subprocess.STDOUT,timeout=420);status=dict(returncode=r.returncode,timed_out=False)
  except subprocess.TimeoutExpired:status=dict(returncode=None,timed_out=True)
 status.update(command=cmd,elapsed_seconds=time.monotonic()-start,log_sha256=digest(log),source_sha256=source_hashes)
 (out/(name+'.json')).write_text(json.dumps(status,indent=2)+'\n')
 print(json.dumps(dict(name=name,returncode=status["returncode"],timed_out=status["timed_out"],elapsed_seconds=status["elapsed_seconds"],source_files=len(source_hashes))),flush=True)
 if status['returncode']!=0:raise SystemExit(1)
 assert source_hashes=={p:digest(root/p) for p in sources},'Source changed during qualification'
 return status
run('configure',['cmake','-S',str(root),'-B',str(root/'build-hyperbolic-mpi')])
run('build',['cmake','--build',str(root/'build-hyperbolic-mpi'),'-j2','--target','hyperbolic_shift_3d_mesh_plan_bundle','hyperbolic_shift_3d_sparse_exchange'])
binary_hash=digest(binary)
for ranks in [1,2]:
 name=f'{ranks}-ranks';assert digest(binary)==binary_hash
 status=run(name,['/opt/homebrew/bin/mpiexec','-n',str(ranks),str(binary)])
 assert digest(binary)==binary_hash,'Binary changed during qualification'
 status['binary_sha256']=binary_hash;(out/(name+'.json')).write_text(json.dumps(status,indent=2)+'\n')

sparse_binary=binary.with_name('hyperbolic_shift_3d_sparse_exchange');sparse_hash=digest(sparse_binary)
status=run('sparse-2-ranks',['/opt/homebrew/bin/mpiexec','-n','2',str(sparse_binary)])
assert digest(sparse_binary)==sparse_hash
status['binary_sha256']=sparse_hash;(out/'sparse-2-ranks.json').write_text(json.dumps(status,indent=2)+'\n')
