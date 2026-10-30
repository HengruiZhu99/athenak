#!/usr/bin/env python3
"""Fixed-layout lifecycle tests against complete frozen-control checkpoint files."""
from pathlib import Path
import argparse,subprocess,hashlib,json,time,os
root=next(p for p in Path(__file__).resolve().parents if (p/'CMakeLists.txt').is_file())
p=argparse.ArgumentParser()
for name in ['binary','legacy','initial','mpiexec','parity-dir','output-dir']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);cases=a.parity_dir.resolve()/'cases'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources=[root/'tst/hyperbolic_shift_3d'/name for name in ['CMakeLists.txt','coupled_mpi.cpp','fixed_composite_mesh.hpp']]
for folder in ['src/z4c/hyperboloidal','src/mesh/cubed_sphere']:sources+=sorted((root/folder).glob('*.hpp'))
pins={str(q.relative_to(root)):sha(q) for q in sources};bins={str(q.resolve()):sha(q) for q in [a.binary,a.legacy]}
assert json.loads((a.parity_dir/'cases/verification.json').read_text())['status']=='passed'
records=[]
def run(name,reference,kind='pulse',end=.00075,restart='-',initial='-',legacy=False,failure=False):
 prefix=out/name;ref=cases/reference;n=16 if kind=='puncture' else 12
 cmd=[str(a.mpiexec.resolve()),'-n','2',str((a.legacy if legacy else a.binary).resolve()),str(n),str(prefix),kind,str(end),'.00025',str(initial),str(restart),'140','.001','.001','6','12','2']
 start=time.monotonic()
 with prefix.with_suffix('.log').open('w') as log:
  try:r=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=150,env=dict(os.environ,HYPERBOLIC_SPARSE_POISON='1',HYPERBOLIC_PLAN_REBUILD_INTERVAL='0' if legacy else '1'));code=r.returncode
  except subprocess.TimeoutExpired:code=None
 record=dict(name=name,command=cmd,returncode=code,seconds=time.monotonic()-start,environment=dict(HYPERBOLIC_SPARSE_POISON='1',HYPERBOLIC_PLAN_REBUILD_INTERVAL='0' if legacy else '1'))
 records.append(record);(out/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
 assert code==(2 if failure else 0),(name,code)
 report=json.loads(prefix.with_suffix('.report.json').read_text())
 assert report['status']==('failed' if failure else 'completed')
 assert report['checkpoint_admitted']==(not failure)
 if not legacy:
  assert report['mesh_plan_lifetime']=='coordinated_generation'
  expected=1 if failure else round((end-(.00025 if restart!='-' else 0))/.00025)
  assert report['forced_plan_rebuilds']==expected and report['mesh_plan_generations']==expected+1,report
  assert report['plan_rebuild_interval']==1
 else:assert report['mesh_plan_lifetime']=='independent_fixed'
 assert report['lapse_floor_node_interventions']==0
 assert report['max_null_surface']<5e-13 and report['max_null_ghost']<5e-13
 for suffix in (['.checkpoint','.failed-stage.bin'] if failure else ['.checkpoint']):
  current=prefix.with_suffix(suffix);frozen=ref.with_suffix(suffix)
  assert current.read_bytes()==frozen.read_bytes(),(name,suffix,'trajectory differs')
  record[suffix]=dict(sha256=sha(current),reference=str(frozen),reference_sha256=sha(frozen))
 assert pins=={str(q.relative_to(root)):sha(q) for q in sources},'source changed'
 assert bins=={str(q.resolve()):sha(q) for q in [a.binary,a.legacy]},'binary changed'
 print(json.dumps(dict(name=name,seconds=record['seconds'],generations=report.get('mesh_plan_generations'),status='passed')),flush=True)
run('legacy-refactored','control',legacy=True)
run('pulse-rebuild','control')
run('restart-rebuild','control',restart=cases/'part.checkpoint')
run('minkowski-rebuild','minkowski-control',kind='minkowski',end=.0005)
run('puncture-rebuild','puncture-control',kind='puncture',end=.0005,initial=a.initial.resolve())
run('failure-rebuild','failure-control',end=.00025,initial=cases/'stress.txt',failure=True)
(out/'verification.json').write_text(json.dumps(dict(status='passed',scope='fixed-layout plan replacement with unchanged physical fields and compensation; no regridding',source_sha256=pins,binary_sha256=bins,commands=records),indent=2)+'\n')
