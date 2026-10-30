#!/usr/bin/env python3
"""Bounded tests of live ownership changes, accepted state transfer and recovery."""
from pathlib import Path
import argparse,subprocess,hashlib,json,time,os,csv,math
root=next(p for p in Path(__file__).resolve().parents if (p/'CMakeLists.txt').is_file())
p=argparse.ArgumentParser()
for name in ['binary','transaction','control','initial','mpiexec','reference-dir','output-dir']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);reference=a.reference_dir.resolve()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources=[root/'tst/hyperbolic_shift_3d'/name for name in ['CMakeLists.txt','coupled_mpi.cpp','fixed_composite_mesh.hpp','test_repartition_transaction.cpp']]
for folder in ['src/z4c/hyperboloidal','src/mesh/cubed_sphere']:sources+=sorted((root/folder).glob('*.hpp'))
pins={str(q.relative_to(root)):sha(q) for q in sources};bins={str(q.resolve()):sha(q) for q in [a.binary,a.transaction,a.control]};initial_hash=sha(a.initial)
assert sha(a.control)=='e4137758a1315a2e6b9fc033fc8d1bb2e9ac32de948c3a98db7848acd6e43353','use the preserved V24 control'
records=[]
def pinned():
 assert pins=={str(q.relative_to(root)):sha(q) for q in sources},'source changed during test'
 assert bins=={str(q.resolve()):sha(q) for q in [a.binary,a.transaction,a.control]},'binary changed during test'
 assert initial_hash==sha(a.initial),'initial data changed'
def launch(name,cmd,env,limit):
 pinned();start=time.monotonic()
 with (out/(name+'.log')).open('w') as log:
  try:r=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=limit,env=dict(os.environ,**env));code=r.returncode
  except subprocess.TimeoutExpired:code=None
 record=dict(name=name,command=cmd,environment=env,returncode=code,seconds=time.monotonic()-start,log_sha256=sha(out/(name+'.log')))
 records.append(record);(out/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
 pinned();assert code==0,(name,code,(out/(name+'.log')).read_text())
 return record
record=launch('transaction',[str(a.mpiexec.resolve()),'-n','2',str(a.transaction.resolve()),str(out/'transaction')],{'HYPERBOLIC_SPARSE_POISON':'1'},420)
assert 'PASS: one-rank invalid repartition recovered;' in (out/'transaction.log').read_text()
record['recovered_checkpoint_sha256']=sha(out/'transaction.recovered.checkpoint');record['failed_checkpoint_sha256']=sha(out/'transaction.failed.checkpoint')
print(json.dumps(dict(name='transaction',status='passed',seconds=record['seconds'])),flush=True)
def run(name,kind='pulse',end=.00075,restart='-',control=False,rebuild=False,expected=None):
 prefix=out/name;n=16 if kind=='puncture' else 12
 cmd=[str(a.mpiexec.resolve()),'-n','1' if control else '2',str((a.control if control else a.binary).resolve()),str(n),str(prefix),kind,str(end),'.00025',str(a.initial.resolve()) if kind=='puncture' else '-',str(restart),'140','.00025','.001','6','12','2']
 env=dict(HYPERBOLIC_SPARSE_POISON='1',HYPERBOLIC_PLAN_REBUILD_INTERVAL='1' if rebuild else '0',HYPERBOLIC_PLAN_REPARTITION_INTERVAL='0' if control else '1')
 record=launch(name,cmd,env,150);report=json.loads(prefix.with_suffix('.report.json').read_text());cp=prefix.with_suffix('.checkpoint')
 assert report['status']=='completed' and report['checkpoint_admitted'] and report['diagnostics_current']
 assert report['t']==end and report['lapse_floor_node_interventions']==0
 assert report['max_null_surface']<5e-13 and report['max_null_ghost']<5e-13
 if not control:
  count=round(end/.00025)-max(1,round(report['start_time']/.00025))
  assert report['ownership_changes']==count and report['rejected_repartitions']==0
  assert report['mesh_plan_generations']==1+count+report['forced_plan_rebuilds']
  assert report['ownership_partition']=='repartitioned_contiguous'
  assert report['ownership_change_times']==[.00025*j for j in range(max(1,round(report['start_time']/.00025)),round(end/.00025))]
  assert report['ownership_history'][0]==[0,report['cells']//10,report['cells']]
  assert report['ownership_history'][-1]==report['ownership_cuts']
  if count>1:assert report['ownership_history'][1]==[0,report['cells']//2,report['cells']]
  if kind=='puncture':
   rows=list(csv.DictReader(prefix.with_suffix('.history.csv').open()))
   assert len(rows)==3 and all(int(row['horizon_found'])==1 and int(row['horizon_complete_surface'])==1 for row in rows)
   frozen=json.loads((out/'puncture-control.report.json').read_text())
   for key in ['horizon_radius','horizon_areal_radius','horizon_mass','horizon_expansion_rms','H_exterior_L2','M_exterior_L2']:
    assert math.isclose(report[key],frozen[key],rel_tol=2e-12,abs_tol=1e-14),(key,report[key],frozen[key])
 record['checkpoint_sha256']=sha(cp)
 if expected:
  assert cp.read_bytes()==expected.read_bytes(),(name,'complete checkpoint differs')
  record['reference']=str(expected);record['reference_sha256']=sha(expected)
 print(json.dumps(dict(name=name,status='passed',seconds=record['seconds'],ownership_changes=report.get('ownership_changes',0))),flush=True)
 return cp
half_control=run('half-control',end=.0005,control=True)
half=run('half-repartition',end=.0005,expected=half_control)
run('restart-repartition',restart=half,expected=reference/'control.checkpoint')
run('pulse-repartition',expected=reference/'control.checkpoint')
run('minkowski-repartition',kind='minkowski',end=.0005,expected=reference/'minkowski-control.checkpoint')
run('puncture-control',kind='puncture',end=.0005,control=True,expected=reference/'puncture-control.checkpoint')
run('puncture-repartition',kind='puncture',end=.0005,expected=reference/'puncture-control.checkpoint')
run('combined-rebuild-repartition',rebuild=True,expected=reference/'control.checkpoint')
(out/'verification.json').write_text(json.dumps(dict(status='passed',scope='native accepted-state/compensation transfer, invalid-descriptor recovery, restart, rejected-stage capture and short frozen-control fidelity; no AMR or performance claim',source_sha256=pins,binary_sha256=bins,initial_sha256=initial_hash,commands=records),indent=2)+'\n')
