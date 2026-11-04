#!/usr/bin/env python3
"""One RF2 evolution per immutable PBS allocation; failed states never restart."""
import os, pathlib, subprocess, sys, time, threading
from control import read,write,digest,bindings,verify
from storage import capacity

def check_affinity(text,nodes,ranks):
 records=[l.split()[1:] for l in text.splitlines() if l.startswith('RF2AFF ')]
 if len(records)!=ranks or any(len(x)<2 for x in records):raise RuntimeError('incomplete affinity')
 hosts={x[0].split('.')[0] for x in records};expected={f'{i//2}.{i%2}' for i in range(12)}
 if len(hosts)!=nodes:raise RuntimeError('wrong host count')
 result={}
 for host in hosts:
  masks=[x[-1] for x in records if x[0].split('.')[0]==host]
  if len(masks)!=12 or set(masks)!=expected:raise RuntimeError('wrong GPU masks on '+host)
  result[host]=sorted(masks)
 return result

def main():
 root=pathlib.Path(sys.argv[1]).resolve();pf=pathlib.Path(sys.argv[2]).resolve();plan=read(pf);bindings(root,plan)
 if not os.environ.get('PBS_JOBID'):raise RuntimeError('PBS allocation required')
 nodes=set(pathlib.Path(os.environ['PBS_NODEFILE']).read_text().splitlines())
 if len(nodes)!=plan['nodes'] or plan['ranks']!=12*len(nodes):raise RuntimeError('allocation mismatch')
 out=root/plan['output'];out.mkdir(exist_ok=False)
 allocation_start=time.monotonic()
 s=dict(status='running',job_id=os.environ['PBS_JOBID'],plan_sha256=digest(pf),nodes=sorted(nodes),started=time.time(),artifacts={});write(out/'worker.json',s)
 cpu='1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
 mpi=['mpiexec','-n',str(plan['ranks']),'-ppn','12','--cpu-bind=list:'+cpu,'gpu_tile_compact.sh']
 code=2
 try:
  submission=read(pf.parent/'submission.json')
  if submission['job_id']!=os.environ['PBS_JOBID'] or submission['plan_sha256']!=digest(pf):raise RuntimeError('submission identity mismatch')
  if submission['storage']['headroom_bytes']<0:raise RuntimeError('unadmitted storage budget')
  s['storage_at_submission']=submission['storage']
  s['storage_snapshot_age_seconds']=time.time()-submission['time']
  if s['storage_snapshot_age_seconds']>300:
   s['fresh_storage']=capacity(root,planned=plan['planned_output_bytes'],reserve=plan['atomic_reserve_bytes'])
  storage_result={};stop_guard=threading.Event()
  def global_guard():
   while not stop_guard.is_set():
    try:storage_result['latest']=capacity(root,planned=plan['planned_output_bytes'],reserve=plan['atomic_reserve_bytes']);storage_result['checked_at']=time.time()
    except Exception as e:storage_result['error']=str(e);return
    stop_guard.wait(60)
  threading.Thread(target=global_guard,daemon=True).start()
  write(out/'worker.json',s)
  # Placement inspection only: no physics smoke or GPU audit gate.
  affinity=mpi+['bash','-c','printf "RF2AFF %s %s\\n" "$(hostname)" "$ZE_AFFINITY_MASK"']
  r=subprocess.run(affinity,capture_output=True,text=True,timeout=60);(out/'affinity.log').write_text(r.stdout+r.stderr)
  if r.returncode:raise RuntimeError('affinity launch failed')
  placements=check_affinity(r.stdout,plan['nodes'],plan['ranks'])
  if {h.split('.')[0] for h in nodes}!=set(placements):raise RuntimeError('placement hosts differ from allocation')
  s['verified_placements']=placements
  restart=str(root/plan['restart']) if plan['restart'] else '-'
  cmd=mpi+[str(root/plan['binary']),str(plan['n']),'boost','0',str(plan['dt']),str(out/'state'),restart,'smr',str(root/plan['config']),str(max(1,min(2920,3200-(time.monotonic()-allocation_start)-180))),str(plan['diagnostic_interval']),str(plan['target_time'])]
  s['command']=cmd;write(out/'worker.json',s);began=time.monotonic()
  with (out/'native.log').open('w') as log:
   r=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
   while r.poll() is None:
    used=sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in out.rglob('*') if p.is_file())
    error=storage_result.get('error')
    if used>plan['planned_output_bytes']+plan['atomic_reserve_bytes']:error='per-allocation output/atomic reserve exceeded'
    if time.monotonic()-allocation_start>3180 or time.monotonic()-began>3100:error='worker exit reserve reached'
    if error:
     r.terminate()
     try:r.wait(timeout=20)
     except subprocess.TimeoutExpired:r.kill();r.wait()
     raise RuntimeError(error)
    time.sleep(2)
   r.wait()
  s.update(native_returncode=r.returncode,native_seconds=time.monotonic()-began)
  if r.returncode not in (0,3):raise RuntimeError('native terminal failure')
  v=verify(root,plan,out);expected=0 if v['status']=='completed' else 3
  if r.returncode!=expected:raise RuntimeError('native disposition mismatch')
  write(out/'checkpoint-verification.json',v);s['status']=v['status'];code=expected
 except Exception as e:s.update(status='terminal_failure',error=str(e))
 finally:
  s['finished']=time.time()
  if 'stop_guard' in locals():stop_guard.set();s['background_storage']=dict(storage_result)
  for p in sorted(out.rglob('*')):
   if p.is_file() and p.name!='worker.json':s['artifacts'][str(p.relative_to(out))]=dict(bytes=p.stat().st_size,sha256=digest(p))
  write(out/'worker.json',s)
 return code
if __name__=='__main__':sys.exit(main())
