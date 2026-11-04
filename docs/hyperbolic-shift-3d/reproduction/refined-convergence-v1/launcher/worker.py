#!/usr/bin/env python3
"""One RF2 evolution per immutable PBS allocation; failed states never restart."""
import os, pathlib, subprocess, sys, time
from control import read,write,digest,bindings,verify
from storage import capacity

def main():
 root=pathlib.Path(sys.argv[1]).resolve();pf=pathlib.Path(sys.argv[2]).resolve();plan=read(pf);bindings(root,plan)
 if not os.environ.get('PBS_JOBID'):raise RuntimeError('PBS allocation required')
 nodes=set(pathlib.Path(os.environ['PBS_NODEFILE']).read_text().splitlines())
 if len(nodes)!=plan['nodes'] or plan['ranks']!=12*len(nodes):raise RuntimeError('allocation mismatch')
 out=root/plan['output'];out.mkdir(exist_ok=False)
 s=dict(status='running',job_id=os.environ['PBS_JOBID'],plan_sha256=digest(pf),nodes=sorted(nodes),started=time.time(),artifacts={});write(out/'worker.json',s)
 cpu='1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
 mpi=['mpiexec','-n',str(plan['ranks']),'-ppn','12','--cpu-bind=list:'+cpu,'gpu_tile_compact.sh']
 code=2
 try:
  s['storage']=capacity(root,planned=plan['planned_output_bytes'],reserve=plan['atomic_reserve_bytes'])
  # Placement inspection only: no physics smoke or GPU audit gate.
  affinity=mpi+['bash','-c','printf "RF2AFF %s %s %s %s\\n" "$(hostname)" "$PMI_RANK" "$MPI_LOCALRANKID" "$ZE_AFFINITY_MASK"']
  r=subprocess.run(affinity,capture_output=True,text=True,timeout=60);(out/'affinity.log').write_text(r.stdout+r.stderr)
  if r.returncode:raise RuntimeError('affinity launch failed')
  records=[l.split()[1:] for l in r.stdout.splitlines() if l.startswith('RF2AFF ')]
  if len(records)!=plan['ranks'] or {int(x[1]) for x in records}!=set(range(plan['ranks'])):raise RuntimeError('incomplete affinity')
  hosts={x[0] for x in records}
  if len(hosts)!=plan['nodes']:raise RuntimeError('wrong host count')
  for host in hosts:
   rows=[x for x in records if x[0]==host]
   if {int(x[2]) for x in rows}!=set(range(12)) or any(x[3]!=f'{int(x[2])//2}.{int(x[2])%2}' for x in rows):raise RuntimeError('wrong local GPU masks')
  restart=str(root/plan['restart']) if plan['restart'] else '-'
  cmd=mpi+[str(root/plan['binary']),str(plan['n']),'boost','0',str(plan['dt']),str(out/'state'),restart,'smr',str(root/plan['config']),'2920',str(plan['diagnostic_interval']),str(plan['target_time'])]
  s['command']=cmd;write(out/'worker.json',s);began=time.monotonic()
  with (out/'native.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=3100)
  s.update(native_returncode=r.returncode,native_seconds=time.monotonic()-began)
  if r.returncode not in (0,3):raise RuntimeError('native terminal failure')
  v=verify(root,plan,out);expected=0 if v['status']=='completed' else 3
  if r.returncode!=expected:raise RuntimeError('native disposition mismatch')
  write(out/'checkpoint-verification.json',v);s['status']=v['status'];code=expected
 except Exception as e:s.update(status='terminal_failure',error=str(e))
 finally:
  s['finished']=time.time()
  for p in sorted(out.rglob('*')):
   if p.is_file() and p.name!='worker.json':s['artifacts'][str(p.relative_to(out))]=dict(bytes=p.stat().st_size,sha256=digest(p))
  write(out/'worker.json',s)
 return code
if __name__=='__main__':sys.exit(main())
