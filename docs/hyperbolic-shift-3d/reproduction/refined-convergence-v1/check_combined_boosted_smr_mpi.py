"""Full combined SMR/radial/angular mesh: fresh MPI2 versus verified MPI1."""
from pathlib import Path
import json,subprocess,time,os,signal,hashlib,shutil
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists())
B=R/'output/refined-convergence/sealed-topology-candidate-v1';O=B/'mpi-checks';O.mkdir(exist_ok=False)
E=D/'combined-smr-mpi';E.mkdir(exist_ok=False)
reference=json.loads((D/'sealed-topology-controls/verification.json').read_text());assert reference['passed']
config=R/'inputs/hyperbolic_shift_3d/refined_convergence/central20_boosted_smr_tube_radial24_angular2_reference.cfg'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
assert sha(B/'evolve')==reference['candidate_binary_sha256'] and sha(config)==reference['config_sha256']
prefix=O/'state';cmd=[str(x) for x in ['mpiexec','-n',2,B/'evolve',12,'boost',2,1/24000,prefix,'-','smr',config]]
start=time.monotonic();peak=0;abort=None
with (E/'native.log').open('w') as log:
 p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 with (E/'resources.jsonl').open('w') as resources:
  while p.poll() is None:
   entries=[list(map(int,line.split())) for line in subprocess.check_output(['ps','-axo','pid=,ppid=,rss='],text=True).splitlines() if line.strip()];children={p.pid}
   while True:
    new=children|{pid for pid,ppid,rss in entries if ppid in children}
    if new==children:break
    children=new
   rss=1024*sum(rss for pid,ppid,rss in entries if pid in children);peak=max(peak,rss);elapsed=time.monotonic()-start
   resources.write(json.dumps(dict(elapsed=elapsed,aggregate_rss=rss,pids=sorted(children)))+'\n');resources.flush()
   if rss>43_000_000_000 or elapsed>1800:
    abort='Local resource limit, not numerical failure';os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    break
   time.sleep(3)
record=dict(command=cmd,returncode=p.returncode,elapsed=time.monotonic()-start,aggregate_peak_rss=peak,resource_abort=abort,binary_sha256=sha(B/'evolve'),config_sha256=sha(config))
(E/'command.json').write_text(json.dumps(record,indent=2)+'\n');assert p.returncode==0 and abort is None
report=json.loads((O/'state.report.json').read_text());assert report['status']=='completed' and report['checkpoint_admitted'] and report['ranks']==2 and report['steps']==2 and report['dt']==1/24000
v=json.loads(subprocess.check_output([str(R/'output/refined-convergence/verify-checkpoint'),str(O/'state.checkpoint')],text=True))
assert v['verified'] and v['identity']==reference['full_smr_checkpoint']['identity'] and v['n']==12 and v['kind']==2 and v['leaves']==400
v.update(sha256=sha(O/'state.checkpoint'),bytes=(O/'state.checkpoint').stat().st_size)
assert v['sha256']==reference['full_smr_checkpoint']['sha256'] and v['bytes']==reference['full_smr_checkpoint']['bytes']
for suffix in ['report.json','history.jsonl','rank0.performance.json','rank1.performance.json','rank0.generations.jsonl','rank1.generations.jsonl']:
 shutil.copy2(O/('state.'+suffix),E/('state.'+suffix))
rows=[json.loads(x) for x in (O/'state.history.jsonl').read_text().splitlines()]
assert all(x['horizon_found'] and x['candidate_finest_margin']>0 for x in rows)
result=dict(passed=True,checkpoint_byte_equal=True,checkpoint=v,reference=reference['full_smr_checkpoint'],command=record,qualification='Fresh two-step local Serial-MPI1/2 full combined400-leaf SMR/radial24/angular2 control, not GPU, long-time or convergence evidence. Unchanged equations and timestep.')
(E/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('full combined MPI1/2 checkpoints byte-identical',flush=True)
