"""Bounded local qualification of fixed-topology buffer retirement and full boosted mesh.
Never changes numerical parameters on a resource abort; preserves the evidence.
"""
from pathlib import Path
import hashlib,json,subprocess,time,shutil,os,signal,importlib.util
import numpy as np
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists())
B=R/'output/refined-convergence/sealed-topology-candidate-v1';O=B/'physics';O.mkdir(exist_ok=False)
E=D/'sealed-topology-controls';E.mkdir(exist_ok=False)
config=R/'inputs/hyperbolic_shift_3d/refined_convergence/central20_boosted_smr_tube_radial24_angular2_reference.cfg'
oldconfig=R/'inputs/hyperbolic_shift_3d/refined_convergence/central20_boosted_smr_tube_radial18_reference.cfg'
commands=[]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def run(name,cmd):
 start=time.monotonic();cmd=[str(x) for x in cmd];peak=0;guard=None
 with (E/(name+'.log')).open('w') as f:
  p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  with (E/(name+'-resources.jsonl')).open('w') as resources:
   while p.poll() is None:
    entries=[list(map(int,line.split())) for line in subprocess.check_output(['ps','-axo','pid=,ppid=,rss='],text=True).splitlines() if line.strip()]
    children={p.pid}
    while True:
     new=children|{pid for pid,ppid,rss in entries if ppid in children}
     if new==children:break
     children=new
    rss=1024*sum(rss for pid,ppid,rss in entries if pid in children);peak=max(peak,rss)
    elapsed=time.monotonic()-start
    resources.write(json.dumps(dict(elapsed=elapsed,aggregate_rss=rss,pids=sorted(children)))+'\n');resources.flush()
    if rss>43_000_000_000 or elapsed>1800:
     guard='local resource limit, not a numerical failure';os.killpg(p.pid,signal.SIGTERM)
     try:p.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
     break
    time.sleep(3)
 commands.append(dict(name=name,command=cmd,returncode=p.returncode,elapsed=time.monotonic()-start,aggregate_peak_rss=peak,resource_abort=guard))
 (E/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 assert p.returncode==0 and guard is None,(name,p.returncode,guard)
 print(name,'completed',flush=True)
for ranks in [1,2]:
 cached=B/'checks'/('topology-mpi'+str(ranks)+'.json')
 if cached.exists():
  assert json.loads(cached.read_text())['passed'];shutil.copy2(cached,E/cached.name)
 else:run('topology-mpi'+str(ranks),['mpiexec','-n',ranks,B/'support',E/('topology-mpi'+str(ranks)+'.json')])
verifier=R/'output/refined-convergence/verify-checkpoint'
def evolve(name,cfg,policy,steps,restart='-'):
 prefix=O/name;run(name,['mpiexec','-n',1,B/'evolve',12,'boost',steps,1/24000,prefix,restart,policy,cfg])
 report=json.loads(Path(str(prefix)+'.report.json').read_text());assert report['checkpoint_admitted'] and report['status']=='completed' and report['steps']==steps and report['dt']==1/24000
 v=json.loads(subprocess.check_output([str(verifier),str(prefix)+'.checkpoint'],text=True));assert v['identity']==report['identity'] and v['n']==12
 v.update(sha256=sha(str(prefix)+'.checkpoint'),bytes=Path(str(prefix)+'.checkpoint').stat().st_size)
 (E/(name+'-verification.json')).write_text(json.dumps(v,indent=2)+'\n')
 for suffix in ['.report.json','.history.jsonl','.rank0.generations.jsonl','.rank0.performance.json']:
  shutil.copy2(str(prefix)+suffix,E/(name+suffix))
 rows=[json.loads(x) for x in Path(str(prefix)+'.history.jsonl').read_text().splitlines()]
 assert max(x['cfl'] for x in rows)<=.2+1e-12
 generations=[json.loads(x) for x in Path(str(prefix)+'.rank0.generations.jsonl').read_text().splitlines()]
 assert all(x['memory_bytes']['native_state_compensation']==0 for x in generations)
 return prefix,v,rows
uniform=R/'output/refined-convergence/boosted-radial24-control/uniform-core.cfg'
one,a,rows=evolve('uniform-parity',uniform,'uniform',2)
assert a['sha256']==sha(R/'output/refined-convergence/boosted-radial24-control/state.checkpoint')
old=R/'output/refined-convergence/smr-tube-candidate-v1/checks'
resume,b,rows=evolve('smr-restart-parity',oldconfig,'smr',4,old/'boost-mpi2-step2.checkpoint')
spec=importlib.util.spec_from_file_location('reader',D/'check_smr_restart.py');reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
x,xt,xa,xf=reader.read(str(old/'boost-mpi1-restart-step4'));y,yt,ya,yf=reader.read(str(resume))
assert x['identity']==y['identity'] and x['config']==y['config'] and xt==yt and xa==ya and x['time']==y['time'] and x['dt']==y['dt']
assert all(np.array_equal(u,v) for u,v in zip(xf,yf))
del xf,yf
full,c,rows=evolve('full-smr-radial24-angular2',config,'smr',2)
assert c['leaves']==400
assert all(x['finest_level']==3 and x['puncture_finest_margin']>0 and x['horizon_complete'] and x['horizon_found'] for x in rows)
assert all(x['horizon_finest_margin']>0 for x in rows)
shutil.copy2(B/'build.json',E/'build.json')
result=dict(passed=True,uniform_checkpoint_byte_equal=True,smr_restart_all8_arrays_byte_equal=True,full_smr_checkpoint=c,full_smr_initial=rows[0],full_smr_final=rows[-1],candidate_binary_sha256=sha(B/'evolve'),config_sha256=sha(config),qualification='Full400-leaf boosted SMR plus radial24/angular2 combined local two-step check. No GPU, long-time, dynamic AMR or convergence claim. Memory retirement changes no numerical identity.')
(E/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('combined full-SMR verified',flush=True)
