#!/usr/bin/env python3
"""Bounded one-thread source-frozen Python component experiment."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=120,choices=range(1,301));p.add_argument('--dependency',type=Path,action='append',default=[]);p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
root=Path(__file__).resolve().parents[2];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[a.source.resolve(),*[d.resolve() for d in a.dependency],Path(__file__).resolve()]
source_hashes={str(f.relative_to(root)):sha(f) for f in files}
for f in files:
 dest=out/'source'/f.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes())
arguments=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
cmd=[sys.executable,str(out/'source'/a.source.resolve().relative_to(root)),'--output',str(out/'result'),*arguments]
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1')
m={'scope':'Bounded component experiment, not 3D evolution acceptance','source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'source_hashes':source_hashes,'command':cmd,'maximum_seconds':a.seconds,'environment':{k:env[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']}}
def save():(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
start=time.monotonic()
with (out/'probe.stdout').open('w') as so,(out/'probe.stderr').open('w') as se:
 child=subprocess.Popen(cmd,stdout=so,stderr=se,cwd=out,env=env,start_new_session=True);m['owned_process_group']=child.pid;save()
 try:m['exit_code']=child.wait(timeout=a.seconds)
 except subprocess.TimeoutExpired:
  assert os.getpgid(child.pid)==child.pid;os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
  m['exit_code']=124
m['seconds']=time.monotonic()-start;m['source_unchanged']=all(sha(root/f)==h for f,h in source_hashes.items());save();assert m['source_unchanged']
print(json.dumps({k:m[k] for k in ['exit_code','seconds','source_unchanged']}));raise SystemExit(m['exit_code'])
