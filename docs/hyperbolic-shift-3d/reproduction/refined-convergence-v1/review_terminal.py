"""Independent post-run review of the immutable verifier permission failure.
Never changes the original controller, verifier, worker disposition or checkpoint.
"""
import argparse, hashlib, importlib.util, json, pathlib, shutil, subprocess, sys
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('campaign');a=p.parse_args()
root=pathlib.Path(a.root).resolve();d=root/a.campaign;review=d/'independent-review';review.mkdir(exist_ok=True)
sys.path.insert(0,str(root/'launcher-v2'))
spec=importlib.util.spec_from_file_location('frozen_control',root/'launcher-v2/control.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
worker=c.read(d/'output/worker.json');terminal=c.read(d/'terminal-verification.json');plan=c.read(d/'plan.json')
assert worker['native_returncode']==0 and worker['status']=='terminal_failure'
original=root/'launcher-v2/verify-checkpoint'
assert worker['error']==f"[Errno 13] Permission denied: '{original}'"
assert terminal['scheduler_exit']==2 and terminal['admitted'] is False
assert c.read(d/'allocation-released.json')['job_id']==worker['job_id']
assert not original.stat().st_mode & 0o111
copy=review/'verify-checkpoint-executable'
if copy.exists():assert c.digest(copy)==c.digest(original)
else:shutil.copyfile(original,copy)
copy.chmod(0o700)
assert c.digest(copy)==c.digest(original)
for name,b in worker['artifacts'].items():
 f=d/'output'/name;assert f.stat().st_size==b['bytes'] and c.digest(f)==b['sha256'],name
real_check=subprocess.check_output
calls=[]
def checked_command(argv,*args,**kwargs):
 assert argv[0]==str(original)
 calls.append(dict(original=argv,executed=[str(copy)]+argv[1:]))
 return real_check([str(copy)]+argv[1:],*args,**kwargs)
c.subprocess.check_output=checked_command
v=c.verify(root,plan,d/'output')
result=dict(status='independently_verified_completed_endpoint',original_worker_admitted=False,
 original_terminal=terminal,reason=worker['error'],original_verifier_mode=oct(original.stat().st_mode & 0o777),
 verifier_sha256=c.digest(original),controller_sha256=c.digest(root/'launcher-v2/control.py'),
 review_script_sha256=c.digest(pathlib.Path(__file__)),calls=calls,artifact_count=len(worker['artifacts']),
 checkpoint=v,qualification='Original worker/PBS failure remains immutable. Identical verifier bytes executed from a separate executable copy; all frozen scientific checks and artifact hashes passed. Endpoint is approved for read-only analysis, not a continuation or accuracy claim.')
c.write(review/'verification.json',result);print(json.dumps(result,indent=2))
