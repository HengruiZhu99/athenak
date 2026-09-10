from pathlib import Path
import sys,json
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'bisection'))
from controller import sha,validate_termination
from criterion import read_history
from resume_case import checkpoint_info
v=root/'early-lapse/validation'
a,b=v/'above_old',v/'above_new'
ah,bh=[read_history(next(d.glob('*.hst'))) for d in [a,b]]
if ah!=bh:raise RuntimeError('Above-threshold histories differ')
checkpoints=[sorted((d/'rst').glob('*.rst'))[-1] for d in [a,b]]
payloads=[p.read_bytes().split(b'<par_end>\n',1)[1] for p in checkpoints]
if payloads[0]!=payloads[1]:raise RuntimeError('Above-threshold checkpoint payloads differ')
for d in [a,b]:
 if list(d.glob('*.termination.json')):raise RuntimeError('Premature stop above threshold')
 if 'Terminating on cycle limit' not in (d/'stdout.log').read_text():raise RuntimeError('Unexpected short-test ending')
d=v/'below_new';h=read_history(next(d.glob('*.hst')))
reason=validate_termination(d,h)
p=sorted((d/'rst').glob('*.rst'))[-1];info=checkpoint_info(p)
manifest=json.loads((v/'manifest.json').read_text())
if info['cycle']!=manifest[-1]['checkpoint_info']['cycle']+1:raise RuntimeError('Stop not at first completed step')
if info['time']!=h[-1]['time'] or info['cycle']!=h[-1]['cycle']:raise RuntimeError('Early-stop checkpoint/history mismatch')
for d in [a,b,v/'below_new']:
 if (d/'run-status').read_text().strip()!='0':raise RuntimeError('Validation evolution failed')
report=dict(status='PASS',job_id=(v/'job-id.txt').read_text().strip(),above_threshold_history_exact=True,above_threshold_checkpoint_payload_exact=True,above_threshold_steps=8,early_stop_reason=reason,early_stop_time=info['time'],early_stop_cycle=info['cycle'],early_stop_minLapse=h[-1]['minLapse'],early_stop_checkpoint_sha256=sha(p),new_executable_sha256=manifest[-1]['executable_sha256'],tests=manifest)
(v/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='tests'},indent=2))
