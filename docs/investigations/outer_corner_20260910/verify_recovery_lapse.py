from pathlib import Path
import sys,json,re
root=Path(__file__).resolve().parent
sys.path.insert(0,'/pscratch/sd/h/hzhu/lapse-bisection-recovery-20260910')
from controller import sha,validate_termination
from criterion import read_history
from resume_case import checkpoint_info
v=root/'recovery-lapse/validation';manifest=json.loads((v/'manifest.json').read_text());results={}
for entry in manifest:
 label=entry['label'];d=v/label;h=read_history(next(d.glob('*.hst')))
 if (d/'run-status').read_text().strip()!='0':raise RuntimeError('Validation run failed')
 rst=sorted((d/'rst').glob('*.rst'))[-1];info=checkpoint_info(rst)
 if info['time']!=h[-1]['time'] or info['cycle']!=h[-1]['cycle']:raise RuntimeError('Checkpoint/history mismatch')
 if label in ['recovered','collapse']:
  reason=validate_termination(d,h)
  expected='early_lapse_recovery' if label=='recovered' else 'early_global_lapse'
  if reason!=expected or info['cycle']!=entry['checkpoint_info']['cycle']+1:raise RuntimeError('Wrong early stop')
 else:
  if list(d.glob('*.termination.json')) or 'Terminating on cycle limit' not in (d/'stdout.log').read_text():raise RuntimeError('Unexpected early stop')
  if info['cycle']!=entry['checkpoint_info']['cycle']+8:raise RuntimeError('Wrong step count')
 if label=='above':
  old=sorted((root/'early-lapse/validation/above_old/rst').glob('*.rst'))[-1]
  if rst.read_bytes().split(b'<par_end>\n',1)[1]!=old.read_bytes().split(b'<par_end>\n',1)[1]:raise RuntimeError('Dynamics changed')
  if h!=read_history(next((root/'early-lapse/validation/above_old').glob('*.hst'))):raise RuntimeError('History changed')
 results[label]=dict(final=h[-1],checkpoint=str(rst),checkpoint_sha256=sha(rst))
d=v/'persisted';h=read_history(next(d.glob('*.hst')));p=json.loads((d/'provenance.json').read_text())
if (d/'run-status').read_text().strip()!='0' or validate_termination(d,h)!='early_lapse_recovery':raise RuntimeError('Persisted dip was not restored')
if h[-1]['cycle']!=p['checkpoint_info']['cycle']+1:raise RuntimeError('Persisted stop delayed')
if re.search(r'^\s*termination_min_lapse\s*=',(d/'input.athinput').read_text(),re.M):raise RuntimeError('Persistence test re-seeded input')
results['persistence']=dict(final=h[-1],input_sha256=sha(d/'input.athinput'))
report=dict(status='PASS',job_id=(v/'job-id.txt').read_text().strip(),executable_sha256=manifest[0]['executable_sha256'],results=results,above_threshold_checkpoint_and_history_exact=True,prior_dip_restored_without_override=True)
(v/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
