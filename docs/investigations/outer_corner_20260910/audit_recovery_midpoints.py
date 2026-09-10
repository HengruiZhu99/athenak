"""Read-only audit of real completed midpoint evolutions and automatic successor."""
from pathlib import Path
from decimal import Decimal
import sys,json,subprocess,math
root=Path(__file__).resolve().parent
campaign=Path('/pscratch/sd/h/hzhu/lapse-bisection-recovery-20260910')
sys.path.insert(0,str(campaign))
from controller import sha,validate_termination
from criterion import read_history,classify,relative_width
from resume_case import checkpoint_info
state=json.loads((campaign/'state.json').read_text())
sub,sup=Decimal('-.0485'),Decimal('-.05');audited=[]
for record in state['completed']:
 if record['adopted']:continue
 case=Path(record['directory']);expected=(sub+sup)/2
 if Decimal(record['amplitude'])!=expected:raise RuntimeError('Wrong midpoint amplitude')
 if (case/'run-status').read_text().strip()!='0':raise RuntimeError('Evolution did not complete cleanly')
 job=(case/'job-id.txt').read_text().strip()
 accounting=subprocess.check_output(['sacct','-j',job,'-X','-n','-P','--format=JobID,State,ExitCode'],text=True)
 rows=[r.split('|') for r in accounting.splitlines() if r.split('|')[0]==job]
 if len(rows)!=1 or rows[0][1:3]!=['COMPLETED','0:0']:raise RuntimeError('Allocation completion unverified')
 history=read_history(next(case.glob('*.hst')));reason=validate_termination(case,history)
 result=('disperse' if reason=='early_lapse_recovery' else ('collapse' if reason=='early_global_lapse' else classify(history,200)))
 if record['classification']!=result or record['final']!=history[-1]:raise RuntimeError('Classification or final diagnostics mismatch')
 rst=Path(record['checkpoint']);info=checkpoint_info(rst)
 if info['time']!=history[-1]['time'] or info['cycle']!=history[-1]['cycle']:raise RuntimeError('Final checkpoint/history mismatch')
 if sha(rst)!=record['checkpoint_sha256']:raise RuntimeError('Final checkpoint hash mismatch')
 for name,key in [('initial.coefficients','coefficient_sha256'),('input.athinput','input_sha256')]:
  if sha(case/name)!=record[key]:raise RuntimeError('Input artifact changed')
 if result=='collapse':sup=expected
 else:sub=expected
 audited.append(dict(case=case.name,job_id=job,amplitude=str(expected),classification=result,reason=reason,time=info['time'],minLapse=history[-1]['minLapse'],checkpoint=str(rst),checkpoint_sha256=sha(rst),sub=str(sub),supercritical=str(sup)))
if len(audited)<2:
 print(json.dumps(dict(status='WAITING_FOR_TWO_REAL_MIDPOINTS',verified_midpoints=audited,active=state.get('active')),indent=2));sys.exit(0)
if Decimal(state['sub'])!=sub or Decimal(state['super'])!=sup:raise RuntimeError('State bracket mismatch; retry if controller is updating')
if relative_width(sub,sup)>Decimal('1e-8'):
 active=state.get('active')
 if not active or Decimal(active['amplitude'])!=(sub+sup)/2:raise RuntimeError('Automatic successor missing')
 case=Path(active['directory']);jobfile=case/'job-id.txt'
 if jobfile.exists():job=jobfile.read_text().strip()
 else:
  import re
  ids=re.findall(r'Pending job allocation (\d+)',(case/'allocation.log').read_text())
  if not ids:raise RuntimeError('Successor allocation missing')
  job=ids[-1]
 live=subprocess.check_output(['squeue','-h','-u','hzhu','-o','%i'],text=True).split()
 if job not in live:raise RuntimeError('Successor not currently live; inspect accounting/state before claiming supervision complete')
else:
 if state['status']!='COMPLETE':raise RuntimeError('Tolerance met but controller not complete')
 job=None
print(json.dumps(dict(status='TWO_REAL_MIDPOINTS_AND_SUCCESSOR_VERIFIED',midpoints=audited,successor_job=job,active=state.get('active')),indent=2))
