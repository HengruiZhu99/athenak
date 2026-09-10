"""Join a preserved historical prefix to a clean recovery-stop certificate."""
from pathlib import Path
import sys,json,shutil
root=Path(__file__).resolve().parent
sys.path.insert(0,'/pscratch/sd/h/hzhu/lapse-bisection-recovery-20260910')
from controller import sha,input_parameters,validate_termination
from criterion import read_history,classify
from resume_case import checkpoint_info
v=root/'recovery-lapse/validation';report=json.loads((v/'verification.json').read_text())
if report['status']!='PASS':raise RuntimeError('Recovery validation not passed')
certificate=v/'recovered';original=Path('/pscratch/sd/h/hzhu/lapse-bisection-t200-corner-fixed-20260910/cycle_01')
entry=next(x for x in json.loads((v/'manifest.json').read_text()) if x['label']=='recovered')
history=next(original.glob('*.hst'))
if sha(history)!=entry['history_evidence_sha256']:raise RuntimeError('Original history changed after certification')
out=root/'recovery-lapse/endpoints';out.mkdir();d=out/'sub';d.mkdir()
for name in ['input.athinput','initial.coefficients','stdout.log','stderr.log','run-status']:shutil.copyfile(certificate/name,d/name)
(d/'job-id.txt').write_text(report['job_id']+'\n');(d/'amplitude.txt').write_text('-.0485\n')
prefix=[x for x in history.read_text().splitlines() if x.startswith('#') or not x.strip() or float(x.split()[0])<=entry['checkpoint_info']['time']]
newhistory=next(certificate.glob('*.hst'))
suffix=[x for x in newhistory.read_text().splitlines() if x.strip() and not x.startswith('#') and float(x.split()[0])>entry['checkpoint_info']['time']]
(d/newhistory.name).write_text('\n'.join(prefix+suffix)+'\n')
rows=read_history(d/newhistory.name)
if classify(rows,200)!='disperse' or validate_termination(certificate,read_history(newhistory))!='early_lapse_recovery':raise RuntimeError('Recovery not independently supported')
shutil.copyfile(next(certificate.glob('*.termination.json')),d/next(certificate.glob('*.termination.json')).name)
(d/'rst').mkdir();rst=sorted((certificate/'rst').glob('*.rst'))[-1];shutil.copyfile(rst,d/'rst'/rst.name)
info=checkpoint_info(rst)
if info['time']!=rows[-1]['time'] or info['cycle']!=rows[-1]['cycle']:raise RuntimeError('Merged history/checkpoint mismatch')
provenance=dict(exe_sha256=report['executable_sha256'],input_sha256=sha(d/'input.athinput'),original_directory=str(original),original_history_sha256=sha(history),history_prefix_end=entry['checkpoint_info'],input_checkpoint_sha256=entry['checkpoint_sha256'],certificate_directory=str(certificate),certificate_history_sha256=sha(newhistory),note='Cancelled old continuation remains cancelled. Import uses its finite prefix through saved t74 checkpoint and a separate successful one-step recovery certificate; not counted as a midpoint iteration.')
(d/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
(d/'inputs.sha256').write_text(''.join(sha(d/name)+'  '+name+'\n' for name in ['input.athinput','initial.coefficients']))
params=input_parameters((d/'input.athinput').read_text())
record=dict(executable_sha256=report['executable_sha256'],restart_certificate=True,allowed_analysis_overrides={name:params[tuple(name.split('/'))] for name in ['time/nlim','problem/termination_min_lapse']},files={str(p.relative_to(d)):sha(p) for p in d.rglob('*') if p.is_file()})
old=root/'early-lapse/historical-endpoints';evidence=json.loads((old/'evidence.json').read_text())
superpath=str(old/'super');records={str(d):record,superpath:evidence[superpath]}
(out/'evidence.json').write_text(json.dumps(records,indent=2)+'\n');print(d)
