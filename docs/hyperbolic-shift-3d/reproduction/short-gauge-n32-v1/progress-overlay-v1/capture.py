"""Capture an immutable, complete-row live snapshot; not terminal verification."""
import base64, hashlib, json, shlex, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
script=r'''
import base64,csv,datetime,hashlib,io,json,pathlib,subprocess
root=pathlib.Path('/home/hzhu/hyperbolic-core-shell-20261028')
p=root/'campaigns/n32-short-gauge-fresh10m-v1-segment01'
plan=p/'plan.json'
assert hashlib.sha256(plan.read_bytes()).hexdigest()=='125aa58211bcf47b3cdc74b1daf0d65e2f90bec3d3b7da63e32ab35751271120'
raw=(p/'output/puncture.history.csv').read_bytes()
# Only complete newline-terminated rows are considered, even during a write.
lines=raw.splitlines(keepends=True);lines=[line for line in lines if line.endswith(b'\n')]
header=lines[0];rows=list(csv.DictReader(io.StringIO(b''.join(lines).decode())))
selected=[(line,row) for line,row in zip(lines[1:],rows)]
assert selected, 'No diagnostic ready'
data=header+b''.join(line for line,row in selected)
q=subprocess.run(['qstat','-f','9027014'],capture_output=True,text=True)
assert q.returncode==0,q.stderr
worker=json.loads((p/'worker-status.json').read_text())
result=dict(job='9027014',captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 scope='Provisional live history snapshot through latest complete row; not a terminal report, checkpoint verification, or completed10M result.',
 plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),binding=json.loads((p/'binding.json').read_text()),
 worker_status=worker['status'],scheduler=q.stdout,rows=len(selected),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
 payload_base64=base64.b64encode(data).decode())
print(json.dumps(result))
'''
command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ControlPath=/Users/hz0693/.ssh/codex-control/%C','hzhu@aurora.alcf.anl.gov', 'python3.11 -c '+shlex.quote(script)]
result=json.loads(subprocess.check_output(command,text=True))
data=base64.b64decode(result.pop('payload_base64'))
assert len(data)==result['bytes'] and hashlib.sha256(data).hexdigest()==result['sha256']
for name in ['n32.history.csv','snapshot.json']:
 if (HERE/name).exists():raise RuntimeError('Snapshot already exists; do not overwrite')
(HERE/'n32.history.csv').write_bytes(data)
(HERE/'snapshot.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['job','captured_utc','worker_status','rows','sha256']},indent=2))
