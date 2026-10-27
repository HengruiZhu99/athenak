#!/usr/bin/env python3
"""Run exact-time controls, then continue selected verified prefixes.

Physical failures remain data. No failed prefix is resumed or overwritten.
"""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--library',required=True);p.add_argument('--fast-library');a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
driver=Path(__file__).with_name('run_matrix.py');data=json.loads(a.input.read_text());(a.output/'protocol.json').write_text(a.input.read_text())
command=[sys.executable,str(driver),'--input',str(a.input.resolve()),'--output',str(a.output/'prefix'),'--library',str(Path(a.library).resolve())]
if a.fast_library:command+=['--fast-library',str(Path(a.fast_library).resolve())]
receipt=dict(pid=os.getpid(),protocol_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),prefix_command=command,continuations=[],status='prefix-running')
def save():(a.output/'pipeline.json').write_text(json.dumps(receipt,indent=2)+'\n')
save()
with (a.output/'prefix.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
if result.returncode:
    receipt.update(status='orchestration-failed',prefix_returncode=result.returncode);save();sys.exit(result.returncode)
receipt['status']='continuations-running';save();continued=a.output/'continued';continued.mkdir()
helper=a.output/'prefix/matrix-source/tst/elliptic_shift_1d/continue_verified.py'
for case in data['cases']:
    if 'continue_to' not in case:continue
    parent=a.output/'prefix'/case['name'];m=json.loads((parent/'manifest.json').read_text())
    if m['status']!='finished':
        receipt['continuations'].append(dict(name=case['name'],status='skipped-failed-prefix'));save();continue
    command=[sys.executable,str(helper),'--checkpoint',str(parent/'checkpoint.npz'),'--output',str(continued/case['name']),'--t',str(case['continue_to'])]
    with (continued/(case['name']+'.log')).open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    receipt['continuations'].append(dict(name=case['name'],command=command,returncode=result.returncode));save()
receipt['status']='completed-with-failures' if any(z.get('returncode',1)!=0 for z in receipt['continuations']) else 'completed';save();print(json.dumps(receipt))
