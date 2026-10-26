#!/usr/bin/env python3
"""Replay a frozen native wave-map failure from its last saved checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--resolution',type=int,choices=[16,24,32],default=32)
args=parser.parse_args()
root=Path(__file__).resolve().parents[2]
cap=root/('docs/validation/hyperboloidal-reference-wave-map-native-t2-wave-N%d-failure-20261009'%args.resolution)
receipt=json.loads((cap/'launch-receipt.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
binary=Path(receipt['executable']);assert sha(binary)==receipt['executable_sha256']
key=max(k for k in receipt['outputs'] if k.endswith('.rst'))
restart=Path(receipt['output_directory'])/key;assert sha(restart)==receipt['outputs'][key]['sha256']
work=root/('build-cubed-sphere/research-20261012/old-wave-replay-N%d'%args.resolution)
work.mkdir(exist_ok=False)
shutil.copy2(binary,work/'athena_frozen');shutil.copy2(restart,work/'input.rst')
command=[str(work/'athena_frozen'),'-r',str(work/'input.rst'),'-d',str(work),'time/ndiag=100']
result={'scope':'Last saved checkpoint replay, not a fresh evolution from t=0.',
        'executable_sha256':sha(work/'athena_frozen'),'restart_sha256':sha(work/'input.rst'),
        'original_restart':str(restart),'command':command}
resource.setrlimit(resource.RLIMIT_CORE,(0,0));start=time.monotonic()
with (work/'run.log').open('w') as log,(work/'stderr.log').open('w') as err:
    try:
        proc=subprocess.run(command,stdout=log,stderr=err,timeout=180)
        result['exit_code']=proc.returncode
    except subprocess.TimeoutExpired:
        result['timeout']=True
result['elapsed_seconds']=time.monotonic()-start
def fatal(path):
    lines=[s for s in path.read_text().splitlines() if 'invalid hyperboloidal' in s]
    return lines[-1] if lines else None
result['original_fatal']=fatal(cap/'native.stderr');result['replayed_fatal']=fatal(work/'stderr.log')
result['terminal_error_byte_identical']=result['original_fatal'] is not None and result['original_fatal']==result['replayed_fatal']
(work/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
destination=root/('docs/validation/cubed-sphere-refinement-20261012/old-wave-replay-N%d'%args.resolution)
destination.mkdir(exist_ok=False)
for name in ['run.log','stderr.log','receipt.json']:
    shutil.copy2(work/name,destination/name)
shutil.copy2(Path(__file__),destination/'driver.py')
print(json.dumps(result,indent=2))
assert result['terminal_error_byte_identical'],result
