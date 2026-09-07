#!/usr/bin/env python3
"""Diagnostic cadence, restart continuity and exclusive-output controls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.dont_write_bytecode=True
from intrinsic_restart import read_restart
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);records=[]
seed=a.fixtures/'fd6-3d-seed-multi.rst'
base=(a.fixtures/'fd6-3d-seeded-multi/used.athinput').read_text().replace('formulation = intrinsic_clean','formulation = intrinsic_clean\nintrinsic_diagnostics = true\nintrinsic_diagnostic_dcycle = 2')
def run(name,text,restart,expected=None,collision=False):
    d=a.output/name;d.mkdir();inp=d/'used.athinput';inp.write_text(text)
    if collision:(d/'intrinsic-diagnostics-c0-s0.csv').write_text('preserved sentinel\n')
    command=[str(a.binary.resolve()),'-i',str(inp.resolve()),'-r',str(restart.resolve())]
    with (d/'run.log').open('w') as log:r=subprocess.run(command,cwd=d,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    ok=r.returncode==0 if expected is None else r.returncode!=0 and expected in (d/'run.log').read_text()
    records.append(dict(name=name,command=command,returncode=r.returncode,status='PASS' if ok else 'FAIL',input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),restart_sha256=hashlib.sha256(restart.read_bytes()).hexdigest()))
    (a.output/'runs.json').write_text(json.dumps(records,indent=2)+'\n');assert ok
    return d
run('invalid-cadence',base.replace('dcycle = 2','dcycle = 0'),seed,'diagnostic cadence must be positive')
full=run('full',base,seed)
assert sorted(p.name for p in full.glob('intrinsic-diagnostics-*.csv'))==['intrinsic-diagnostics-c0-s0.csv','intrinsic-diagnostics-c2-s3.csv']
restart=sorted((full/'rst').glob('*.rst'))[0]
assert read_restart(restart)['cycle']==1
resumed=run('resumed',base,restart)
assert sorted(p.name for p in resumed.glob('intrinsic-diagnostics-*.csv'))==['intrinsic-diagnostics-c1-s0.csv','intrinsic-diagnostics-c2-s3.csv']
assert (full/'intrinsic-diagnostics-c2-s3.csv').read_bytes()==(resumed/'intrinsic-diagnostics-c2-s3.csv').read_bytes()
assert np.array_equal(read_restart(sorted((full/'rst').glob('*.rst'))[-1])['state'],read_restart(sorted((resumed/'rst').glob('*.rst'))[-1])['state'])
collision=run('collision',base,seed,'cannot exclusively create intrinsic-diagnostics-c0-s0.csv',True)
assert (collision/'intrinsic-diagnostics-c0-s0.csv').read_text()=='preserved sentinel\n'
(a.output/'results.json').write_text(json.dumps(dict(status='PASS',runs=len(records),cadence_correct=True,restart_history_bitwise=True,restart_state_bitwise=True,collision_preserved=True,binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest()),indent=2)+'\n')
