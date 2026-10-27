#!/usr/bin/env python3
"""Frozen nonzero angular default-path regression against historical CSV bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--baseline', type=Path, default=Path('docs/elliptic-shift-3d/evidence/pre-supported-default-control-20261027'))
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
binary = a.output/'pilot'
shutil.copy2('build-shift-3d/elliptic_shift_3d_coupled_pilot', binary)
(a.output/'.gitignore').write_text('/pilot\n')
baseline = json.loads((a.baseline/'manifest.json').read_text())
command = list(baseline['command'])
command[0] = str(binary)
command[8] = str(a.output/'coupled.csv')
command[9] = str(a.output/'checkpoint.bin')
command[10] = sha(binary)
report = dict(command=command, maximum_seconds=60, binary_sha256=sha(binary),
              baseline_manifest_sha256=sha(a.baseline/'manifest.json'))
(a.output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
start = time.monotonic()
with (a.output/'pilot.stdout').open('w') as out, (a.output/'pilot.stderr').open('w') as err:
    result = subprocess.run(command, stdout=out, stderr=err, timeout=60)
report.update(exit_code=result.returncode, seconds=time.monotonic()-start,
              binary_stable=sha(binary)==report['binary_sha256'])
report['bitwise_files'] = {name: dict(before=sha(a.baseline/name), after=sha(a.output/name),
                                     equal=(a.baseline/name).read_bytes()==(a.output/name).read_bytes())
                           for name in ('coupled.csv', 'coupled.csv.spheres.csv')}
(a.output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
if result.returncode or not report['binary_stable'] or not all(row['equal'] for row in report['bitwise_files'].values()):
    raise SystemExit(1)
