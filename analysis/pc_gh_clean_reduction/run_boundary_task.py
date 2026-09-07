#!/usr/bin/env python3
"""Check physical boundary completion in an actual one-step legacy task graph."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', type=Path, required=True)
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
binary = args.binary.resolve()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
source = args.input.read_text()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rows = []
for dimension in [2, 3]:
    for boundary in ['outflow', 'mixed']:
        folder = output / f'{boundary}-{dimension}d'
        folder.mkdir()
        text = source.replace('nx3 = 1', 'nx3 = 8') if dimension == 3 else source
        for axis in range(1, dimension+1):
            for side in ['i', 'o']:
                reflect = boundary == 'mixed' and (
                    (axis != 2 and side == 'i') or (axis == 2 and side == 'o'))
                text = text.replace(f'{side}x{axis}_bc = periodic',
                                    f'{side}x{axis}_bc = ' + ('reflect' if reflect else 'outflow'))
        inp = folder / 'used_input.athinput'
        inp.write_text(text)
        start = time.time()
        with (folder / 'run.log').open('w') as log:
            run = subprocess.run([str(binary), '-i', str(inp)], cwd=folder,
                                 stdout=log, stderr=subprocess.STDOUT, timeout=120)
        raw = folder / 'legacy_equivalence.pcgh-reduction.csv.state-budget.rank0.bin'
        checker = Path(__file__).with_name('check_transfer_task_budget.py').resolve()
        check = subprocess.run([sys.executable, str(checker), str(raw), '--physical-boundary'],
                               capture_output=True, text=True)
        (folder / 'checker.log').write_text(check.stdout+check.stderr)
        row = dict(dimension=dimension, boundary=boundary,
                   status='PASS' if run.returncode == 0 and check.returncode == 0 else 'FAIL',
                   run_returncode=run.returncode, check_returncode=check.returncode,
                   wall_seconds=time.time()-start, binary_sha256=sha(binary),
                   input_sha256=sha(inp), raw_path=str(raw),
                   raw_sha256=sha(raw) if raw.exists() else None,
                   raw_bytes=raw.stat().st_size if raw.exists() else 0)
        if check.returncode == 0:
            row['brackets'] = json.loads(check.stdout)
        rows.append(row)
        (output / 'results.json').write_text(json.dumps(rows, indent=2)+'\n')
        print(dimension, boundary, row['status'], flush=True)
sys.exit(0 if all(row['status'] == 'PASS' for row in rows) else 1)
