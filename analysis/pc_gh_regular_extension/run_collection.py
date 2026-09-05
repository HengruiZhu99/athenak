"""Run an explicitly selected collection; never advance physics gates automatically."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--inputs', type=Path, required=True)
ap.add_argument('--build', type=Path, required=True)
ap.add_argument('--output', type=Path, required=True)
ap.add_argument('--pattern', default='*.athinput')
ap.add_argument('--wall-segment', default='00:15:00')
args = ap.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
inputs = sorted(args.inputs.glob(args.pattern))
if not inputs:
    raise SystemExit('No inputs matched the explicit selection')
for path in inputs:
    out = args.output/path.stem
    if out.exists():
        raise SystemExit(f'Existing run {out}; inspect before explicitly resuming or selecting new runs')
    argv = [sys.executable, str(HERE/'cuda_driver.py'), 'run',
            '--build', str(args.build.resolve()), '--input', str(path.resolve()),
            '--output', str(out.resolve()), '--wall-segment', args.wall_segment]
    begin = time.monotonic()
    print('START', path.name, flush=True)
    result = subprocess.run(argv)
    record = dict(input=str(path), output=str(out), argv=argv,
                  exit_code=result.returncode, elapsed_seconds=time.monotonic()-begin)
    with (args.output/'collection.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n')
    if result.returncode:
        raise SystemExit(result.returncode)
    print('COMPLETE', path.name, flush=True)
print('COLLECTION EXITED CLEANLY; PHYSICS ANALYSIS REQUIRED', flush=True)
