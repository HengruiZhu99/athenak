#!/usr/bin/env python3
"""Continue best frozen MOTS trials through increasing angular band limits."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


def best_row(output):
    history = next((output/'search').glob('*.cartoon_m0_horizon_0.txt'))
    rows = [r.split() for r in history.read_text().splitlines() if r and not r.startswith('#')]
    valid = [(i, r) for i, r in enumerate(rows) if float(r[7]) > 0 and math.isfinite(float(r[13]))]
    if not valid:
        raise RuntimeError(f'No evaluated surface: {output}')
    i, row = min(valid, key=lambda v: float(v[1][13]))
    return i, row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--athena', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--levels', nargs='+', type=int, default=[8, 16, 32, 64])
    p.add_argument('--initial-result', type=Path)
    p.add_argument('--iterations', type=int, default=500)
    p.add_argument('--ntheta-factor', type=int, default=2)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    previous = args.initial_result
    records = []
    for l in args.levels:
        output = args.output/f'L{l}'
        cmd = [sys.executable, str(Path(__file__).with_name('search_checkpoint.py')),
               '--athena', str(args.athena), '--checkpoint', str(args.checkpoint),
               '--output', str(output), '--lmax', str(l), '--iterations', str(args.iterations),
               '--ntheta', str(args.ntheta_factor*l+4), '--profile-points', '1061', '--radii', '4']
        if previous:
            _, row = best_row(previous)
            coefficients = row[17:-6]
            if len(coefficients) > l+1:
                raise ValueError('Continuation requires nondecreasing angular order')
            seed = args.output/f'L{l}.seed'
            seed.write_text(f'{row[4]} {len(coefficients)}\n'+'\n'.join(coefficients)+'\n')
            cmd += ['--seed', str(seed), '--seed-only']
        subprocess.run(cmd, check=True)
        index, row = best_row(output)
        state = json.loads((output/'search/frozen_mots.json').read_text())
        record = dict(lmax=l, ntheta=args.ntheta_factor*l+4, area=float(row[7]),
                      epsilon2=float(row[13]), epsilon_inf=float(row[-5]),
                      failure=row[15], iterations=int(row[-1]), row=index,
                      verified=bool(int(row[-6])), previous=str(previous), output=str(output),
                      state=state)
        records.append(record)
        (args.output/'summary.json').write_text(json.dumps(records, indent=2)+'\n')
        print(json.dumps(record), flush=True)
        previous = output


if __name__ == '__main__':
    main()
