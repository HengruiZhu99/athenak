#!/usr/bin/env python3
"""Report native histories in total-mass units without changing evolved fields.

Only combine continuation segments of the same run; timestep experiments are
separate histories. Compact-chart quadrature volumes retain their native units.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('histories', type=Path, nargs='+')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--internal-mass', type=float, default=1/3)
args = parser.parse_args()
mass = args.internal_mass
if not math.isfinite(mass) or mass <= 0:
    parser.error('internal mass must be finite and positive')
rows = {}
for path in args.histories:
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        values = list(map(float, line.split()))
        if len(values) != 12 or not all(math.isfinite(v) for v in values):
            raise ValueError(f'Invalid native history row in {path}')
        rows[values[0]] = values
if not rows:
    raise ValueError('No history samples')
factors = [1/mass, 1/mass, 1, mass**2, mass**2, mass, mass,
           1, mass**2, mass**2, mass, mass]
labels = ['t/M0', 'dt/M0', 'coordinate_volume', 'M0^2_H_rms',
          'M0^2_M_rms', 'M0_Theta_rms', 'M0_Z_rms', 'exterior_coordinate_volume',
          'M0^2_exterior_H_rms', 'M0^2_exterior_M_rms',
          'M0_exterior_Theta_rms', 'M0_exterior_Z_rms']
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open('w') as stream:
    stream.write(f'# Total-mass convention M0=1; internal M0={mass:.17g}\n')
    stream.write('# Compact-coordinate quadrature volumes are not physical volumes.\n')
    stream.write('# ' + ' '.join(f'[{i+1}]={v}' for i, v in enumerate(labels)) + '\n')
    for time in sorted(rows):
        stream.write(' '.join(f'{v*f:.12e}' for v, f in zip(rows[time], factors)) + '\n')
metadata = {'internal_mass': mass, 'samples': len(rows),
            'last_t_over_M0': max(rows)/mass,
            'source_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in args.histories},
            'column_factors': factors, 'columns': labels}
args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
print(json.dumps({'samples': len(rows), 'last_t_over_M0': max(rows)/mass}))
