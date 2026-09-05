"""Aggregate complete pulse runs with rates, drift, amplitude checks and convergence."""
import argparse
import json
import math
from pathlib import Path

from make_inputs import parse
from verify_pulses import measure

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('groups', type=Path, nargs='+')
ap.add_argument('--output', type=Path, required=True)
args = ap.parse_args()
rows = []
for group in args.groups:
    for done in sorted(group.glob('*/completed.json')):
        run = done.parent
        params = parse((run/'used_input.athinput').read_text())
        result = measure(run)
        result.update(run=str(run), n=int(params['mesh']['nx1']),
                      cfl=float(params['time']['cfl_number']))
        rows.append(result)
ladder = {}
for row in rows:
    if row['time'] != 1 or row['amplitude'] != 1e-8:
        continue
    key = (row['family'], row['direction'], row['rate'])
    ladder.setdefault(key, []).append(row)
orders = []
for key, group in sorted(ladder.items()):
    group.sort(key=lambda row: row['n'])
    for a, b in zip(group[:-1], group[1:]):
        if a['n'] == b['n']:
            raise ValueError('Duplicate resolution in a pulse ladder')
        order = math.log(a['error_l2_over_amplitude']/b['error_l2_over_amplitude'])/math.log(b['n']/a['n'])
        orders.append(dict(family=key[0], direction=key[1], rate=key[2],
                           coarse=a['n'], fine=b['n'], l2_order=order))
report = dict(runs=rows, convergence=orders,
              scope='continuum linear prediction; no puncture or AMR stability conclusion')
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2)+'\n')
normal = [r for r in rows if r['rate'] in [0, 1, 4]]
print('Complete pulse runs:', len(rows))
print('Maximum fitted damping error (rates 0,1,4):',
      max(abs(r['damping_fit']-r['rate']) for r in normal))
for family in ['p','Q','L','B']:
    selected = [r for r in rows if r['family']==family and r['direction']==1 and r['rate']==1 and r['amplitude']==1e-8]
    finest = max(selected, key=lambda row: row['n'])
    print(family, 'finest n=', finest['n'], 'rate=', finest['damping_fit'],
          'speed=', finest['speed_fit'])
    print(' adjacent L2 orders:', [(r['coarse'],r['fine'],round(r['l2_order'],6))
          for r in orders if r['family']==family and r['direction']==1 and r['rate']==1])
