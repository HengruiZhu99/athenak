#!/usr/bin/env python3
"""Matched-time closure consistency, spatial convergence and timestep controls."""
import argparse
import csv
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--quintic', nargs=3, type=Path, required=True)
p.add_argument('--interior', nargs=3, type=Path, required=True)
p.add_argument('--half-interior', type=Path, required=True)
p.add_argument('--radial', type=Path, required=True)
p.add_argument('--half-radial', type=Path, required=True)
p.add_argument('output', type=Path)
a = p.parse_args()

def read(root):
    values, sampling = {}, {}
    for filename, fields in [
            ('cubed_sphere_conformal_regions.csv', ['H_l2', 'H_linf', 'M_conformal_l2', 'Z_conformal_l2']),
            ('cubed_sphere_conformal_maxima.csv', ['M_conformal_linf', 'Z_conformal_linf'])]:
        rows = list(csv.DictReader((root / filename).open()))
        assert max(float(x['time']) for x in rows) == .1, root
        for x in rows:
            if float(x['time']) != .1:
                continue
            region = x['region'] + '/' + x['subset']
            sampling.setdefault(region, {}).update({k: v for k, v in x.items()
                                                    if k in ['cells', 'volume'] or k.startswith('max_')})
            for field in fields:
                value = float(x[field])
                assert math.isfinite(value) and value >= 0
                values[region + '/' + field] = value
    state = [x for x in csv.DictReader((root / 'cubed_sphere_conformal.csv').open())
             if float(x['time']) == .1]
    minimum_omega = {x['region']: float(x['min_Omega']) for x in state}
    receipt = json.loads((root / 'receipt.json').read_text())
    assert receipt['reference_unchanged'] and receipt['initial_profile_unchanged']
    return dict(values=values, sampling=sampling, minimum_omega=minimum_omega,
                directory=str(root.resolve()), grid=[receipt['bulk'], receipt['nangular'], receipt['nr']])

q, i = [read(x) for x in a.quintic], [read(x) for x in a.interior]
ih, radial, rh = [read(x) for x in [a.half_interior, a.radial, a.half_radial]]
for b, c in zip(q, i):
    assert b['grid'] == c['grid'] and b['minimum_omega'] == c['minimum_omega']
assert ih['grid'] == i[1]['grid'] and radial['grid'] == rh['grid']
runs = q + i + [ih, radial, rh]
assert all(x['values'].keys() == q[0]['values'].keys() for x in runs)
checks = []
def orders(values):
    return [math.log(values[n]/values[n+1])/math.log([32/24, 40/32][n])
            if values[n] > 0 and values[n+1] > 0 else None for n in range(2)]

for key in q[0]['values']:
    vq, vi = [x['values'][key] for x in q], [x['values'][key] for x in i]
    differences = [abs(x-y) for x, y in zip(vq, vi)]
    time_i = abs(ih['values'][key] - vi[1])
    time_r = abs(rh['values'][key] - radial['values'][key])
    # No numerical floor in the evolution: this tolerance only classifies
    # comparisons below diagnostic roundoff, and is explicitly reported.
    roundoff = 1.e-12
    monotone = lambda v: all(v[n+1] < v[n] or max(v[n:n+2]) < roundoff for n in range(2))
    scale_i = min(differences[1], abs(vi[0]-vi[1]), abs(vi[1]-vi[2]))
    scale_r = abs(radial['values'][key] - vq[1])
    checks.append(dict(region_field=key, quintic=vq, interior=vi,
                       absolute_closure_difference=differences,
                       quintic_orders=orders(vq), interior_orders=orders(vi),
                       closure_difference_orders=orders(differences),
                       closure_difference_decreases=monotone(differences),
                       quintic_constraints_decrease=monotone(vq),
                       interior_constraints_decrease=monotone(vi),
                       interior_time_difference=time_i, radial_time_difference=time_r,
                       interior_spatial_comparison_scale=scale_i, radial_comparison_scale=scale_r,
                       interior_time_small_relative_refinement=(
                           time_i < .1*min(abs(vi[0]-vi[1]), abs(vi[1]-vi[2])) or time_i < roundoff),
                       interior_time_small_relative_closure=(time_i < .1*differences[1] or time_i < roundoff),
                       interior_time_small=(time_i < .1*scale_i or time_i < roundoff),
                       radial_time_small=(time_r < .1*scale_r or time_r < roundoff)))
fixed = [x for x in checks if '/radial_' not in x['region_field']]
result = dict(time=.1, resolutions=[24, 32, 40], checks=checks, runs=runs,
              comparison_roundoff_tolerance=1.e-12,
              all_fixed_closure_differences_decrease=all(x['closure_difference_decreases'] for x in fixed),
              all_fixed_interior_constraints_decrease=all(x['interior_constraints_decrease'] for x in fixed),
              all_fixed_time_controls_small=all(x['interior_time_small'] and x['radial_time_small'] for x in fixed),
              scope='Native operator sensitivity at finite radius, no SBP energy estimate or incoming characteristic BC. '
                    'Mapped-coordinate-volume RMS, conformal M/Z norms, all horizon-interior cells retained. '
                    'Timestep controls include differing restart milestones. Original failed baselines remain authoritative; '
                    'no long-time stability or near-scri acceptance from this short-time comparison.')
a.output.write_text(json.dumps(result, indent=2) + '\n')
print({k: v for k, v in result.items() if k.startswith('all_fixed')})
