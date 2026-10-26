#!/usr/bin/env python3
"""Common-time constraint and pointwise-state closure comparisons; retain failures."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('manifest', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
m = json.loads(a.manifest.read_text())

def read(directory):
    root = Path(directory)
    values, sampling, times = {}, {}, []
    for filename, fields in [
        ('cubed_sphere_conformal_regions.csv', ['H_l2','H_linf','M_conformal_l2','Z_conformal_l2']),
        ('cubed_sphere_conformal_maxima.csv', ['M_conformal_linf','Z_conformal_linf'])]:
        rows = list(csv.DictReader((root/filename).open()))
        time = max(float(x['time']) for x in rows)
        times.append(time)
        for x in rows:
            if float(x['time']) != time:
                continue
            region = x['region']+'/'+x['subset']
            sampling.setdefault(region, {}).update({k:v for k,v in x.items()
                if k in ['cells','volume'] or k.startswith('max_')})
            for field in fields:
                value = float(x[field])
                assert math.isfinite(value) and value >= 0
                values[region+'/'+field] = value
    assert len(set(times)) == 1
    state = list(csv.DictReader((root/'cubed_sphere_conformal.csv').open()))
    omega = {x['region']:float(x['min_Omega']) for x in state if float(x['time']) == time}
    receipt = json.loads((root/'receipt.json').read_text())
    assert receipt['reference_unchanged'] and receipt['all_rows_finite']
    return dict(time=time, values=values, sampling=sampling, minimum_Omega=omega,
                grid=[receipt['bulk'],receipt['nangular'],receipt['nr']],
                checkpoint=receipt['checkpoint_payload'], directory=str(root.resolve()))

q, i = [read(x) for x in m['quintic']], [read(x) for x in m['interior']]
assert len(q) == len(i) == 3
for first, second in zip(q, i):
    assert first['grid'] == second['grid'] and first['minimum_Omega'] == second['minimum_Omega']
common = all(x['time'] == .2 for x in q+i)
out = dict(common_target_reached=common, target=.2, resolutions=[16,24,32],
           runs=dict(quintic=q,interior=i), constraint_checks=None, field_checks=None,
           source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
           scope='Native H/M/Z mapped-coordinate-volume norms, conformal M/Z, all regions retained. '
                 'State-component norms of pointwise differences are separate diagnostics. '
                 'No invariant field norm or characteristic-boundary consistency proof. '
                 'Original growth-monitor failures remain authoritative; no long-stability acceptance.')
if common:
    def decreasing(v):
        return v[0] > v[1] > v[2] or all(x == 0 for x in v)
    def orders(v):
        return [math.log(v[n]/v[n+1])/math.log([1.5,4/3][n])
                if v[n]*v[n+1] > 0 else None for n in range(2)]
    assert all(x['values'].keys() == q[0]['values'].keys() for x in q+i)
    rows = []
    for key in q[0]['values']:
        vq, vi = [x['values'][key] for x in q], [x['values'][key] for x in i]
        d = [abs(x-y) for x,y in zip(vq,vi)]
        rows.append(dict(region_field=key,quintic=vq,interior=vi,absolute_norm_difference=d,
                         quintic_decreases=decreasing(vq),interior_decreases=decreasing(vi),
                         norm_difference_decreases=decreasing(d),quintic_orders=orders(vq),
                         interior_orders=orders(vi),norm_difference_orders=orders(d)))
    fields = [json.loads(Path(x).read_text()) for x in m['field_differences']]
    for n, x in enumerate(fields):
        assert x['time'] == .2 and [z['sha256'] for z in x['payloads']] == [q[n]['checkpoint']['sha256'],i[n]['checkpoint']['sha256']]
    assert all(x['regions'].keys() == fields[0]['regions'].keys() for x in fields)
    frows = []
    for region in fields[0]['regions']:
        for component in range(25):
            for norm in ['l2','linf']:
                v = [x['regions'][region]['components'][component][norm] for x in fields]
                frows.append(dict(region=region,component=component,norm=norm,values=v,
                                  decreases=decreasing(v),below_diagnostic_roundoff=max(v)<=1e-12,
                                  orders=orders(v)))
    time = json.loads(Path(m['time_control']).read_text())
    assert time['time'] == .2
    out.update(constraint_checks=rows,field_checks=frows,
               all_constraint_sequences_decrease=all(x['quintic_decreases'] and x['interior_decreases'] for x in rows),
               all_constraint_norm_differences_decrease=all(x['norm_difference_decreases'] for x in rows),
               all_resolved_state_differences_decrease=all(x['decreases'] or x['below_diagnostic_roundoff'] for x in frows),
               all_baseline_constraint_time_controls_small=all(x['temporal_small'] for x in time['checks']),
               diagnostic_roundoff_tolerance=1e-12,
               unresolved='Alternative closure temporal/background controls are unmeasured; tiny closure effects '
                          'below measured timestep uncertainty are not resolved. No adoption from this comparison.')
a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print({k:v for k,v in out.items() if k.startswith('all_') or k=='common_target_reached'})
