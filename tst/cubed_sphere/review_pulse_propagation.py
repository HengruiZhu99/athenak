#!/usr/bin/env python3
"""Bounded pulse diagnosis; incomplete endpoints never establish convergence."""
import argparse
import csv
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('manifest', type=Path)
p.add_argument('initial_review', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()

def read(root):
    root = Path(root)
    values, peaks, sampling = {}, {}, {}
    times = []
    for file, fields in [('cubed_sphere_conformal_regions.csv', ['H_l2','H_linf','M_conformal_l2','Z_conformal_l2']),
                         ('cubed_sphere_conformal_maxima.csv', ['M_conformal_linf','Z_conformal_linf'])]:
        rows = list(csv.DictReader((root/file).open()))
        time = max(float(x['time']) for x in rows)
        times.append(time)
        for x in rows:
            region = x['region']+'/'+x['subset']
            for field in fields:
                key = region+'/'+field
                v = float(x[field])
                assert math.isfinite(v) and v >= 0
                peaks[key] = max(peaks.get(key, 0), v)
                if float(x['time']) == time:
                    values[key] = v
            if float(x['time']) == time:
                sampling.setdefault(region, {}).update({k:v for k,v in x.items()
                    if k in ['cells','volume'] or k.startswith('max_')})
    assert len(set(times)) == 1
    state = list(csv.DictReader((root/'cubed_sphere_conformal.csv').open()))
    assert all(float(x[k]) > 0 for x in state for k in ['min_alpha','min_chi','min_Omega'])
    omega = {x['region']:float(x['min_Omega']) for x in state if float(x['time']) == time}
    return dict(time=time, values=values, peaks=peaks, sampling=sampling, minimum_Omega=omega,
                max_field_drift=max(float(x['max_deviation']) for x in state))

manifest = json.loads(a.manifest.read_text())
initial_orders = {x['region_field']:x['orders'][-1] for x in json.loads(a.initial_review.read_text())['checks']}
runs = {}
for name, spec in manifest.items():
    initial = read(spec['initial'])
    assert initial['time'] == .1
    segments = [read(x) for x in spec['directories']]
    assert segments and all(.1 <= x['time'] <= .2 for x in segments)
    latest = segments[-1]
    peaks = {k:max(x['peaks'][k] for x in segments) for k in initial['values']}
    growth = {k:dict(initial=v, final=latest['values'][k], peak=peaks[k],
                     endpoint_factor=latest['values'][k]/v if v > 0 else None,
                     peak_factor=peaks[k]/v if v > 0 else None,
                     secant_log_rate=(math.log(latest['values'][k]/v)/(latest['time']-.1)
                         if v > 0 and latest['values'][k] > 0 and latest['time'] > .1 else None),
                     effective_time_power=(math.log(latest['values'][k]/v)/math.log(latest['time']/.1)
                         if v > 0 and latest['values'][k] > 0 and latest['time'] > .1 else None),
                     above_tenfold=(peaks[k] > 10*v and max(peaks[k],v) >= 1e-10))
              for k,v in initial['values'].items()}
    runs[name] = dict(latest=latest, growth=growth, directories=spec)
common = all(x['latest']['time'] == .2 for x in runs.values())
checks = []
if common:
    for key in initial_orders:
        values = [runs[n]['latest']['values'][key] for n in ['pulse-coarse','pulse-middle','pulse-fine']]
        order = initial_orders[key]
        checks.append(dict(region_field=key, values=values, decreases=values[0]>values[1]>values[2],
            orders=[math.log(values[n]/values[n+1])/math.log([1.5,4/3][n])
                    if values[n]*values[n+1] > 0 else None for n in range(2)],
            initial_finest_order=order,
            scaled_errors_using_initial_order=[v*N**order for v,N in zip(values,[16,24,32])]
                if order is not None else None))
stationary = runs['stationary-middle']['latest']
fine_concerns = [k for k,v in runs['pulse-fine']['growth'].items()
                 if '/radial_' not in k and v['above_tenfold']]
out = dict(common_target_reached=common, target=.2, runs=runs, spatial_checks=checks,
           all_fixed_constraints_decrease=(all(x['decreases'] for x in checks
               if '/radial_' not in x['region_field']) if common else None),
           all_radial_constraints_decrease=(all(x['decreases'] for x in checks
               if '/radial_' in x['region_field']) if common else None),
           finest_fixed_tenfold_concerns=fine_concerns,
           stationary_within_limits=(max(stationary['values'].values()) <= 1e-8
                                     and stationary['max_field_drift'] <= 1e-8),
           scope='Additional diagnostic only; original100x/doubling flags remain separately authoritative. '
                 'Time error and boundary sensitivity at .2 unmeasured. Mapped-coordinate volume norms, '
                 'conformal M/Z, no horizon exclusions, immutable Minkowski reference. '
                 'Secant log rates and time powers describe regional norms, not eigenmode growth: '
                 'forcing and transport across region boundaries can change these rates. '
                 'Neither finite fields nor short-time refinement establishes long stability.')
a.output.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
print({k:v for k,v in out.items() if k not in ['runs','spatial_checks','scope']})
