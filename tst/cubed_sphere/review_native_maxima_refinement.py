#!/usr/bin/env python3
"""Regional native M/Z maximum refinement and timestep screen.

The layer aggregate reproduces the existing review's radial bins 6..11:
0.375 <= r < 0.75 for unit reference radius. It does not resolve the entire
0.35 <= r < 0.75 transition separately; global and shell include that strip.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from compare_native_maxima import read_case


def regional_maxima(case):
    receipt = json.loads((case/'receipt.json').read_text())
    assert receipt['outcome'] == 'complete' and receipt['reference_unchanged']
    assert receipt['reached_time_target'], 'step-only admission is not a completed refinement endpoint'
    assert receipt.get('initial_profile_unchanged', True)
    time = float(receipt['last_rows'][-1]['time'])
    domain, all_rows = read_case(case)
    rows = [v for key, v in all_rows.items() if float(key[0]) == time]
    selectors = {
        'bulk': lambda v: v['region'] == 'bulk' and v['subset'] == 'all',
        'shell': lambda v: v['region'] == 'shell' and v['subset'] == 'all',
        'global': lambda v: v['subset'] == 'all',
        'layer_bins_6_through_11': lambda v: v['subset'] in ['radial_'+str(i) for i in range(6, 12)],
    }
    for subset in ['fixed_interior', 'fixed_interface', 'fixed_outer_collar']:
        selectors[subset] = lambda v, subset=subset: v['subset'] == subset
    result = {}
    for region, select in selectors.items():
        selected = [v for v in rows if select(v)]
        assert selected, region
        result[region] = {}
        for field in ['M', 'Z']:
            key = field+'_conformal_linf'
            peak = max(selected, key=lambda v: float(v[key]))
            result[region][field] = dict(value=float(peak[key]),
                xyz=[float(peak['max_'+field+'_'+axis]) for axis in 'xyz'],
                native_region=peak['region'], native_subset=peak['subset'])
    return dict(time=time, domain=domain, regions=result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    launch = json.loads((a.results/'launch.json').read_text())
    grids = {g['name']: g for g in launch['grids']}
    names = ['coarse', 'medium', 'fine', 'fine-halfstep']
    cases = {n: regional_maxima(a.results/n) for n in names}
    assert len({cases[n]['time'] for n in names}) == 1
    assert all(cases[n]['domain'] == cases['coarse']['domain'] for n in names)
    assert grids['fine-halfstep']['dt'] == grids['fine']['dt']/2
    for component in ['bulk', 'na', 'nr']:
        assert grids['fine-halfstep'][component] == grids['fine'][component]
    checks = {}
    for region in cases['coarse']['regions']:
        checks[region] = {}
        for field in ['M', 'Z']:
            values = [cases[n]['regions'][region][field]['value'] for n in names]
            orders = [math.log(values[i]/values[i+1]) /
                      math.log(grids[names[i+1]]['bulk']/grids[names[i]]['bulk'])
                      if values[i] > 0 and values[i+1] > 0 else None
                      for i in range(2)]
            spatial = abs(values[1]-values[2])
            temporal = abs(values[3]-values[2])
            checks[region][field] = dict(values=values, orders=orders,
                strictly_decreases=values[0] > values[1] > values[2],
                temporal_change=temporal, finest_spatial_change=spatial,
                temporal_to_spatial=temporal/spatial if spatial else None,
                temporal_below_ten_percent=temporal < .1*spatial)
    out = dict(cases=cases, checks=checks,
        all_maxima_strictly_decrease=all(v['strictly_decreases'] for d in checks.values() for v in d.values()),
        all_temporal_changes_below_ten_percent=all(v['temporal_below_ten_percent'] for d in checks.values() for v in d.values()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope=__doc__, complete_physical_acceptance=False)
    a.output.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:out[k] for k in ['all_maxima_strictly_decrease', 'all_temporal_changes_below_ten_percent']}))


if __name__ == '__main__':
    main()
