#!/usr/bin/env python3
"""Apply the declared N1025 duration budget to a validated collector export.

Run paper/collect_long_puncture.py first, with explicit --case and --protocol.
This assesses saved-state accuracy and recorded stage extrema, not convergence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def assess(summary, manifest, protocol):
    args = summary['parameters']
    if args != manifest['arguments'] or summary['status'] != manifest['status']:
        raise ValueError('Export and captured manifest disagree')
    if summary['resolution'] != 1025 or args['n'] != 1025:
        raise ValueError('This budget applies only to the principal N1025 trial')
    if args['mass'] != 1/3:
        raise ValueError('Collector mass-error convention requires M=1/3')
    budget = protocol['principal_n1025']
    if args['t'] != budget['target']:
        raise ValueError('Input target differs from declared duration')
    rows = summary['rows']
    times = [r['t'] for r in rows]
    end = summary['last_saved_time']
    cadence = args['report_dt']
    expected = [round(i*cadence, 9) for i in range(math.floor(end/cadence+1e-8)+1)]
    if not expected or abs(expected[-1]-end) > 1e-9:
        expected.append(end)
    if times != expected:
        raise ValueError('Incomplete or duplicated saved trajectory')
    if end > manifest['time']+1e-9 or summary['integrator_time'] != manifest['time']:
        raise ValueError('Invalid captured time coverage')
    checks = {}

    def limit(name, values, maximum):
        if not values or any(v is None or not math.isfinite(v) for v in values):
            checks[name] = dict(passed=False, measured=None, limit=maximum)
        else:
            value = max(abs(v) for v in values)
            checks[name] = dict(passed=value <= maximum, measured=value, limit=maximum)

    limit('relative_AH_mass_error', [r['AH_mass_error']/args['mass']
          if r['AH_mass_error'] is not None else None for r in rows], budget['max_relative_AH_error'])
    for key, maximum in budget['max_full_trajectory_exterior_RMS'].items():
        limit(key+'_exterior_full', [r[key+'_exterior_rms'] for r in rows], maximum)
    late = [r for r in rows if r['t'] >= 10]
    for key, maximum in budget['max_late_exterior_RMS_after_t10'].items():
        limit(key+'_exterior_late', [r[key+'_exterior_rms'] for r in late], maximum)
    stage = manifest['all_RK_stages']
    for key in ['scri_clock', 'scri_null']:
        limit(key, [r[key] for r in rows]+[stage['max_'+key]], budget[key])
    limit('all_stage_scri_pole', [stage['max_scri_pole']], budget['all_stage_scri_pole'])
    # These are the native integrator's existing guards, not relaxed tolerances.
    limit('mapped_CFL', [stage['max_cfl']], .4)
    limit('relaxation_dt', [stage['max_relaxation_dt']], .4)
    checks['positive_stage_geometry'] = dict(passed=all(
        math.isfinite(stage[k]) and stage[k] > 0 for k in ['min_chi', 'min_alpha']),
        min_chi=stage['min_chi'], min_alpha=stage['min_alpha'])
    accuracy = all(check['passed'] for check in checks.values())
    reached = (summary['status'] == 'finished' and
               abs(end-budget['target']) < 1e-9 and
               abs(manifest['time']-budget['target']) < 1e-9)
    verdict = 'passed_duration_budget' if reached and accuracy else (
        'failed' if summary['status'] == 'failed' or not accuracy else 'pending')
    return dict(verdict=verdict, reached_target=reached, accuracy_to_saved_cutoff=accuracy,
                saved_cutoff=end, stage_extrema_through=manifest['time'], checks=checks,
                floor_diagnostics={k:v for k,v in stage.items() if 'floor' in k or 'clamps' in k},
                scope='N1025 finite-duration evidence only. Exterior RMS excludes the apparent-horizon interior and normalizes over exterior weights. Saved-state accuracy is not an all-stage constraint bound or convergence evidence.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    names = ['summary.json', 'manifest-long.json', 'protocol.json', 'source-manifest.json']
    files = {name:(args.export/name).read_bytes() for name in names}
    data = {name:json.loads(content) for name,content in files.items()}
    receipts = data['source-manifest.json']
    for name in ['manifest-long.json', 'protocol.json']:
        digest = hashlib.sha256(files[name]).hexdigest()
        if not any(r['sha256'] == digest for r in receipts):
            raise ValueError('Copied file does not match collector receipt: '+name)
    result = assess(data['summary.json'], data['manifest-long.json'], data['protocol.json'])
    result['export_sha256'] = {name:hashlib.sha256(content).hexdigest() for name,content in files.items()}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
