#!/usr/bin/env python3
"""Qualify a contiguous checkpoint chain without counting a late restart alone."""
import argparse
import json
import math
from pathlib import Path
from analyze import analyze, numeric_rows


def chain(runs, max_gap):
    records = [analyze(r) for r in runs]
    times = []
    links = []
    for i, (run, record) in enumerate(zip(runs, records)):
        if record['finder'] == 'fastflow':
            times.extend(record.get('successful_horizon_times', []))
        else:
            paths = list((run / 'horizon').glob('BHaHAHA_diagnostics.ah1.gp'))
            times.extend(r[1] for p in paths for r in numeric_rows(p)
                         if len(r) >= 26 and math.isfinite(r[14]))
        if i:
            ref = run / 'restart_source.txt'
            source = ref.read_text().strip() if ref.exists() else ''
            previous = runs[i - 1]
            # The archived local job groups retain their remote directory names.
            linked = f'/{previous.parent.name}/{previous.name}/rst/' in source
            histories = list(run.glob('*.hst'))
            history = numeric_rows(histories[0]) if histories else []
            start = history[0][0] if history else None
            previous_end = records[i - 1]['evolution_time']
            continuous = (start is not None and previous_end is not None and
                          abs(start - previous_end) <= 1e-3)
            links.append({'restart_source': source, 'matches_previous_run': linked,
                          'first_history_time': start,
                          'previous_final_time': previous_end,
                          'time_continuous': continuous})
    times = sorted(set(times))
    gaps = [b - a for a, b in zip(times, times[1:])]
    observed_gap = max(gaps, default=0)
    samples_cover = bool(times and times[0] <= 1e-3 and times[-1] >= 19 and
                         observed_gap <= max_gap)
    segments_pass = all(r.get('failures') == 0 and r.get('pending_searches', 0) == 0
                        and r.get('reported_successes_above_rms_tolerance') == 0
                        and r.get('constraints_finite') is True
                        and r.get('exit_status') == '0' for r in records)
    connected = all(x['matches_previous_run'] and x['time_continuous'] for x in links)
    return {'segments': records, 'checkpoint_links': links,
            'first_horizon_time': times[0] if times else None,
            'last_horizon_time': times[-1] if times else None,
            'unique_successful_samples': len(times),
            'max_horizon_sample_gap': observed_gap, 'allowed_sample_gap': max_gap,
            'qualified_tracking_0_to_20M': bool(records[-1]['completed_20M'] and
                samples_cover and segments_pass and connected),
            'note': 'Qualification retains each segment’s recorded residual tolerance; '
                    'this is a tracking test, not angular or spatial convergence.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path,
                        help='Case directories in chronological checkpoint order')
    parser.add_argument('--max-gap', type=float, default=0.55)
    args = parser.parse_args()
    print(json.dumps(chain(args.runs, args.max_gap), indent=2, allow_nan=False))
