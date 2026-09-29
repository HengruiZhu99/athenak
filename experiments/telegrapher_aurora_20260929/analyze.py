#!/usr/bin/env python3
"""Summarize actual horizon convergence and evolution progress, without dependencies."""
import argparse
import json
import math
import re
from pathlib import Path


def numeric_rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            try:
                rows.append([float(x) for x in line.split()])
            except ValueError:
                pass
    return rows


def analyze(run):
    log = (run / 'run.log').read_text()
    result = {'run': str(run)}
    progress = re.findall(r'time=([\d.eE+-]+)', log)
    result['evolution_time'] = float(progress[-1]) if progress else None
    result['completed_20M'] = ('Terminating on time limit' in log and
                               result['evolution_time'] >= 19.999)
    result['termination'] = [x for x in log.splitlines() if 'Terminating on' in x]
    result['exit_status'] = ((run / 'exit_status.txt').read_text().strip()
                             if (run / 'exit_status.txt').exists() else None)
    summaries = list(run.glob('*horizon_summary_0.txt'))
    if summaries:
        rows = numeric_rows(summaries[0])
        verbose = list(run.glob('*horizon_verbose_0.txt'))
        text = verbose[0].read_text() if verbose else ''
        result['finder'] = 'fastflow'
        result['successes'] = text.count('Found horizon')
        result['mass_stall_successes'] = text.count('(mass stall)')
        result['searches'] = text.count('Searching for horizon')
        result['failures'] = result['searches'] - result['successes']
        # Fastflow column 9 is the area-weighted mean SQUARE expansion.
        usable = [r for r in rows if len(r) >= 12 and
                  all(math.isfinite(r[i]) for i in [1, 2, 7, 8, 9, 10, 11])]
        if usable:
            r = usable[-1]
            result.update(last_horizon_time=r[1], mass=r[2], area=r[7],
                          expansion_rms_times_mass=math.sqrt(max(0, r[8])) * r[2],
                          mean_expansion_times_mass=abs(r[9]) * r[2] / r[7],
                          min_radius=r[11])
    else:
        paths = list((run / 'horizon').glob('BHaHAHA_diagnostics.ah*.gp'))
        rows = numeric_rows(paths[0]) if paths else []
        result['finder'] = 'bhahaha'
        result['successes'] = log.splitlines().count('Success')
        result['failures'] = len(re.findall(r'Failed.*(?:code|error)', log, re.IGNORECASE))
        usable = [r for r in rows if len(r) >= 26 and
                  all(math.isfinite(r[i]) for i in [1, 5, 11, 12, 13, 14, 24])]
        if usable:
            r = usable[-1]
            result.update(last_horizon_time=r[1], mass=r[24], mass_irr=r[12],
                          area=r[11], expansion_linf_times_mass=r[13],
                          expansion_rms_times_mass=r[14], min_radius=r[5])
    if 'mass' in result:
        result['mass_relative_error_to_1M'] = abs(result['mass'] - 1)
        result['small_expansion'] = result['expansion_rms_times_mass'] < 1e-2
    hst = list(run.glob('*.hst'))
    history = numeric_rows(hst[0]) if hst else []
    if history:
        result['constraints_finite'] = all(all(math.isfinite(x) for x in r)
                                            for r in history)
        for i, label in [(2, 'C'), (3, 'H'), (4, 'M')]:
            finite = [r for r in history if len(r) > i and math.isfinite(r[i])]
            if finite:
                peak = max(finite, key=lambda r: r[i])
                result[label + '_norm2_peak'] = {'time': peak[0], 'value': peak[i]}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps([analyze(p.parent) for p in sorted(args.root.rglob('run.log'))],
                     indent=2, allow_nan=False))
