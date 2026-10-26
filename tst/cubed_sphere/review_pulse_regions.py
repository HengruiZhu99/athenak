#!/usr/bin/env python3
"""Recount exact pulse region histories and endpoint closure sensitivity.

Uses only segments admitted by collect_pulse_chain.py. Does not infer spatial
convergence or a causal explanation from one grid's survival.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from analyze_spherical import rows

NORMS = ['H_l2', 'M_conformal_l2', 'Z_conformal_l2']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('evidence', nargs='+', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    merged = {}; provenance = {}
    for directory in args.evidence:
        review_path = directory/'chain-review.json'
        provenance[str(review_path)] = hashlib.sha256(review_path.read_bytes()).hexdigest()
        for segment in json.loads(review_path.read_text())['segments']:
            base = directory/segment['job_id'].split('.')[0]
            for name in ['degree2', 'degree4', 'half-degree2', 'half-degree4']:
                path = base/name/'cubed_sphere_conformal_regions.csv'
                if not path.exists():
                    continue
                provenance[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                target = merged.setdefault(name, {})
                for row in rows(path):
                    key = (float(row['time']), int(row['cycle']), row['region'], row['subset'])
                    assert all(math.isfinite(float(v)) for k, v in row.items()
                               if k not in ['region', 'subset'])
                    if key in target:
                        assert target[key] == row, 'restart region diagnostic changed'
                    target[key] = row
    result = {'input_sha256': provenance, 'cases': {}, 'endpoint_comparisons': {},
              'scope': 'Exact recorded samples only. RMS uses Cartesian coordinate-volume weights, not physical-volume weights. M and Z use native conformal contractions. Regions overlap; only radial bins or bulk+shell form partitions. This is diagnostic localization and boundary sensitivity, not spatial convergence or proof of causality.'}
    for name, indexed in merged.items():
        data = [row for _, row in sorted(indexed.items())]
        with (args.output/(name+'-regions.csv')).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
        end = max(float(row['time']) for row in data)
        last = {row['region']+'/'+row['subset']: row for row in data if float(row['time']) == end}
        total_volume = sum(float(last[r+'/all']['volume']) for r in ['bulk', 'shell'])
        global_rms = {key: math.sqrt(sum(float(last[r+'/all']['volume'])*float(last[r+'/all'][key])**2
                                      for r in ['bulk', 'shell'])/total_volume) for key in NORMS}
        fractions = {}
        for region in ['bulk', 'shell']:
            total = last[region+'/all']
            for label, row in last.items():
                if row['region'] != region or row['subset'] == 'all':
                    continue
                fractions[label] = {key: float(row['volume'])*float(row[key])**2 /
                                    max(float(total['volume'])*float(total[key])**2, 1e-300)
                                    for key in NORMS}
        # Describe [1.5,2] only when the run has actually completed to t=2.
        late = {}
        if end == 2.:
            for label in last:
                region, subset = label.split('/')
                selected = [r for r in data if r['region'] == region and r['subset'] == subset
                            and 1.5 <= float(r['time']) <= 2.]
                if not selected:
                    continue
                late[label] = {'first_recorded_time': float(selected[0]['time']),
                    'last_recorded_time': float(selected[-1]['time']),
                    'end_over_first': {key: float(selected[-1][key])/max(float(selected[0][key]), 1e-300)
                                       for key in NORMS},
                    'max_over_first': {key: max(float(r[key]) for r in selected)/max(float(selected[0][key]), 1e-300)
                                       for key in NORMS}}
        result['cases'][name] = {'end_time': end, 'last_rows': last, 'global_rms': global_rms,
                                 'fraction_of_parent_squared_norm': fractions, 'late_samples': late}
    for first, second in [('degree2', 'degree4'), ('degree2', 'half-degree2'),
                          ('degree4', 'half-degree4'), ('half-degree2', 'half-degree4')]:
        if not all(n in result['cases'] and result['cases'][n]['end_time'] == 2. for n in [first, second]):
            continue
        a, b = [result['cases'][n]['last_rows'] for n in [first, second]]
        assert a.keys() == b.keys()
        result['endpoint_comparisons'][first+' versus '+second] = {
            region: {key: abs(float(a[region][key])-float(b[region][key])) /
                     max(abs(float(a[region][key])), abs(float(b[region][key])), 1e-300)
                     for key in NORMS} for region in a}
    (args.output/'region-review.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    colors = {'degree2': '#0072b2', 'degree4': '#d55e00',
              'half-degree2': '#009e73', 'half-degree4': '#cc79a7'}
    for name, indexed in merged.items():
        collar = [row for _, row in sorted(indexed.items())
                  if row['subset'] == 'fixed_outer_collar' and float(row['time']) >= .5]
        for ax, key in zip(axes, NORMS):
            ax.plot([float(r['time']) for r in collar], [float(r[key]) for r in collar],
                    color=colors[name], label=name, ls='--' if name.startswith('half') else '-')
    for ax, label in zip(axes, ['Hamiltonian RMS', 'Conformal momentum RMS', 'Conformal Z RMS']):
        ax.set(xlim=(.5, 2), xlabel='Evolution time', ylabel=label, yscale='log')
        ax.grid(alpha=.2)
        for time in [.7927734374996852, 1.5837303635638276]:
            ax.axvline(time, color='#777777', ls=':', lw=1)
    axes[0].legend(fontsize=8)
    fig.suptitle('Matched-collar pulse: outer collar r > 0.9')
    fig.supxlabel('Late view t ≥ 0.5; exact recorded samples. Earlier data remain in full histories.\n'
                   'Coordinate-volume norms; single spatial grid. Dotted lines: archived Cartesian failure times.', fontsize=9)
    for suffix in ['png', 'pdf']:
        fig.savefig(args.output/('pulse-collar.'+suffix), dpi=180)
    print(json.dumps({name: r['end_time'] for name, r in result['cases'].items()}))


if __name__ == '__main__':
    main()
