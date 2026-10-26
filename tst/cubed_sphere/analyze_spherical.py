#!/usr/bin/env python3
"""Recount retained spherical-control diagnostics and localize stage failures.

An evidence check passing does not accept the evolution as stable or converged.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re


def rows(path):
    if path.exists():
        with path.open() as f:
            return list(csv.DictReader(f))
    with gzip.open(str(path)+'.gz', 'rt') as f:
        return list(csv.DictReader(f))


def close(a, b):
    assert abs(a-b) <= 2e-11*max(abs(a), abs(b), 1e-20), (a, b)


def analyze_case(case):
    data = rows(case/'cubed_sphere_conformal.csv')
    assert data and all(math.isfinite(float(v)) for r in data for k, v in r.items() if k != 'region')
    profiles = rows(case/'cubed_sphere_conformal_regions.csv')
    groups = {}
    for r in profiles:
        groups.setdefault((r['time'], r['region']), {})[r['subset']] = r
    result = {'path': str(case), 'diagnostic_rows': len(data), 'all_rows_finite': True,
              'last_rows': data[-2:]}
    for r in data:
        group = groups[(r['time'], r['region'])]
        total = group['all']
        radial = [v for k, v in group.items() if k.startswith('radial_')]
        assert sum(int(v['cells']) for v in radial) == int(total['cells'])
        close(sum(float(v['volume']) for v in radial), float(total['volume']))
        for key in ['H_l2', 'M_conformal_l2', 'Z_conformal_l2']:
            close(sum(float(v['volume'])*float(v[key])**2 for v in radial),
                  float(total['volume'])*float(total[key])**2)
            close(float(r[key]), float(total[key]))
        close(float(r['H_linf']), float(total['H_linf']))
        close(float(r['volume']), float(total['volume']))
    result['radial_partition_and_norms_match'] = True
    t = float(data[-1]['time'])
    result['passed_old_failure_times'] = {
        'N24': t >= .79277343749968521, 'N32': t >= 1.5837303635638276}
    receipt = case/'receipt.json'
    if receipt.exists():
        receipt = json.loads(receipt.read_text())
        result.update({k: receipt[k] for k in ['outcome', 'reached_target', 'reference_unchanged',
                      'elapsed_seconds', 'terminal_messages'] if k in receipt})
        close(float(receipt['last_rows'][-1]['time']), t)
        if 'input_sha256' in receipt:
            assert hashlib.sha256((case/'input.athinput').read_bytes()).hexdigest() == receipt['input_sha256']
    bad = case/'cubed_sphere_first_bad_stage.csv'
    if bad.exists() or Path(str(bad)+'.gz').exists():
        opener = bad.open if bad.exists() else lambda: gzip.open(str(bad)+'.gz', 'rt')
        with opener() as f:
            header = next(f).strip()
            match = re.search(r'first_kji=(\d+),(\d+),(\d+)', header)
            assert match, header
            k, j, i = match.groups()
            selected = [r for r in csv.DictReader(f) if (r['k'], r['j'], r['i']) == (k, j, i)]
        assert len(selected) == 25
        values = {int(r['variable']): float(r['current']) for r in selected}
        # Field order is fixed by Z4c::I_Z4C_* in z4c.hpp.
        xx, xy, xz, yy, yz, zz = [values[n] for n in range(1, 7)]
        det = xx*(yy*zz-yz*yz)-xy*(xy*zz-yz*xz)+xz*(xy*yz-yy*xz)
        finite_or_text = lambda v: v if math.isfinite(v) else str(v)
        result['first_bad_stage'] = {'header': header,
            'xyz': [float(selected[0][a]) for a in ['x', 'y', 'z']],
            'alpha': finite_or_text(values[18]), 'chi': finite_or_text(values[0]),
            'metric_determinant': finite_or_text(det),
            'all_25_fields_finite': all(math.isfinite(v) for v in values.values())}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('cases', nargs='+', type=Path)
    args = ap.parse_args()
    result = [analyze_case(case) for case in args.cases]
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps([{'path': r['path'], 'last_time': r['last_rows'][-1]['time'],
                       'outcome': r.get('outcome', 'see completion.json')} for r in result]))


if __name__ == '__main__':
    main()
