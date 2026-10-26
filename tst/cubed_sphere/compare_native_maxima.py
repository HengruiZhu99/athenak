#!/usr/bin/env python3
"""CPU/GPU admission for native regional M/Z maxima, not convergence proof."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read_case(case):
    domain = json.loads((case/'cubed_sphere_conformal_domain.json').read_text())
    rows = list(csv.DictReader((case/'cubed_sphere_conformal_maxima.csv').open()))
    norms = list(csv.DictReader((case/'cubed_sphere_conformal_regions.csv').open()))
    key = lambda row: tuple(row[k] for k in ['time', 'cycle', 'region', 'subset'])
    result = {key(row): row for row in rows}
    assert len(result) == len(rows) == len(norms), 'duplicate/missing maxima rows'
    assert result.keys() == {key(row) for row in norms}
    for norm in norms:
        row = result[key(norm)]
        assert row['cells'] == norm['cells'] and int(row['cells']) > 0
        for field in ['M', 'Z']:
            peak = float(row[field+'_conformal_linf'])
            rms = float(norm[field+'_conformal_l2'])
            assert math.isfinite(peak) and 0 <= rms <= peak*(1+1e-14)
            xyz = [float(row['max_'+field+'_'+axis]) for axis in 'xyz']
            assert all(math.isfinite(x) for x in xyz)
            assert math.sqrt(sum(x*x for x in xyz)) <= domain['domain_radius']*(1+1e-14)
    return domain, result


def compare(cpu, gpu):
    summaries = [json.loads((p/'summary.json').read_text()) for p in [cpu, gpu]]
    assert summaries[0]['grids'] == summaries[1]['grids']
    assert summaries[0]['cases'].keys() == summaries[1]['cases'].keys()
    passed = True
    cases = {}
    for name in summaries[0]['cases']:
        for summary in summaries:
            receipt = summary['cases'][name]
            assert receipt['exit_code'] == 0 and receipt['reference_unchanged']
            assert receipt['all_rows_finite'] and receipt['reached_step_target']
        da, a = read_case(cpu/name)
        db, b = read_case(gpu/name)
        assert da == db and a.keys() == b.keys()
        checks = []
        for key, first in a.items():
            second = b[key]
            assert first['cells'] == second['cells']
            for field in ['M', 'Z']:
                x, y = [float(row[field+'_conformal_linf']) for row in [first, second]]
                error = abs(x-y)
                tolerance = 1e-10+1e-6*max(abs(x), abs(y))
                ok = error <= tolerance
                passed &= ok
                checks.append(dict(key=key, field=field, cpu=x, gpu=y,
                                   absolute_error=error, tolerance=tolerance, passed=ok))
        cases[name] = checks
    return dict(passed=passed, cases=cases,
                binary_hashes=[s['binary_sha256'] for s in summaries],
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Native maxima CPU/GPU admission; existing field/RMS/source checks remain separate. Locations must be in-domain, but degenerate symmetric peaks may select different cells. No spatial or long-time acceptance.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cpu', type=Path)
    p.add_argument('gpu', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    result = compare(a.cpu, a.gpu)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('passed', result['passed'])
    raise SystemExit(0 if result['passed'] else 1)
