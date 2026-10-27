"""Manufactured derivative audit of the fixed wormhole reference chart.

No Einstein RHS is evaluated and no evolution is advanced. Direct derivatives
are a diagnostic control, not an adopted replacement for the live scheme.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from continue_series import verify
from puncture import PunctureEvolution


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cases', type=Path, nargs=3, required=True)
    p.add_argument('--frame', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    frame = json.loads((a.frame/'assessment.json').read_text())
    rows, observed, arrays = [], [], {}
    for k, case in enumerate(a.cases):
        m = verify(case)
        # This read-only audit uses current modules only after verifying the
        # relevant implementation matches the frozen numerical source.
        for rel in ['tst/hyperbolic_shift_1d/puncture.py',
                    'tst/elliptic_shift_1d/puncture_chart.py',
                    'tst/elliptic_shift_1d/numerics.py']:
            assert digest(Path(rel)) == m['sources'][rel], rel
        kwargs = {key:value for key,value in m['arguments'].items()
                  if key in inspect.signature(PunctureEvolution).parameters
                  and key not in ['n', 'library']}
        lib = case/'source'/Path(m['arguments']['library']).name
        e = PunctureEvolution(m['arguments']['n'], lib.resolve(), **kwargs)
        r, positive = e.r, e.r > 0
        n = len(r)
        assert frame['runs'][k]['n'] == n
        with np.load(a.frame/'physical-profiles.npz') as z:
            assert np.array_equal(z[f'r{k}'], r)
            q = z[f'fields{k}']
            slopes = {}
            for field in ['chi', 'alpha']:
                values = q[:,frame['fields'].index(field)]
                slopes[field] = {f'nodes{lo}-{hi}':float(np.polyfit(
                    np.log(r[lo:hi+1]), np.log(values[lo:hi+1]), 1)[0])
                    for lo,hi in [(1,4), (5,10), (11,20)]}
            observed.append(dict(n=n, fitted_log_slopes=slopes))
        for field, powers in [('chi', [4., 2.]), ('alpha', [2., 1., .2])]:
            f = 0 if field == 'chi' else 6
            for power in powers:
                state = np.zeros((n,8))
                total = r**power
                state[positive,f] = np.log(total[positive]/e.chart.refs[field][positive,0])
                d, dd = e.op.derivatives(state, np.ones(8))
                _, _, _, _, first, second = e.chart.reconstruct(state, d, dd)
                chart_first = first[:,f].copy()
                chart_second = second[:,f].copy()
                if field == 'alpha':
                    chart_first += r
                    chart_second += 1
                direct_first, direct_second = e.op.derivatives(total, 1)
                exact_first = np.zeros(n)
                exact_second = np.zeros(n)
                exact_first[positive] = power*r[positive]**(power-1)
                exact_second[positive] = power*(power-1)*r[positive]**(power-2)
                # Normalize by r^(p-d), including powers with zero exact
                # derivative; this is a dimensionless scaled error, not an
                # undefined relative error for the linear-lapse second derivative.
                scales = [r[positive]**(power-1), r[positive]**(power-2)]
                errors = {}
                for name, first_values, second_values in [
                        ('reference_chart', chart_first, chart_second),
                        ('direct_total_control', direct_first, direct_second)]:
                    values = [abs(first_values[positive]-exact_first[positive])/scales[0],
                              abs(second_values[positive]-exact_second[positive])/scales[1]]
                    fixed = int(np.argmin(abs(r-.046875)))
                    assert abs(r[fixed]-.046875) < 1e-15
                    errors[name] = dict(
                        first16_max=[float(max(v[:16])) for v in values],
                        node1=[float(v[0]) for v in values],
                        node6=[float(v[5]) for v in values],
                        fixed_radius=[float(v[fixed-1]) for v in values])
                    arrays[f'n{n}-{field}-p{power}-{name}'] = np.column_stack([r[positive], *values])
                rows.append(dict(n=n, field=field, power=power, errors=errors,
                                 manifest_sha256=digest(case/'manifest.json')))
    result = dict(observed_t_over_M=frame['t_over_M'], observed_slopes=observed,
                  tests=rows, fixed_radius=.046875,
                  normalization='Absolute derivative error divided by r^(p-d), d=1 or2; arrays list first then second derivative.',
                  scope='Manufactured differentiation only. Power laws are diagnostic fields, not constraint-satisfying initial data. Observed slopes are local fits, not asymptotic exponents. Nonvanishing fixed-index chart errors do not by themselves identify the nonlinear oscillation mechanism. Direct-total controls do not qualify an alternative evolution.',
                  inputs={str(a.frame/name):digest(a.frame/name)
                          for name in ['assessment.json','physical-profiles.npz']},
                  script_sha256=digest(Path(__file__)))
    (a.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(a.output/'scaled-errors.npz', **arrays)
    print(json.dumps([dict(n=r['n'], field=r['field'], power=r['power'],
                          node1=r['errors']['reference_chart']['node1'],
                          node6=r['errors']['reference_chart']['node6']) for r in rows],indent=2))


if __name__ == '__main__':
    main()
