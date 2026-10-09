"""Independent arithmetic readback of saved nonlinear-oracle map/metric jets."""
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import mpmath as mp

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def coefficient(jet, *axes):
    target = [axes.count(i) for i in range(4)]
    matches = [row for row in jet['ordinary'] if row['multiindex'] == target]
    assert len(matches) == 1
    return mp.mpf(matches[0]['value'])


def number_tree(value):
    if isinstance(value, list):
        return [number_tree(v) for v in value]
    return mp.mpf(value)


started = time.monotonic()
pin = json.loads((HERE/'readback-pins.json').read_text())
for row in pin['inputs']:
    assert sha(HERE/row['path']) == row['sha256']
assert sha(__file__) == pin['source_sha256']
errors = {}
checks = 0
def compare(kind, actual, expected):
    global checks
    assert mp.isfinite(actual) and mp.isfinite(expected)
    error = abs(actual-expected)/max(1, abs(actual), abs(expected))
    errors[kind] = max(errors.get(kind, mp.mpf(0)), error)
    assert error <= mp.mpf('1e-55'), (kind, error)
    checks += 1


with mp.workdps(130):
    for digits in (80, 110):
        payload = json.loads((HERE/('attempt002/oracle-%d.json' % digits)).read_text())
        assert payload['digits'] == digits and len(payload['cases']) == 48
        for row in payload['cases']:
            x1 = number_tree(row['inverse_first'])
            x2 = number_tree(row['inverse_second'])
            x3 = number_tree(row['inverse_third'])
            y1 = number_tree(row['reference_first'])
            y2 = number_tree(row['reference_second'])
            y3 = number_tree(row['reference_third'])
            g = row['physical_metric']
            for kind, j2, j3 in (('inverse', x2, x3), ('reference', y2, y3)):
                for a, i, j in itertools.product(range(4), repeat=3):
                    compare(kind+'_symmetric2', j2[a][i][j], j2[a][j][i])
                for a, i, j, k in itertools.product(range(4), repeat=4):
                    compare(kind+'_symmetric3', j3[a][i][j][k], j3[a][k][j][i])
            eta = [-1, 1, 1, 1]
            for a, b in itertools.product(range(4), repeat=2):
                compare('metric_value', coefficient(g[a][b]), sum(eta[c]*x1[c][a]*x1[c][b] for c in range(4)))
                for i in range(4):
                    expected = sum(eta[c]*(x2[c][a][i]*x1[c][b]+x1[c][a]*x2[c][b][i]) for c in range(4))
                    compare('metric_first', coefficient(g[a][b], i), expected)
                    for j in range(4):
                        expected = sum(eta[c]*(x3[c][a][i][j]*x1[c][b]+x2[c][a][i]*x2[c][b][j]
                                               +x2[c][a][j]*x2[c][b][i]+x1[c][a]*x3[c][b][i][j]) for c in range(4))
                        compare('metric_second', coefficient(g[a][b], i, j), expected)
            physical = mp.matrix([[coefficient(v) for v in rr] for rr in g])
            assert mp.det(physical) < 0
            spatial = physical[1:4, 1:4]
            assert spatial[0, 0] > 0 and mp.det(spatial[0:2, 0:2]) > 0 and mp.det(spatial) > 0
            omega = coefficient(row['fields']['omega'])
            yi = mp.inverse(mp.matrix(y1))
            for a in range(4):
                for i, j in itertools.product(range(3), repeat=2):
                    expected = omega*sum(yi[a, b]*y2[b][i+1][j+1] for b in range(4))
                    compare('scaled_reference_connection', mp.mpf(row['scaled_reference_connection'][a][i][j]), expected)
            gt = mp.matrix([[coefficient(v) for v in rr] for rr in row['fields']['metric']])
            aa = mp.matrix([[coefficient(v) for v in rr] for rr in row['fields']['A']])
            ti = mp.inverse(gt)
            compare('stored_det', mp.det(gt), mp.mpf(1))
            compare('stored_A_trace', sum(ti[i, j]*aa[i, j] for i, j in itertools.product(range(3), repeat=2)), 0)
            for key in ('alpha', 'chi', 'P', 'Theta'):
                compare('exact_time_'+key, mp.mpf(row['exact_time_derivatives'][key]), coefficient(row['fields'][key], 0))
            for key in ('beta', 'Lambda'):
                for i in range(3):
                    compare('exact_time_'+key, mp.mpf(row['exact_time_derivatives'][key][i]), coefficient(row['fields'][key][i], 0))
            for key in ('metric', 'A'):
                for i, j in itertools.product(range(3), repeat=2):
                    compare('exact_time_'+key, mp.mpf(row['exact_time_derivatives'][key][i][j]), coefficient(row['fields'][key][i][j], 0))
receipt = dict(kind='Independent saved map/metric/connection/storage arithmetic readback',
               source_sha256=sha(__file__), input_pins_sha256=sha(HERE/'readback-pins.json'),
               cases=96, checks=checks, scaled_max=float(max(errors.values())),
               category_scaled_max={key: float(value) for key, value in errors.items()},
               seconds=time.monotonic()-started, mpmath=mp.__version__, passed=True,
               scope='Saved decimal arithmetic only; no oracle/helper/kernel execution, eigenvalue or propagation. Exact time rates are retained coefficients, not an independent PDE verification.')
(HERE/'saved-readback.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))
