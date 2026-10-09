"""Separate direct-Ricci contraction from saved full physical metric jets."""
import hashlib
import itertools
import json
import time
from pathlib import Path

import mpmath as mp

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def derivative(jet, *axes):
    m = [axes.count(i) for i in range(4)]
    return mp.mpf(next(row['value'] for row in jet['ordinary'] if row['multiindex'] == m))


start = time.monotonic()
pins = json.loads((HERE/'ricci-readback-pins.json').read_text())
assert sha(__file__) == pins['source_sha256']
for row in pins['inputs']:
    assert sha(HERE/row['path']) == row['sha256']
results = []
with mp.workdps(130):
    for digits in (80, 110):
        data = json.loads((HERE/('attempt002/oracle-%d.json' % digits)).read_text())
        for case in data['cases']:
            g = case['physical_metric']
            inverse = mp.inverse(mp.matrix([[derivative(v) for v in row] for row in g]))
            gamma = [[[0 for _ in range(4)] for _ in range(4)] for _ in range(4)]
            dg = [[[[0 for _ in range(4)] for _ in range(4)] for _ in range(4)] for _ in range(4)]
            for a, b, c in itertools.product(range(4), repeat=3):
                gamma[a][b][c] = sum(inverse[a, d]*(derivative(g[d][c], b)+derivative(g[d][b], c)
                                                    -derivative(g[b][c], d))/2 for d in range(4))
            for e in range(4):
                dinverse = -inverse*mp.matrix([[derivative(v, e) for v in row] for row in g])*inverse
                for a, b, c in itertools.product(range(4), repeat=3):
                    dg[e][a][b][c] = sum(
                        dinverse[a, d]*(derivative(g[d][c], b)+derivative(g[d][b], c)-derivative(g[b][c], d))/2
                        +inverse[a, d]*(derivative(g[d][c], b, e)+derivative(g[d][b], c, e)
                                        -derivative(g[b][c], d, e))/2 for d in range(4))
            maximum, absolute = mp.mpf(0), mp.mpf(0)
            for a, b in itertools.product(range(4), repeat=2):
                terms = [dg[c][c][a][b]-dg[b][c][a][c] for c in range(4)]
                terms += [gamma[c][c][d]*gamma[d][a][b]-gamma[c][b][d]*gamma[d][a][c]
                          for c, d in itertools.product(range(4), repeat=2)]
                value = abs(sum(terms))
                error = value/max(1, sum(abs(v) for v in terms))
                assert error <= mp.mpf('1e-55'), (case['point_name'], case['epsilon'], a, b, error)
                maximum, absolute = max(maximum, error), max(absolute, value)
            results.append(dict(digits=digits, point=case['point_name'], epsilon=case['epsilon'],
                                scaled_max=mp.nstr(maximum, 60), absolute_max=mp.nstr(absolute, 60)))
receipt = dict(kind='Independent direct physical Ricci saved-data check', source_sha256=sha(__file__),
               pins_sha256=sha(HERE/'ricci-readback-pins.json'), cases=len(results), components=len(results)*16,
               scaled_max=max(float(row['scaled_max']) for row in results),
               absolute_max=max(float(row['absolute_max']) for row in results),
               seconds=time.monotonic()-start, results=results, passed=True,
               scope='Saved full physicalg2 arithmetic only, no geometric oracle rerun or scientific kernel query.')
(HERE/'ricci-readback.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps({k: v for k, v in receipt.items() if k != 'results'}, indent=2))
