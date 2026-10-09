"""Independent saved-data/schema check; never imports or invokes a kernel."""
from pathlib import Path
from decimal import Decimal, localcontext
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BINDER = ROOT / 'build-layer-research/nonlinear-wave-map-RHS-root-20261009'
PAYLOAD = ROOT / ('build-layer-research/continuum/'
                  'nonlinear-minkowski-wave-map-oracle-20261009/'
                  'attempt002/oracle-110.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert sha(PAYLOAD) == ('411594e363e92e0d1998150cc19b4d627'
                            '549cda6ec625841282f430461b84f38')
    payload = json.loads(PAYLOAD.read_text())
    rows = [list(map(float, s.split())) for s in
            (BINDER / 'prepared-input001/input.txt').read_text().splitlines()]
    expected = json.loads((BINDER / 'prepared-input001/expected.json').read_text())
    assert len(rows) == len(expected) == len(payload['cases']) == 48
    tensor = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    paths = [('chi',)] + [('metric', i, j) for i, j in tensor]
    paths += [('P',)] + [('A', i, j) for i, j in tensor]
    paths += [('Lambda', i) for i in range(3)]
    paths += [('Theta',), ('alpha',)] + [('beta', i) for i in range(3)]
    assert len(paths) == 22
    configuration = {0, 1, 2, 3, 4, 5, 6, 18, 19, 20, 21}
    zero = (0, 0, 0, 0)
    dt = (1, 0, 0, 0)
    ds = [(0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
    dss = [tuple(a[k] + b[k] for k in range(4)) for a in ds for b in ds]
    checks = {'input_binary64_values': 0, 'time_rates': 0,
              'scaled_connection': 0, 'scaled_source': 0,
              'metadata': 0, 'complete_ordinary_jet_tables': 0}

    def take(x, path):
        for k in path:
            x = x[k]
        return x

    def table(jet):
        vals = {tuple(x['multiindex']): x['value'] for x in jet['ordinary']}
        wanted = {(a, b, c, d) for a in range(jet['order'] + 1)
                  for b in range(jet['order'] + 1)
                  for c in range(jet['order'] + 1)
                  for d in range(jet['order'] + 1)
                  if a + b + c + d <= jet['order']}
        assert set(vals) == wanted and len(vals) == len(jet['ordinary'])
        checks['complete_ordinary_jet_tables'] += 1
        return vals

    def same(a, b, category):
        assert math.isfinite(a) and math.isfinite(b)
        assert a.hex() == b.hex(), (category, a.hex(), b.hex())
        checks[category] += 1

    for i, (case, row, exp) in enumerate(zip(payload['cases'], rows, expected)):
        assert case['passed'] and exp['index'] == i
        assert exp['point_name'] == case['point_name']
        same(exp['epsilon'], float(case['epsilon']), 'metadata')
        values = [float(x) for x in case['point']]
        for x, y in zip(exp['point'], values):
            same(x, y, 'metadata')
        for col, path in enumerate(paths):
            jet = take(case['fields'], path)
            assert jet['order'] == (2 if col in configuration else 1)
            t = table(jet)
            values.extend(float(t[m]) for m in [zero] + ds +
                          (dss if col in configuration else []))
            rate = take(case['exact_time_derivatives'], path)
            assert Decimal(rate) == Decimal(t[dt])
            same(exp['rhs22'][col], float(t[dt]), 'time_rates')
        t = table(case['fields']['omega'])
        values.extend(float(t[m]) for m in [zero] + ds + dss)
        assert len(values) == len(row) == 204
        for a, b in zip(row, values):
            same(a, b, 'input_binary64_values')
        same(exp['omega'], float(t[zero]), 'metadata')
        connection = [float(case['scaled_reference_connection'][a][j][k])
                      for a in range(4) for j in range(3) for k in range(3)]
        assert len(exp['scaled_reference_connection36']) == 36
        for a, b in zip(exp['scaled_reference_connection36'], connection):
            same(a, b, 'scaled_connection')
        with localcontext() as ctx:
            ctx.prec = 256
            source = [float(Decimal(t[zero]) * Decimal(x))
                      for x in case['source_Fbar']]
        assert len(exp['scaled_source4']) == len(source) == 4
        for a, b in zip(exp['scaled_source4'], source):
            same(a, b, 'scaled_source')
    result = {'passed': True, 'scope': 'saved decimal payload/input arithmetic '
              'and schema only; no compile, kernel query, eigen or propagation',
              'cases': 48, 'input_columns': 204, 'checks': checks,
              'source_sha256': sha(__file__), 'payload_sha256': sha(PAYLOAD),
              'input_sha256': sha(BINDER / 'prepared-input001/input.txt'),
              'expected_sha256': sha(BINDER / 'prepared-input001/expected.json'),
              'python_version': sys.version}
    (HERE / 'saved-input-readback.json').write_text(
        json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
