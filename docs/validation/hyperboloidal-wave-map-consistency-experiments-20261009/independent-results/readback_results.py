"""Recheck saved actual point outputs only, with independent Decimal arithmetic."""
from pathlib import Path
from decimal import Decimal, localcontext
from collections import Counter
import hashlib
import json
import math
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
B = ROOT / 'build-layer-research/nonlinear-wave-map-RHS-root-20261009'
A = B / 'attempts/1791564981754626000'
D = Decimal.from_float


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def inverse(g):
    a, b, c = g[0]
    _, d, e = g[1]
    _, _, f = g[2]
    det = a*(d*f-e*e)-b*(b*f-c*e)+c*(b*e-c*d)
    assert a > 0 and a*d-b*b > 0 and det > 0
    inv = [[d*f-e*e, c*e-b*f, b*e-c*d],
           [c*e-b*f, a*f-c*c, b*c-a*e],
           [b*e-c*d, b*c-a*e, a*d-b*b]]
    return det, [[x/det for x in row] for row in inv]


def main():
    start = time.monotonic()
    assert sha(HERE/'source-review.json') == (
        'ff5081846cd9419ab21311c05f5d89c222bc3878d3202a1d6bd2900fe095c75e')
    assert sha(A/'receipt.json') == (
        '518dcec198bf9dde402c70d0b485531162522845388a4e5d3580c39648acdf2a')
    receipt = json.loads((A/'receipt.json').read_text())
    assert receipt['passed'] and receipt['release_debug_equal']
    assert receipt['source_before'] == receipt['source_after']
    for path, h in receipt['source_before'].items():
        assert sha(ROOT/path) == h
    assert len(receipt['commands']) == 5
    for cmd in receipt['commands']:
        assert cmd['exit_code'] == 0
        for stream in ['stdout', 'stderr']:
            path = A/(cmd['name']+'.'+stream)
            assert sha(path) == cmd[stream+'_sha256']
            if stream == 'stderr':
                assert path.read_bytes() == b''
    assert (A/'run-release.stdout').read_bytes() == (A/'run-debug.stdout').read_bytes()
    expected = json.loads((B/'prepared-input001/expected.json').read_text())
    input_rows = [list(map(float, s.split())) for s in
                  (B/'prepared-input001/input.txt').read_text().splitlines()]
    actual = [json.loads(s) for s in (A/'run-release.stdout').read_text().splitlines()]
    saved = json.loads((A/'analysis-release/comparisons.json').read_text())
    assert len(actual) == len(expected) == len(input_rows) == len(saved) == 48
    sizes = {'actual_rhs22': 22, 'physical_constraints8': 8,
             'input_normals2': 2, 'rate_normals2': 2,
             'submitted_minus_native_omega13': 13, 'scaled_source4': 4,
             'scaled_reference_connection36': 36}
    thresholds = {'rhs22_scaled': D(5e-9), 'connection_scaled': D(5e-9),
                  'source_scaled': D(5e-9), 'physical_constraints_absolute': D(5e-9),
                  'input_normals_absolute': D(5e-11), 'rate_normals_scaled': D(5e-11),
                  'omega_difference_absolute': D(2e-10)}
    maxima = {k: Decimal(0) for k in list(thresholds) +
              ['rhs22_absolute', 'rate_normals_absolute']}
    normal_difference = Decimal(0)
    records = []
    sym = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    counts = Counter()
    with localcontext() as ctx:
        ctx.prec = 100
        for index, (a, e, row, old) in enumerate(zip(actual, expected, input_rows, saved)):
            assert len(row) == 204 and a['point'] == e['point'] == row[:4]
            assert e['index'] == old['index'] == index
            assert e['point_name'] == old['point_name']
            assert e['epsilon'] == old['epsilon']
            counts[e['point_name']] += 1
            for key, size in sizes.items():
                assert len(a[key]) == size and all(math.isfinite(x) for x in a[key])
            x = a['actual_rhs22']
            exact_errors = [abs(D(v)-D(w)) for v, w in zip(x, e['rhs22'])]
            scaled = [v/max(Decimal(1), abs(D(w)))
                      for v, w in zip(exact_errors, e['rhs22'])]
            conn = [abs(D(v)-D(w))/max(Decimal(1), abs(D(w))) for v, w in
                    zip(a['scaled_reference_connection36'], e['scaled_reference_connection36'])]
            src = [abs(D(v)-D(w))/max(Decimal(1), abs(D(w))) for v, w in
                   zip(a['scaled_source4'], e['scaled_source4'])]
            norm_abs = max(abs(D(v)) for v in a['rate_normals2'])
            values = {'rhs22_scaled': max(scaled), 'rhs22_absolute': max(exact_errors),
                      'connection_scaled': max(conn), 'source_scaled': max(src),
                      'physical_constraints_absolute': max(abs(D(v)) for v in a['physical_constraints8']),
                      'input_normals_absolute': max(abs(D(v)) for v in a['input_normals2']),
                      'rate_normals_absolute': norm_abs,
                      'rate_normals_scaled': norm_abs/max(Decimal(1), *(abs(D(v)) for v in x)),
                      'omega_difference_absolute': max(abs(D(v)) for v in a['submitted_minus_native_omega13'])}
            for key, value in values.items():
                maxima[key] = max(maxima[key], value)
                assert abs(float(value)-old['metrics'][key]) <= 1e-27
                if key in thresholds:
                    assert value <= thresholds[key]
            # Independent inverse/normal readback from submitted state and
            # returned raw22 rates, without any RHS or geometry implementation.
            state = []
            cursor = 4
            for col in range(22):
                state.append(D(row[cursor]))
                cursor += 13 if col <= 6 or col >= 18 else 4
            assert cursor == 191
            assert state[0] > 0 and state[18] > 0 and D(row[cursor]) > 0
            g = [[Decimal(0) for _ in range(3)] for _ in range(3)]
            av = [[Decimal(0) for _ in range(3)] for _ in range(3)]
            gd = [[Decimal(0) for _ in range(3)] for _ in range(3)]
            ad = [[Decimal(0) for _ in range(3)] for _ in range(3)]
            for t, (i, j) in enumerate(sym):
                g[i][j] = g[j][i] = state[1+t]
                av[i][j] = av[j][i] = state[8+t]
                gd[i][j] = gd[j][i] = D(x[1+t])
                ad[i][j] = ad[j][i] = D(x[8+t])
            det, inv = inverse(g)
            nin = [det-1, sum(inv[i][j]*av[i][j] for i in range(3) for j in range(3))]
            nout = [sum(inv[i][j]*gd[i][j] for i in range(3) for j in range(3)),
                    sum(inv[i][j]*ad[i][j] for i in range(3) for j in range(3)) -
                    sum(inv[i][k]*inv[j][l]*av[k][l]*gd[i][j]
                        for i in range(3) for j in range(3)
                        for k in range(3) for l in range(3))]
            for v, w in zip(nin+nout, a['input_normals2']+a['rate_normals2']):
                normal_difference = max(normal_difference, abs(v-D(w)))
            assert max(abs(v) for v in nin) <= D(5e-11)
            assert max(abs(v) for v in nout)/max(Decimal(1), *(abs(D(v)) for v in x)) <= D(5e-11)
            for arr, name in [(scaled, 'rhs22_scaled_errors'), (exact_errors, 'rhs22_absolute_errors'),
                              (conn, 'connection_scaled_errors'), (src, 'source_scaled_errors')]:
                assert len(arr) == len(old[name])
                for v, w in zip(arr, old[name]):
                    assert abs(float(v)-w) <= 1e-27
            records.append({'index': index, 'point_name': e['point_name'], 'epsilon': e['epsilon'],
                            'maxima': {k: str(v) for k, v in values.items()},
                            'input_normals_independent': [str(v) for v in nin],
                            'rate_normals_independent': [str(v) for v in nout]})
    assert len(counts) == 12 and set(counts.values()) == {4}
    assert normal_difference < D(2e-15)
    dependencies = {}
    for mode in ['release', 'debug']:
        build = json.loads((A/('build-'+mode+'.json')).read_text())
        assert sha(A/('probe-'+mode)) == build['executable_sha256']
        assert len(build['compiler_dependencies']) == build['compiler_dependencies_count']
        for path, h in build['compiler_dependencies'].items():
            assert sha(path) == h, path
            if path in dependencies:
                assert dependencies[path] == h
            dependencies[path] = h
        assert sha(A/('build-'+mode+'.json')) == receipt['modes'][mode]['build_receipt_sha256']
        assert sha(A/('analysis-'+mode+'/receipt.json')) == receipt['modes'][mode]['analysis_receipt_sha256']
    result = {'passed': True, 'scope': 'independent saved finite-radius 48-case output and dependency readback only; no new compile/query/eigen/propagation',
              'source_sha256': sha(__file__), 'source_review_sha256': sha(HERE/'source-review.json'),
              'root_receipt_sha256': sha(A/'receipt.json'), 'actual_sha256': sha(A/'run-release.stdout'),
              'expected_sha256': sha(B/'prepared-input001/expected.json'),
              'cases': 48, 'point_count': 12, 'amplitudes_per_point': 4,
              'returned_numeric_entries_per_case': sum(sizes.values()),
              'rhs_entries_checked': 1056, 'all_output_dimensions_checked': sizes,
              'maxima': {k: float(v) for k, v in maxima.items()},
              'maxima_decimal': {k: str(v) for k, v in maxima.items()},
              'independent_normal_max_difference': float(normal_difference),
              'release_debug_byte_equal': True, 'commands_zero_exit_empty_stderr': 5,
              'release_dependencies': 1049, 'debug_dependencies': 1051,
              'unique_dependency_hashes_rechecked': len(dependencies),
              'recipe_source_hashes_rechecked': 381, 'seconds': time.monotonic()-start,
              'limitations': 'Manufactured flat point-action consistency only, at the prescribed finite Omega and amplitudes. No principal, stability, boundary, scri or wormhole-to-trumpet acceptance.'}
    (HERE/'saved-result-readback.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (HERE/'saved-result-cases.json').write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
