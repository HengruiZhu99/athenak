"""Independent saved-matrix arithmetic; no original analyzer or numerical imports."""
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
import traceback

P = Path(__file__).resolve().parent
Q = P / 'inputs-before-review'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    def bad(token):
        raise ValueError('nonfinite JSON token ' + token)
    return json.loads(Path(path).read_text(), parse_constant=bad)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def need(condition, detail):
    if not condition:
        raise ValueError(detail)


def finite(value):
    if isinstance(value, dict):
        return all(finite(x) for x in value.values())
    if isinstance(value, list):
        return all(finite(x) for x in value)
    return not isinstance(value, float) or math.isfinite(value)


def registry():
    rows = []
    for a, chi, w, g0, oblique in itertools.product((.05, 1., 3.), (.1, 1.),
                                                 (0., .5, 1.), (.375, .75), (0, 1)):
        rows.append((1, a, chi, w, g0, oblique, None, None))
    for w, g0, oblique in itertools.product((0., .5), (.375, .75), (0, 1)):
        rows.append((1, 1., g0, w, g0, oblique, None, None))
    a = 54/29
    for oblique in (0, 1):
        rows.append((1, a, .375/(2*a*a), 0., .375, oblique, None, None))
    for mu in (.125, .375, .75, 1., 2., 8.):
        ec = 2*mu*mu/(1+mu)**2
        for f, oblique in itertools.product((1., 3., (4*mu-2*ec)/3), (0, 1)):
            rows.append((0, 1., 1., 0., .375, oblique, f, mu))
    need(len(rows) == 118, 'independent registry size')
    return rows


def scalar(f, mu, ec):
    return [[0, 0, 0, -f, 0, 0, 0, 0],
            [0, 0, 0, 2/3, 4/3, 0, 0, -2/3],
            [0, 0, 0, 0, 0, -2, 0, 4/3],
            [-1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1/2, 0],
            [-2/3, 1/3, -1/2, 0, 0, 0, 2/3, 0],
            [0, 0, 0, -4/3, -2/3, 0, 0, 4/3],
            [-1, ec, 0, 0, 0, 0, mu, 0]]


def complete(f, mu, ec):
    matrix = [[0.] * 20 for _ in range(20)]
    blocks = [(0, scalar(f, mu, ec))]
    vector = [[0, -2, 0, 1], [-.5, 0, .5, 0], [0, 0, 0, 1], [0, 0, mu, 0]]
    blocks += [(8, vector), (12, vector), (16, [[0, -2], [-.5, 0]]),
               (18, [[0, -2], [-.5, 0]])]
    for start, block in blocks:
        for i, row in enumerate(block):
            for j, value in enumerate(row):
                matrix[start+i][start+j] = value
    return matrix


def row_action(row, matrix):
    # Deliberately direct row composition with compensated summation; no T inverse.
    return [math.fsum(row[k]*matrix[k][j] for k in range(len(row)))
            for j in range(len(row))]


def waves(f, mu, matrix):
    C = 2*(1+mu)**2/(4*mu*mu+5*mu+3)
    q = (4*mu-4*mu*mu/(1+mu)**2)/3
    vectors = {'ell': ([1, 0, 0, 0, 0, 0, 0, 0], f),
               'H': ([0, 2, 1, 0, 0, 0, 0, 0], 1.),
               'V': ([0, 2, 0, 0, 0, 0, 1, 0], 1.),
               'X': ([0, 1-2*C, 0, 0, 0, 0, -C, 0], q)}
    errors = {}
    for name, (row, speed2) in vectors.items():
        second = row_action(row_action(row, matrix), matrix)
        errors[name] = max(abs(x-speed2*y) for x, y in zip(second, row))
        need(errors[name] <= 1e-10, 'direct wave identity ' + name)
    return errors


def main():
    if sys.flags.optimize != 0:
        raise RuntimeError('optimized reviewer is forbidden')
    out = P / 'attempt001'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    result = dict(passed=False, saved_only=True, no_compiler_probe_original_analyzer_or_gate_execution=True)
    capture = load(P / 'review-inputs.json')
    try:
        for row in capture['copied'] + capture['metadata_only']:
            need(sha(row['original']) == row['sha256'], 'changed original ' + row['original'])
            if 'copy' in row:
                need(sha(row['copy']) == row['sha256'], 'changed captured file')
        child = load(Q / 'attempt/receipt.json')
        outer = load(Q / 'outer/invocation001/receipt.json')
        recipe = load(Q / 'attempt/recipe.json')
        index = load(Q / 'attempt/source-index.json')
        need(child['completed'] is True and child['passed'] is True and child['returncode'] == 0
             and child['inputs_unchanged'] is True and child['input_drift'] == [], 'child did not pass')
        need(outer['completed'] is True and outer['accepted'] is True and outer['returncode'] == 0
             and outer['source_drift'] == [], 'outer did not pass')
        need(child['python_flags'] == {'optimize': 0, 'isolated': 1}, 'assertion guard flags')
        need(len(child['commands']) == 10, 'command count')
        for command in child['commands']:
            name = command['name']
            need(command['returncode'] == 0 and command['stderr_bytes'] == 0, name + ' failed')
            for stream in ('stdout', 'stderr'):
                need(sha(Q / 'attempt' / (name+'.'+stream)) == command[stream+'_sha256'], name+' log differs')
        for mode in ('release', 'debug'):
            command = next(x['command'] for x in child['commands'] if x['name'] == 'compile-'+mode)
            need(command[1:1+len(recipe[mode+'_flags'])] == recipe[mode+'_flags'], 'compile flags differ')
            if mode == 'debug':
                need('-fsanitize=address,undefined' in command, 'sanitizer absent')
            dependencies = load(Q / 'attempt' / ('dependencies-'+mode+'.json'))
            need(len(dependencies) == child[mode+'_dependency_count'], 'dependency count')
            for pin in dependencies:
                need(sha(pin['path']) == pin['sha256'], 'dependency changed ' + pin['path'])
        protected = index['files'] + index['external_inputs'] + recipe['protected_inputs']
        unique = {pin['path']: pin['sha256'] for pin in protected}
        for path, digest in unique.items():
            need(sha(path) == digest, 'protected source/runtime changed ' + path)
        release, debug = Q / 'attempt/probe-release.stdout', Q / 'attempt/probe-debug.stdout'
        need(release.read_bytes() == debug.read_bytes(), 'matrix stdout differs')
        data = load(release)
        need(len(data) == 118 and finite(data), 'matrix payload size/finite')
        matrix_max, wave_max = 0., dict(ell=0., H=0., V=0., X=0.)
        for number, (case, params) in enumerate(zip(data, registry())):
            candidate, a, chi, w, g0, oblique, f, mu = params
            for key, value in zip(('candidate', 'alpha', 'chi', 'W', 'G0', 'oblique'),
                                  (candidate, a, chi, w, g0, oblique)):
                need(case[key] == value, 'registry mismatch '+str(number)+'/'+key)
            if candidate:
                f, mu = 1+2*(1-w)/a, w+(1-w)*g0/(a*a*chi)
            ec = 2*mu*mu/(1+mu)**2
            m, expected = case['M'], complete(f, mu, ec)
            need(len(m) == 20 and all(len(row) == 20 for row in m), 'matrix dimension')
            error = max(abs(x-y) for row, target in zip(m, expected) for x, y in zip(row, target))
            need(error <= 2e-12, 'matrix mismatch '+str(number))
            matrix_max = max(matrix_max, error)
            for key, value in waves(f, mu, [row[:8] for row in m[:8]]).items():
                wave_max[key] = max(wave_max[key], value)
        exact = load(Q / 'attempt/exact-scalar.stdout')
        need(exact['passed'] is True and exact['exact_scalar_cases'] == 18, 'exact saved count')
        for row in exact['cases']:
            f, mu, ec, q, C = (Fraction(row[k]) for k in ('f', 'mu', 'ec', 'q', 'C'))
            need(f > 0 and mu > 0 and q > 0 and ec == 2*mu*mu/(1+mu)**2
                 and q == (4*mu-2*ec)/3 and C*(q-1) == 2*(mu-1)/3, 'saved exact rational relation')
        for mode in ('release', 'debug'):
            summary = load(Q / 'attempt' / ('analyze-'+mode+'.stdout'))
            need(summary == child['analysis_summaries'][mode], 'summary/receipt mismatch')
            need(summary['passed'] is True and summary['actual20_cases'] == 118, 'saved analyzer failed')
            maxima = summary['maxima']
            for key, tolerance in [('matrix', 2e-12), ('conjugacy_absolute', 1e-10),
                                    ('conjugacy_scaled', 1e-11), ('left_absolute', 1e-10),
                                    ('left_scaled', 1e-10), ('basis_inverse', 1e-10)]:
                need(math.isfinite(maxima[key]) and maxima[key] <= tolerance, 'saved summary threshold')
        result.update(passed=True, cases=118, exact_records=18, commands=10,
                      release_debug_byte_equal=True, independent_matrix_max=matrix_max,
                      independent_direct_second_row_wave_maxima=wave_max,
                      original_analyzer_maxima=child['analysis_summaries']['release']['maxima'],
                      actual_dependency_counts={k:child[k+'_dependency_count'] for k in ('release','debug')},
                      unique_protected_source_header_runtime_pins=len(unique),
                      outer_seconds=outer['seconds'], production_implementation=child['production_implementation'],
                      launch_HEAD=child['launch_HEAD'], executable_hashes={k:child[k+'_executable_sha256'] for k in ('release','debug')},
                      source_and_run_originals_unchanged=True,
                      scope='Finite positive alpha/chi constant-reference principal only; no nonlinear/puncture/BH admission')
    except BaseException as error:
        result['exception'] = repr(error)
        (out / 'exception.txt').write_text(traceback.format_exc())
    finally:
        result['seconds'] = time.monotonic()-started
        result['reviewer_source_sha256'] = sha(__file__)
        result['captured_inputs_sha256'] = sha(P / 'review-inputs.json')
        write(out / 'receipt.json', result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
