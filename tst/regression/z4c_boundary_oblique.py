#!/usr/bin/env python3
"""Exercise skew-metric CPBC at physical faces/internal tangential MPI edges.

Requires an MPI AthenaK z4c_tov_ks build and NumPy. Runs the same eight-block
oblique pulse for three RK3 cycles with 1/2/4 ranks; does not submit jobs.
Checks actual stage arrays by global block id, the triggered raised normals,
and all ten zero-rate equations at sampled physical/internal-edge intersections.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

import numpy as np

OPS = ('rhs_full_vs_bg', 'pre_boundary_rhs', 'post_boundary_rhs', 'post_rk')


def set_parameter(text, section, name, value):
    pattern = rf'(?ms)(^<{re.escape(section)}>\s*\n)(.*?)(?=^<|\Z)'
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise ValueError(f'expected one input section {section}')
    match = matches[0]
    body = match.group(2)
    assignment = rf'(?m)^{re.escape(name)}\s*=.*$'
    if re.search(assignment, body):
        body, count = re.subn(assignment, f'{name} = {value}', body)
        assert count == 1
    else:
        body += f'{name} = {value}\n'
    return text[:match.start()] + match.group(1) + body + text[match.end():]


def input_text(repo):
    text = (repo/'inputs/tests/z4c_characteristic_cpbc_plane_pulse.athinput').read_text()
    settings = {
        'mesh': {**{f'nx{a}': 16 for a in (1, 2, 3)},
                 **{f'x{a}{side}': value for a in (1, 2, 3)
                    for side, value in [('min', -1), ('max', 1)]},
                 **{f'{side}x{a}_bc': 'outflow' for a in (1, 2, 3) for side in ('i', 'o')}},
        'meshblock': {f'nx{a}': 8 for a in (1, 2, 3)},
        'time': {'nlim': 3},
        'z4c': {'debug_balance': 'true', 'debug_reduction_stride': 1,
                'debug_snapshot_operations': ','.join(OPS)},
        'problem': {'characteristic_test_amplitude': .01,
                    'characteristic_test_center': .75,
                    'characteristic_test_width': .5,
                    'characteristic_test_oblique': 'true',
                    'characteristic_test_oblique_dimensions': 3,
                    'characteristic_test_transverse_width': 1}}
    for section, params in settings.items():
        for name, value in params.items():
            text = set_parameter(text, section, name, value)
    return text


def tensor(v, first):
    a, b, c, d, e, f = v[first:first+6]
    return np.array([[a, b, c], [b, d, e], [c, e, f]])


def frame(v, side):
    metric = tensor(v, 1)
    inv = np.linalg.inv(metric)
    assert np.linalg.eigvalsh(metric)[0] > 0 and v[0] > 0 and v[18] > 0
    nd = side/np.sqrt(side@inv@side)
    nu = inv@nd
    candidates = [np.eye(3)[a] - nu*nd[a] for a in range(3)]
    norm2 = [t@metric@t for t in candidates]
    t1 = candidates[int(np.argmax(norm2))]/np.sqrt(max(norm2))
    t2 = np.cross(nd, metric@t1)/np.sqrt(np.linalg.det(metric))
    t2 /= np.sqrt(t2@metric@t2)
    return metric, inv, nd, nu, (t1, t2)


def normal_derivative(array, point, normal, start, count, dx):
    # point and active bounds use x,y,z; array uses variable,z,y,x.
    derivative = np.zeros(array.shape[0])
    for axis in range(3):
        if abs(normal[axis]) <= 128*np.finfo(float).eps:
            continue
        pos = point[axis]
        lo, hi = start[axis], start[axis]+count[axis]-1
        side = -1 if pos == lo else (1 if pos == hi else 0)
        def value(offset):
            p = list(point)
            p[axis] += offset
            assert lo <= p[axis] <= hi
            return array[:, p[2], p[1], p[0]]
        if side:
            result = side*(3*value(0)-4*value(-side)+value(-2*side))/(2*dx[axis])
        else:
            result = (value(1)-value(-1))/(2*dx[axis])
        derivative += normal[axis]*result
    return derivative


def characteristic_rate_error(v, rhs, derivative, side, scalar_left):
    metric, inv, nd, nu, tangents = frame(v, side)
    A = tensor(rhs, 8)
    dg = tensor(derivative, 1)
    p = np.array([rhs[7], rhs[17], nu@A@nu - np.sum(inv*A)/3,
                  nd@rhs[14:17]])
    d = np.array([derivative[0], nu@dg@nu - np.sum(inv*dg)/3,
                  derivative[18], nd@derivative[19:22]])
    alpha, chi = v[18], v[0]
    beta = nd@v[19:22]
    # Background-adapted Minkowski gauge: alpha_bg=1, lapse_driver=2,
    # shift_driver=1, beta_bg=0. Do not substitute the perturbed lapse driver.
    rows, _ = scalar_left(alpha, chi, 2., 1., 0., beta, 1.)
    scalar_error = np.max(abs(rows@np.r_[p, d])/np.linalg.norm(rows, axis=1))
    shift_root = .5*(beta+np.sqrt(beta*beta+4))
    errors = [float(scalar_error)]
    for tangent in tangents:
        td = metric@tangent
        gamma = td@rhs[14:17]
        gauge = shift_root*gamma + td@derivative[19:22]
        constraint = -2*(nu@A@tangent)/np.sqrt(chi) - gamma + nu@dg@tangent
        errors.extend([abs(gauge), abs(constraint)])
    t1, t2 = tangents
    errors.extend([
        abs(-(t1@A@t1-t2@A@t2)/np.sqrt(chi) + .5*(t1@dg@t1-t2@dg@t2)),
        abs(-2*(t1@A@t2)/np.sqrt(chi) + t1@dg@t2)])
    return max(errors), nu


def load_array(meta_path):
    meta = json.loads(meta_path.read_text())
    assert meta['scalar_bytes'] == 8 and meta['byte_order'] in ('little', 'big')
    dtype = '<f8' if meta['byte_order'] == 'little' else '>f8'
    data = np.fromfile(meta_path.with_suffix('.bin'), dtype=dtype).reshape(meta['shape'])
    return meta, data


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def inspect(run, scalar_left):
    hashes, residual_payloads, stages, blocks = {}, {}, set(), set()
    samples = 0
    min_skew = np.inf
    min_metric_determinant = min_lapse = min_chi = np.inf
    max_skew = rate_error = correction = 0.
    for operation in OPS:
        for path in sorted(run.glob(f'z4c_snapshot_{operation}_*.json')):
            meta, data = load_array(path)
            ng = meta['ng']
            count = meta['active_count']
            start = meta['active_start']
            assert count == [8, 8, 8] and start == [ng]*3
            active = data[:, :, ng:ng+8, ng:ng+8, ng:ng+8]
            assert np.isfinite(active).all(), path
            stages.add((meta['cycle'], meta['stage']))
            for m, block in enumerate(meta['blocks']):
                gid = block['gid']
                blocks.add(gid)
                key = f'{operation}:c{meta["cycle"]}:s{meta["stage"]}:b{gid}'
                assert key not in hashes
                hashes[key] = digest(active[m])
                if operation == 'post_rk':
                    # Includes every residual field and stored ghost as well as
                    # active cells; unchanged block layout is important here.
                    assert np.isfinite(data[m]).all()
                    residual_payloads[key] = digest(data[m])
            if operation != 'rhs_full_vs_bg':
                continue
            gxx, gxy, gxz, gyy, gyz, gzz = [active[:, f] for f in range(1, 7)]
            determinant = (gxx*(gyy*gzz-gyz*gyz) - gxy*(gxy*gzz-gyz*gxz)
                           + gxz*(gxy*gyz-gyy*gxz))
            assert np.min(gxx) > 0 and np.min(gxx*gyy-gxy*gxy) > 0
            assert np.min(determinant) > 0 and np.min(active[:, 0]) > 0
            assert np.min(active[:, 18]) > 0
            min_metric_determinant = min(min_metric_determinant, float(np.min(determinant)))
            min_lapse = min(min_lapse, float(np.min(active[:, 18])))
            min_chi = min(min_chi, float(np.min(active[:, 0])))
            pre_path = Path(str(path).replace('rhs_full_vs_bg', 'pre_boundary_rhs'))
            post_path = Path(str(path).replace('rhs_full_vs_bg', 'post_boundary_rhs'))
            _, pre = load_array(pre_path)
            _, post = load_array(post_path)
            configuration = [0, 1, 2, 3, 4, 5, 6, 18, 19, 20, 21]
            assert (pre[:, configuration].tobytes() == post[:, configuration].tobytes()), (
                'CPBC modified configuration RHS stencil inputs', path)
            for m, block in enumerate(meta['blocks']):
                sign = -1 if block['xmin'][0] == -1 else 1
                assert (block['xmin'][0] == -1 or block['xmax'][0] == 1)
                # Pure physical x face, intersecting each internal y/z edge;
                # other tangent coordinate is at a block midpoint, not a face.
                for tangent_axis in (1, 2):
                    point = [ng+4]*3
                    point[0] = ng if sign < 0 else ng+7
                    point[tangent_axis] = (ng+7 if block['xmax'][tangent_axis] == 0 else ng)
                    assert (block['xmax'][tangent_axis] == 0 or block['xmin'][tangent_axis] == 0)
                    i, j, k = point
                    v, q = data[m, :, k, j, i], post[m, :, k, j, i]
                    side = np.array([sign, 0., 0.])
                    _, _, _, nu, _ = frame(v, side)
                    skew = abs(nu[tangent_axis])
                    min_skew, max_skew = min(min_skew, skew), max(max_skew, skew)
                    assert skew > 1e-8, ('test did not trigger skew tangent', block, point, skew)
                    dq = normal_derivative(post[m], point, nu, start, count, block['dx'])
                    err, _ = characteristic_rate_error(v, q, dq, side, scalar_left)
                    rate_error = max(rate_error, err)
                    correction = max(correction, float(np.max(abs(q-pre[m, :, k, j, i]))))
                    samples += 1
    assert stages == {(c, s) for c in range(3) for s in (1, 2, 3)}, stages
    assert blocks == set(range(8)), blocks
    assert len(hashes) == len(OPS)*9*8 and len(residual_payloads) == 9*8
    assert samples == 9*8*2 and rate_error < 2e-12 and correction > 1e-6
    lines = [line for line in (run/'run.log').read_text().splitlines()
             if line.startswith('Z4C_CHARACTERISTIC_CPBC ')]
    assert lines
    enforcement = max(float(re.search(r' enforcement=(\S+)', line).group(1)) for line in lines)
    assert enforcement < 2e-12
    return {'ranks': int(run.name[1:]), 'stages': len(stages), 'blocks': len(blocks),
            'sampled_physical_internal_intersections': samples,
            'min_abs_triggered_tangent_normal': float(min_skew),
            'max_abs_triggered_tangent_normal': float(max_skew),
            'min_active_metric_determinant': min_metric_determinant,
            'min_active_lapse': min_lapse, 'min_active_chi': min_chi,
            'max_independent_ten_mode_rate_error': rate_error,
            'max_logged_enforcement_error': enforcement,
            'max_boundary_rhs_change': correction,
            'active_array_sha256': hashes, 'all_post_rk_payload_sha256': residual_payloads}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launcher', default='mpiexec', help='MPI command, optionally with flags')
    parser.add_argument('--ranks', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--analyze-only', action='store_true', help='read existing run outputs')
    args = parser.parse_args()
    assert args.ranks and all(8 % ranks == 0 for ranks in args.ranks)
    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('characteristics', repo /
        'analysis/z4c_characteristic/check_residual_characteristics_numeric.py')
    algebra = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(algebra)
    results = {'scope': '3D Minkowski oblique finite lapse pulse; original zero_rate; same eight blocks',
               'cycles': 3, 'mpi_ranks': args.ranks,
               'executable_sha256': digest(np.fromfile(args.exe, dtype=np.uint8)), 'runs': {}}
    baseline = input_text(repo)
    results['input_sha256'] = hashlib.sha256(baseline.encode()).hexdigest()
    reference = None
    for ranks in args.ranks:
        run = args.output.resolve()/f'r{ranks}'
        if not args.analyze_only:
            run.mkdir(parents=True, exist_ok=False)
            (run/'input.athinput').write_text(baseline)
            environment = dict(os.environ, OMP_NUM_THREADS='1')
            with (run/'run.log').open('w') as log:
                subprocess.run(shlex.split(args.launcher) + ['-n', str(ranks), str(args.exe.resolve()),
                               '-i', 'input.athinput'], cwd=run, env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        assert (run/'input.athinput').read_text() == baseline, f'{run}: input changed'
        result = inspect(run, algebra.finite_scalar_left)
        if reference is None:
            reference = result
        else:
            for key in ('active_array_sha256', 'all_post_rk_payload_sha256'):
                assert result[key] == reference[key], f'{run.name}: MPI mismatch: {key}'
        result['bitwise_matches_reference'] = True
        results['runs'][run.name] = result
        (args.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
        print(f'{run.name} PASS: {result["sampled_physical_internal_intersections"]} skew-edge samples, '
              f'rate error={result["max_independent_ten_mode_rate_error"]:.3e}', flush=True)
    assert digest(np.fromfile(args.exe, dtype=np.uint8)) == results['executable_sha256'], (
        'executable changed while testing; rerun with an immutable executable copy')
    results['pass'] = True
    (args.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
