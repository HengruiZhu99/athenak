"""Independent characteristic quadrature for finite smooth-rate CUDA pulses.

Tests all 33 reduction and all 33 uncontracted curl components. The prediction
uses the linear flat-background subsidiary PDE, not an exponential fitted to
the evolved norm. Moving centers follow the analytic flat shift trajectory.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from make_inputs import parse
from verify_pulses import table


def rate_gradient(points, pc, time, shift):
    outer = float(pc['reduction_rate'])
    if pc.get('reduction_profile', 'constant') == 'constant':
        return np.full(len(points), outer), np.zeros_like(points)
    core = float(pc['reduction_core_radius'])
    taper = float(pc['reduction_taper_radius'])
    following = pc.get('reduction_follow_trackers', 'false') == 'true'
    centers = []
    if following:
        n = 0
        while pc.get(f'co_{n}', 'false') == 'true':
            centers.append([float(pc.get(f'co_{n}_{a}', 0)) for a in 'xyz'])
            centers[-1][0] -= shift*time
            n += 1
        if not centers:
            raise ValueError('Moving profile without centers')
    else:
        centers = [[float(pc.get(f'reduction_center_{a}', 0)) for a in 'xyz']]
    outside = np.ones(len(points))
    grad_outside = np.zeros_like(points)
    for center in centers:
        offset = points-np.array(center)
        q = ((offset**2).sum(axis=1)-core**2)/(taper**2-core**2)
        weight = (q <= 0).astype(float)
        grad_weight = np.zeros_like(points)
        interior = (q > 0) & (q < 1)
        z = q[interior]
        # Independent logistic representation of the C-infinity switch.
        log_ratio = -1/z+1/(1-z)
        weight[interior] = np.exp(-np.logaddexp(0, log_ratio))
        derivative = -weight[interior]*(1-weight[interior])*(1/z**2+1/(1-z)**2)
        grad_weight[interior] = derivative[:, None]*2*offset[interior]/(taper**2-core**2)
        grad_outside = grad_outside*(1-weight[:, None])-outside[:, None]*grad_weight
        outside *= 1-weight
    difference = float(pc['reduction_inner_rate'])-outer
    return outer+difference*(1-outside), -difference*grad_outside


def integrated_rate(points, time, pc, shift, nodes):
    total = np.zeros(len(points))
    gradient = np.zeros_like(points)
    if pc.get('reduction_follow_trackers', 'false') == 'true':
        # Points and centers share dx/dt=-beta on this flat background.
        rate, grad = rate_gradient(points, pc, time, shift)
        return time*rate, time*grad
    abscissae, weights = np.polynomial.legendre.leggauss(nodes)
    for z, weight in zip(abscissae, weights):
        s = time*(z+1)/2
        along = points.copy()
        along[:, 0] += shift*(time-s)
        rate, grad = rate_gradient(along, pc, s, shift)
        total += time*weight*rate/2
        gradient += time*weight*grad/2
    return total, gradient


def prediction(data, params, nodes=128):
    pc, problem = params['pc_gh'], params['problem']
    time = float(data['time'][0])
    assert np.all(data['time'] == time)
    points = np.column_stack([data[a] for a in 'xyz'])
    shift = float(problem.get('pulse_shift', .5))
    width = float(problem.get('pulse_width', .75))
    amplitude = float(problem.get('pulse_amplitude', 1e-8))
    initial = points.copy()
    initial[:, 0] += shift*time-float(problem.get('pulse_center_x', 0))
    if problem.get('pulse_radial', 'false') != 'true':
        initial[:, 1:] = 0
    squared = (initial**2).sum(axis=1)/width**2
    inside = squared < 1
    bump = np.zeros(len(points))
    gradient = np.zeros_like(points)
    # Chunk the characteristic integral to keep full-volume memory bounded.
    for start in range(0, len(points), 4096):
        sl = slice(start, start+4096)
        integral, di = integrated_rate(points[sl], time, pc, shift, nodes)
        good = inside[sl]
        b = np.zeros(len(integral))
        b[good] = amplitude*np.exp(1-1/(1-squared[sl][good])-integral[good])
        gradient[sl] = -b[:, None]*di
        gradient[sl][good] -= (2*b[good]/width**2/(1-squared[sl][good])**2)[:, None]*initial[sl][good]
        bump[sl] = b
    d = int(problem.get('pulse_direction', 1))
    family = problem['pulse_family']
    tensor = np.zeros((len(data), 3, 11))
    derivative = np.zeros((len(data), 3, 3, 11))
    fibers = {'p': [(0, 1)], 'Q': [(1, 1), (6, -1)],
              'L': [(7, 1)], 'B': [(10, 1)]}[family]
    for fiber, sign in fibers:
        tensor[:, d, fiber] = sign*bump
        derivative[:, :, d, fiber] = sign*gradient
    reductions = np.column_stack([tensor[:, :, 0], tensor[:, :, 1:7].reshape(-1, 18),
                                  tensor[:, :, 7], tensor[:, :, 8:11].reshape(-1, 9)])
    curl = np.column_stack([derivative[:, a, b, :]-derivative[:, b, a, :]
                            for a, b in [(0, 1), (0, 2), (1, 2)]])
    return reductions, curl


def measure(directory):
    params = parse((directory/'used_input.athinput').read_text())
    amplitude = float(params['problem'].get('pulse_amplitude', 1e-8))
    data = table(directory, 'final')
    if not all(np.isfinite(data[name]).all() for name in data.dtype.names):
        raise ValueError('Nonfinite uncensored pulse dump')
    expected, curl = prediction(data, params)
    finer, curl_finer = prediction(data, params, 256)
    quadrature = float(max(np.abs(expected-finer).max(), np.abs(curl-curl_finer).max())/amplitude)
    if quadrature > 1e-9:
        raise ValueError(f'Characteristic quadrature unresolved: {quadrature}')
    reductions = np.column_stack([data[f'E{v}'] for v in range(33)])
    actual_curl = np.column_stack([data[f'C{v}'] for v in range(33)])
    def norms(error):
        return dict(l2_over_amplitude=float(np.sqrt((data['volume'][:, None]*error**2).sum())/amplitude),
                    linf_over_amplitude=float(np.abs(error).max()/amplitude))
    result = dict(time=float(data['time'][0]), family=params['problem']['pulse_family'],
        reductions=norms(reductions-expected), curls=norms(actual_curl-curl),
        quadrature_difference_over_amplitude=quadrature,
        scope='All 33 reductions and curls versus flat linear characteristic solution; no fitted constant rate')
    if params['pc_gh'].get('reduction_follow_trackers', 'false') == 'true':
        tracks = {}
        for n, path in enumerate(sorted(directory.glob('*.co_*.txt'))):
            a = np.atleast_2d(np.loadtxt(path))
            origin = np.array([float(params['pc_gh'].get(f'co_{n}_{d}', 0)) for d in 'xyz'])
            exact = np.tile(origin, (len(a), 1))
            exact[:, 0] -= float(params['problem'].get('pulse_shift', .5))*a[:, 1]
            tracks[path.name] = float(np.abs(a[:, 2:5]-exact).max())
        if not tracks:
            raise ValueError('Missing moving-center trajectories')
        result['tracker_max_coordinate_errors'] = tracks
    (directory/'smooth-pulse-metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', type=Path, nargs='+')
    for run in ap.parse_args().runs:
        print(run, json.dumps(measure(run), sort_keys=True), flush=True)
