"""Measure continuum pulse error, damping, drift and trace-domain bounds.

Results measure the *linearized* flat prediction. They are not automatically
puncture/AMR acceptance: compare resolutions, amplitudes, and transfer brackets.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from make_inputs import parse


def table(directory, phase):
    paths = sorted(directory.glob(f'pulse-{phase}-rank*.csv'))
    if not paths:
        raise ValueError(f'No {phase} pulse data in {directory}')
    return np.concatenate([np.atleast_1d(np.genfromtxt(p, delimiter=',', names=True))
                           for p in paths])


def prediction(data, problem, rate):
    t = data['time']
    x = data['x']+float(problem.get('pulse_shift', .5))*t
    x -= float(problem.get('pulse_center_x', 0))
    radius2 = x*x
    if problem.get('pulse_radial', 'false') == 'true':
        radius2 += data['y']**2+data['z']**2
    s2 = radius2/float(problem.get('pulse_width', .75))**2
    bump = np.zeros_like(s2)
    inside = s2 < 1
    bump[inside] = np.exp(1-1/(1-s2[inside]))
    bump *= float(problem.get('pulse_amplitude', 1e-8))*np.exp(-rate*t)
    expected = np.zeros((len(data), 33))
    d = int(problem.get('pulse_direction', 1))
    family = problem['pulse_family']
    if family == 'p':
        expected[:, d] = bump
    elif family == 'Q':
        expected[:, 3+6*d] = bump
        expected[:, 3+6*d+5] = -bump
    elif family == 'L':
        expected[:, 21+d] = bump
    elif family == 'B':
        expected[:, 24+3*d+2] = bump
    return expected


def measure(directory):
    params = parse((directory/'used_input.athinput').read_text())
    rate = float(params['pc_gh']['reduction_rate'])
    problem = params['problem']
    init, final = table(directory, 'initial'), table(directory, 'final')
    initial_e = np.column_stack([init[f'E{i}'] for i in range(33)])
    final_e = np.column_stack([final[f'E{i}'] for i in range(33)])
    expected = prediction(final, problem, rate)
    amp = float(problem.get('pulse_amplitude', '1e-8'))
    error = final_e-expected
    if not all(np.isfinite(final[name]).all() for name in final.dtype.names):
        raise ValueError('Nonfinite value in uncensored pulse component dump')
    energy0 = (initial_e**2).sum(axis=1)*init['volume']
    energy1 = (final_e**2).sum(axis=1)*final['volume']
    norm0, norm1 = np.sqrt(energy0.sum()), np.sqrt(energy1.sum())
    time = float(final['time'][0])
    centroid0 = float((energy0*init['x']).sum()/energy0.sum())
    centroid1 = float((energy1*final['x']).sum()/energy1.sum())
    alpha = final['u0']*final['u18']
    summary = dict(time=time, rate=rate, family=problem['pulse_family'],
        direction=int(problem.get('pulse_direction', 1)), amplitude=amp,
        error_l2_over_amplitude=float(np.sqrt((error**2*final['volume'][:, None]).sum())/amp),
        error_linf_over_amplitude=float(np.abs(error).max()/amp),
        damping_fit=float(-np.log(norm1/norm0)/time) if time else None,
        speed_fit=(centroid1-centroid0)/time if time else None,
        expected_speed=-float(problem.get('pulse_shift', .5)),
        initial_error=float(np.abs(initial_e-prediction(init, problem, rate)).max()),
        max_alpha=float(alpha.max()), max_alpha2_chi=float((alpha**2*final['u0']**2).max()),
        min_w=float(final['u0'].min()), min_rho=float(final['u18'].min()),
        scope='flat linear prediction; compare truncation and finite-amplitude effects')
    (directory/'pulse-metrics.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', type=Path, nargs='+')
    args = ap.parse_args()
    for directory in args.runs:
        print(directory, json.dumps(measure(directory), sort_keys=True))
