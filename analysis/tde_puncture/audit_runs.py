#!/usr/bin/env python3
"""Summarize finite evolution, target/walltime, and measured Theta amplification.

No completion marker is interpreted as a perturbation-stability pass.
Requires NumPy. Raw histories remain authoritative and should be archived.
"""
import argparse
import json
from pathlib import Path
import re
import numpy as np


def history(path):
    lines = path.read_text().splitlines()
    header = next(line for line in lines if '[1]=' in line)
    names = re.findall(r'\[\d+\]=([^\s]+)', header)
    rows = np.array([[float(x) for x in line.split()] for line in lines
                     if line.strip() and not line.startswith('#')])
    assert rows.ndim == 2 and rows.shape[1] == len(names)
    return {name: rows[:, i] for i, name in enumerate(names)}, bool(np.isfinite(rows).all())


def audit(run):
    log = (run/'run.log').read_text(errors='replace')
    text = (run/'input.athinput').read_text()
    target = float(re.search(r'^tlim\s*=\s*(\S+)', text, re.M)[1])
    records, all_finite = {}, True
    for path in sorted(run.glob('*.hst')):
        data, finite = history(path)
        all_finite &= finite
        records[path.name] = data
    output = dict(run=str(run.resolve()), target_M=target, histories_finite=all_finite,
                  termination=('target' if 'Terminating on time limit' in log else
                               'walltime' if 'Terminating on wall clock limit' in log else
                               'cycle_limit' if 'Terminating on cycle limit' in log else
                               'fatal' if 'FATAL' in log or 'MPI_Abort' in log else 'unfinished'))
    if (run/'operator-stop.json').exists():
        output['termination'] = 'operator_stop'
        output['operator_stop'] = json.loads((run/'operator-stop.json').read_text())
    fatal = [line for line in log.splitlines() if 'FATAL' in line or 'BAD_METRIC' in line]
    if fatal:
        output['failure_diagnostics'] = fatal[-3:]
    metrics = next((d for d in records.values() if 'Theta-max' in d), None)
    if metrics is None:
        output['status'] = 'no metric history'
        return output
    t = metrics['time']; theta = metrics['Theta-max']
    output.update(last_history_time_M=float(t[-1]), target_reached=bool(abs(t[-1]-target)<1e-8),
                  max_Theta=float(np.max(theta)), final_Theta=float(theta[-1]),
                  bad_metric_max=float(np.max(metrics['bad-metric'])),
                  min_lapse=float(np.min(metrics['alpha-min'])),
                  min_chi=float(np.min(metrics['chi-min'])),
                  min_metric_determinant=float(np.min(metrics['detg-min'])),
                  density_max_initial=float(metrics['rho-max'][0]),
                  density_max_final=float(metrics['rho-max'][-1]))
    output['exact_zero_history_residuals'] = all(np.count_nonzero(metrics[k]) == 0
        for k in ['Theta-max', 'Khat-res', 'alpha-res', 'beta-res', 'B-res', 'Gam-res'])
    for start, stop in [(10,20), (20,30), (30,40), (40,60), (60,100)]:
        selected = (t >= start) & (t <= stop) & (theta > 0) & np.isfinite(theta)
        if np.count_nonzero(selected) >= 5 and t[-1] >= stop-.04:
            tx, ty = t[selected], np.log(theta[selected])
            rate, offset = np.polyfit(tx, ty, 1)
            output[f'Theta_fit_{start}_{stop}'] = dict(rate_per_M=float(rate),
                observed_factor=float(np.exp(ty[-1]-ty[0])), actual_window=[float(tx[0]),float(tx[-1])])
    cycles = re.findall(r'elapsed=([0-9.eE+-]+) cycle=(\d+) time=([0-9.eE+-]+)', log)
    if len(cycles) >= 2:
        e0, n0, t0 = map(float, cycles[0]); e1, n1, t1 = map(float, cycles[-1])
        if t1 > t0:
            output['measured_seconds_per_M'] = (e1-e0)/(t1-t0)
            output['estimated_hours_to_1000M_at_measured_rate'] = 1000*(e1-e0)/(t1-t0)/3600
    fits = [v for k,v in output.items() if k.startswith('Theta_fit')]
    if not all_finite or output['bad_metric_max'] > 0 or output['termination'] == 'fatal':
        output['status'] = 'failed'
    elif any(v['rate_per_M'] > 0 and v['observed_factor'] > 10 for v in fits):
        output['status'] = 'growing perturbation; not stable'
    elif output['exact_zero_history_residuals']:
        output['status'] = 'zero equilibrium preserved over recorded interval; perturbation stability separate'
    else:
        output['status'] = 'finite over recorded interval; stability not established'
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    results={str(run):audit(run) for run in args.runs}
    args.output.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    print(json.dumps(results,indent=2,allow_nan=False))


if __name__=='__main__':main()
