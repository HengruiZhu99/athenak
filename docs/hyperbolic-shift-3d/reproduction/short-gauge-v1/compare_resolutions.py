#!/usr/bin/env python3
"""Compare collected SG1 N16/N24 campaigns; never submits or admits a restart."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = 'a5fd4a1cf9924927dfa19ddf9d27e91c6d13a91a0d12b8e997e6868853801cff'
BINARY = 'build-sycl-mpi-short-gauge-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_short_gauge'
BINARY_SHA = '1b75a50ec11f8b3908de9172bd5baec57eb8bc17d382d1fba25173ab4352996f'
REVIEW_SHA = '723e5cda25c9496b12b487bbe7f37c6578f95a2743feecb682c1fbb673ccc224'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_campaign(raw, plans, n):
    """Require terminal, hash-verified artifacts; retain numerical failures as evidence."""
    reports, histories, records = [], [], []
    previous = None
    for segment, relative in enumerate(plans, 1):
        path = raw / relative
        plan, worker = read(path), read(path.with_name('worker-status.json'))
        binding, collected = read(path.with_name('binding.json')), read(path.with_name('collected.json'))
        require(plan['n'] == n and plan['segment'] == segment, 'Wrong resolution or segment order')
        require(collected['scheduler']['job_state'] in ['F', 'X'], 'Job is not terminal')
        require(digest(path) == worker['plan_sha256'] == binding['plan_sha256'], 'Plan identity changed')
        for record in [worker, binding]:
            require(record['source_manifest_sha256'] == SOURCE and record['binaries'] == {BINARY: BINARY_SHA}, 'Scientific binding changed')
        if n == 24:
            review = plan['n16_accuracy_review']
            require(digest(raw / review) == binding['inputs'][review] == REVIEW_SHA, 'N16 review identity changed')
        require(len(worker['cases']) == 1, 'Expected one puncture case')
        case = worker['cases'][0]
        require(not case['timed_out'] and not case['storage_error'], 'Infrastructure failure requires diagnosis')
        for name, identity in case['artifacts'].items():
            artifact = raw / name
            require(artifact.stat().st_size == identity['bytes'] and digest(artifact) == identity['sha256'], 'Artifact identity changed: ' + name)
        command = plan['commands'][0]
        report_path = raw / command['report']
        history_path = raw / (command['args'][1] + '.history.csv')
        for artifact in [report_path, history_path]:
            require(str(artifact.relative_to(raw)) in case['artifacts'], 'Required artifact was not recorded')
        report = read(report_path)
        require(report['n'] == n and report['dt'] == .0005 * 16 / n and report['checkpoint_schema'] == 'HYP3D_MPI_SG1', 'Wrong native configuration')
        if previous is None:
            require(plan['start_verification']['kind'] == 'fresh_wormhole' and plan['start_verification']['time'] == 0, 'Must start from fresh data')
        else:
            require(previous['checkpoint_admitted'] and plan['start_verification']['time'] == previous['t'], 'Invalid segment continuity')
            require(plan['start_verification']['sha256'] == records[-1]['checkpoint_sha256'], 'Restart identity mismatch')
        history = np.atleast_1d(np.genfromtxt(history_path, delimiter=',', names=True))
        require(len(history) > 0 and np.all(np.diff(history['t']) >= 0), 'Invalid history ordering')
        require(history['t'][-1] <= report['t'] + 1e-10, 'History exceeds last accepted state')
        cp_name = command['args'][1] + '.checkpoint'
        cp_sha = case['artifacts'].get(cp_name, {}).get('sha256')
        records.append(dict(plan=relative, status=worker['status'], report_status=report['status'],
                            checkpoint_sha256=cp_sha, artifacts=case['artifacts']))
        reports.append(report)
        histories.append(history)
        previous = report
    data = np.concatenate(histories)
    # A restart repeats its first diagnostic. Keep the preceding observation;
    # require all physical diagnostic fields to agree, excluding elapsed wall time.
    keep = []
    for i in range(len(data)):
        if keep and abs(data['t'][i] - data['t'][keep[-1]]) < 1e-12:
            for key in data.dtype.names:
                if key != 'wall_seconds':
                    require(np.isclose(data[key][i], data[key][keep[-1]], rtol=1e-10, atol=1e-12, equal_nan=True), 'Restart diagnostic mismatch: ' + key)
        else:
            require(not keep or data['t'][i] > data['t'][keep[-1]], 'Overlapping histories')
            keep.append(i)
    return dict(n=n, reports=reports, history=data[keep], records=records, folder=path.parent)


def counters(reports):
    names = ['chi_floor_stage_interventions', 'chi_floor_acceptance_interventions',
             'surface_chi_floor_stage_interventions', 'surface_chi_floor_final_interventions',
             'lapse_floor_node_interventions', 'scri_interpolated_chi_negative_count',
             'scri_interpolated_chi_zero_count']
    groups = {}
    for report in reports:
        for row in report['candidate_sign_events']:
            event = dict(zip(report['candidate_sign_columns'], row))
            name = ('surface' if event['point_id'] < 0 else 'volume') + '_field_' + str(event['field'])
            item = groups.setdefault(name, dict(negative_stage=0, negative_final=0, zero_stage=0, zero_final=0,
                                               minimum=None, first_negative_time=None, first_zero_time=None))
            for key in ['negative_stage', 'negative_final', 'zero_stage', 'zero_final']:
                item[key] += event[key]
            for key in ['minimum', 'first_negative_time', 'first_zero_time']:
                if event[key] is not None:
                    item[key] = event[key] if item[key] is None else min(item[key], event[key])
    return dict(totals={k: sum(r[k] for r in reports) for k in names}, candidate_groups=groups,
                scope='Segment-local counts include rejected attempts; event times are coordinate time. Boundary interpolation counts are separate.')


def compare(raw, plans16, plans24, out):
    cases = {n: load_campaign(raw, plans, n) for n, plans in [(16, plans16), (24, plans24)]}
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fields = [('H_exterior_L2', 'Candidate-exterior Hamiltonian RMS'),
              ('M_exterior_L2', 'Candidate-exterior momentum magnitude RMS'),
              ('Mr_exterior_L2', 'Candidate-exterior radial momentum RMS'),
              ('H_L2', 'Full-domain Hamiltonian RMS'), ('M_L2', 'Full-domain momentum magnitude RMS'),
              ('horizon_irreducible_mass', 'Accepted horizon mass / M − 1')]
    for n, case in cases.items():
        data, last = case['history'], case['reports'][-1]
        for ax, (field, label) in zip(axes.flat, fields):
            value = data[field]
            if field == 'horizon_irreducible_mass':
                value = np.where(data['horizon_found'] == 1, 3 * value - 1, np.nan)
            ax.plot(data['t_over_M'], value, '.-', ms=3, label=f'N{n}')
            ax.set(xlabel='t / M', ylabel=label)
            ax.grid(alpha=.2)
        summary[n] = dict(last_accepted_t_over_M=last['t_over_M'], last_diagnostic_t_over_M=float(data['t_over_M'][-1]),
                          status=last['status'], accepted_horizon_rows=int(np.sum(data['horizon_found'] == 1)),
                          rows=len(data), counters=counters(case['reports']), segments=case['records'],
                          final_accepted_mass_error=(3 * last['horizon_irreducible_mass'] - 1)
                          if last['horizon_found'] and last['horizon_irreducible_mass'] is not None else None)
    axes[0, 0].legend()
    fig.suptitle('SG1: matched fresh N16 and N24 evolutions, fixed dt/dx')
    fig.tight_layout(rect=(0, .07, 1, .96))
    fig.text(.5, .02, 'Exterior histories use each run’s candidate horizon; missing diagnostics remain gaps. Two grids do not establish convergence order.', ha='center', fontsize=9)
    fig.savefig(out / 'histories.png', dpi=150)
    # Only compare endpoints at the SAME accepted time and with verified healthy
    # checkpoints. A failed checkpoint must never be treated as an accepted state.
    result = dict(resolutions=summary, common_endpoint=None)
    last = [case['reports'][-1] for case in cases.values()]
    if all(r['status'] == 'completed' and r['checkpoint_admitted'] and r['t_over_M'] == 10 for r in last):
        radii = [r['horizon_trial_radius'] for r in last]
        require(all(r is not None and math.isfinite(r) and 0 < r < 1 for r in radii), 'No usable candidate cutoff')
        cutoff = math.ceil(max(radii) * 10000) / 10000 + .0001
        bands = sorted(set([0, .03, .06, cutoff, .13, .16, .18, .2, .25, .3, .4, .45, .55, .6, .675, .7, .725, .75, .775, .8, .85, .9, .95, 1]))
        exe = ROOT / 'build-hyperbolic-mpi/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_checkpoint_constraints_short_gauge'
        endpoints = {}
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for n, case in cases.items():
            verified = read(case['folder'] / 'checkpoint-verified.json')
            require(verified['verified'] and verified['completed'] and verified['n'] == n and verified['t_over_M'] == 10, 'Missing endpoint verification')
            cp = raw / verified['checkpoint']
            require(digest(cp) == verified['sha256'] == case['records'][-1]['checkpoint_sha256'], 'Checkpoint verification identity mismatch')
            dest = out / f'n{n}-radial.json'
            command = [str(exe), str(n), str(cp), str(.0005 * 16 / n), str(cutoff), str(dest), ','.join(map(str, bands))]
            with (out / f'n{n}-radial.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            d = read(dest)
            require(d['t_over_M'] == 10 and d['cutoff'] == cutoff, 'Diagnostic state mismatch')
            for key in ['H_L2', 'M_L2', 'Mr_L2']:
                require(math.isclose(d['norms']['total'][key], case['reports'][-1][key], rel_tol=1e-9, abs_tol=1e-12), 'Native norm reconstruction mismatch')
            cumulative = d['radial_cumulative']
            r = np.array([row['cutoff'] for row in cumulative])
            volume = -np.diff([row['volume'] for row in cumulative])
            for ax, key in zip(axes, ['H2', 'M2']):
                power = -np.diff([row[key] for row in cumulative])
                value = np.full_like(volume, np.nan)
                np.divide(np.maximum(power, 0), volume, out=value, where=volume > 0)
                ax.stairs(np.sqrt(value), r, label=f'N{n}')
            regions = {}
            for lower, upper in [(.16, .3), (.775, .9)]:
                below = next(row for row in cumulative if row['cutoff'] == lower)
                above = next(row for row in cumulative if row['cutoff'] == upper)
                exterior = next(row for row in cumulative if row['cutoff'] == cutoff)
                regions[f'{lower}-{upper}'] = {key: (below[key] - above[key]) / exterior[key]
                                              if exterior[key] > 0 else None for key in ['H2', 'M2']}
            endpoints[n] = dict(norms=d['norms'], checkpoint_sha256=verified['sha256'], command=command,
                                regional_fractions_of_exterior_squared_norm=regions)
        for ax, label in zip(axes, ['Hamiltonian RMS', 'Momentum magnitude RMS']):
            ax.set(xlabel='compactified radius', ylabel=label, yscale='log')
            ax.grid(alpha=.2)
            ax.legend()
            ax.axvspan(.16, .18, color='k', alpha=.1)
        fig.suptitle('SG1 constraints at 10M: N16 versus N24')
        fig.tight_layout()
        fig.savefig(out / 'radial.png', dpi=150)
        ratios = {key: endpoints[16]['norms']['exterior'][key] / endpoints[24]['norms']['exterior'][key] for key in ['H_L2', 'M_L2', 'Mr_L2', 'Theta_L2'] if endpoints[24]['norms']['exterior'][key] > 0}
        result['common_endpoint'] = dict(t_over_M=10, cutoff=cutoff, candidate_radii=radii, diagnostic_binary_sha256=digest(exe), endpoints=endpoints, n16_over_n24_ratios=ratios)
    (out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--n16-plans', nargs='+', required=True)
    parser.add_argument('--n24-plans', nargs='+', required=True)
    parser.add_argument('--output', type=Path, default=HERE / 'resolution-comparison')
    args = parser.parse_args()
    compare(args.root.resolve(), args.n16_plans, args.n24_plans, args.output.resolve())
