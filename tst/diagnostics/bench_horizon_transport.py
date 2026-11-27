"""Paired whole-solver timing and history parity for modeled-horizon diagnostics."""
import argparse
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'vis/python'))
import athena_read


def timed(exe, launcher, inp, directory):
    directory.mkdir()
    result = subprocess.run([*launcher, str(exe), '-i', str(inp), '-d', str(directory)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (directory / 'log').write_text(result.stdout)
    result.check_returncode()
    # Exclude initialization, the first eight cycles, and final output.
    clocks = {int(c): float(t) for t, c in re.findall(r'elapsed=([\d.eE+-]+) cycle=(\d+)', result.stdout)}
    return clocks[32] - clocks[8]


def parity(old, new):
    a, b = [athena_read.hst(str(d / 'matched_bondi.user.hst')) for d in (old, new)]
    assert set(a) <= set(b)
    worst = 0
    for k, values in a.items():
        np.testing.assert_array_equal(np.isfinite(values), np.isfinite(b[k]), err_msg=k)
        np.testing.assert_allclose(values, b[k], atol=2e-10, rtol=2e-10, equal_nan=True, err_msg=k)
        good = np.isfinite(values)
        if good.any():
            worst = max(worst, float(np.max(np.abs(values[good] - b[k][good]))))
    return dict(common_columns=len(a), added_columns=len(b)-len(a), max_abs_difference=worst)


def benchmark(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    base = Path(__file__).with_name('binary_shell.athinput').read_text().split('<output2>')[0]
    for key, old, new in [('nlim', 2, 40), ('ndiag', 100, 8), ('flux_ntheta', 24, 64),
                          ('flux_nphi', 48, 128), ('b_inf', 0, 0.002),
                          ('flux_shell_budget', 'true', 'false')]:
        assert f'{key} = {old}\n' in base
        base = base.replace(f'{key} = {old}\n', f'{key} = {new}\n')
    base = base.replace('flux_horizon1 = false', 'flux_horizon1 = true\nflux_horizon2 = true')
    base = base.replace('flux_radius1 = 3', 'flux_radius1 = 1.55\nflux_radius2 = 1.55')
    exes = {'baseline': args.baseline.resolve(), 'candidate': args.candidate.resolve(),
            'inventory': args.candidate.resolve()}
    report = {}
    for cadence in [1, 8]:
        inp = root / f'cadence-{cadence}.in'
        inp.write_text(base.replace('dcycle = 1\n', f'dcycle = {cadence}\n'))
        inventory = root / f'inventory-{cadence}.in'
        inventory.write_text(inp.read_text().replace('<problem>', '<problem>\nflux_transfer_inventory = true'))
        times = {k: [] for k in exes}
        for rep in range(args.repeats):
            order = list(exes)
            for name in order[rep % len(order):] + order[:rep % len(order)]:
                directory = root / f'{cadence}-{name}-{rep}'
                case = inventory if name == 'inventory' else inp
                times[name].append(timed(exes[name], shlex.split(args.launcher), case, directory))
        summary = parity(root / f'{cadence}-baseline-0', root / f'{cadence}-candidate-0')
        summary['inventory_parity'] = parity(root / f'{cadence}-candidate-0', root / f'{cadence}-inventory-0')
        if args.compare_ranks is not None:
            summary['rank_parity'] = parity(args.compare_ranks / f'{cadence}-inventory-0',
                                            root / f'{cadence}-inventory-0')
        sample = athena_read.hst(str(root / f'{cadence}-inventory-0' / 'matched_bondi.user.hst'))
        for hole in (1, 2):
            assert np.all(sample[f'valid_BH{hole}_Kerr'] == 1)
            assert np.all(sample[f'transport_interval_valid_BH{hole}_Kerr'][1:] == 1)
            assert np.all(sample[f'inventory_H{hole}_storage_valid'] == 1)
        summary['inventory_cost_ratio'] = statistics.median(times['inventory']) / statistics.median(times['candidate'])
        summary.update(seconds_24_cycles=times, speedup=statistics.median(times['baseline']) /
                       statistics.median(times['candidate']))
        report[cadence] = summary
        print(json.dumps(summary), flush=True)
    (root / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ['baseline', 'candidate', 'output']:
        parser.add_argument(arg, type=Path)
    parser.add_argument('--launcher', default='')
    parser.add_argument('--compare-ranks', type=Path,
                        help='compare all native channels to another rank count in this directory')
    parser.add_argument('--repeats', type=int, default=5)
    benchmark(parser.parse_args())
