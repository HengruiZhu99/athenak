#!/usr/bin/env python3
"""Verify additive inner-sponge RHS semantics, including zero damping rate.

Requires a double-precision MPI executable and NumPy. Checks every field at
all three RK stages, exact vacuum, and bitwise MPI partition independence.
This is a source-operator regression, not a long-time stability test.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from z4c_gauge_pulse import snapshot


def verify_source(run, rate, perturbed):
    nonzero_annulus_rhs = False
    damping_response = False
    records = {}
    for stage in (1, 2, 3):
        state = snapshot(run, 'pre_rhs_state', 0, stage)
        before = snapshot(run, 'pre_excision_rhs', 0, stage)
        after = snapshot(run, 'post_excision_rhs', 0, stage)
        final = snapshot(run, 'post_excision_state', 0, stage)
        assert state.keys() == before.keys() == after.keys() == final.keys()
        worst = 0.
        for gid, (block, pre) in before.items():
            post = after[gid][1]
            u = state[gid][1]
            nz, ny, nx = pre.shape[1:]
            axes = [block['xmin'][a] + (np.arange(n)+.5)
                    * (block['xmax'][a]-block['xmin'][a])/n
                    for a, n in enumerate((nx, ny, nz))]
            z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
            radius = np.sqrt(x*x+y*y+z*z)
            core, annulus, exterior = radius <= .5, (radius > .5) & (radius < 1), radius >= 1
            q = np.clip((radius-.5)/.5, 0., 1.)
            ramp = q**3*(10+q*(-15+6*q))
            sigma = rate*(1-ramp)
            expected = pre - sigma[None]*u
            expected[:, core] = 0.
            assert np.count_nonzero(post[:, core]) == 0
            assert np.count_nonzero(final[gid][1][:, core]) == 0
            assert np.array_equal(post[:, exterior], pre[:, exterior])
            if rate == 0:
                assert np.array_equal(post[:, annulus], pre[:, annulus]), \
                    'Zero damping changed the physical/KO RHS in the annulus'
            scale = max(float(np.max(abs(pre))), float(np.max(abs(sigma[None]*u))), 1e-300)
            error = float(np.max(abs(post-expected)))/scale
            assert error < 2e-13, (stage, gid, error)
            worst = max(worst, error)
            nonzero_annulus_rhs |= bool(np.any(pre[:, annulus]))
            damping_response |= bool(np.any(post[:, annulus] != pre[:, annulus]))
            if not perturbed:
                assert np.count_nonzero(pre) == np.count_nonzero(post) == 0
                assert np.count_nonzero(u) == np.count_nonzero(final[gid][1]) == 0
        records[str(stage)] = worst
    if perturbed:
        assert nonzero_annulus_rhs
        assert damping_response == (rate > 0)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--launcher', default='mpiexec')
    p.add_argument('--ranks', nargs='+', type=int, default=[1, 4])
    args = p.parse_args()
    base = (Path(__file__).resolve().parents[1]/'inputs/z4c_ks_background.athinput').read_text()
    base = re.sub(r'^nlim\s*=.*', 'nlim = 1', base, flags=re.M)
    base = base.replace('excision_freeze_radius = 1.0', 'excision_freeze_radius = 0.5')
    base = base.replace('excision_ramp_radius = 1.4', 'excision_ramp_radius = 1.0')
    base = base.replace('<problem>', '<problem>\nzero_tmunu = true\n'
                        'vacuum_gauge_pulse_x1 = 0.75\nvacuum_gauge_pulse_width = 0.75')
    base = base.replace('<z4c>', '<z4c>\ndebug_balance_freeze = 0.5\n'
                        'debug_balance_ramp = 1.0\ndebug_snapshot_operations = '
                        'pre_rhs_state,pre_excision_rhs,post_excision_rhs,post_excision_state')
    results = {}
    references = {}
    for ranks in args.ranks:
        for rate in (0, 5):
            for perturbed in (False, True):
                name = f'rate{rate}_{"pulse" if perturbed else "zero"}_r{ranks}'
                run = args.output.resolve()/name
                run.mkdir(parents=True, exist_ok=False)
                text = base.replace('excision_damp_rate = 5', f'excision_damp_rate = {rate}')
                text = text.replace('<problem>', '<problem>\n'
                                    f'vacuum_gauge_pulse_amplitude = {1e-8 if perturbed else 0}')
                (run/'input.athinput').write_text(text)
                with (run/'run.log').open('w') as log:
                    subprocess.run([args.launcher, '-n', str(ranks), str(args.exe.resolve()),
                                    '-i', 'input.athinput'], cwd=run, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
                log = (run/'run.log').read_text()
                assert 'Terminating on cycle limit' in log and '### FATAL ERROR' not in log
                results[name] = verify_source(run, rate, perturbed)
                # Exact arrays, not only reduced maxima, must agree across ranks.
                for stage in (1, 2, 3):
                    for operation in ('pre_rhs_state', 'pre_excision_rhs', 'post_excision_rhs',
                                      'post_excision_state'):
                        for gid, (_, values) in snapshot(run, operation, 0, stage).items():
                            key = (rate, perturbed, stage, operation, gid)
                            if ranks == args.ranks[0]:
                                references[key] = values
                            else:
                                assert np.array_equal(references[key], values), key
                print(name, 'passed', flush=True)
                (args.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    print('All source semantics, exact-zero, and MPI parity checks passed.')


if __name__ == '__main__':
    main()
