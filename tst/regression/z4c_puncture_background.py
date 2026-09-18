#!/usr/bin/env python3
"""Puncture residual stages: independent background values, MPI parity, matter response.

This short regression checks correctness, not long-time perturbation stability.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from z4c_background_balance import read_csv
from z4c_gauge_pulse import snapshot


def replace(text, key, value):
    return re.sub(r'^'+re.escape(key)+r'\s*=.*', f'{key} = {value}', text, flags=re.M)


def check_background(run):
    error = 0.
    paths = list(run.glob('z4c_snapshot_init_state_rank*_cycle0_stage0.json'))
    assert paths
    for path in paths:
        meta = json.loads(path.read_text())
        values = np.fromfile(path.with_suffix('.background.bin'), '<f8').reshape(meta['shape'])
        ng = meta['ng']
        for block, bg in zip(meta['blocks'], values):
            axes = [block['xmin'][a] + (np.arange(bg.shape[3-a])-ng+.5)*block['dx'][a]
                    for a in range(3)]
            z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
            psi = 1 + .5/np.sqrt(x*x+y*y+z*z)
            expected = np.zeros_like(bg)
            expected[0] = psi**meta['chi_psi_power']
            expected[1] = expected[4] = expected[6] = 1.
            expected[18] = psi**-2
            error = max(error, float(np.max(abs(expected-bg))))
    assert error < 2e-15, error
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launcher', default='mpiexec')
    parser.add_argument('--ranks', type=int, nargs='+', default=[1, 4])
    args = parser.parse_args()
    base = (Path(__file__).resolve().parents[1]/'inputs/z4c_puncture_background.athinput').read_text()
    results = {}
    for ranks in args.ranks:
        for refined in [False, True]:
            for case in ['vacuum', 'lapse', 'atmosphere']:
                text = base
                if refined:
                    text = replace(text, 'refinement', 'static')
                    text = replace(text, 'max_nmb_per_rank', 128)
                    text += ('\n<refined_region0>\nlevel = 1\n'
                             'x1min = -1\nx1max = 1\nx2min = -1\nx2max = 1\nx3min = -1\nx3max = 1\n')
                if case == 'lapse':
                    text = text.replace('<problem>', '<problem>\nvacuum_gauge_pulse_amplitude = 1e-8')
                if case == 'atmosphere':
                    text = replace(text, 'zero_tmunu', 'false')
                    text = replace(text, 'zero_tmunu_feedback', 'false')
                    text = replace(text, 'dfloor', '1e-14')
                    text = replace(text, 'pfloor', '1e-26')
                name = f'{case}_{"refined" if refined else "uniform"}_r{ranks}'
                run = args.output.resolve()/name
                run.mkdir(parents=True, exist_ok=False)
                (run/'input.athinput').write_text(text)
                with (run/'run.log').open('w') as stream:
                    subprocess.run([args.launcher, '-n', str(ranks), str(args.exe.resolve()),
                                    '-i', 'input.athinput'], cwd=run, stdout=stream,
                                   stderr=subprocess.STDOUT, check=True)
                assert 'Terminating on cycle limit' in (run/'run.log').read_text()
                rows = [row for path in run.glob('z4c_balance_rank*.csv') for row in read_csv(path)]
                assert rows
                if refined:
                    assert {'post_restrict', 'post_prolong'} <= {row['operation'] for row in rows}
                assert {'1', '2', '3'} <= {row['stage'] for row in rows}
                for row in rows:
                    assert int(row['nonfinite']) == 0
                    assert np.isfinite(float(row['max_abs']))
                    if case == 'vacuum':
                        assert float(row['max_abs']) == 0 and int(row['bit_mismatch']) == 0
                if case == 'vacuum':
                    for path in run.glob('z4c_geometry_rank*.csv'):
                        for row in read_csv(path):
                            assert float(row['max_abs']) == 0 and int(row['bit_mismatch']) == 0
                final = snapshot(run, 'post_recast', 2, 3)
                response = max(float(np.max(abs(v[:18]))) for _, v in final.values())
                assert (response == 0) if case == 'vacuum' else (response > 0)
                hashes = {str(gid): hashlib.sha256(v.tobytes()).hexdigest()
                          for gid, (_, v) in final.items()}
                result = {'background_max_error_including_ghosts': check_background(run),
                          'geometric_response': response, 'active_block_hashes': hashes}
                if ranks != args.ranks[0]:
                    assert result == results[name.rsplit('_r', 1)[0]+f'_r{args.ranks[0]}']
                results[name] = result
                (args.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
                print(name, response, 'PASS', flush=True)


if __name__ == '__main__':
    main()
