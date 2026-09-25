#!/usr/bin/env python3
"""Check stationary-provider default/opt-out equality or time-dependent rejection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from z4c_kerr_trumpet import replace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gauge-wave', action='store_true')
    p.add_argument('--linear-wave', action='store_true')
    p.add_argument('--case', help='Run one stationary provider case')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    if a.linear_wave:
        base = (root / 'inputs/tests/linear_wave_z4c.athinput').read_text()
        for key in ['nx1', 'nx2', 'nx3']:
            base = re.sub(r'(?m)^' + key + r'\s*=.*$', key + ' = 8', base)
        base = replace(base, 'nlim', 3).replace('<z4c>', '<z4c>\nuse_analytic_background = true\ndebug_balance = true\ndebug_snapshot_operations = init_state,post_recast')
        cases = {'linear_wave_flat': {}}
    elif a.gauge_wave:
        base = (root / 'inputs/z4c/awa/z4c_gauge_wave.athinput').read_text()
        base = replace(base, 'nlim', 3).replace('<z4c>', '<z4c>\nuse_analytic_background = true\ndebug_balance = true\ndebug_snapshot_operations = init_state,post_recast')
        cases = {'gauge_wave': {}}
    else:
        base = (root / 'tst/inputs/z4c_puncture_background.athinput').read_text()
        cases = {name: {'bh_background': name} for name in
                 ['schwarzschild_puncture', 'schwarzschild_trumpet', 'kerr_schild']}
        cases['kerr_schild_indirect'] = {'bh_background': 'kerr_schild',
                                       'use_direct_z4c_background': 'false'}
        cases['flat'] = {'bh_background': 'kerr_schild', 'bh_mass': 0}
    results = {'binary_sha256': hashlib.sha256(a.exe.read_bytes()).hexdigest(), 'cases': {}}
    for name, changes in cases.items():
        if a.case and name != a.case:
            continue
        text = base
        if name == 'flat':
            text = text.replace('<coord>', '<coord>\nminkowski = true')
        for k, v in changes.items():
            text = replace(text, k, v)
        snapshots = []
        for mode in ['false', 'default', 'true']:
            run = out / (name + '_' + mode); run.mkdir()
            config = text if mode == 'default' else text.replace('<problem>', '<problem>\ncache_stationary_background = ' + mode)
            (run / 'input').write_text(config)
            with (run / 'run.log').open('w') as log:
                r = subprocess.run(['mpiexec', '-n', '1', str(a.exe.resolve()), '-i', 'input'], cwd=run,
                                   env=dict(os.environ, OMP_NUM_THREADS='1'), stdout=log,
                                   stderr=subprocess.STDOUT, timeout=120)
            log = (run / 'run.log').read_text()
            if a.gauge_wave and mode == 'true':
                assert r.returncode != 0 and 'explicitly time-independent analytic' in log
                continue
            assert r.returncode == 0 and 'Terminating on cycle limit' in log, log[-2000:]
            stats = re.findall(r'Z4C_BACKGROUND_CACHE rank=\d+ fills=(\d+) hits=(\d+)', log)
            if mode != 'false' and not a.gauge_wave:
                assert stats and all(int(f) > 0 and int(h) > 0 for f, h in stats)
            else:
                assert not stats
            data = {x.name: x.read_bytes() for x in run.iterdir() if x.suffix in ['.bin', '.hst', '.tab']}
            assert data
            snapshots.append(data)
            if a.gauge_wave:
                backgrounds = [v for k, v in data.items() if k.endswith('.background.bin')]
                assert len(backgrounds) > 1 and len(set(backgrounds)) > 1, 'background did not evolve'
        assert all(s == snapshots[0] for s in snapshots[1:]), name
        results['cases'][name] = {'passed': True, 'identical_files': len(snapshots[0])}
        print(name, 'PASS', flush=True)
        (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
