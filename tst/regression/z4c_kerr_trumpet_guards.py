#!/usr/bin/env python3
"""Spin-zero identity and input guards for stationary Kerr trumpet registration.

Compares every saved active/ghost residual and background byte against the
unchanged Schwarzschild trumpet branch, including a physical matter response.
This short correctness regression makes no long-time stability claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import numpy as np


def replace(text, key, value):
    pattern = r'^' + re.escape(key) + r'\s*=.*'
    assert re.search(pattern, text, flags=re.M), key
    return re.sub(pattern, f'{key} = {value}', text, flags=re.M)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--launcher', default='mpiexec')
    p.add_argument('--ranks', type=int, default=2)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    exe = a.exe.resolve()
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    base = (Path(__file__).resolve().parents[1] / 'inputs/z4c_puncture_background.athinput').read_text()
    for key, value in dict(bh_background='schwarzschild_trumpet', extrap_order=2,
                           residual_lapse_damping='.1', shift_eta='.02', damp_kappa1=0).items():
        base = replace(base, key, value)
    results = {'scope': __doc__, 'binary_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
               'identity': {}, 'rejections': {}}
    for case in ['vacuum', 'lapse', 'atmosphere']:
        text = base
        if case == 'lapse':
            text = text.replace('<problem>', '<problem>\nvacuum_gauge_pulse_amplitude = 1e-8')
        elif case == 'atmosphere':
            for key, value in dict(zero_tmunu='false', zero_tmunu_feedback='false',
                                   dfloor='1e-14', pfloor='1e-26').items():
                text = replace(text, key, value)
        cases = []
        for background in ['schwarzschild_trumpet', 'kerr_trumpet']:
            run = out / f'{case}-{background}'; run.mkdir()
            (run / 'input.athinput').write_text(replace(text, 'bh_background', background))
            with (run / 'run.log').open('w') as log:
                result = subprocess.run([a.launcher, '-n', str(a.ranks), str(exe),
                                         '-i', 'input.athinput'], cwd=run, stdout=log,
                                        stderr=subprocess.STDOUT, env=env)
            assert result.returncode == 0 and 'Terminating on cycle limit' in (run/'run.log').read_text(), run
            snapshots = sorted(run.glob('z4c_snapshot_*.json'))
            assert snapshots, run
            hashes = {}
            response = 0.
            for path in snapshots:
                meta = json.loads(path.read_text())
                assert meta['shape'][0] > 0 and meta['ng'] == 4
                for suffix in ['.bin', '.background.bin']:
                    f = path.with_suffix(suffix)
                    values = np.fromfile(f, '<f8').reshape(meta['shape'])
                    assert np.isfinite(values).all(), f
                    hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
                    if suffix == '.bin' and meta['operation'] == 'post_recast':
                        response = max(response, float(np.max(abs(values[:, :18]))))
            assert (response == 0) if case == 'vacuum' else (response > 0), (case, response)
            cases.append((hashes, response))
        assert cases[0] == cases[1], case
        results['identity'][case] = dict(all_snapshot_bytes_equal=True,
                                         includes_active_and_ghost=True,
                                         files_compared=len(cases[0][0]),
                                         geometric_response=cases[0][1],
                                         hashes=cases[0][0])
        print(case, 'active+ghost identity PASS', flush=True)
    kerr = replace(replace(replace(base, 'bh_background', 'kerr_trumpet'), 'bh_spin', '.9'), 'a', '.9')
    checks = {
        'primitive_prolongation': (kerr.replace('<mesh_refinement>', '<mesh_refinement>\nprolong_primitives = true'), 'does not support mesh_refinement/prolong_primitives=true'),
        'extremal_spin': (replace(replace(kerr, 'bh_spin', 1), 'a', 1), 'bh_background must'),
        'superextremal_spin': (replace(replace(kerr, 'bh_spin', 1.01), 'a', 1.01), 'bh_background must'),
        'nonunit_mass': (replace(kerr, 'bh_mass', 2), 'supports bh_mass'),
        'unsupported_chi': (replace(kerr, 'boundary_rhs', 'sommerfeld').replace('<z4c>', '<z4c>\nchi_psi_power = -2'), 'requires z4c/chi_psi_power=-4'),
        'spin_mismatch': (replace(kerr, 'a', '.8'), 'requires <coord>/a to match'),
        'core_projection': (replace(kerr, 'excision_project_state', 'true'), 'Puncture control requires'),
        'inner_damping': (replace(kerr, 'excision_damp_rate', 1), 'Puncture control requires'),
        'indirect_background': (replace(kerr, 'use_direct_z4c_background', 'false'), 'Puncture control requires'),
        'ks_orbit': (kerr.replace('<problem>', '<problem>\nstar_orbit = circular_geodesic'), 'Puncture coordinates require'),
        'hamiltonian_balance': (kerr.replace('<z4c>', '<z4c>\nresidual_hamiltonian_balance = true'), 'only the direct Schwarzschild trumpet'),
    }
    for name, (config, expected) in checks.items():
        run = out / ('reject-' + name); run.mkdir()
        (run / 'input.athinput').write_text(config)
        r = subprocess.run([a.launcher, '-n', '1', str(exe), '-i', 'input.athinput'], cwd=run,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        (run / 'run.log').write_text(r.stdout)
        assert r.returncode != 0 and expected in r.stdout, (name, r.returncode, r.stdout)
        results['rejections'][name] = {'passed': True, 'exit_code': r.returncode,
                                      'diagnostic': expected}
        print(name, 'rejection PASS', flush=True)
    (out / 'results.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
