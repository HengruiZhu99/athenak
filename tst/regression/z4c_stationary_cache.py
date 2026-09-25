#!/usr/bin/env python3
"""Short cache on/off equality tests; not long-time perturbation stability.

Run with an MPI z4c_tov_ks executable. Checks all saved active/ghost spacetime
and background bytes and histories, plus exact zero/genuine matter response.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from z4c_kerr_trumpet import inspect, replace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--launcher', default='mpiexec')
    p.add_argument('--ranks', type=int, default=4)
    p.add_argument('--guards-only', action='store_true')
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    exe = a.exe.resolve()
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    base = (Path(__file__).resolve().parents[1] /
            'inputs/z4c_puncture_background.athinput').read_text()
    for key, value in dict(bh_background='kerr_trumpet', bh_spin='.9', a='.9',
                           extrap_order=2, residual_lapse_damping='.1',
                           shift_eta='.02', damp_kappa1=0,
                           max_nmb_per_rank=128).items():
        base = replace(base, key, value)
    results = {'scope': __doc__, 'binary_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
               'pairs': {}, 'rejections': {}}
    for refined in ([] if a.guards_only else [False, True]):
        for case in ['vacuum', 'lapse', 'atmosphere']:
            text = base
            if refined:
                text = replace(text, 'refinement', 'static')
                text += ('\n<refined_region0>\nlevel = 1\nx1min = -1\nx1max = 1\n'
                         'x2min = -1\nx2max = 1\nx3min = -1\nx3max = 1\n')
            if case == 'lapse':
                text = text.replace('<problem>', '<problem>\nvacuum_gauge_pulse_amplitude = 1e-8')
            elif case == 'atmosphere':
                for key, value in dict(zero_tmunu='false', zero_tmunu_feedback='false',
                                       dfloor='1e-14', pfloor='1e-26').items():
                    text = replace(text, key, value)
            name = case + ('_refined' if refined else '_uniform')
            runs = []
            for enabled in ['false', 'true']:
                run = out / (name + '_' + enabled); run.mkdir()
                config = text.replace('<problem>', '<problem>\ncache_stationary_background = ' + enabled)
                (run / 'input.athinput').write_text(config)
                with (run / 'run.log').open('w') as log:
                    subprocess.run([a.launcher, '-n', str(a.ranks), str(exe), '-i', 'input.athinput'],
                                   cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=300)
                inspect(run, case == 'vacuum', refined, .9)
                log = (run / 'run.log').read_text()
                assert 'Terminating on cycle limit' in log
                if enabled == 'true':
                    stats = re.findall(r'Z4C_BACKGROUND_CACHE rank=(\d+) fills=(\d+) hits=(\d+) invalidations=(\d+)', log)
                    assert sorted(int(s[0]) for s in stats) == list(range(a.ranks)), stats
                    assert all(int(s[1]) > 0 and int(s[2]) > 0 for s in stats), stats
                runs.append(run)
            files = sorted(x.name for x in runs[0].iterdir() if x.suffix in ['.bin', '.hst'])
            assert files and files == sorted(x.name for x in runs[1].iterdir() if x.suffix in ['.bin', '.hst'])
            assert any(x.endswith('.background.bin') for x in files)
            hashes = {}
            for name2 in files:
                left, right = [(run / name2).read_bytes() for run in runs]
                assert left == right, name2
                hashes[name2] = hashlib.sha256(left).hexdigest()
            results['pairs'][name] = {'passed': True, 'files': hashes, 'cached_rank_statistics': stats}
            (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            print(name, 'PASS', len(files), 'byte-identical files', flush=True)
    for name, key, value in [('wrong_provider', 'bh_background', 'schwarzschild_trumpet'),
                             ('indirect_provider', 'use_direct_z4c_background', 'false')]:
        run = out / name; run.mkdir()
        text = replace(base, key, value).replace('<problem>', '<problem>\ncache_stationary_background = true')
        if name == 'wrong_provider':
            text = replace(replace(text, 'bh_spin', '0'), 'a', '0')
        (run / 'input.athinput').write_text(text)
        with (run / 'run.log').open('w') as log:
            r = subprocess.run([a.launcher, '-n', '1', str(exe), '-i', 'input.athinput'], cwd=run,
                               env=env, stdout=log, stderr=subprocess.STDOUT, timeout=60)
        assert r.returncode != 0
        expected = ('Puncture control requires direct background' if name == 'indirect_provider'
                    else 'Stationary cache requires the fixed direct Kerr trumpet provider.')
        assert expected in (run / 'run.log').read_text()
        results['rejections'][name] = True
    (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
