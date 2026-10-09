"""Reproducible native Cartesian layer sweeps; all outputs stay in output_dir.

Usage: python run_layer_validation.py /abs/athena output_dir [--suite long]
Small runs are finite-duration consistency evidence, never a stability proof.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import shutil
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'reader', ROOT / 'vis/python/bin_convert.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
spec = importlib.util.spec_from_file_location('budget', Path(
    __file__).with_name('analyze_native_constraints.py'))
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


def summarize(directory):
    history = np.atleast_2d(np.loadtxt(directory / 'hyp.z4c.user.hst'))
    row = history[-1]
    paths = sorted((directory / 'bin').glob('*.z4c.*.bin'))
    fields = {k: np.asarray(v) for k, v in reader.read_binary(
        str(paths[-1]))['mb_data'].items()}
    mask = fields['z4c_active'].astype(bool)
    shape = fields['z4c_chi'][mask].shape
    metric = np.zeros(shape + (3, 3))
    for i, j, suffix in [(0, 0, 'xx'), (0, 1, 'xy'), (0, 2, 'xz'),
                         (1, 1, 'yy'), (1, 2, 'yz'), (2, 2, 'zz')]:
        metric[:, i, j] = metric[:, j, i] = fields['z4c_g' +
                                                   suffix][mask] / fields['z4c_chi'][mask]
    eigenvalues = np.linalg.eigvalsh(metric)
    constraints = budget.analyze(
        sorted((directory / 'bin').glob('*.con.*.bin'))[-1], [0, .35, .75, .9, 1])
    return {'time': row[0], 'H': row[2], 'M': row[3], 'Z': row[4], 'Theta': row[5],
            'alpha_min': row[8], 'chi_min': row[9], 'pole_deviation_max': row[12],
            'null_deviation_max': row[13], 'metric_eigen_min': eigenvalues.min(),
            'metric_eigen_max': eigenvalues.max(),
            'metric_stretch_ratio_max': (eigenvalues[:, -1] / eigenvalues[:, 0]).max(),
            'constraint_budget': constraints}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--suite',
                        choices=['quick', 'long', 'control', 'reference-long'],
                        default='quick')
    args = parser.parse_args()
    executable, output = args.executable.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    immutable_executable = output / 'athena-validation'
    if immutable_executable.exists():
        raise ValueError('Use a fresh output directory to preserve run provenance')
    shutil.copy2(executable, immutable_executable)
    executable = immutable_executable
    template = (ROOT / 'inputs/z4c/hyperboloidal_layer.athinput').read_text()
    if args.suite == 'long':
        cases = [('finite-angular-long', 24, .1, .02, True, 2., '')]
    elif args.suite == 'reference-long':
        cases = [('reference-long', 24, 0., 0., True, 2., '')]
    elif args.suite == 'control':
        cases = [('finite-angular', 24, .1, .02, True, .01, '')]
    else:
        cases = [('reference', 24, 0., 0., True, .05, '')]
        for label, amplitude, shift, angular in [('small-angular', .001, .0002, True),
                                                 ('finite-radial', .1, .02, False)]:
            for n in [24, 36, 48]:
                cases += [(label, n, amplitude, shift, angular, .01, '')]
        for label, extra in [
                ('wide-layer', (
                    '<z4c>\n'
                    'hyperboloidal_layer_r0=.2\n'
                    'hyperboloidal_layer_r1=.8\n')),
                ('narrow-layer', (
                    '<z4c>\n'
                    'hyperboloidal_layer_r0=.5\n'
                    'hyperboloidal_layer_r1=.7\n'
                    'hyperboloidal_gauge_r0=.55\n')),
                ('no-damping', (
                    '<z4c>\n'
                    'hyperboloidal_kappa1=0\n'
                    'hyperboloidal_layer_lapse_outer=0\n'
                    'hyperboloidal_layer_shift_outer=0\n')),
                ('half-pole-step', '<z4c>\nhyperboloidal_pole_cfl=.02\n'),
                ('degree-two', '<z4c>\nhyperboloidal_ghost_degree=2\n'),
                ('degree-four', '<z4c>\nhyperboloidal_ghost_degree=4\n'),
                ('source-disabled', '<z4c>\nhyperboloidal_preferred_source=false\n')]:
            cases += [(label, 24, .1, .02, True, .01, extra)]
    provenance = {'executable': str(executable),
                  'sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
                  'source_commit': subprocess.check_output(['git',
                                                            'rev-parse',
                                                            'HEAD'],
                                                           cwd=ROOT,
                                                           text=True).strip(),
                  'source_dirty': bool(subprocess.check_output(['git',
                                                                'status',
                                                                '--porcelain'],
                                                               cwd=ROOT,
                                                               text=True)),
                  'cases': []}
    for label, n, lapse, shift, angular, duration, extra in cases:
        directory = output / f'{label}-N{n}'
        directory.mkdir(exist_ok=True)
        text = template + f'''
<mesh>
nx1={n}
nx2={n}
nx3={n}
<meshblock>
nx1={n}
nx2={n}
nx3={n}
<time>
nlim=-1
tlim={duration}
ndiag=100
<problem>
lapse_pulse={lapse}
shift_pulse={shift}
pulse_angular={str(angular).lower()}
'''
        for i in [1, 2, 3, 4, 5]:
            cadence = duration if 'long' not in args.suite else .002
            text += f'\n<output{i}>\ndt={cadence}\n'
        text += '\n' + extra
        input_path = directory / 'layer.athinput'
        input_path.write_text(text)
        command = [str(executable), '-i', str(input_path)]
        started = time.monotonic()
        with (directory / 'run.log').open('w') as log:
            result = subprocess.run(
                command,
                cwd=directory,
                stdout=log,
                stderr=subprocess.STDOUT)
        record = {
            'name': directory.name,
            'command': command,
            'exit_status': result.returncode,
            'wall_seconds': time.monotonic() - started,
            'requested_time': duration}
        try:
            record['diagnostics'] = summarize(directory)
        except (OSError, ValueError, IndexError, KeyError) as error:
            record['diagnostic_error'] = str(error)
        provenance['cases'].append(record)
        receipt = json.dumps(provenance, indent=2, default=float) + '\n'
        (output / 'results.json').write_text(receipt)
        status = {k: v for k, v in record.items() if k not in ['command', 'diagnostics']}
        diagnostics = {k: v for k, v in record.get('diagnostics', {}).items()
                       if k != 'constraint_budget'}
        print(json.dumps(status) + ' ' + json.dumps(diagnostics, default=float),
              flush=True)


if __name__ == '__main__':
    main()
