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


def input_parameters(text):
    """Read the last assignment in each Athena input section, including overrides."""
    values, section = {}, ''
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].strip()
        if line.startswith('<') and line.endswith('>'):
            section = line[1:-1]
        elif '=' in line:
            key, value = line.split('=', 1)
            values[section + '/' + key.strip()] = value.strip()
    return values


def snapshot_workspace(output):
    """Preserve the launch workspace without asserting it built the executable."""
    snapshot = output / 'source-at-launch'
    snapshot.mkdir()
    patch = subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=ROOT)
    (snapshot / 'changes.patch').write_bytes(patch)
    changed = subprocess.check_output(
        ['git', 'diff', '--name-only', '-z', 'HEAD'], cwd=ROOT).split(b'\0')
    untracked = subprocess.check_output(
        ['git', 'ls-files', '--others', '--exclude-standard', '-z'],
        cwd=ROOT).split(b'\0')
    files = {}
    for name in sorted(set(changed + untracked) - {b''}):
        relative = Path(name.decode())
        source = ROOT / relative
        if source == output or output in source.parents:
            continue
        if source.is_file():
            contents = source.read_bytes()
            destination = snapshot / 'files' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(contents)
            files[str(relative)] = hashlib.sha256(contents).hexdigest()
        else:
            files[str(relative)] = None
    manifest = {
        'scope': 'Workspace at launch; executable build-source equality is not inferred',
        'patch_sha256': hashlib.sha256(patch).hexdigest(),
        'changed_files_sha256': files}
    (snapshot / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


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
    parameters = input_parameters((directory / 'layer.athinput').read_text())
    r0 = float(parameters.get('z4c/hyperboloidal_layer_r0', '.35'))
    r1 = float(parameters.get('z4c/hyperboloidal_layer_r1', '.75'))
    constraints = budget.analyze(
        sorted((directory / 'bin').glob('*.con.*.bin'))[-1],
        sorted(set([0, r0, r1, .9, 1])))
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
                        choices=['quick', 'long', 'control', 'reference-long',
                                 'gauge-scan', 'geometry-scan'],
                        default='quick')
    parser.add_argument('--overrides', type=Path,
                        help='Append this parameter-file fragment to every input')
    parser.add_argument('--duration', type=float,
                        help='Override every requested coordinate duration')
    parser.add_argument('--output-cadence', type=float,
                        help='Override diagnostics/checkpoint cadence')
    args = parser.parse_args()
    for value in (args.duration, args.output_cadence):
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error('Duration and output cadence must be finite and positive')
    override_text = args.overrides.read_text() if args.overrides else ''
    override_parameters = input_parameters(override_text)
    executable, output = args.executable.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    immutable_executable = output / 'athena-validation'
    if immutable_executable.exists():
        raise ValueError('Use a fresh output directory to preserve run provenance')
    source_snapshot = snapshot_workspace(output)
    shutil.copy2(executable, immutable_executable)
    executable = immutable_executable
    template = (ROOT / 'inputs/z4c/hyperboloidal_layer.athinput').read_text()
    if args.suite == 'geometry-scan':
        physical = ('<z4c>\nhyperboloidal_physical_trace_lapse=true\n'
                    'hyperboloidal_preferred_source=false\n'
                    'hyperboloidal_curvature_radius=.5\n'
                    'hyperboloidal_symmetric_ghosts=true\n'
                    'hyperboloidal_pole_cfl=.06\n')
        broad = physical + ('hyperboloidal_layer_r0=.2\n'
                            'hyperboloidal_layer_r1=.8\n')
        cases = [('gentle-layer-degree3', 24, .1, .02, True, .5, physical),
                 ('gentle-layer-degree2', 24, .1, .02, True, .5,
                  physical + 'hyperboloidal_ghost_degree=2\n'),
                 ('broad-layer-degree3', 24, .1, .02, True, .5, broad),
                 ('broad-layer-degree2', 24, .1, .02, True, .5,
                  broad + 'hyperboloidal_ghost_degree=2\n')]
    elif args.suite == 'gauge-scan':
        physical = ('<z4c>\nhyperboloidal_physical_trace_lapse=true\n'
                    'hyperboloidal_preferred_source=false\n'
                    'hyperboloidal_pole_cfl=.1\n')
        cases = [('physical-layer-degree3', 24, .1, .02, True, .2, physical),
                 ('physical-layer-degree2', 24, .1, .02, True, .2,
                  physical + 'hyperboloidal_ghost_degree=2\n'),
                 ('physical-cmc-degree3', 24, .1, .02, True, .2,
                  physical + 'hyperboloidal_layer=false\n'),
                 ('physical-cmc-degree2', 24, .1, .02, True, .2,
                  physical + 'hyperboloidal_layer=false\nhyperboloidal_ghost_degree=2\n')]
    elif args.suite == 'long':
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
                  'workspace_at_launch': source_snapshot,
                  'cases': []}
    for label, n, lapse, shift, angular, duration, extra in cases:
        if args.duration is not None:
            duration = args.duration
        n = int(override_parameters.get('mesh/nx1', n))
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
            if args.suite in ['gauge-scan', 'geometry-scan']:
                cadence = min(duration, .025)
            if args.output_cadence is not None:
                cadence = min(duration, args.output_cadence)
            text += f'\n<output{i}>\ndt={cadence}\n'
        text += '\n' + extra
        text += '\n' + override_text
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
            'requested_time': duration,
            'input_text': text,
            'input_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'input_parameters': input_parameters(text)}
        try:
            record['diagnostics'] = summarize(directory)
        except (OSError, ValueError, IndexError, KeyError) as error:
            record['diagnostic_error'] = str(error)
        provenance['cases'].append(record)
        receipt = json.dumps(provenance, indent=2, default=float) + '\n'
        (output / 'results.json').write_text(receipt)
        status = {k: v for k, v in record.items()
                  if k not in ['command', 'diagnostics', 'input_text',
                               'input_parameters']}
        diagnostics = {k: v for k, v in record.get('diagnostics', {}).items()
                       if k != 'constraint_budget'}
        print(json.dumps(status) + ' ' + json.dumps(diagnostics, default=float),
              flush=True)


if __name__ == '__main__':
    main()
