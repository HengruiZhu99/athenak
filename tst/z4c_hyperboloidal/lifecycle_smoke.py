#!/usr/bin/env python3
"""Exercise native dynamic initialization and fixed dt with the actual executable.

These are lifecycle tests on local CPU MPI ranks, not a numerical trajectory
oracle or GPU performance result. No test integrator or timestep overrides enter
the executable: all options below are ordinary athinput parameters.
"""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--build', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
parser.add_argument('--ranks', type=int, default=2)
parser.add_argument('--case', action='append', help='rerun selected cases, retaining prior results')
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parents[2]
exe = args.build.resolve()/'src/athena'
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
fixture = root/'tst/z4c_hyperboloidal/n12_initialize.athinput'
dt = 0.00046944203735617445
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
report = {'scope': 'actual executable lifecycle; local CPU MPI ranks',
          'ranks': args.ranks, 'executable_sha256': digest(exe),
          'fixture_sha256': digest(fixture), 'expected_dt': dt, 'cases': {}}
if args.case and (output/'results.json').exists():
    previous = json.loads((output/'results.json').read_text())
    for key in ('ranks', 'executable_sha256', 'fixture_sha256', 'expected_dt'):
        assert previous[key] == report[key], f'cannot reuse changed {key}'
    report['cases'] = previous['cases']
cycle_pattern = re.compile(r'cycle=(\d+) time=(\S+) dt=(\S+)')


def run(name, options, count=None, error=None):
    if args.case and name not in args.case:
        return
    directory = output/name
    directory.mkdir(exist_ok=True)
    settings = {}
    for option in ['time/evolution=dynamic', 'time/tlim=1', 'time/ndiag=1', *options]:
        key, value = option.split('=', 1)
        section, parameter = key.split('/', 1)
        settings.setdefault(section, {})[parameter] = value
    lines = []
    section = None
    for line in fixture.read_text().splitlines():
        if line.startswith('<'):
            section = line.strip('<>')
            lines.append(line)
            lines.extend(f'{key} = {value}' for key, value in settings.get(section, {}).items())
        elif '=' not in line or line.split('=', 1)[0].strip() not in settings.get(section, {}):
            lines.append(line)
    input_file = directory/'case.athinput'
    input_file.write_text('\n'.join(lines)+'\n')
    command = ['mpiexec', '-n', str(args.ranks), str(exe), '-i', str(input_file)]
    with (directory/'run.log').open('w') as log:
        result = subprocess.run(command, cwd=directory, stdout=log,
                                stderr=subprocess.STDOUT, timeout=1200)
    text = (directory/'run.log').read_text()
    case = {'command': command, 'returncode': result.returncode}
    report['cases'][name] = case
    if error:
        assert result.returncode != 0 and error in text, (name, text[-3000:])
        case['expected_rejection'] = error
    else:
        assert result.returncode == 0, (name, text[-3000:])
        cycles = [(int(c), float(t), float(d)) for c, t, d in cycle_pattern.findall(text)]
        assert [c for c, _, _ in cycles] == list(range(count+1)), cycles
        expected_time = 0.
        for c, time, step in cycles:
            assert step == dt and time == expected_time, (name, c, time, expected_time, step)
            expected_time += dt
        case.update({'cycles': count, 'final_time': cycles[-1][1],
                     'all_dt_and_times_exact': True})
    (output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(name, 'PASS', flush=True)


# Omitted integrator selects the module's validated classical RK4 default.
run('one-step', ['time/nlim=1'], count=1)
run('eight-steps', ['time/integrator=rk4_classical', 'time/nlim=8'], count=8)
run('different-cfl', ['time/nlim=1', 'time/cfl_number=.1'], count=1)
run('cross-tlim', ['time/nlim=8', f'time/tlim={1.5*dt:.17g}'], count=2)
run('reject-integrator', ['time/nlim=1', 'time/integrator=rk4'],
    error='Hyperboloidal evolution requires rk4_classical')
run('reject-kinematic', ['time/nlim=1', 'time/evolution=kinematic'],
    error='Hyperboloidal evolution requires static or dynamic mode')
run('reject-adaptive', ['time/nlim=1', 'mesh_refinement/refinement=adaptive',
                        'mesh_refinement/num_levels=2',
                        'mesh_refinement/max_nmb_per_rank=106'],
    error='Hyperboloidal AMR state transfer is not yet integrated')
run('reject-relaxation', ['time/nlim=1', 'time/dt=.01'],
    error='Hyperboloidal fixed timestep exceeds the relaxation limit')
assert digest(exe) == report['executable_sha256'], 'executable changed during tests'
report['status'] = 'PASS' if len(report['cases']) == 8 else 'PARTIAL'
(output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
