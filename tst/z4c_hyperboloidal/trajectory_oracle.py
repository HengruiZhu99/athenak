#!/usr/bin/env python3
"""Compare the real native Driver with independent N12 classical RK4 at 1/8 steps.

The private legacy overlay adds candidate-floor/ledger and diagnostic hooks only.
Its RHS and RK arithmetic remain unchanged; the short test requires zero floors.
"""
import argparse
import hashlib
import json
import pathlib
import re
import shlex
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--build', type=pathlib.Path, required=True)
parser.add_argument('--legacy', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
parser.add_argument('--run-output', type=pathlib.Path,
                    help='separate run artifacts while reusing the exact build executable')
parser.add_argument('--ranks', type=int, default=2)
parser.add_argument('--build-only', action='store_true')
parser.add_argument('--run-existing', action='store_true')
args = parser.parse_args()
assert not (args.build_only and args.run_existing)
root = pathlib.Path(__file__).resolve().parents[2]
build, legacy, output = args.build.resolve(), args.legacy.resolve(), args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
tracked = sorted((root/'src/z4c_hyperboloidal').glob('*.[ch]pp'))
tracked += [root/p for p in ['src/mesh/chart_atlas.cpp', 'src/mesh/chart_atlas.hpp',
    'src/mesh/mesh.cpp', 'src/driver/driver.cpp', 'src/driver/driver.hpp',
    'src/bvals/mapped_exchange.cpp', 'src/bvals/mapped_exchange.hpp',
    'tst/z4c_hyperboloidal/trajectory_oracle.cpp']]
source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}

relative = pathlib.Path('z4c/hyperboloidal/spherical_transfer_3d.hpp')
original = (legacy/'src'/relative).read_text()
replacements = [
    ('#include <chrono>', '#include <chrono>\n#include <functional>'),
    ('  bool experimental_timestep=false;',
     '  bool experimental_timestep=false;\n'
     '  std::function<void(V&,V&,Real,bool)> trajectory_candidate_observer;\n'
     '  std::function<void(Real,Real)> trajectory_rhs_observer;'),
    ('        failed_state=tmp;failed_boundary=scri.values;};',
     '        if(trajectory_candidate_observer) trajectory_candidate_observer(tmp,scri.values,time+h,false);\n'
     '        failed_state=tmp;failed_boundary=scri.values;};'),
    ('      failed_stage=5;failed_state=state;failed_boundary=scri.values;Materialize(state);MaterializeBoundary(state);',
     '      if(trajectory_candidate_observer) trajectory_candidate_observer(state,scri.values,time+dt,true);\n'
     '      failed_stage=5;failed_state=state;failed_boundary=scri.values;Materialize(state);MaterializeBoundary(state);'),
    ('    for(double v:out)if(!std::isfinite(v))throw std::runtime_error("nonfinite RHS");\n    return out;',
     '    for(double v:out)if(!std::isfinite(v))throw std::runtime_error("nonfinite RHS");\n'
     '    if(trajectory_rhs_observer) trajectory_rhs_observer(normal_defect,clock);\n    return out;'),
]
modified = original
for before, after in replacements:
    assert modified.count(before) == 1, 'legacy instrumentation location changed'
    modified = modified.replace(before, after)
recovered = modified
for before, after in reversed(replacements):
    recovered = recovered.replace(after, before)
assert recovered == original, 'legacy arithmetic modified beyond declared hooks'
overlay = output/'legacy-test-overlay'/relative
legacy_provenance = {
    'source': str(legacy/'src'/relative),
    'source_sha256': hashlib.sha256(original.encode()).hexdigest(),
    'overlay_sha256': hashlib.sha256(modified.encode()).hexdigest(),
    # JSON stores sequences as lists; normalize the in-memory pairs as well so
    # a verified existing executable can pass the identical-provenance check.
    'replacements': [list(pair) for pair in replacements],
    'floor_helper': str(legacy/'tst/hyperbolic_shift_3d/endpoint_policy/chi_policy.hpp'),
    'floor_helper_sha256': hashlib.sha256((legacy/'tst/hyperbolic_shift_3d/endpoint_policy/chi_policy.hpp').read_bytes()).hexdigest(),
    'gpu_step_source_sha256': hashlib.sha256((legacy/'tst/hyperbolic_shift_3d/coupled_mpi.cpp').read_bytes()).hexdigest(),
    # Include all legacy source headers, covering transitive quoted includes as
    # well as the two direct reference headers. These files are read only.
    'legacy_header_sha256': {str(p.relative_to(legacy)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(list((legacy/'src').rglob('*.hpp'))+list((legacy/'src').rglob('*.h')))},
    'floor_events_required': 0,
}
if args.run_existing:
    stored = json.loads((output/'build-provenance.json').read_text())
    assert stored.get('legacy_provenance') == legacy_provenance, \
        'legacy reference provenance changed or absent; rebuild before running'
overlay.parent.mkdir(parents=True, exist_ok=True)
overlay.write_text(modified)
(output/'overlay-provenance.json').write_text(json.dumps(legacy_provenance, indent=2)+'\n')

input_text = (root/'tst/z4c_hyperboloidal/n12_initialize.athinput').read_text()
assert input_text.count('evolution = static') == 1
input_text = input_text.replace('evolution = static',
    'evolution = dynamic\nintegrator = rk4_classical\ntlim = 1\nnlim = 8\nndiag = 1')
input_path = output/'n12_trajectory.athinput'
input_path.write_text(input_text)
flags = (build/'src/CMakeFiles/athena.dir/flags.make').read_text()
includes = shlex.split(re.search(r'^CXX_INCLUDES = (.*)$', flags, re.M).group(1))
options = ['BROAD_SMOOTH_A', 'CONFIGURED_FORMULATION', 'DIRECT_OUTER', 'INNER_TELEGRAPHER',
    'NORMALIZED_INNER_SHIFT', 'RESTORE_INNER_DAMPING', 'SHORT_INNER_GAUGE',
    'SMOOTH_GAUGE_BLEND', 'SPHERICAL_POLICY']
obj, exe = output/'trajectory_oracle.o', output/'trajectory_oracle'
compile = ['/usr/bin/c++', '-std=c++17', '-O2', '-ffp-contract=off']+includes
compile += ['-DHYPERBOLIC_'+x+'=1' for x in options]
compile += ['-I'+str(output/'legacy-test-overlay'), '-I'+str(legacy/'src'), '-I'+str(legacy),
    '-c', str(root/'tst/z4c_hyperboloidal/trajectory_oracle.cpp'), '-o', str(obj)]
link = shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
link = [str(obj) if x == 'CMakeFiles/athena.dir/main.cpp.o' else x for x in link]
assert str(obj) in link
link[link.index('-o')+1] = str(exe)
makefile = output/'trajectory_oracle.mk'
makefile.write_text('all:\n\t'+shlex.join(compile)+'\n\t'+shlex.join(link)+'\n')
if not args.run_existing:
    with (output/'build.log').open('w') as log:
        subprocess.run(['make', '-j64', '-f', str(makefile)], cwd=build/'src', stdout=log,
                       stderr=subprocess.STDOUT, check=True)
build_provenance = {'source_sha256': source_hashes,
    'legacy_provenance': legacy_provenance,
    'executable_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
    'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
    'native_executable_sha256': hashlib.sha256((build/'src/athena').read_bytes()).hexdigest(),
    'native_compile_flags': flags, 'oracle_compile_command': compile}
if args.run_existing:
    assert json.loads((output/'build-provenance.json').read_text()) == build_provenance, \
        'existing trajectory executable/source provenance changed; rebuild before running'
else:
    (output/'build-provenance.json').write_text(json.dumps(build_provenance, indent=2)+'\n')
if args.build_only:
    print('Built trajectory oracle; run separately with --run-existing')
    sys.exit(0)
command = ['mpiexec', '-n', str(args.ranks), str(exe), str(input_path)]
run_output = args.run_output.resolve() if args.run_output else output
run_output.mkdir(parents=True, exist_ok=True)
if run_output != output:
    for name in ['build-provenance.json', 'overlay-provenance.json']:
        (run_output/name).write_bytes((output/name).read_bytes())
with (run_output/'run.log').open('w') as log:
    result = subprocess.run(command, cwd=run_output, stdout=log, stderr=subprocess.STDOUT)
text = (run_output/'run.log').read_text()
metrics, ledgers, history = {}, {}, []
for m in re.finditer(r'^(\w+) comparisons=(\d+) max_scaled=([^ ]+) max_absolute=([^ ]+) max_tolerance_ratio=([^ ]+) failures=(\d+) weighted_rms=([^ ]+) weight_sum=([^\n]+)$', text, re.M):
    metrics[m.group(1)] = {'comparisons': int(m.group(2)), 'max_scaled_error': float(m.group(3)),
        'max_absolute_error': float(m.group(4)), 'max_tolerance_ratio': float(m.group(5)),
        'failed_values': int(m.group(6)), 'weighted_rms': float(m.group(7)),
        'weight_sum': float(m.group(8)), 'per_field': [], 'worst_location': {}}
for m in re.finditer(r'^(\w+) field=(\d+) max_absolute=([^ ]+) max_scaled=([^ ]+) weighted_rms=([^ ]+) failures=(\d+)$', text, re.M):
    metrics[m.group(1)]['per_field'].append({'field': int(m.group(2)),
        'max_absolute_error': float(m.group(3)), 'max_scaled_error': float(m.group(4)),
        'weighted_rms': float(m.group(5)), 'failed_values': int(m.group(6))})
for m in re.finditer(r'^(\w+) worst=(absolute|scaled) field=(\d+) gid=(\d+) ijk=([^ ]+) xyz=([^ ]+) step=(\d+) rank=(\d+) actual=([^ ]+) expected=([^\n]+)$', text, re.M):
    metrics[m.group(1)]['worst_location'][m.group(2)] = {'field': int(m.group(3)),
        'native_gid': int(m.group(4)), 'cell_indices': [int(x) for x in m.group(5).split(',')],
        'coordinates': [float(x) for x in m.group(6).split(',')], 'step': int(m.group(7)),
        'rank': int(m.group(8)), 'actual': float(m.group(9)), 'expected': float(m.group(10))}
for m in re.finditer(r'^step(\d+) chi_records=(\d+) failures=(\d+) max_absolute=([^ ]+) native_floor_events=([^ ]+) reference_floor_events=([^ ]+) time=([^\n]+)$', text, re.M):
    ledgers[m.group(1)] = {'comparisons': int(m.group(2)), 'failed_values': int(m.group(3)),
        'max_absolute_error': float(m.group(4)), 'native_floor_events': float(m.group(5)),
        'reference_floor_events': float(m.group(6)), 'accepted_time': float(m.group(7))}
for m in re.finditer(r'^step(\d+) native_candidate_min=([^ ]+) reference_candidate_min=([^ ]+) native_accepted_min=([^ ]+) reference_accepted_min=([^ ]+) native_state_min=([^ ]+) reference_state_min=([^\n]+)$', text, re.M):
    for key, group in zip(['native_candidate_min', 'reference_candidate_min', 'native_accepted_min',
                          'reference_accepted_min', 'native_state_min', 'reference_state_min'], range(2, 8)):
        value = float(m.group(group))
        ledgers[m.group(1)][key] = value if value < 1e300 else None
for line in text.splitlines():
    if line.startswith('stage_history '):
        row = dict(re.findall(r'(\w+)=([^ ]+)', line))
        history.append({k: int(v) if k in ['step', 'stage'] else float(v) for k, v in row.items()})
unchanged = source_hashes == {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
status = {'passed': result.returncode == 0, 'local_mpi_ranks': args.ranks,
    'scope': 'Original native Driver eight-step fixed-N12 trajectory; snapshots at0,1,8; no Aurora/AMR/restart claim',
    'gate': 'abs(delta)<=1e-12+5e-10*abs(reference); chi<=1e-14+5e-10*abs(reference)',
    'metrics': metrics, 'chi_records': ledgers, 'stage_history': history,
    'source_sha256': source_hashes, 'source_unchanged_during_run': unchanged,
    'runner_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    'executable_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
    'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
    'native_compile_flags': flags, 'oracle_compile_command': compile, 'command': command,
    'snapshot_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(run_output.glob('*.bin'))},
    'rms_weights': 'legacy evolved-point physical volumes including overlap/endpoint halfweight; scri physical area; per-field and all31 aggregate'}
(run_output/'results.json').write_text(json.dumps(status, indent=2)+'\n')
print('\n'.join(line for line in text.splitlines() if ' comparisons=' in line or
    ' chi_records=' in line or line.startswith('PASS ') or 'mismatch' in line))
result.check_returncode()
assert unchanged, 'native/test sources changed during trajectory run'
assert set(metrics) == {f'step{s}_{kind}' for s in [0, 1, 8] for kind in ['bulk', 'surface']}
assert all(len(x['per_field']) == 31 and len(x['worst_location']) == 2 for x in metrics.values())
assert len(history) == 32 and set(ledgers) == {'0', '1', '8'}
