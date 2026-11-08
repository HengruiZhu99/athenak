#!/usr/bin/env python3
"""Link/run independent native surface transfer gates with frozen provenance."""
import argparse
import hashlib
import json
import pathlib
import re
import shlex
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--build', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
parser.add_argument('--ranks', type=int, default=2)
parser.add_argument('--build-only', action='store_true')
mode=parser.add_mutually_exclusive_group()
mode.add_argument('--diagnostic', action='store_true')
mode.add_argument('--reviewed-admission', action='store_true')
parser.add_argument('--reuse-executable', type=pathlib.Path)
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parents[2]
build, output = args.build.resolve(), args.output.resolve()
output.mkdir(parents=True, exist_ok=True)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

sources = [root/name for name in [
    'src/z4c_hyperboloidal/surface_remesh.cpp', 'src/z4c_hyperboloidal/surface_remesh.hpp',
    'src/z4c_hyperboloidal/remesh.hpp', 'src/z4c_hyperboloidal/cartesian_algebraic_chart.hpp',
    'src/bvals/mapped_exchange.cpp', 'src/bvals/mapped_exchange.hpp',
    'tst/z4c_hyperboloidal/surface_remesh.cpp', 'tst/z4c_hyperboloidal/surface_remesh.py',
    'tst/z4c_hyperboloidal/n12_initialize.athinput', 'tst/z4c_hyperboloidal/surface_smr.athinput',
    'tst/z4c_hyperboloidal/surface_remesh_migration.athinput',
    'tst/z4c_hyperboloidal/surface_remesh_contract.md',
    'tst/z4c_hyperboloidal/surface_remesh_prediction.py']]
if args.reuse_executable:
    exe = args.reuse_executable.resolve()
    obj = exe.parent/'surface_remesh.o'
    source_manifest = json.loads((exe.parent/'source-manifest.json').read_text())
    object_manifest = json.loads((exe.parent/'object-manifest.json').read_text())
    assert all(sha256(root/path) == digest for path,digest in source_manifest.items())
    assert all(sha256(build/path) == digest for path,digest in object_manifest.items())
else:
    source_manifest = {str(path.relative_to(root)):sha256(path) for path in sources}
    flags = (build/'src/CMakeFiles/athena.dir/flags.make').read_text()
    includes = shlex.split(re.search(r'^CXX_INCLUDES = (.*)$', flags, re.M).group(1))
    obj, exe = output/'surface_remesh.o', output/'surface_remesh'
    compile_command = ['/usr/bin/c++', '-std=c++17', '-O2', '-ffp-contract=off', *includes,
        '-c', str(root/'tst/z4c_hyperboloidal/surface_remesh.cpp'), '-o', str(obj)]
    link = shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
    link = [str(obj) if x == 'CMakeFiles/athena.dir/main.cpp.o' else x for x in link]
    assert str(obj) in link
    link[link.index('-o')+1] = str(exe)
    objects = [build/'src'/part for part in link if part.endswith('.o') and part != str(obj)]
    object_manifest = {str(path.relative_to(build)):sha256(path) for path in objects}
    makefile = output/'surface_remesh.mk'
    makefile.write_text('all:\n\t'+shlex.join(compile_command)+'\n\t'+shlex.join(link)+'\n')
    with (output/'build.log').open('w') as log:
        subprocess.run(['make', '-j64', '-f', str(makefile)], cwd=build/'src', stdout=log,
                       stderr=subprocess.STDOUT, check=True)
(output/'source-manifest.json').write_text(json.dumps(source_manifest, indent=2)+'\n')
(output/'object-manifest.json').write_text(json.dumps(object_manifest, indent=2)+'\n')
if args.build_only:
    raise SystemExit(0)
command = ['mpiexec', '-n', str(args.ranks), str(exe),
           str(root/'tst/z4c_hyperboloidal/n12_initialize.athinput'),
           str(root/'tst/z4c_hyperboloidal/surface_smr.athinput'),
           str(root/'tst/z4c_hyperboloidal/surface_remesh_migration.athinput')]
if args.diagnostic:
    command.append('--diagnostic')
if args.reviewed_admission:
    command.append('--reviewed-admission')
with (output/'run.log').open('w') as log:
    result = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT,
                            timeout=300)
text = (output/'run.log').read_text()
status = {'exit_code': result.returncode, 'local_mpi_ranks': args.ranks,
          'native_build': str(build), 'command': command, 'diagnostic_mode': args.diagnostic, 'reviewed_admission_mode': args.reviewed_admission,
          'finite_remainder_gate': 'pointwise Bu+Lu*Bv plus fixed roundoff allowance at N12/24/48/96/192',
          'polynomial_gate': '2048*epsilon*max(1,row_L1)',
          'smooth_gate': 'order >= 5.5 at both N12->24 and N24->48',
          'history_gate': 'all8 members bitwise; unique old packets; exact global ledger',
          'executable_sha256': sha256(exe), 'harness_object_sha256': sha256(obj),
          'source_manifest_sha256': sha256(output/'source-manifest.json'),
          'object_manifest_sha256': sha256(output/'object-manifest.json'),
          'sources_unchanged': all(sha256(root/path) == digest
                                    for path,digest in source_manifest.items()),
          'objects_unchanged': all(sha256(build/path) == digest
                                    for path,digest in object_manifest.items()),
          'cases': {}}
for label in ['refine','coarsen','migration']:
    match = re.search(label+r' unchanged_values=(\d+) changed_values=(\d+) '
                      r'migrated_unchanged_points=(\d+) failures=(\d+) '
                      r'remote_donors=(\d+) max_scaled_error=([^ ]+) '
                      r'zero_face_ranks=(\d+),(\d+)', text)
    if match:
        status['cases'][label] = dict(unchanged_values=int(match[1]),
            changed_values=int(match[2]), migrated_unchanged_points=int(match[3]),
            failures=int(match[4]), remote_donors=int(match[5]),
            max_scaled_error=float(match[6]),
            zero_old_face_ranks=int(match[7]), zero_new_face_ranks=int(match[8]))
    match = re.search(label+r' polynomial_checks=(\d+) max_roundoff_ratio=([^ ]+) '
                      r'smooth_errors=([^ ]+) orders=([^ ]+) remainder_bounds=([^ ]+) '
                      r'remainder_ratios=([^\n]+)', text)
    if match:
        status['cases'].setdefault(label, {}).update(polynomial_checks=int(match[1]),
            max_roundoff_ratio=float(match[2]), smooth_errors=list(map(float,match[3].split(','))),
            convergence_orders=list(map(float,match[4].split(','))),
            remainder_bounds=list(map(float,match[5].split(','))),
            max_pointwise_remainder_ratios=list(map(float,match[6].split(','))))
status['required_cases_present'] = all('failures' in status['cases'].get(case,{})
                                      for case in ['refine','coarsen','migration'])
status['required_convergence_present'] = all('convergence_orders' in status['cases'].get(case,{})
                                            for case in ['refine','coarsen'])
transport_gates = status['required_cases_present']
analytic_gates = status['required_convergence_present']
original_rates = analytic_gates
if transport_gates:
    transport_gates = all(case['failures'] == 0 and case['max_scaled_error'] <= 1
                          for case in status['cases'].values())
    transport_gates &= status['cases']['migration']['migrated_unchanged_points'] > 0
    transport_gates &= status['cases']['migration']['remote_donors'] > 0
    if args.ranks == 8:
        transport_gates &= all(case['zero_old_face_ranks'] > 0 and case['zero_new_face_ranks'] > 0
                               for case in status['cases'].values())
if analytic_gates:
    for name in ['refine','coarsen']:
        case=status['cases'][name]
        analytic_gates &= (case['polynomial_checks'] > 0 and case['max_roundoff_ratio'] <= 1
            and len(case['smooth_errors']) == 5 and len(case['convergence_orders']) == 4
            and len(case['max_pointwise_remainder_ratios']) == 5
            and max(case['max_pointwise_remainder_ratios']) <= 1)
        original_rates &= min(case['convergence_orders'][:2]) >= 5.5
status['transport_gates_passed']=transport_gates
status['polynomial_and_remainder_gates_passed']=analytic_gates
status['original_convergence_gate_passed']=original_rates
status['diagnostic_completed']=(args.diagnostic and transport_gates and analytic_gates and
    not original_rates and 'DIAGNOSTIC transport passed; original smooth-rate gate failed' in text)
status['original_gate_passed']=original_rates
status['asymptotic_rate_gate_passed']=(analytic_gates and all(
    min(status['cases'][name]['convergence_orders'][2:]) >= 5.5 for name in ['refine','coarsen']))
status['revised_admission_passed']=(args.reviewed_admission and result.returncode == 0 and
    transport_gates and analytic_gates and status['asymptotic_rate_gate_passed'] and
    status['sources_unchanged'] and status['objects_unchanged'] and
    'PASS reviewed native sparse surface AMR transfer; original_gate_passed=' in text)
(output/'results.json').write_text(json.dumps(status, indent=2)+'\n')
print(text)
result.check_returncode()
if args.reviewed_admission:
    assert status['revised_admission_passed'], 'reviewed admission or provenance failed'
else:
    assert (transport_gates and analytic_gates and original_rates and status['sources_unchanged']
            and status['objects_unchanged']), 'original surface transfer gates or provenance failed'
