#!/usr/bin/env python3
"""Real FastFlow external-sampler gates plus preserved Cartesian executable oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--build', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--ranks', type=int, default=2)
parser.add_argument('--executable', type=Path, help='reuse an already linked test executable')
parser.add_argument('--skip-cartesian', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
build, output = args.build.resolve(), args.output.resolve()
output.mkdir(parents=True, exist_ok=True)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(command, directory, logfile, timeout=180):
    with logfile.open('w') as log:
        result = subprocess.run(command, cwd=directory, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        print(logfile.read_text())
    result.check_returncode()
    return {'command': command, 'log_sha256': digest(logfile)}

sources = [root/'src/z4c'/name for name in
           ('fastflow.hpp', 'fastflow.cpp', 'horizon_geometry.hpp', 'fastflow_surface.hpp')]
sources += [root/'src/geodesic-grid'/name for name in
            ('gauss_legendre.hpp', 'gauss_legendre.cpp')]
sources += [root/'tst/z4c_hyperboloidal/fastflow_external.cpp', Path(__file__).resolve(),
            root/'tst/inputs/z4c_boosted.athinput']
source_manifest = {str(path.relative_to(root)): digest(path) for path in sources}
(output/'source-manifest.json').write_text(json.dumps(source_manifest, indent=2)+'\n')
link = shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
objects = [build/'src'/arg for arg in link if arg.endswith('.o')]
object_manifest = {str(path.relative_to(build)): digest(path) for path in objects}
(output/'object-manifest.json').write_text(json.dumps(object_manifest, indent=2)+'\n')
if args.executable:
    executable = args.executable.resolve()
else:
    flags = (build/'src/CMakeFiles/athena.dir/flags.make').read_text()
    includes = shlex.split(re.search(r'^CXX_INCLUDES = (.*)$', flags, re.M).group(1))
    obj, executable = output/'fastflow_external.o', output/'fastflow_external'
    compile_command = ['/usr/bin/c++', '-std=c++17', '-O2', '-ffp-contract=off', *includes,
                       '-c', str(sources[-3]), '-o', str(obj)]
    link = [str(obj) if arg == 'CMakeFiles/athena.dir/main.cpp.o' else arg for arg in link]
    assert str(obj) in link
    link[link.index('-o')+1] = str(executable)
    makefile = output/'build.mk'
    makefile.write_text('all:\n\t'+shlex.join(compile_command)+'\n\t'+shlex.join(link)+'\n')
    run(['make', '-j64', '-f', str(makefile)], build/'src', output/'build.log')
report = {'ranks': args.ranks, 'executable_sha256': digest(executable),
          'source_manifest_sha256': digest(output/'source-manifest.json'),
          'object_manifest_sha256': digest(output/'object-manifest.json')}
report['external'] = run(['mpiexec', '-n', str(args.ranks), str(executable)], output,
                         output/'external.log')
required = {'schwarzschild_scalar_batch', 'skipped_search', 'schwarzschild_nontrivial_flow',
            'missing', 'duplicate', 'invalid_geometry', 'bad_flag', 'batch_exception',
            'scalar_exception', 'invalid_radius', 'failed_after_success', 'nonconverged',
            'collective_output_failure', 'full_harmonic_mask', 'fastflow_external_all'}
found = re.findall(r'^PASS (\w+)$', (output/'external.log').read_text(), re.M)
assert len(found) == len(set(found)) and set(found) == required, found
report['external']['required_gates'] = sorted(found)
zero_owner_match = re.search(r'^ZERO_OWNER_RANKS (\d+)$', (output/'external.log').read_text(), re.M)
assert zero_owner_match and int(zero_owner_match.group(1)) == args.ranks-2
report['external']['zero_owner_ranks'] = int(zero_owner_match.group(1))
if not args.skip_cartesian:
    baseline = build/'fastflow-baseline'
    expected = json.loads((baseline/'manifest.json').read_text())
    assert all(digest(baseline/name) == sha for name, sha in expected.items())
    report['cartesian_baseline_sha256'] = digest(baseline/'athena')
    report['cartesian_current_sha256'] = digest(build/'src/athena')
    arguments = ['-i', str(root/'tst/inputs/z4c_boosted.athinput'), 'time/nlim=2',
                 'mesh_refinement/refinement=none', 'problem/punc_velocity_x1=0',
                 'job/basename=fastflow', 'fastflow/use_puncture_0=-1']
    for axis in range(1, 4):
        arguments += [f'mesh/nx{axis}=32', f'meshblock/nx{axis}=16',
                      f'mesh/x{axis}min=-2', f'mesh/x{axis}max=2']
    runs = {}
    for label, binary in [('before', baseline/'athena'), ('after', build/'src/athena')]:
        directory = output/('cartesian-'+label)
        directory.mkdir(exist_ok=True)
        runs[label] = run(['mpiexec', '-n', str(args.ranks), str(binary), *arguments],
                          directory, directory/'run.log')
    hashes = {}
    for filename in ['fastflow.horizon_summary_0.txt', 'fastflow.horizon_shape_0.txt']:
        before = output/'cartesian-before'/filename
        after = output/'cartesian-after'/filename
        assert before.read_bytes() == after.read_bytes(), filename
        hashes[filename] = digest(after)
    summary = (output/'cartesian-after/fastflow.horizon_summary_0.txt').read_text()
    assert not re.search(r'\b(?:nan|inf)\b', summary, re.I), summary
    assert any(line and not line.startswith('#') for line in summary.splitlines())
    report['cartesian'] = {'runs': runs, 'exact_file_sha256': hashes}
report['sources_unchanged'] = all(digest(root/path) == sha for path, sha in source_manifest.items())
report['objects_unchanged'] = all(digest(build/path) == sha for path, sha in object_manifest.items())
assert report['sources_unchanged'] and report['objects_unchanged']
report['passed'] = True
(output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
print((output/'external.log').read_text())
print('PASS FastFlow gates and Cartesian regression' if not args.skip_cartesian
      else 'PASS FastFlow external gates')
