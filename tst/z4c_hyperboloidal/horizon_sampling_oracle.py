#!/usr/bin/env python3
"""Build/run the independent native horizon sampler comparison on frozen sources."""
import argparse
import hashlib
import json
import pathlib
import re
import shlex
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--build', type=pathlib.Path, required=True)
parser.add_argument('--legacy', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
parser.add_argument('--fields', type=pathlib.Path)
parser.add_argument('--ranks', type=int, choices=[2, 8], default=2)
parser.add_argument('--build-only', action='store_true')
parser.add_argument('--reuse-executable', type=pathlib.Path)
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parents[2]
build, legacy, output = args.build.resolve(), args.legacy.resolve(), args.output.resolve()
fields = (args.fields or root/'build-native/n12-reference-step3551/fields.bin').resolve()
output.mkdir(parents=True, exist_ok=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def snapshot(paths):
    return {str(path.resolve()): sha(path) for path in sorted(set(paths))}

def unchanged(manifest):
    return all(pathlib.Path(path).exists() and sha(pathlib.Path(path)) == digest
               for path, digest in manifest.items())

assert sha(fields) == '7d37c940eea84a04dc471bb90178c3e2facf153f14382cd2a47cf1fba6a320be', \
    'saved input is not the pinned step3551 full fields'
helper = legacy/'tst/hyperbolic_shift_3d/endpoint_policy/fastflow_history.hpp'
source = helper.read_text()
start = source.index('class EndpointHorizonSampler {')
end = source.index('// The resident research build', start)
extracted = source[start:end]
assert extracted.rstrip().endswith('};')

if args.reuse_executable:
    exe = args.reuse_executable.resolve()
    provenance = json.loads((exe.parent/'build-provenance.json').read_text())
    assert str(exe) == provenance['executable'] and sha(exe) == provenance['executable_sha256']
    for key in ['source_sha256', 'dependency_sha256', 'object_sha256']:
        assert unchanged(provenance[key]), f'changed {key} from frozen build'
    assert sha(helper) == provenance['legacy_sampler_source_sha256']
    assert hashlib.sha256(extracted.encode()).hexdigest() == provenance['extracted_sampler_sha256']
else:
    generated = output/'legacy_horizon_sampler.hpp'
    generated.write_text(extracted)
    tracked = [path for path in (root/'src').rglob('*')
               if path.suffix in ['.cpp', '.hpp']]
    tracked += [root/'tst/z4c_hyperboloidal'/name for name in [
        'horizon_sampling_oracle.cpp', 'horizon_sampling_oracle.py',
        'horizon_sampling_contract.md', 'n12_initialize.athinput']]
    tracked += [helper, generated]
    source_manifest = snapshot(tracked)
    flags = (build/'src/CMakeFiles/athena.dir/flags.make').read_text()
    includes = shlex.split(re.search(r'^CXX_INCLUDES = (.*)$', flags, re.M).group(1))
    options = ['BROAD_SMOOTH_A', 'CONFIGURED_FORMULATION', 'DIRECT_OUTER',
               'INNER_TELEGRAPHER', 'NORMALIZED_INNER_SHIFT', 'RESTORE_INNER_DAMPING',
               'SHORT_INNER_GAUGE', 'SMOOTH_GAUGE_BLEND', 'SPHERICAL_POLICY']
    obj, exe, dep = output/'horizon_sampling_oracle.o', output/'horizon_sampling_oracle', output/'oracle.d'
    compile_command = ['/usr/bin/c++', '-std=c++17', '-O2', '-ffp-contract=off',
        *includes, *['-DHYPERBOLIC_'+x+'=1' for x in options],
        '-I'+str(output), '-I'+str(legacy/'src'), '-I'+str(legacy), '-MMD', '-MF', str(dep),
        '-c', str(root/'tst/z4c_hyperboloidal/horizon_sampling_oracle.cpp'), '-o', str(obj)]
    link = shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
    link = [str(obj) if x == 'CMakeFiles/athena.dir/main.cpp.o' else x for x in link]
    assert str(obj) in link
    assert any(x.endswith('/horizon_sampling.cpp.o') for x in link), 'native sampler object absent'
    link[link.index('-o')+1] = str(exe)
    objects = [(build/'src'/part).resolve() for part in link
               if part.endswith('.o') and part != str(obj)]
    object_manifest = snapshot(objects)
    makefile = output/'oracle.mk'
    makefile.write_text('all:\n\t'+shlex.join(compile_command)+'\n\t'+shlex.join(link)+'\n')
    with (output/'build.log').open('w') as log:
        subprocess.run(['make', '-j64', '-f', str(makefile)], cwd=build/'src',
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    dependencies = shlex.split(dep.read_text().replace('\\\n', ' ').split(':', 1)[1])
    dependency_manifest = snapshot([(build/'src'/path).resolve() for path in dependencies])
    assert unchanged(source_manifest) and unchanged(object_manifest), 'sources changed during build'
    provenance = {'executable': str(exe), 'executable_sha256': sha(exe),
        'source_sha256': source_manifest, 'dependency_sha256': dependency_manifest,
        'object_sha256': object_manifest, 'compile_command': compile_command,
        'native_compile_flags': flags, 'legacy_sampler_source': str(helper),
        'legacy_sampler_source_sha256': sha(helper),
        'extracted_sampler_sha256': hashlib.sha256(extracted.encode()).hexdigest(),
        'reference_fields': str(fields), 'reference_fields_sha256': sha(fields)}
    (output/'build-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
if args.build_only:
    print('Built', exe)
    raise SystemExit(0)

command = ['mpiexec', '-n', str(args.ranks), str(exe),
           str(root/'tst/z4c_hyperboloidal/n12_initialize.athinput'), str(fields)]
with (output/'run.log').open('w') as log:
    run = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, timeout=900)
text = (output/'run.log').read_text()
scenarios = {}
for m in re.finditer(r'^scenario=(\d+) comparisons=(\d+) failures=(\d+) max_tolerance_ratio=(\S+) '
                     r'worst_point=(-?\d+) worst_component=(-?\d+) worst_xyz=(\S+)$', text, re.M):
    scenarios[m[1]] = {'comparisons': int(m[2]), 'failures': int(m[3]),
        'max_tolerance_ratio': float(m[4]), 'worst_point': int(m[5]),
        'worst_component': int(m[6]), 'worst_coordinates': [float(x) for x in m[7].split(',')],
        'components': []}
for m in re.finditer(r'^scenario=(\d+) component=(\d+) count=(\d+) failures=(\d+) '
                    r'max_absolute=(\S+) max_scaled=(\S+) max_tolerance_ratio=(\S+) rms=(\S+)$', text, re.M):
    scenarios[m[1]]['components'].append({'component': int(m[2]), 'count': int(m[3]),
        'failures': int(m[4]), 'max_absolute': float(m[5]), 'max_scaled': float(m[6]),
        'max_tolerance_ratio': float(m[7]), 'rms': float(m[8])})
summary = re.search(r'^horizon_sampling queries=(\d+) scenarios=(\d+) failures=(\d+) '
                    r'ghost_nodes=(\d+) remote_donors=(\d+) unchanged_values=(\d+) '
                    r'cache_checks=(\d+) zero_owner_batches=(\d+) tensor_corners=(\d+)$', text, re.M)
invalidation = re.search(r'^remote_invalidation checks=(\d+) changed_query=(\d+) '
                         r'donor_rank=(\d+) reader_rank=(\d+) query=(\d+)$', text, re.M)
stable = all(unchanged(provenance[key]) for key in ['source_sha256', 'dependency_sha256', 'object_sha256'])
result = {'passed': run.returncode == 0 and stable, 'exit_code': run.returncode,
    'source_dependencies_objects_unchanged': stable, 'local_mpi_ranks': args.ranks,
    'executable': str(exe), 'executable_sha256': sha(exe), 'command': command,
    'gate': 'abs(delta)<=1e-12+5e-10*abs(reference)',
    'rms_weights': 'equal positive weight per prescribed query, per component; no volume integral',
    'scenario_names': ['fresh reference', 'nonspherical perturbation', 'saved step3551'],
    'scenarios': scenarios, 'build_provenance': str(exe.parent/'build-provenance.json'),
    'complete_query_outputs_sha256': {p.name: sha(p) for p in sorted(output.glob('samples-*.bin'))},
    'reference_fields_sha256': sha(fields)}
if summary:
    result.update(dict(zip(['queries', 'scenario_count', 'failures', 'ghost_nodes',
                           'remote_donors', 'unchanged_values', 'cache_checks',
                           'zero_owner_batches', 'tensor_corners'],
                          map(int, summary.groups()))))
if invalidation:
    result['one_rank_invalidation'] = dict(zip(
        ['checks', 'changed_remote_query', 'donor_rank', 'reader_rank', 'query'],
        map(int, invalidation.groups())))
(output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
print(text)
run.check_returncode()
assert stable and summary and invalidation and len(scenarios) == 3
assert result['one_rank_invalidation']['changed_remote_query'] == 1
assert all(len(s['components']) == 45 and s['failures'] == 0 for s in scenarios.values())
