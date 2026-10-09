"""Read-only private composed-operator/halo gate; no evolution or tracked edits."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PRIVATE = HERE.parent/'composed-derivative-build'
FAMILY = ROOT/'build-layer-research/continuum/preferred/native-overlay/spatial-norm-family'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


build = json.loads((PRIVATE/'build-receipt.json').read_text())
assert all(sha(ROOT/key) == value for key, value in build['private_source_sha256'].items())
assert build['link_exit_status'] == 0
private_bridge = PRIVATE/'include/z4c/hyperboloidal/athenak_bridge.hpp'
private_cart = PRIVATE/'include/z4c/hyperboloidal/cartesian_patch.hpp'
bridge = (ROOT/'src/z4c/hyperboloidal/athenak_bridge.hpp').read_text()
cart = (ROOT/'src/z4c/hyperboloidal/cartesian_patch.hpp').read_text()
assert private_bridge.read_text().count('auto first = FirstDerivativeField') == 1
assert private_cart.read_text() == cart.replace(
    'auto u = LoadMeshJet<3>(udev,idx,0,k,j,i);',
    'auto u = LoadComposedMeshJet<3>(udev,idx,0,k,j,i);').replace(
    'PlanSymmetricSphericalGhosts(grid,3,ghost_degree)',
    'PlanSymmetricSphericalGhosts(grid,4,ghost_degree)').replace(
    'PlanSphericalGhosts(grid,3,ghost_degree)',
    'PlanSphericalGhosts(grid,4,ghost_degree)')
begin = bridge.index('template <int NGHOST, typename Field>')
end = bridge.index('// Add an analytic, time-independent initial-data jet')
assert bridge[begin:end] in private_bridge.read_text()
critical = sorted((ROOT/'src/z4c/hyperboloidal').glob('*.hpp'))
critical += [ROOT/key for key in build['private_source_sha256']]
critical += [FAMILY/'native_injection.hpp', FAMILY/'spatial_norm_control.hpp',
             HERE/'test_composed_patch.cpp', Path(__file__)]
before = {str(p.relative_to(ROOT)): sha(p) for p in critical}
flags = (ROOT/'build-layer-release/CMakeFiles/hyperboloidal_layer_constraint_tangent.dir/flags.make').read_text()


def makevar(name):
    return shlex.split(re.search('^'+name+' = (.*)$', flags, re.M).group(1))


common = ['/usr/bin/c++', '-I'+str(PRIVATE/'include'), '-I'+str(FAMILY)]
common += makevar('CXX_DEFINES')+makevar('CXX_INCLUDES')+['-std=c++17']
common += ['-include', str(FAMILY/'native_injection.hpp')]
libs = [str(ROOT/f'build-layer-release/kokkos/{part}/src/libkokkos{part}.a')
        for part in ['containers', 'algorithms', 'core', 'simd']]
started = time.monotonic()
rows = []


def save():
    after = {str(p.relative_to(ROOT)): sha(p) for p in critical}
    receipt = {
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                        text=True).strip(),
        'scope': ('Actual private CartesianConformalPatch N24/ng4/S1/span2.1 '
                  'a.5 wide(.05,.95), kappa10, spatial-norm gauge injection, '
                  'cube symmetric degree2/3. Original diagnostic loader is '
                  'byte-preserved. No native evolution or stability claim.'),
        'seconds': time.monotonic()-started,
        'private_native_build_receipt_sha256': sha(PRIVATE/'build-receipt.json'),
        'private_native_executable_sha256': build['executable_sha256'],
        'source_sha256_before': before, 'source_sha256_after': after,
        'all_sources_unchanged': before == after,
        'original_diagnostic_loader_byte_preserved': True,
        'private_patch_only_rhs_loader_and_plan_radius_changes': True,
        'sanitizers': 'address,undefined; ASAN leak checking disabled on macOS',
        'results': rows,
    }
    (HERE/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')


def run(label, command):
    start = time.monotonic()
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                             env={**os.environ, 'ASAN_OPTIONS': 'detect_leaks=0',
                                  'UBSAN_OPTIONS': 'halt_on_error=1'})
    (HERE/(label+'.log')).write_text(process.stdout)
    (HERE/(label+'.stderr')).write_text(process.stderr)
    row = {'label': label, 'command': command, 'cwd': str(ROOT),
           'returncode': process.returncode, 'seconds': time.monotonic()-start,
           'stdout_sha256': sha(HERE/(label+'.log')),
           'stderr_sha256': sha(HERE/(label+'.stderr'))}
    if label.startswith('test-'):
        row['measurements'] = [json.loads(s) for s in process.stdout.splitlines()]
    rows.append(row)
    save()
    print(label, 'PASS' if process.returncode == 0 else 'FAIL',
          process.stdout, process.stderr[-3000:], flush=True)
    if process.returncode:
        failure = HERE/('failed-'+label+'-'+str(time.time_ns()))
        failure.mkdir()
        for file in [Path(__file__), HERE/'test_composed_patch.cpp', HERE/'receipt.json',
                     HERE/(label+'.log'), HERE/(label+'.stderr')]:
            (failure/file.name).write_bytes(file.read_bytes())
        raise RuntimeError(label)


for mode, compiler_flags in [
        ('release', ['-O3', '-DNDEBUG']),
        ('asan-ub', ['-O0', '-g', '-fsanitize=address,undefined',
                     '-fno-omit-frame-pointer', '-DKOKKOS_ENABLE_DEBUG_BOUNDS_CHECK'])]:
    binary = HERE/('test-composed-'+mode)
    command = common+compiler_flags+[str(HERE/'test_composed_patch.cpp'), '-o', str(binary)]+libs
    run('build-'+mode, command)
    rows[-1]['binary_sha256'] = sha(binary)
    run('test-'+mode, [str(binary)])
save()
assert all(sha(ROOT/key) == value for key, value in before.items())
print('PASS gate', HERE/'receipt.json', flush=True)
