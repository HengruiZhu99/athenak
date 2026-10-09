"""Build private Dx(Dx) evolution jets; preserve standard constraint diagnostics."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BUILD = ROOT/'build-layer-spatial-norm-native'
FAMILY = ROOT/'build-layer-research/continuum/preferred/native-overlay/spatial-norm-family'
PRIVATE = HERE/'composed-derivative-build'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


PRIVATE.mkdir(exist_ok=False)
include = PRIVATE/'include'
overlay = include/'z4c/hyperboloidal'
overlay.mkdir(parents=True)
base_receipt_path = FAMILY/'native-build-receipt.json'
base_receipt = json.loads(base_receipt_path.read_text())
assert all(sha(ROOT/k) == v for k, v in base_receipt['source_sha256'].items())
assert all(sha(FAMILY/k) == v for k, v in base_receipt['overlay_sha256'].items())
base_exe = BUILD/'src/athena'
assert sha(base_exe) == 'dd1d189210abd4e094da339dd73e3014357924b343c9f08f658eaf8cd4ae172d'
bridge_path = ROOT/'src/z4c/hyperboloidal/athenak_bridge.hpp'
bridge = bridge_path.read_text()
start = bridge.index('template <int NGHOST, typename Field>')
end = bridge.index('// Add an analytic, time-independent initial-data jet')
original_loaders = bridge[start:end]
assert original_loaders.count('LoadMeshScalar') == 5
assert original_loaders.count('LoadMeshJet') == 1
composed_loaders = original_loaders.replace('LoadMeshScalar', 'LoadComposedMeshScalar')
composed_loaders = composed_loaders.replace('LoadMeshJet', 'LoadComposedMeshJet')
composed_loaders = composed_loaders.replace('Dxx<NGHOST>', 'ComposedDxx<NGHOST>')
helper = '''// Private experiment: standard Dx4 composed with itself, radius four.
// Original loaders remain intact for independent standard constraint diagnostics.
template <int NGHOST, typename Field>
struct FirstDerivativeField {
  int axis;
  Real idx[3];
  Field field;
  KOKKOS_INLINE_FUNCTION
  FirstDerivativeField(int a, const Real inv[3], const Field &q) : axis(a), field(q) {
    for (int d = 0; d < 3; ++d) idx[d] = inv[d];
  }
  KOKKOS_INLINE_FUNCTION
  Real operator()(int m, int k, int j, int i) const {
    return Dx<NGHOST>(axis,idx,field,m,k,j,i);
  }
  KOKKOS_INLINE_FUNCTION
  Real operator()(int m, int a, int k, int j, int i) const {
    return Dx<NGHOST>(axis,idx,field,m,a,k,j,i);
  }
  KOKKOS_INLINE_FUNCTION
  Real operator()(int m, int a, int b, int k, int j, int i) const {
    return Dx<NGHOST>(axis,idx,field,m,a,b,k,j,i);
  }
};
template <int NGHOST, typename Field, typename... Indices>
KOKKOS_INLINE_FUNCTION
Real ComposedDxx(int axis, const Real idx[3], const Field &q, Indices... indices) {
  static_assert(NGHOST == 3, "private composed experiment requires fourth-order Dx");
  return Dx<NGHOST>(axis,idx,FirstDerivativeField<NGHOST,Field>(axis,idx,q),indices...);
}

'''
private_bridge = overlay/bridge_path.name
private_bridge.write_text(bridge[:end]+helper+composed_loaders+bridge[end:])
cart_path = ROOT/'src/z4c/hyperboloidal/cartesian_patch.hpp'
cart = cart_path.read_text()
assert cart.count('auto u = LoadMeshJet<3>(udev,idx,0,k,j,i);') == 1
assert cart.count('PlanSymmetricSphericalGhosts(grid,3,ghost_degree)') == 1
assert cart.count('PlanSphericalGhosts(grid,3,ghost_degree)') == 1
cart = cart.replace('auto u = LoadMeshJet<3>(udev,idx,0,k,j,i);',
                    'auto u = LoadComposedMeshJet<3>(udev,idx,0,k,j,i);')
cart = cart.replace('PlanSymmetricSphericalGhosts(grid,3,ghost_degree)',
                    'PlanSymmetricSphericalGhosts(grid,4,ghost_degree)')
cart = cart.replace('PlanSphericalGhosts(grid,3,ghost_degree)',
                    'PlanSphericalGhosts(grid,4,ghost_degree)')
private_cart = overlay/cart_path.name
private_cart.write_text(cart)
init_path = ROOT/'src/z4c/z4c_hyperboloidal.cpp'
init = init_path.read_text()
assert init.count('ind.ng != 3 || opt.fd_stencil != 3') == 1
assert init.count('ng=3, fourth-order Z4c') == 1
init = init.replace('ind.ng != 3 || opt.fd_stencil != 3',
                    'ind.ng != 4 || opt.fd_stencil != 3')
init = init.replace('ng=3, fourth-order Z4c', 'ng=4, private composed fourth-order Z4c')
private_init = PRIVATE/init_path.name
private_init.write_text(init)
commands = json.loads((BUILD/'compile_commands.json').read_text())
link_cmd = shlex.split((BUILD/'src/CMakeFiles/athena.dir/link.txt').read_text())
link_cwd = BUILD/'src'
link_inputs = {}
for arg in link_cmd:
    if arg.endswith(('.o', '.a')):
        path = (link_cwd/arg).resolve()
        link_inputs[str(path.relative_to(ROOT))] = {'sha256': sha(path),
                                                   'bytes': path.stat().st_size}
assert len([x for x in link_inputs if x.endswith('.o')]) == 182
selected = []
for entry in commands:
    cmd = shlex.split(entry['command'])
    if '-o' not in cmd or not Path(entry['file']).is_relative_to(ROOT/'src'):
        continue
    obj = (Path(entry['directory'])/cmd[cmd.index('-o')+1]).resolve()
    deps = Path(str(obj)+'.d')
    assert deps.is_file()
    text = deps.read_text()
    if str(bridge_path) in text or str(cart_path) in text or Path(entry['file']) == init_path:
        selected.append((entry, cmd, obj))
assert len(selected) >= 4
compiled, dependencies = [], {}
started = time.monotonic()
with (PRIVATE/'build.log').open('w') as log:
    for index, (entry, cmd, original_object) in enumerate(selected):
        output = PRIVATE/('object-'+str(index)+'.o')
        dep_path = PRIVATE/('object-'+str(index)+'.d')
        cmd[cmd.index('-o')+1] = str(output)
        if Path(entry['file']) == init_path:
            cmd[cmd.index('-c')+1] = str(private_init)
        cmd[1:1] = ['-I'+str(include)]
        cmd.extend(['-MD', '-MF', str(dep_path)])
        result = subprocess.run(cmd, cwd=entry['directory'], stdout=log,
                                stderr=subprocess.STDOUT)
        assert result.returncode == 0
        for arg in link_cmd:
            if arg.endswith('.o') and (link_cwd/arg).resolve() == original_object:
                link_cmd[link_cmd.index(arg)] = str(output)
                break
        else:
            raise AssertionError('compiled object missing from base link')
        tokens = shlex.split(dep_path.read_text().replace('\\\n', ' '))
        for name in tokens[1:]:
            path = Path(name)
            if path.is_file() and path.is_relative_to(ROOT):
                dependencies[str(path.relative_to(ROOT))] = sha(path)
        compiled.append({'base_source': entry['file'], 'command': cmd,
                         'cwd': entry['directory'], 'exit_status': result.returncode,
                         'base_object_sha256': sha(original_object),
                         'private_object_sha256': sha(output),
                         'dependency_file_sha256': sha(dep_path)})
    executable = PRIVATE/'athena-composed-derivative'
    link_cmd[link_cmd.index('-o')+1] = str(executable)
    link_result = subprocess.run(link_cmd, cwd=link_cwd, stdout=log,
                                 stderr=subprocess.STDOUT)
    assert link_result.returncode == 0
assert all(sha(ROOT/k) == v['sha256'] for k, v in link_inputs.items())
assert all(sha(ROOT/k) == v for k, v in base_receipt['source_sha256'].items())
assert all(sha(FAMILY/k) == v for k, v in base_receipt['overlay_sha256'].items())
out = {
    'scope': ('Private spatial discretization experiment: only evolution diagonal second '
              'jets use Dx4(Dx4). Existing standard diagnostic derivatives stay intact. '
              'Extra allocated halo and strictly interior donor ghost planning have radius4; '
              'first/mixed/upwind/KO/gauge/finite-RK-stage projection otherwise unchanged. '
              'No adopted stability fix or continuum regularity claim.'),
    'launch_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                           cwd=ROOT, text=True).strip(),
    'compiled_implementation': base_receipt['implementation_commit'],
    'base_build_receipt_sha256': sha(base_receipt_path),
    'base_executable_sha256': sha(base_exe), 'script_sha256': sha(Path(__file__)),
    'private_source_sha256': {str(p.relative_to(ROOT)): sha(p)
                             for p in [private_bridge, private_cart, private_init]},
    'all_compiled_repository_dependencies_sha256': dependencies,
    'compile_results': compiled, 'link_command': link_cmd, 'link_cwd': str(link_cwd),
    'link_exit_status': link_result.returncode, 'elapsed_seconds': time.monotonic()-started,
    'reused_base_link_inputs_sha256': link_inputs,
    'base_source_overlay_and_link_inputs_unchanged': True,
    'executable': str(executable.relative_to(ROOT)), 'executable_sha256': sha(executable),
    'build_log_sha256': sha(PRIVATE/'build.log')}
(PRIVATE/'build-receipt.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
print('PASS private composed build', len(compiled), 'TUs', out['executable_sha256'])
