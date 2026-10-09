"""Private finite-Omega bulk-C1 build gated by tensor, principal and subsidiary checks."""
import hashlib
import json
from pathlib import Path
import shlex
import sys
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BUILD = ROOT/'build-layer-spatial-norm-native'
FAMILY = ROOT/'build-layer-research/continuum/preferred/native-overlay/spatial-norm-family'
PRIVATE = HERE/'native-build'
EXPECTED_BLEND_INDEX = '1bb5f697691404c7a8ed19c20fc77aa79c5aa3fa31a81248d10c7e958d3ac2f0'
GATE = ROOT/'build-layer-research/continuum/covariant-constraint-propagation/immutable-C1-blend-constraint-20261009'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert EXPECTED_BLEND_INDEX != 'PENDING', 'Scientific blend gate not pinned yet'
gate_path = Path(sys.argv[1]).resolve()
assert sha(gate_path) == EXPECTED_BLEND_INDEX
gate = json.loads(gate_path.read_text())
for name, value in gate.get('files', gate.get('small_files')).items():
    source = gate_path.parent/name
    wanted = value if isinstance(value, str) else value['sha256']
    assert sha(source) == wanted, source
gate_receipt = json.loads((gate_path.parent/'receipt.json').read_text())
assert gate_receipt['passed_reference_tangent_and_blend_gates'] is True
assert gate_receipt['global_native_or_scri_stability_accepted'] is False
assert gate_receipt['sources_unchanged'] is True
assert all(row['returncode'] == 0 for row in gate_receipt['commands'])
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
cart_path = ROOT/'src/z4c/hyperboloidal/cartesian_patch.hpp'
assert sha(cart_path) == 'fecbebdbb8f82cf5aef72e2bf083a0e25314978a840f103771d1530c1f369280'
assert sha(gate_path.parent/'receipt.json') == 'a307b684547de8fc2be267e3e7989ed577c5bc2146695f12165d1a5315988c78'
assert len(gate_receipt['source_before']) == 380
assert gate_path.parent == GATE
prepared = cart_path.read_text()
assert '#include "z4c/hyperboloidal/layer_gauge.hpp"' in prepared
prepared = prepared.replace('#include "z4c/hyperboloidal/layer_gauge.hpp"',
                            '#include "z4c/hyperboloidal/layer_gauge.hpp"\n#include "blend_injection.hpp"')
anchor = '      AddMeshUpwindAdvectionWithVelocity<3>(udev,full.beta_u,idx,0,k,j,i,rhs,gauge_rhs);'
assert prepared.count(anchor) == 1
injection = ('      // Private bulk-C1 candidate: all tensor additions and covector repair use 1-W.\n'
             '      if (!ResearchAddBlendedC1(u,omega,p.radius,lg,rhs)) { ++bad; return; }\n')
prepared = prepared.replace(anchor, injection+anchor)
# Extend the existing pole diagnostic numerator without changing any evolution term.
diag_capture = '    const Real damping = kappa1;'
assert prepared.count(diag_capture) == 1
prepared = prepared.replace(diag_capture, diag_capture+'\n    const auto diagnostic_lg = layer_gauge;')
prepared = prepared.replace('        const auto pole = parts.pole;',
                            '        auto pole = parts.pole;\n'
                            '        ResearchAddBlendedPole(u,CartesianOmega(u,p),p.radius,diagnostic_lg,pole);')
old_pole0 = ('        const auto pole0 = ConformalRHS(background,CartesianOmega(background,p),\n'
             '                                        damping,Real(0)).pole;')
new_pole0 = old_pole0.replace('const auto pole0','auto pole0') + ('\n'
    '        ResearchAddBlendedPole(background,CartesianOmega(background,p),p.radius,diagnostic_lg,pole0);')
assert prepared.count(old_pole0) == 1
prepared = prepared.replace(old_pole0,new_pole0)
restored = prepared.replace('#include "blend_injection.hpp"\n','').replace(injection,'')
restored = restored.replace(diag_capture+'\n    const auto diagnostic_lg = layer_gauge;',diag_capture)
restored = restored.replace('        auto pole = parts.pole;\n'
    '        ResearchAddBlendedPole(u,CartesianOmega(u,p),p.radius,diagnostic_lg,pole);',
    '        const auto pole = parts.pole;')
restored = restored.replace(new_pole0,old_pole0)
assert restored == cart_path.read_text()
private_cart = overlay/cart_path.name
private_cart.write_text(prepared)
private_helpers = []
for name, expected in [
    ('c1_additions.hpp', '908d655ad8a43261d8e0cd66b3c0da485fc131b57a5203019aea01acc21771b7'),
    ('bulk_c1_additions.hpp', '4c7b6637fc9c38339134d5d5824e589986110edc9efa55489ebe0fc941bafbe2')]:
    source = GATE/name
    assert sha(source) == expected, source
    output = include/name
    output.write_bytes(source.read_bytes())
    private_helpers.append(output)
injection_path = HERE/'blend_injection.hpp'
output = include/injection_path.name
output.write_bytes(injection_path.read_bytes())
private_helpers.append(output)
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
    if str(cart_path) in text:
        selected.append((entry, cmd, obj))
assert len(selected) >= 1
compiled, dependencies = [], {}
started = time.monotonic()
with (PRIVATE/'build.log').open('w') as log:
    for index, (entry, cmd, original_object) in enumerate(selected):
        output = PRIVATE/('object-'+str(index)+'.o')
        dep_path = PRIVATE/('object-'+str(index)+'.d')
        cmd[cmd.index('-o')+1] = str(output)
        cmd[1:1] = ['-I'+str(include)]
        cmd.extend(['-MD', '-MF', str(dep_path)])
        attempt = {'base_source': entry['file'], 'command': cmd, 'cwd': entry['directory']}
        (PRIVATE/('attempt-'+str(index)+'.json')).write_text(json.dumps(attempt, indent=2)+'\n')
        result = subprocess.run(cmd, cwd=entry['directory'], stdout=log,
                                stderr=subprocess.STDOUT)
        if result.returncode:
            (PRIVATE/'failed-recipe.py').write_bytes(Path(__file__).read_bytes())
            (PRIVATE/'failure.json').write_text(json.dumps({
                'phase': 'compile', 'attempt': attempt, 'returncode': result.returncode,
                'gate_index_sha256': EXPECTED_BLEND_INDEX,
                'base_build_receipt_sha256': sha(base_receipt_path)}, indent=2)+'\n')
            raise RuntimeError('private bulk-C1 compilation failed; preserved complete attempt/log')
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
    executable = PRIVATE/'athena-blended-c1'
    link_cmd[link_cmd.index('-o')+1] = str(executable)
    link_result = subprocess.run(link_cmd, cwd=link_cwd, stdout=log,
                                 stderr=subprocess.STDOUT)
    if link_result.returncode:
        (PRIVATE/'failed-recipe.py').write_bytes(Path(__file__).read_bytes())
        (PRIVATE/'failure.json').write_text(json.dumps({
            'phase': 'link', 'command': link_cmd, 'returncode': link_result.returncode,
            'gate_index_sha256': EXPECTED_BLEND_INDEX}, indent=2)+'\n')
        raise RuntimeError('private bulk-C1 link failed; preserved complete attempt/log')
assert all(sha(ROOT/k) == v['sha256'] for k, v in link_inputs.items())
assert all(sha(ROOT/k) == v for k, v in base_receipt['source_sha256'].items())
assert all(sha(FAMILY/k) == v for k, v in base_receipt['overlay_sha256'].items())
out = {
    'scope': ('Private finite-Omega physical-P tensor additions and covector repair with '
              'shared C=1-W_gauge coefficient; explicit simple/double-pole assembly. Original '
              'spatial-norm gauge, derivatives, ng3 ghosts, KO and final-only algebraic '
              'projection unchanged. No C1 reference subtraction or regular-scri claim.'),
    'blend_gate_index': str(gate_path.relative_to(ROOT)),
    'blend_gate_index_sha256': EXPECTED_BLEND_INDEX,
    'launch_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                           cwd=ROOT, text=True).strip(),
    'compiled_implementation': base_receipt['implementation_commit'],
    'base_build_receipt_sha256': sha(base_receipt_path),
    'base_executable_sha256': sha(base_exe), 'script_sha256': sha(Path(__file__)),
    'private_source_sha256': {str(p.relative_to(ROOT)): sha(p)
                             for p in [private_cart]+private_helpers},
    'all_compiled_repository_dependencies_sha256': dependencies,
    'compile_results': compiled, 'link_command': link_cmd, 'link_cwd': str(link_cwd),
    'link_exit_status': link_result.returncode, 'elapsed_seconds': time.monotonic()-started,
    'reused_base_link_inputs_sha256': link_inputs,
    'base_source_overlay_and_link_inputs_unchanged': True,
    'executable': str(executable.relative_to(ROOT)), 'executable_sha256': sha(executable),
    'build_log_sha256': sha(PRIVATE/'build.log')}
(PRIVATE/'build-source.py').write_bytes(Path(__file__).read_bytes())
(PRIVATE/'build-receipt.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
print('PASS private blended C1 build', len(compiled), 'TUs', out['executable_sha256'])
