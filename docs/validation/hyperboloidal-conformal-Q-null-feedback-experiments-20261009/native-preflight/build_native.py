"""Private factored Q/null-feedback gauge build; compilation requires the immutable local gate."""
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
MODE = 'physical-inner'
PRIVATE = HERE/'native-build'
EXPECTED_TRACE_INDEX = 'dac759becbe666c71443328b61324a50aeaf0f72e76b4e751a9cabfa3a87cd96'
EXPECTED_RECEIPT = '0768080c9c251829f68d9446d7537da38a4ab026c93e0c6f033bc11a2911f0f2'
EXPECTED_HELPER = 'f3f3acfe16ba3ce36d0b687225aa1e3b33c159781e46d05e1441ccea7c57e049'
EXPECTED_FACTORED = '216bb80c18ee33e553de8defd027e6d3e0394eab38a19a56f66699583e0eca7a'
GATE = ROOT/'build-layer-research/continuum/conformal-q-null-feedback/immutable-conformal-Q-null-feedback-local-20261009'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert EXPECTED_TRACE_INDEX != 'PENDING', 'Scientific profile gate not pinned yet'
gate_path = Path(sys.argv[1]).resolve()
assert sha(gate_path) == EXPECTED_TRACE_INDEX
gate = json.loads(gate_path.read_text())
entries = gate.get('files', gate.get('small_files'))
for name, value in (entries.items() if isinstance(entries,dict) else ((row['path'],row) for row in entries)):
    source = gate_path.parent/name
    wanted = value if isinstance(value,str) else value['sha256']
    assert sha(source) == wanted, source
gate_receipt = json.loads((gate_path.parent/'receipt.json').read_text())
assert sha(gate_path.parent/'receipt.json') == EXPECTED_RECEIPT
assert gate_receipt['passed_local_pole_source_principal_corner_gates'] is True
assert gate_receipt['release_debug_json_equal'] and gate_receipt['sources_unchanged']
assert len(gate_receipt['source_before'])==382 and len(gate_receipt['commands'])==14
assert gate_receipt['source_before']==gate_receipt['source_after']
assert all(sha(ROOT/name)==value for name,value in gate_receipt['source_before'].items())
assert all(row['returncode']==0 and (gate_path.parent/row['stderr']).read_bytes()==b'' for row in gate_receipt['commands'])
assert gate['native_or_global_accepted'] is False
# Independent review is pinned before compilation.
INDEPENDENT_INDEX = 'a456408b97c35ddf34ffe0af9c52411a13d990c9fa95f83a987234de24a3b650'
INDEPENDENT_PATH = 'build-layer-research/continuum/independent-q-null-feedback-review/immutable-independent-Q-null-review-20261009/index.json'
assert INDEPENDENT_INDEX != 'PENDING' and sha(ROOT/INDEPENDENT_PATH)==INDEPENDENT_INDEX
review_path=ROOT/INDEPENDENT_PATH
review=json.loads(review_path.read_text())
for name,entry in review['files'].items():
    wanted=entry if isinstance(entry,str) else entry['sha256']
    assert sha(review_path.parent/name)==wanted
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
assert gate_path.parent == GATE
private_helpers=[]
for name,digest in [('q_null_feedback.hpp',EXPECTED_HELPER),('factored_base.hpp',EXPECTED_FACTORED)]:
    helper=GATE/name
    assert sha(helper)==digest
    output=include/name
    output.write_bytes(helper.read_bytes())
    private_helpers.append(output)
native_injection=include/'native_q_injection.hpp'
native_injection.write_text("""#ifndef PRIVATE_Q_NULL_NATIVE_INJECTION_HPP_
#define PRIVATE_Q_NULL_NATIVE_INJECTION_HPP_
#include <Kokkos_Core.hpp>
#include <type_traits>
#include "q_null_feedback.hpp"
namespace z4c { namespace hyperboloidal {
template<typename T> KOKKOS_INLINE_FUNCTION GaugeRHSParts<T> ResearchNativeQNullGauge(
    const LayerPoint<T>&p,const Z4cJet<T>&u,const LayerGaugeParameters&g) {
  if constexpr(std::is_same<T,double>::value) {
    if(!g.physical_trace_lapse && g.preferred_source)
      return qnf::Gauge(p,u,g,qnf::Parameters{.85,.95,5,true});
  }
  return InteriorLayerGauge(p,u,g);
}
template<typename T> KOKKOS_INLINE_FUNCTION bool ResearchNativeQNullAssemble(
    const GaugeRHSParts<T>&p,T omega,GaugeRHS<T>&r) {
  if(!AssembleGaugeInterior(p,omega,r)) return false;
  for(int i=0;i<3;++i) {r.beta[i]+=p.pole.beta[i]/omega;
    if(!Kokkos::isfinite(r.beta[i])) return false;}
  return true;
}
}}
#define InteriorLayerGauge ResearchNativeQNullGauge
#define AssembleGaugeInterior ResearchNativeQNullAssemble
#endif
""")
private_helpers.append(native_injection)
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
        assert cmd.count('-include')==1
        forced=cmd.index('-include')+1
        assert Path(cmd[forced]).resolve()==FAMILY/'native_injection.hpp'
        cmd[forced]=str(native_injection)
        cmd.extend(['-MD', '-MF', str(dep_path)])
        attempt = {'base_source': entry['file'], 'command': cmd, 'cwd': entry['directory']}
        (PRIVATE/('attempt-'+str(index)+'.json')).write_text(json.dumps(attempt, indent=2)+'\n')
        result = subprocess.run(cmd, cwd=entry['directory'], stdout=log,
                                stderr=subprocess.STDOUT)
        if result.returncode:
            (PRIVATE/'failed-recipe.py').write_bytes(Path(__file__).read_bytes())
            (PRIVATE/'failure.json').write_text(json.dumps({
                'phase': 'compile', 'attempt': attempt, 'returncode': result.returncode,
                'gate_index_sha256': EXPECTED_TRACE_INDEX,
                'base_build_receipt_sha256': sha(base_receipt_path)}, indent=2)+'\n')
            raise RuntimeError('private Q/null compilation failed; preserved complete attempt/log')
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
    executable = PRIVATE/'athena-q-null'
    link_cmd[link_cmd.index('-o')+1] = str(executable)
    link_result = subprocess.run(link_cmd, cwd=link_cwd, stdout=log,
                                 stderr=subprocess.STDOUT)
    if link_result.returncode:
        (PRIVATE/'failed-recipe.py').write_bytes(Path(__file__).read_bytes())
        (PRIVATE/'failure.json').write_text(json.dumps({
            'phase': 'link', 'command': link_cmd, 'returncode': link_result.returncode,
            'gate_index_sha256': EXPECTED_TRACE_INDEX}, indent=2)+'\n')
        raise RuntimeError('private Q/null link failed; preserved complete attempt/log')
assert all(sha(ROOT/k) == v['sha256'] for k, v in link_inputs.items())
assert all(sha(ROOT/k) == v for k, v in base_receipt['source_sha256'].items())
assert all(sha(FAMILY/k) == v for k, v in base_receipt['overlay_sha256'].items())
out = {
    'scope': 'Private factored Q/preferred/null-feedback gauge with physical-P alpha at W=0, harmonic Q at W=1, explicit alpha blend, sigma5 beta pole. P geometric evolution/storage, C0 damping, stencils/ng3 ghosts/KO/final projection unchanged. No exact scri/puncture or stability admission.',
    'mode':MODE,
    'scientific_gate_index': str(gate_path.relative_to(ROOT)),
    'scientific_gate_index_sha256': EXPECTED_TRACE_INDEX,
    'independent_review_index': INDEPENDENT_PATH,
    'independent_review_index_sha256': INDEPENDENT_INDEX,
    'launch_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                           cwd=ROOT, text=True).strip(),
    'compiled_implementation': base_receipt['implementation_commit'],
    'base_build_receipt_sha256': sha(base_receipt_path),
    'base_executable_sha256': sha(base_exe), 'script_sha256': sha(Path(__file__)),
    'private_source_sha256': {str(p.relative_to(ROOT)): sha(p)
                             for p in private_helpers},
    'all_compiled_repository_dependencies_sha256': dependencies,
    'compile_results': compiled, 'link_command': link_cmd, 'link_cwd': str(link_cwd),
    'link_exit_status': link_result.returncode, 'elapsed_seconds': time.monotonic()-started,
    'reused_base_link_inputs_sha256': link_inputs,
    'base_source_overlay_and_link_inputs_unchanged': True,
    'executable': str(executable.relative_to(ROOT)), 'executable_sha256': sha(executable),
    'build_log_sha256': sha(PRIVATE/'build.log')}
(PRIVATE/'build-source.py').write_bytes(Path(__file__).read_bytes())
(PRIVATE/'build-receipt.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
print('PASS private Q/null build', len(compiled), 'TUs', out['executable_sha256'])
