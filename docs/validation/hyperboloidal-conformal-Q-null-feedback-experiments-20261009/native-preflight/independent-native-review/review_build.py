"""Independent read-only identity/compile-link check; no compile or evolution."""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import time

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
NATIVE = ROOT/'build-layer-research/q-null-native'
BUILD = NATIVE/'native-build'
BASE = ROOT/'build-layer-spatial-norm-native'
FAMILY = ROOT/'build-layer-research/continuum/preferred/native-overlay/spatial-norm-family'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
start = time.monotonic()
pin = '8f7e60ef5d5df9dad31f8feba359b244b36859aa115d525d2f52b057a8813508'
assert sha(BUILD/'build-receipt.json') == pin
r = json.loads((BUILD/'build-receipt.json').read_text())
assert r['executable_sha256'] == '5d7ce0c747ef0a8519deed61d1d2cbdbba71d8924728127bbbd4f8b1aa39e43e'
assert sha(ROOT/r['executable']) == r['executable_sha256']
assert r['compiled_implementation'] == '27c19d20696ea6dd4704032c51dfd026218f64f2'
assert r['mode'] == 'physical-inner' and r['link_exit_status'] == 0
assert (BUILD/'build.log').read_bytes() == b''
assert sha(BUILD/'build.log') == r['build_log_sha256']
assert sha(BUILD/'build-source.py') == sha(NATIVE/'build_native.py') == r['script_sha256']
base = json.loads((FAMILY/'native-build-receipt.json').read_text())
assert sha(FAMILY/'native-build-receipt.json') == r['base_build_receipt_sha256']
assert len(base['source_sha256']) == 369 and len(base['overlay_sha256']) == 4
for name, digest in base['source_sha256'].items():
    assert sha(ROOT/name) == digest, name
for name, digest in base['overlay_sha256'].items():
    assert sha(FAMILY/name) == digest, name
assert sha(BASE/'src/athena') == r['base_executable_sha256'] == 'dd1d189210abd4e094da339dd73e3014357924b343c9f08f658eaf8cd4ae172d'
for name, digest in r['all_compiled_repository_dependencies_sha256'].items():
    assert sha(ROOT/name) == digest, name
assert len(r['all_compiled_repository_dependencies_sha256']) == 268
assert len(r['private_source_sha256']) == 3
for name, digest in r['private_source_sha256'].items():
    assert sha(ROOT/name) == digest, name
gatepath = ROOT/r['scientific_gate_index']
assert sha(gatepath) == r['scientific_gate_index_sha256'] == 'dac759becbe666c71443328b61324a50aeaf0f72e76b4e751a9cabfa3a87cd96'
for name in ('q_null_feedback.hpp', 'factored_base.hpp'):
    assert (BUILD/'include'/name).read_bytes() == (gatepath.parent/name).read_bytes()
reviewpath = ROOT/r['independent_review_index']
assert sha(reviewpath) == r['independent_review_index_sha256'] == 'a456408b97c35ddf34ffe0af9c52411a13d990c9fa95f83a987234de24a3b650'
for ip in (gatepath, reviewpath):
    index = json.loads(ip.read_text())
    for name, entry in index['files'].items():
        expected = entry if isinstance(entry, str) else entry['sha256']
        assert sha(ip.parent/name) == expected, name
commands = {q['file']: q for q in json.loads((BASE/'compile_commands.json').read_text())}
link = shlex.split((BASE/'src/CMakeFiles/athena.dir/link.txt').read_text())
linkcwd = BASE/'src'
expected_link = link.copy()
expected_link[expected_link.index('-o')+1] = str(ROOT/r['executable'])
assert len(r['compile_results']) == 6
for row in r['compile_results']:
    assert row['exit_status'] == 0
    original = commands[row['base_source']]
    command = shlex.split(original['command'])
    obj = (Path(original['directory'])/command[command.index('-o')+1]).resolve()
    assert sha(obj) == row['base_object_sha256']
    assert command.count('-include') == 1
    assert Path(command[command.index('-include')+1]) == FAMILY/'native_injection.hpp'
    actual = row['command']
    output = Path(actual[actual.index('-o')+1])
    dep = Path(actual[actual.index('-MF')+1])
    assert sha(output) == row['private_object_sha256']
    assert sha(dep) == row['dependency_file_sha256']
    command[command.index('-o')+1] = str(output)
    command[1:1] = ['-I'+str(BUILD/'include')]
    command[command.index('-include')+1] = str(BUILD/'include/native_q_injection.hpp')
    command.extend(['-MD', '-MF', str(dep)])
    assert command == actual and row['cwd'] == original['directory']
    positions = [i for i, token in enumerate(expected_link)
                 if token.endswith('.o') and (linkcwd/token).resolve() == obj]
    assert len(positions) == 1
    expected_link[positions[0]] = str(output)
assert r['link_command'] == expected_link and Path(r['link_cwd']) == linkcwd
inputs = r['reused_base_link_inputs_sha256']
assert len(inputs) == 186 and sum(n.endswith('.o') for n in inputs) == 182
assert sum(n.endswith('.a') for n in inputs) == 4
for name, entry in inputs.items():
    assert sha(ROOT/name) == entry['sha256']
    assert (ROOT/name).stat().st_size == entry['bytes']
wrapper = (BUILD/'include/native_q_injection.hpp').read_text()
assert sha(BUILD/'include/native_q_injection.hpp') == '90809151b8d150054bb82c6ffe1229e5201e1cd96fd40c4d8e78a65936a3622b'
assert 'return qnf::Gauge(p,u,g,qnf::Parameters{.85,.95,5,true});' in wrapper
assert 'if(!g.physical_trace_lapse && g.preferred_source)' in wrapper
assert wrapper.index('return InteriorLayerGauge(p,u,g);') < wrapper.index('#define InteriorLayerGauge')
assert wrapper.index('if(!AssembleGaugeInterior(p,omega,r))') < wrapper.index('#define AssembleGaugeInterior')
assert wrapper.count('r.beta[i]+=p.pole.beta[i]/omega') == 1
baseinput = (FAMILY/'native.athinput').read_text()
expected_input = baseinput.replace('hyperboloidal_physical_trace_lapse=true', 'hyperboloidal_physical_trace_lapse=false').replace('hyperboloidal_preferred_source=false', 'hyperboloidal_preferred_source=true')
assert (NATIVE/'native.athinput').read_text() == expected_input
auditor = (NATIVE/'audit_native.py').read_text()
assert "'8f7e60ef5d5df9dad31f8feba359b244b36859aa115d525d2f52b057a8813508'" in auditor
assert "'5d7ce0c747ef0a8519deed61d1d2cbdbba71d8924728127bbbd4f8b1aa39e43e'" in auditor
assert 'np.array_equal(private["initial"][:, mask], base["initial"][:, mask])' in auditor
assert 'len(private["snapshots"]) == 3' in auditor
launcher = (NATIVE/'launch_preflight.py').read_text()
assert "if stage=='short':" in launcher and "r['status']=='PASS'" in launcher
assert "'--duration','.05' if stage=='reference' else '.02'" in launcher
out = {'status': 'PASS_READ_ONLY_NATIVE_RECIPE_BUILD_SOURCE_REVIEW',
       'seconds': time.monotonic()-start, 'build_receipt_sha256': pin,
       'executable_sha256': r['executable_sha256'], 'private_TUs': 6,
       'repo_dependency_count': 268, 'private_header_count': 3,
       'baseline_sources': 369, 'baseline_overlays': 4,
       'original_objects_rechecked': 182, 'original_libraries_rechecked': 4,
       'exact_compile_link_replacements': True, 'wrapper_beta_pole_added_once': True,
       'input_only_two_gauge_flag_changes': True,
       'reviewed_root_sources': {n: sha(NATIVE/n) for n in ('build_native.py', 'audit_native.py', 'launch_preflight.py', 'native.athinput')},
       'review_HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
       'scope': 'Read-only private host-double routing/build/receipt/audit/launcher wiring. No execution of native binary, state/audit PASS, full snapshot source or pulse/BH/regularity acceptance.'}
(P/'review-result.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
