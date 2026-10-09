"""Read-only independent inspection of trace/combined native source graft.

No native command is invoked and accepted builder/auditor files are unchanged.
"""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
HERE = P.parent
FAMILY = ROOT/'build-layer-research/continuum/preferred/native-overlay/spatial-norm-family'
BASE = ROOT/'build-layer-spatial-norm-native'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
gate = ROOT/'build-layer-research/continuum/inner-conformal-trace-control/immutable-inner-conformal-trace-local-20261009'
assert sha(gate/'index.json') == '801b78e9cdc7755ddda8b0ce6ea375efb02a023f7cfec420ccd8543490c150f8'
pins = {
    'trace': ('2d93e3f2045445a2f6e255f61741397868ff0ed2d3d15f255a6d2e47ce13152c',
              'd1593cfde3cafa88f81faad78c7aeaa23486217dc68a14d2f35f861bbf3f1235'),
    'combined': ('d84e536754f2f2ed44697d689ff32757e67f34d7220bb11979486f0559124f4c',
                 'd5a2c6c5195858306357d1a1ecb68f2a60737e78ce08ed3227fcb1bebf2084bb')}
files = [HERE/'build_trace_native.py', HERE/'audit_trace_native.py',
         FAMILY/'native_injection.hpp', FAMILY/'spatial_norm_control.hpp',
         ROOT/'src/z4c/hyperboloidal/cartesian_patch.hpp', gate/'inner_conformal_trace.hpp']
before = {str(p.relative_to(ROOT)): sha(p) for p in files}
base = json.loads((FAMILY/'native-build-receipt.json').read_text())
for key, value in base['source_sha256'].items():
    assert sha(ROOT/key) == value
for key, value in base['overlay_sha256'].items():
    assert sha(FAMILY/key) == value
assert len(base['source_sha256']) == 369 and len(base['overlay_sha256']) == 4
commands = {Path(x['file']): x for x in json.loads((BASE/'compile_commands.json').read_text())}
rows = []
for mode, (receipt_hash, exe_hash) in pins.items():
    private = HERE/(mode+'-build')
    receipt_path = private/'build-receipt.json'
    assert sha(receipt_path) == receipt_hash
    r = json.loads(receipt_path.read_text())
    assert r['mode'] == mode and r['executable_sha256'] == exe_hash
    assert sha(ROOT/r['executable']) == exe_hash
    assert sha(private/'build-source.py') == sha(HERE/'build_trace_native.py') == r['script_sha256']
    assert sha(private/'build.log') == r['build_log_sha256']
    for key in ('private_source_sha256', 'all_compiled_repository_dependencies_sha256'):
        for name, digest in r[key].items():
            assert sha(ROOT/name) == digest
    for name, value in r['reused_base_link_inputs_sha256'].items():
        path = ROOT/name
        assert sha(path) == value['sha256'] and path.stat().st_size == value['bytes']
    assert len(r['reused_base_link_inputs_sha256']) == 186
    cart = private/'include/z4c/hyperboloidal/cartesian_patch.hpp'
    helper = private/'include/inner_conformal_trace.hpp'
    assert helper.read_bytes() == (gate/'inner_conformal_trace.hpp').read_bytes()
    old = '? InteriorLayerGauge(p,u,lg)'
    new = '? ResearchInnerTraceGauge(p,u,lg,InteriorLayerGauge(p,u,lg),'+str(mode == 'combined').lower()+')'
    text = cart.read_text()
    assert text.count(new) == 1
    original = text.replace('#include "inner_conformal_trace.hpp"\n', '').replace(new, old)
    assert original == (ROOT/'src/z4c/hyperboloidal/cartesian_patch.hpp').read_text()
    # Exact compile and link reconstruction from the original build, independent
    # of the native auditor's verify_build function.
    link = shlex.split((BASE/'src/CMakeFiles/athena.dir/link.txt').read_text())
    link[link.index('-o')+1] = str(ROOT/r['executable'])
    assert len(r['compile_results']) == 6
    for row in r['compile_results']:
        assert row['exit_status'] == 0
        original_command = shlex.split(commands[Path(row['base_source'])]['command'])
        expected = original_command.copy()
        out = row['command'][row['command'].index('-o')+1]
        dep = row['command'][row['command'].index('-MF')+1]
        obj = (Path(row['cwd'])/original_command[original_command.index('-o')+1]).resolve()
        assert sha(obj) == row['base_object_sha256']
        assert sha(Path(out)) == row['private_object_sha256']
        assert sha(Path(dep)) == row['dependency_file_sha256']
        expected[expected.index('-o')+1] = out
        expected[1:1] = ['-I'+str(private/'include')]
        expected += ['-MD', '-MF', dep]
        assert row['command'] == expected
        slots = [i for i, x in enumerate(link) if x.endswith('.o') and (Path(r['link_cwd'])/x).resolve() == obj]
        assert len(slots) == 1
        link[slots[0]] = out
    assert link == r['link_command'] and r['link_exit_status'] == 0
    assert len(r['private_source_sha256']) == 2
    assert len(r['all_compiled_repository_dependencies_sha256']) == 268
    rows.append({'mode': mode, 'receipt_sha256': receipt_hash, 'executable_sha256': exe_hash,
                 'private_sources': 2, 'repository_dependencies': 268,
                 'private_objects': 6, 'reused_base_objects_and_libraries': 186,
                 'whole_cartesian_patch_restored_exactly_after_two_graft_reversals': True})
# The forced include resolves the original-gauge argument to the spatial-norm
# control before the new helper, and assembles its shift pole once afterwards.
inj = (FAMILY/'native_injection.hpp').read_text()
assert '#define InteriorLayerGauge ResearchNativeSpatialGauge' in inj
assert '#define AssembleGaugeInterior ResearchNativeSpatialAssemble' in inj
assert 'r.beta[i]+=p.pole.beta[i]/omega' in inj
h = (gate/'inner_conformal_trace.hpp').read_text()
assert 'parts.regular.alpha+=ResearchInnerConformalTrace(p,u,g)' in h
assert 'parts.pole.' not in h and 'parts.regular.beta' not in h
assert before == {str(p.relative_to(ROOT)): sha(p) for p in files}
report = {'status': 'PASS', 'scope': 'Read-only source/recipe/formula integration review, no native acceptance',
          'reviewer_source_sha256': sha(Path(__file__)), 'reviewed_sources': before,
          'baseline_files': 369, 'baseline_overlays': 4, 'builds': rows,
          'formula_interpretation': 'trace=false adds bounded trace source; combined=true replaces full regular lapse then adds trace. Physical-P pole and norm-shift parts preserved; beta pole assembled exactly once.',
          'scope_caveats': ['Reference/core/outer identities and finite-positive principal gate only.',
                            'No lapse positivity, uniform puncture regularity, scri PDE closure, or BH admission.'],
          'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
(P/'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
print('PASS independent trace/combined source graft, 12 private objects, exact original recipes')
