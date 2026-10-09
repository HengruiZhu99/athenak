"""Read-only saved-output/source review; never runs a scientific executable."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
P = ROOT / 'build-layer-research/boundary/total-j-finite-rb-control-20261009'
OUT = HERE / 'independent-draft-review.json'
assert not OUT.exists()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


pins = {}


def read(path):
    pins[str(path.relative_to(ROOT))] = {'sha256': sha(path),
                                      'bytes': path.stat().st_size}
    return json.loads(path.read_text())


for name in ('audit-draft.md', 'archive-README.md', 'review_build_inputs.py'):
    p = HERE / name
    pins[str(p.relative_to(ROOT))] = {'sha256': sha(p), 'bytes': p.stat().st_size}
root_review = read(HERE / 'root-provenance-review.json')
chunk = read(P / 'source-chunk-002-pins.json')
prior = P / 'independent-chunk-reviews/002-configuration-normalization'
read(prior / 'receipt.json')
read(P / 'independent-chunk-reviews/history/001-source-pin-race.json')
dep_union = {}
outputs = []
builds = []
for mode, attempt, gate, expected_deps in (
        ('release', 'release-004', 'source-gate-release-002', 1060),
        ('debug', 'debug-002', 'source-gate-debug-001', 1062)):
    adir = P / 'build-attempts' / attempt
    gdir = P / gate
    b = read(adir / 'receipt.json')
    r = read(gdir / 'receipt.json')
    report = read(gdir / 'report.json')
    data = read(gdir / 'stdout.json')
    assert b['exit_code'] == r['exit_code'] == 0
    assert b['sources_before'] == b['sources_after']
    assert b['runtime_source_commit'] == '27c19d20696ea6dd4704032c51dfd026218f64f2'
    assert b['launch_HEAD'] == '2e0aa3b0d5ade2fef4d86807a4569fef61f6b202'
    assert len(b['compiler_dependency_hashes']) == expected_deps
    assert len(b['link_archive_hashes']) == 4
    assert b['executable_sha256'] == r['executable_sha256']
    assert b['sources_before']['radial_bridge.cpp'] == r['source_sha256']
    for name, digest in chunk.items():
        assert sha(adir / name) == sha(prior / name) == digest
        pins[str((adir / name).relative_to(ROOT))] = {
            'sha256': digest, 'bytes': (adir / name).stat().st_size}
    for original, digest in dict(b['compiler_dependency_hashes'],
                                 **b['link_archive_hashes']).items():
        path = Path(original)
        captured = adir / path.name
        source = captured if path.parent == P and captured.is_file() else path
        assert sha(source) == digest, str(source)
        assert original not in dep_union or dep_union[original] == digest
        dep_union[original] = digest
    for name in ('stderr',):
        assert (adir / name).stat().st_size == (gdir / name).stat().st_size == 0
    assert (adir / 'dependency.stderr').stat().st_size == 0
    assert sha(gdir / 'stdout.json') == r['stdout_sha256']
    assert report['passed_source_configuration_derivative_gate']
    assert all(report['checks'].values())
    assert report['receipt'] == r
    assert report['source_sha256'] == sha(gdir / 'run_source_gate.py')
    convergence = {}
    for key in ('configuration_derivative_sequences', 'map_derivative_sequences'):
        rows = data[key]
        fourth = sum(any(e[j] > 1e-10 and e[j+1] > 1e-10 and
                         e[j] >= 8*e[j+1] for j in range(3)) for e in rows)
        unknown = sum(not any(e[j] > 1e-10 and e[j+1] > 1e-10 and
                              e[j] >= 8*e[j+1] for j in range(3)) and
                      max(e) <= 2e-7 for e in rows)
        actual = report['convergence'][key]
        assert len(rows) == actual['rows'] == 220
        assert fourth == actual['fourth_order_evidence_rows']
        assert unknown == actual['within_tolerance_order_unclassified_rows']
        assert fourth + unknown == 220 and not actual['unresolved_rows']
        assert max(e[-1] for e in rows) == actual['last_h_max'] <= 2e-7
        convergence[key] = actual
    reconstructed = read(HERE / ('reconstructed-build-' + mode + '-latest.json'))
    assert reconstructed['receipt_sha256'] == sha(adir / 'receipt.json')
    assert reconstructed['executable_sha256'] == r['executable_sha256']
    assert sha(HERE / ('reconstructed-build-' + mode + '-latest.json')) == r['build_latest_sha256']
    builds.append({'mode': mode, 'compiler_dependencies': expected_deps,
                   'link_archives': 4, 'build_seconds': b['seconds'],
                   'gate_seconds': r['seconds'], 'convergence': convergence,
                   'historical_executable_sha256': r['executable_sha256'],
                   'accepted_executable_retained': False,
                   'fresh_accepted_executable_readback': False})
    outputs.append((gdir / 'stdout.json').read_bytes())
assert outputs[0] == outputs[1]
assert len(dep_union) == root_review['unique_dependency_and_link_inputs_verified'] == 1066
draft = (HERE / 'audit-draft.md').read_text()
assert '1060' in draft and '1062' in draft
assert 'preservation error' in draft and 'historical receipt' in draft
assert 'order unclassified' in draft
for path, metadata in pins.items():
    assert sha(ROOT / path) == metadata['sha256'], path
result = {
    'status': 'PASS_read_only_math_source_saved_output_and_provenance_review',
    'scientific_rerun_performed': False,
    'reviewer': '/root/literature_gauge',
    'review_HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                         text=True).strip(),
    'python_version': platform.python_version(),
    'review_script_sha256': sha(Path(__file__)),
    'input_pins': pins,
    'builds': builds,
    'unique_dependency_and_link_inputs_rehashed': len(dep_union),
    'numerical_stdout_byte_equal': True,
    'corrections': [],
    'resolved_during_review': ['Draft count corrected to Release1060/Debug1062 compiler dependencies, four link archives each, union1066.'],
    'math_source_review': [
        'Nested S1 over perturbation dual differentiates the eleven configuration equations only; consumed order is second configuration and first A, with no differentiated momentum RHS.',
        'Reference frame/coframe and all coefficient derivatives are retained; complete U,V normalization and the Aref raised-metric trace tangent subtraction are consistent with the stable chunk002 source review.',
        'The m0/single-oblique-ray/five-radius scope, final-h checks and distinct fourth-order versus within-tolerance-unclassified sequences are stated accurately.',
        'At r=.98 source-binding FD can exceed the later artificial rb; this is explicitly not the strict-inside-rb physical constraint-rate gate.',
        'Energy/SAT assembly, physical subsidiary rates, sector ranks and propagation remain unadmitted by this checkpoint.'
    ],
    'provenance_limitations': [
        'Both accepted binaries were overwritten. Executable hashes are historical receipt values; this review neither rehashes nor reconstructs those missing binaries.',
        'Additive build-pointer reconstructions match the accepted gate hashes; current builder pointers were neither edited nor rerun.',
        'Retained source, build/dependency and scientific-output records permit source/output review, not a fresh execution claim.'
    ],
    'preserved_histories': 'Mechanical compiler failure, selector-crossing derivative failure, and source-pin race remain retained separately.',
    'scope': 'Source/reduction checkpoint only; no radial operator, eigenvalue, PDE stability, finite-pulse or BH acceptance. Later wormhole-to-trumpet BH must retain the Minkowski hyperboloidal reference.'
}
OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps({'status': result['status'], 'receipt': str(OUT),
                  'sha256': sha(OUT), 'bytes': OUT.stat().st_size,
                  'unique_rehashed_inputs': len(dep_union)}))
