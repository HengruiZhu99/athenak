"""Read back the saved scalar mathematical model; no scientific rerun."""
from pathlib import Path
import argparse
import difflib
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys

import numpy as np
import numpy.linalg
import scipy
import scipy.special

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
DEST = P / 'immutable-common-rho-dense-mass-scalar-model-20261009'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_json(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            finite_json(item)
    elif isinstance(value, list):
        for item in value:
            finite_json(item)


parser = argparse.ArgumentParser()
parser.add_argument('--freeze', action='store_true')
args = parser.parse_args()
plan = json.loads((P / 'plan.json').read_text())
result = json.loads((P / 'results.json').read_text())
finite_json(result)
assert result['plan_sha256'] == sha(P / 'plan.json')
assert result['source_sha256'] == sha(P / 'model_gate.py')
assert result['matrix_npz_sha256'] == sha(P / 'model-matrices.npz')
assert result['summary']['cases'] == len(result['results']) == 50
expected = {(n, l, rb) for n in plan['N_values'] for l in plan['L_values'] for rb in plan['rb_values']}
assert {(r['N'], r['L'], r['rb']) for r in result['results']} == expected
assert all(r['passed'] and all(r['checks'].values()) for r in result['results'])
assert result['summary']['passed_all_math_model_gates']
assert result['no_actual_Z4c_matrix_eigensolve_or_evolution']
assert result['L0_energy_is_seminorm_with_static_constant_mode']
assert not (P / 'run.stderr').read_bytes()
assert json.loads((P / 'run.stdout').read_text()) == result['summary']
with np.load(P / 'model-matrices.npz') as matrices:
    assert len(matrices.files) == 50 * 11
    assert all(np.isfinite(matrices[k]).all() for k in matrices.files)
    matrix_entries = sum(matrices[k].size for k in matrices.files)
    matrix_keys = list(matrices.files)
for row in result['results']:
    assert row['origin_flux']['at_zero'] == 0
    if row['L'] == 0:
        assert row['L0_constant']['E'] == 0
        assert row['L0_constant']['generator_max'] == 0
        assert row['L0_constant']['DL_constant_max'] == 0
    for field in ('mass_congruence', 'stiffness_congruence'):
        assert row[field]['scaled'] <= plan['tolerances']['modal_mass_identity_scaled']

assert sha(P / 'history/001-numpy-bool-serialization/model-matrices.npz') == sha(P / 'model-matrices.npz')
old_result = json.loads((P / 'history/002-before-strict-serialization/results.json').read_text())
assert old_result['matrix_npz_sha256'] == result['matrix_npz_sha256']
assert old_result['summary'] == result['summary']

maxima = {}
for key in ('nodal_IBP', 'modal_IBP', 'wave_energy_matrix', 'strong_weak_action',
            'common_collocation_action', 'mass_congruence', 'stiffness_congruence',
            'modal_mass_identity'):
    maxima[key] = {kind: max(row[key][kind] for row in result['results'])
                   for kind in ('absolute', 'scaled', 'absolute_max')}

verification = {
    'passed_saved_scalar_model_readback': True,
    'no_scientific_rerun_by_verifier': True,
    'cases': len(result['results']),
    'case_checks': sum(len(row['checks']) for row in result['results']),
    'all_json_finite': True,
    'all_saved_matrices_finite': True,
    'matrix_arrays': len(matrix_keys),
    'matrix_entries': matrix_entries,
    'source_plan_result_matrix_stdout_identities_passed': True,
    'first_failed_serialization_matrix_bytes_equal_accepted': True,
    'second_serialization_attempt_summary_matrix_bytes_equal_accepted': True,
    'supplementary_saved_mass_stiffness_congruence_below_declared_2e_minus_9': True,
    'maxima': maxima,
    'sampled_energy_rate_absolute_error_max': max(v['absolute_error'] for row in result['results'] for v in row['sampled_energy_rates']),
    'L0_static_constant_preserved_without_regularization': True,
    'actual_Z4c_radial_matrix_or_boundary_admitted': False,
}
(P / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')

# This is an import-time dependency inventory for reproducing the Python model,
# not a claim to have traced all dynamic loader/system-library dependencies.
files = {Path(sys.executable).resolve()}
for module in list(sys.modules.values()):
    name = getattr(module, '__file__', None)
    if name and Path(name).is_file():
        files.add(Path(name).resolve())
for package in (np, scipy):
    package_root = Path(package.__file__).resolve().parent
    for name in ('*.dylib', '*.so'):
        files.update(path.resolve() for path in package_root.rglob(name))
dependencies = [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}
                for path in sorted(files)]
source_diffs = {}
for attempt in ('001-numpy-bool-serialization', '002-before-strict-serialization'):
    previous = P / 'history' / attempt / 'model_gate.py'
    source_diffs[attempt] = ''.join(difflib.unified_diff(
        previous.read_text().splitlines(True), (P / 'model_gate.py').read_text().splitlines(True),
        fromfile=str(previous), tofile=str(P / 'model_gate.py')))
provenance = {
    'working_directory': str(ROOT),
    'launch_documentation_HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'git_status_porcelain_at_freeze': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
    'production_source_inclusions_or_native_build': None,
    'reproduction_command': 'OPENBLAS_NUM_THREADS=1 PYTHONPATH=build-layer-research/boundary/python-deps python3 build-layer-research/boundary/rho-dense-mass-model-20261009/model_gate.py',
    'reproduction_command_note': 'Canonical command for the accepted saved source. Raw accepted stdout/stderr are retained; internal calculation time is recorded in results.json.',
    'python_executable': sys.executable,
    'python': sys.version,
    'numpy': np.__version__,
    'scipy': scipy.__version__,
    'platform': platform.platform(),
    'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS'),
    'PYTHONPATH': os.environ.get('PYTHONPATH'),
    'runtime_import_and_installed_extension_inventory': dependencies,
    'dependency_inventory_scope': 'Loaded module files plus installed NumPy/SciPy .so/.dylib files; metadata-only, not a complete operating-system dynamic-loader trace.',
    'source_diffs_from_preserved_attempts': source_diffs,
}
(P / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')

if args.freeze:
    assert not DEST.exists(), 'Never overwrite an immutable receipt'
    candidates = [p for p in P.rglob('*') if p.is_file() and DEST not in p.parents and '__pycache__' not in p.parts]
    DEST.mkdir()
    records = []
    json_count = 0
    for source in sorted(candidates):
        rel = source.relative_to(P)
        target = DEST / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
        if target.suffix == '.json':
            finite_json(json.loads(target.read_text()))
            json_count += 1
        records.append({'path': str(rel), 'bytes': source.stat().st_size, 'sha256': sha(source)})
    index = {
        'scope': result['scope'],
        'passed_scalar_polynomial_model_gate': True,
        'actual_Z4c_radial_operator_boundary_or_evolution_admitted': False,
        'files': records,
        'small_file_count': len(records),
        'small_file_bytes': sum(r['bytes'] for r in records),
        'finite_json_count': json_count,
        'runtime_dependencies_metadata_only': len(dependencies),
        'source_sha256': sha(P / 'model_gate.py'),
        'plan_sha256': sha(P / 'plan.json'),
        'results_sha256': sha(P / 'results.json'),
        'report_sha256': sha(P / 'REPORT.md'),
        'all_copied_file_hashes_verified': True,
        'all_previous_serialization_attempts_preserved': True,
    }
    (DEST / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    for record in records:
        assert sha(DEST / record['path']) == record['sha256']
    print(json.dumps({'index': str(DEST / 'index.json'), 'sha256': sha(DEST / 'index.json'),
                      'files': len(records), 'bytes': index['small_file_bytes'],
                      'finite_json': json_count, 'passed': True}, indent=2))
else:
    print(json.dumps(verification, indent=2))
