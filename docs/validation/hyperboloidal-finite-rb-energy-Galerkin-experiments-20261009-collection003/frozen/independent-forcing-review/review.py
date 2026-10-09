"""Read-only source/hash/saved-array review; never import or run assembler."""
from pathlib import Path
import ast
import hashlib
import json
import math
import shutil
import sys
import time
import warnings

import numpy as np
from scipy.special import eval_jacobi, roots_jacobi

warnings.simplefilter('error')
np.seterr(all='raise')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P = ROOT/'build-layer-research/boundary/total-j-finite-rb-control-20261009'
OUT = HERE/'receipt.json'
assert not OUT.exists()
START = time.monotonic()
CACHE = {}


def sha(path):
    path = Path(path).resolve()
    if path not in CACHE:
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8*1024*1024), b''):
                h.update(block)
        CACHE[path] = h.hexdigest()
    return CACHE[path]


def rec(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': sha(path),
            'bytes': path.stat().st_size}


def checked(record):
    path = Path(record['path'])
    assert path.stat().st_size == record['bytes']
    assert sha(path) == record['sha256']
    return rec(path)


def norm(a):
    return float(np.sqrt(np.sum(a*a)))


def error(a, b):
    return {'scaled': norm(a-b)/max(1., norm(a), norm(b)),
            'absolute': norm(a-b), 'absolute_max': float(np.max(np.abs(a-b)))}


def mm(a, b):
    return np.einsum('ik,kj->ij', a, b, optimize=False)


def compare(actual, saved):
    for key, value in actual.items():
        assert abs(value-saved[key]) <= 3e-14*max(1., abs(value), abs(saved[key]))


source = P/'replay_forcing_family.py'
source_pin = '34fb0f7a6d1ae23a54a5dfc8a18e8b394c2c29462a272cf0fe3a7a93b19fd6f4'
assert sha(source) == source_pin
tree = ast.parse(source.read_text())
calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
assert not any('subprocess' in c or c in ('a.assemble', 'a.run_queries', 'a.reference_rows') for c in calls)
assert {'a.fit_source', 'a.coefficients', 'a.modal', 'a.angle_work'}.issubset(calls)
dependency_files = {
    'assemble_blas.py': P/'assemble_blas.py',
    'energy_coefficients.py': P/'energy_coefficients.py',
    'canceled_basis_complex.py': P/'canceled_basis_complex.py',
    'basis-data.json': P/'inputs/basis/basis-data.json',
    'angular-plan.json': P.parent/'total-j-local-angular-20261009/immutable-C0-spatialnorm-total-J-local-angular-20261009/angular-plan.json',
    'core-envelope-blocks.npz': P.parent/'total-j-flat-core-envelope-20261009/immutable-total-J-flat-core-envelope-20261009/core-envelope-blocks.npz'}
expected = ['a3b140d7ef41bd5d83ffd9e03cba373f1c10f9486b07c815fd5d80c061926fd7',
            'eb15ff31262e5907d2825c357f055c1f8c0fc3c189f1fe8f7b51e0b0cf0c3e40',
            'a5cc08bff29916f39358602c25842c20e98a86205437ffdf7bd6107e9b10f33b',
            'ddc8c8d91d211aad70a4df4560a75c32807417248abdf025663ad411cfd4e738',
            '91ef3546718754944334536b6881a7de82ee7d23c75ba0e1e15f8e2c611c8b14',
            '379316edf52fd6d137862d313b9edd7977dc58e4ee34afc77212605ff24e3858']
for path, digest in zip(dependency_files.values(), expected):
    assert sha(path) == digest
layout = json.loads(dependency_files['basis-data.json'].read_text())['channel_layouts']
angles = json.loads(dependency_files['angular-plan.json'].read_text())
nfit = len(angles['fit_directions'])+len(angles['heldout_directions'])
report_pins = ['77067f75469f1475a70cf2964b3baf96471652e44ae227445a761da445326978',
               '5f92214cf21585e45c2a579b65808e75b0c60325bcdf09d3ec5869248411715d',
               '44b05f66652393ad6180efd77e37856066fa0ae09633c77845d9f7054d429935']
rows = []
all_pins = [rec(source)]+[rec(p) for p in dependency_files.values()]
for j, digest in enumerate(report_pins):
    folder = P/f'J{j}-N8-forcing-family-replay001'
    report_path = folder/'report.json'
    assert sha(report_path) == digest and sha(folder/'replay_forcing_family.py') == source_pin
    report = json.loads(report_path.read_text())
    plan = json.loads((folder/'plan.json').read_text())
    execution = json.loads((folder/'execution-readback.json').read_text())
    assert report['source_sha256'] == source_pin and execution['source_sha256'] == source_pin
    assert execution['original_report_sha256'] == digest and execution['observed_exec_tool_exit_code'] == 0
    assert not execution['pointwise_kernel_queries_performed'] and execution['exact_launch_wall_time_not_recorded']
    assert plan['coefficient_error_scaled_max'] == 2e-9 and plan['mixed_load_readback_scaled_max'] == 5e-11
    assert plan['pointwise_fit_scaled_max'] == 5e-11
    assert report['J'] == j and report['N'] == 8 and report['rb'] == .98 and report['passed']
    input_pins = [checked(r) for r in report['input_pins']]
    matrix_path = Path(input_pins[0]['path'])
    matrix_folder = matrix_path.parent
    matrix_report = json.loads((matrix_folder/'report.json').read_text())
    assert matrix_report['passed_single_quadrature_algebra'] and matrix_report['Q_per_panel'] == 64
    sr = json.loads((matrix_folder/'source/receipt.json').read_text())
    ir = json.loads((matrix_folder/'input/receipt.json').read_text())
    rr = json.loads((matrix_folder/'reference-receipt.json').read_text())
    with np.load(matrix_path, allow_pickle=False) as archive:
        matrix = {k: archive[k] for k in ('E', 'Kweak', 'Kstrong', 'SATload', 'manufactured_X',
                  'manufactured_load', 'manufactured_boundary_load', 'radial_rho',
                  'radial_weights', 'angular_directions', 'angular_weights')}
    nc = len(layout[str(j)])
    assert nc == (8, 16, 20)[j] and report['family_count'] == 4*nc+1
    radii = np.r_[np.sqrt(matrix['radial_rho']), .98]
    payload = ''.join(format(float(r), '.17g')+'\n' for r in radii).encode()
    assert hashlib.sha256(payload).hexdigest() == rr['query_sha256']
    assert sr['rows'] == len(radii)*nfit*nc*3
    assert ir['rows'] == len(radii)*len(matrix['angular_directions'])*nc*3
    for receipt, mode in ((sr, '--source-batch'), (ir, '--input-binary')):
        assert receipt['exit_code'] == 0 and receipt['stderr_bytes'] == 0
        assert receipt['mode'] == mode
        assert receipt['executable_sha256'] == matrix_report['executable_sha256']
    assert rr['exit_code'] == 0 and rr['executable_sha256'] == matrix_report['executable_sha256']
    assert (matrix_folder/'input/output.bin').stat().st_size == ir['rows']*50*8
    assert len(report['fits']) == len(radii)
    fit_max = max(v['scaled'] for fit in report['fits'] for v in fit.values()
                  if isinstance(v, dict) and 'scaled' in v)
    assert fit_max <= 5e-11
    array_path = folder/'forcing-family.npz'
    assert sha(array_path) == report['array_sha256']
    with np.load(array_path, allow_pickle=False) as archive:
        saved = {k: archive[k] for k in archive.files}
    assert set(saved) == {'family_X', 'pointwise_manufactured_load', 'incoming_boundary_load', 'solved', 'difference'}
    assert all(a.shape == (nc*8, 4*nc+1) and a.dtype == np.float64 and np.isfinite(a).all()
               for a in saved.values())
    x = saved['family_X']
    assert np.array_equal(saved['difference'], saved['solved']-x)
    common = (roots_jacobi(8, 0, .5)[0]+1)*(.98**2)/2
    held = np.linspace(0., .98**2, 25)
    polynomial_error = 0.
    for channel, description in enumerate(layout[str(j)]):
        ell = description['L']
        normalization = np.sqrt(2*(2*np.arange(8)+ell+1.5)/(.98**2)**(ell+1.5))
        for degree in range(4):
            k = 4*channel+degree
            label = report['cases'][k]
            assert label['channel'] == channel and label['name'] == description['name']
            assert label['L'] == ell and label['polynomial_rho_degree'] == degree
            outside = x[:, k].copy();outside[channel*8:(channel+1)*8] = 0.
            assert np.count_nonzero(outside) == 0
            for rho in (common, held):
                v = np.column_stack([normalization[n]*eval_jacobi(n, 0, ell+.5, 2*rho/.98**2-1)
                                     for n in range(8)])
                polynomial_error = max(polynomial_error, error(v@x[channel*8:(channel+1)*8, k], rho**degree)['scaled'])
    assert polynomial_error <= 2e-13
    assert report['cases'][-1]['mixed'] and np.array_equal(x[:, -1], matrix['manufactured_X'])
    for k, case in enumerate(report['cases']):
        compare(error(saved['solved'][:, k], x[:, k]), case['coefficient_error'])
        delta = saved['difference'][:, k]
        e = math.sqrt(float(np.sum(delta*np.einsum('ij,j->i', matrix['E'], delta, optimize=False))))
        e0 = math.sqrt(float(np.sum(x[:, k]*np.einsum('ij,j->i', matrix['E'], x[:, k], optimize=False))))
        assert abs(e-case['energy_error_absolute']) <= 3e-14*max(1., e)
        assert abs(e/max(1., e0)-case['energy_error_scaled']) <= 3e-14
        assert case['passed'] and case['coefficient_error']['scaled'] <= 2e-9
    for key, actual in [('X', error(x[:, -1], matrix['manufactured_X'])),
                        ('pointwise_load', error(saved['pointwise_manufactured_load'][:, -1], matrix['manufactured_load'])),
                        ('incoming_load', error(saved['incoming_boundary_load'][:, -1], matrix['manufactured_boundary_load']))]:
        compare(actual, report['mixed_readback'][key])
        assert actual['scaled'] <= 5e-11
    rhs = mm(matrix['Kweak'], x)+mm(matrix['SATload'], x)+saved['pointwise_manufactured_load']+saved['incoming_boundary_load']
    independent_solution = np.linalg.solve(matrix['E'], rhs)
    solved_error = max(error(independent_solution[:, k], x[:, k])['scaled'] for k in range(4*nc+1))
    assert solved_error <= 2e-9
    backward = error(mm(matrix['E'], saved['solved']), rhs)
    assert backward['scaled'] <= 2e-13
    report_max = max(c['coefficient_error']['scaled'] for c in report['cases'])
    assert report_max == report['max_coefficient_error_scaled']
    row = {'J': j, 'family_count': 4*nc+1, 'report': rec(report_path), 'arrays': rec(array_path),
           'plan': rec(folder/'plan.json'), 'execution_readback': rec(folder/'execution-readback.json'),
           'input_pins': input_pins, 'fixed_family_coverage_complete': True,
           'held_and_common_polynomial_evaluation_error': polynomial_error,
           'all_saved_case_metrics_match': True, 'point_source_fit_max_scaled': fit_max,
           'maximum_saved_coefficient_error_scaled': report_max,
           'maximum_saved_energy_error_scaled': report['max_energy_error_scaled'],
           'independent_saved_load_solve_max_coefficient_error_scaled': solved_error,
           'saved_solution_backward_error': backward, 'passed': True}
    rows.append(row)
    all_pins.extend(input_pins+[row[k] for k in ('report', 'arrays', 'plan', 'execution_readback')])

assert sum(row['family_count'] for row in rows) == 179
copy_dir = HERE/'reviewed-sources'
copy_dir.mkdir(exist_ok=False)
for name, path in {'replay_forcing_family.py': source, **dependency_files}.items():
    shutil.copyfile(path, copy_dir/name)
    assert sha(copy_dir/name) == sha(path)
result = {'status': 'PASS_read_only_source_binding_coverage_and_saved_array_review',
          'reviewer': '/root/literature_gauge', 'source': rec(Path(__file__)),
          'reviewed_replay_source': rec(source), 'dependencies': [rec(p) for p in dependency_files.values()],
          'total_fixed_fields': 179, 'rows': rows, 'seconds': time.monotonic()-START,
          'source_math_findings': [
              'Each channel independently receives W=1,rho,rho^2,rho^3 in the N8 modal envelope; all other channels are exactly zero. The saved alternating mixed cubic is an additional control.',
              'At each cached quadrature point the prescribed input maps and fitted raw22-plus-configuration-derivative action produce Phi and L_actual Phi. Volume forcing is their difference before integration; E/Kweak/Kstrong/SATload do not define that load.',
              'The c*d_r configuration RHS map includes derivative-of-envelope output rows. This retains q=c*d_r U rather than independently evolving q, and the underlying source-normalization/input maps retain the reference metric/A trace tangent.',
              'The positive incoming load rb^2*k_in integral Y_test^T H Pplus Y_prescribed is independently built at rb and cancels the homogeneous negative adjoint SAT on the manufactured field. Radial weights divided by c and boundary rb^2 carry the declared physical measures.',
              'Only after those point loads are constructed does the replay solve E*Xdot=Kweak*X+SATload*X+loads; Xdot=X is the manufactured exponential-time target.',
              'All source/output/input/reference hashes and receipts match the named matrix cache, including zero exit/stderr and executable identity. Imported basis/angular/core data are pinned additively here.',
              'All saved finite arrays and 179 case metrics were read back independently; common and held-node polynomial evaluation verifies intended family coverage. No point-source contractions or matrix assembly were repeated.'
          ],
          'execution_provenance_limit': 'Owner execution-readback records are explicitly additive: observed tool exit and report duration are preserved, but exact launch wall-clock time was not recorded.',
          'corrections': [], 'scientific_kernel_or_assembler_run': False,
          'generator_eigensolve_or_propagation': False,
          'scope': 'Closes the fixed N8 prescribed forcing-family coverage gap only. No general independent continuum-action equivalence, constraint preservation, CPBC, uniform stability, exact scri, finite-pulse or BH acceptance follows. Earlier mixed-only readback freeze remains unchanged.'}
OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({'status': result['status'], 'receipt': rec(OUT),
                  'total_fixed_fields': 179, 'seconds': result['seconds'],
                  'max_errors': {str(r['J']): r['maximum_saved_coefficient_error_scaled'] for r in rows}}, indent=2))
