"""Independent four saved-array comparisons; no assembler/kernel execution."""
from pathlib import Path
import hashlib
import json
import sys
import warnings

import numpy as np

warnings.simplefilter('error')
np.seterr(all='raise')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P = ROOT/'build-layer-research/boundary/total-j-finite-rb-control-20261009'
OUT = HERE/'J1-J2-quadrature-angular-readback.json'
assert not OUT.exists()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rec(path):
    return {'path': str(path.resolve()), 'sha256': sha(path),
            'bytes': path.stat().st_size}


def fro(a):
    return float(np.sqrt(np.sum(a*a)))


def error(a, b):
    return {'scaled': fro(a-b)/max(1., fro(a), fro(b)),
            'absolute': fro(a-b), 'maxabs': float(np.max(np.abs(a-b)))}


source_old = 'd28f652a641293f525c4f6a091dfd17edb97831bdc72d066f8368c78bfd99e0d'
source_blas = 'a3b140d7ef41bd5d83ffd9e03cba373f1c10f9486b07c815fd5d80c061926fd7'
equivalence = P/'BLAS-equivalence-report.json'
assert sha(equivalence) == '8dce9242db7bd5f810abc03d5a67180df6a3a39214f8e782b470ff1c47469d87'
review = HERE/'blas-contraction-review-001/result.json'
assert sha(review) == '2f26ffb1e2e192e084c5eae91d5f5c87fe9b9b974d9bbb96fc674c0931f7735e'
assert json.loads(review.read_text())['complete_array_count'] == 23
matrix_pins = {
    (1, 'radial'): ('afcbaa5da879a3730d4a0358cd940e9df7237efee63c8834ffea29eef677e500',
                    'ce0c07a08ecb9ea0f114c7f4b7b6cd87f1239b8eef6f1f595ef58d99440d4b39'),
    (1, 'angular'): ('ce0c07a08ecb9ea0f114c7f4b7b6cd87f1239b8eef6f1f595ef58d99440d4b39',
                     '0ea2cd92e96912eb0946948782d0b78e425474a656f814f418af7a83f2220a02'),
    (2, 'radial'): ('71a9d7ed88f0a678ca0dede9c7f6895013adb0cdc6e65076a818739a037a3694',
                    '9d559481bf9e54663ea1d62a49c1bee14eff26b20521f24ae29265bcbd68e087'),
    (2, 'angular'): ('9d559481bf9e54663ea1d62a49c1bee14eff26b20521f24ae29265bcbd68e087',
                     '4893c04cba599be97f291157337c91e1263f872ddbf291b2ae808b806bd82125')}
comparison_pins = {
    (1, 'radial'): 'e6662ff2732ee9e3c842d9fde124c45a3833eca2c5a344000e4a9d321ea897e5',
    (1, 'angular'): '634b6585c3bdef32dea1cf471f0c9915c4607a292f0e0535f1ea91a4146f9511',
    (2, 'radial'): 'e77ab92a06dedb8bac69c082660db49f631abbf807398d75d39f358ecc82abbe',
    (2, 'angular'): '1845afc742c3d574ced7ed1f430f5b27ab816fba91513cd14d52b8813f7ef71c'}
keys = ('E', 'Kweak', 'Kstrong', 'Gvolume', 'Fboundary', 'SATload',
        'Jbulk', 'Jsat', 'manufactured_load', 'manufactured_boundary_load')
rows = []
all_input_pins = [rec(equivalence), rec(review)]
for (j, kind), matrix_pair in matrix_pins.items():
    if kind == 'radial':
        names = [f'J{j}-N8-rb.98-segmentedQ{q}-a12x24-refinement001'
                 for q in (32, 64)]
        comparison_name = f'J{j}-segmented-quadrature-comparison.json'
    else:
        names = [f'J{j}-N8-rb.98-segmentedQ64-a12x24-refinement001',
                 f'J{j}-N8-rb.98-segmentedQ64-a16x32-BLAS-angular001']
        comparison_name = f'J{j}-angular-comparison.json'
    reports, arrays, pins = [], [], []
    for n, (name, pin) in enumerate(zip(names, matrix_pair)):
        folder = P/name
        assert sha(folder/'operator.npz') == pin
        report = json.loads((folder/'report.json').read_text())
        assert report['operator_sha256'] == pin
        is_blas = kind == 'angular' and n == 1
        source = folder/('assemble_blas.py' if is_blas else 'assemble_segmented.py')
        assert sha(source) == report['source_sha256'] == (source_blas if is_blas else source_old)
        for filename, key in [('energy_coefficients.py', 'coefficient_source_sha256'),
                              ('canceled_basis_complex.py', 'canceled_source_sha256')]:
            assert sha(folder/filename) == report[key]
        with np.load(folder/'operator.npz', allow_pickle=False) as z:
            arrays.append({k: z[k] for k in keys})
        reports.append(report)
        pins.extend(rec(p) for p in (folder/'operator.npz', folder/'report.json',
                                    source, folder/'energy_coefficients.py',
                                    folder/'canceled_basis_complex.py'))
    for key in ('J', 'N', 'rb', 'coefficient_source_sha256',
                'canceled_source_sha256', 'executable_sha256', 'quadrature_panels_r'):
        assert reports[0][key] == reports[1][key], key
    assert reports[0]['J'] == j and reports[0]['N'] == 8 and reports[0]['rb'] == .98
    if kind == 'radial':
        assert [r['Q_per_panel'] for r in reports] == [32, 64]
        assert all(r['angular_rule'] == [12, 24] for r in reports)
        assert reports[0]['source_sha256'] == reports[1]['source_sha256']
    else:
        assert [r['angular_rule'] for r in reports] == [[12, 24], [16, 32]]
        assert all(r['Q_per_panel'] == 64 for r in reports)
    actual = {}
    for key in keys:
        a, b = arrays[0][key], arrays[1][key]
        assert a.dtype == b.dtype == np.float64 and a.shape == b.shape
        assert np.isfinite(a).all() and np.isfinite(b).all()
        actual[key] = error(a, b)
    owner_path = P/comparison_name
    assert sha(owner_path) == comparison_pins[(j, kind)]
    owner = json.loads(owner_path.read_text())
    assert owner['passed'] and owner['threshold'] == 2e-8
    if kind == 'angular':
        assert owner['BLAS_equivalence_sha256'] == sha(equivalence)
    for key in keys:
        for metric, value in actual[key].items():
            saved = owner['rows'][key][metric]
            assert abs(value-saved) <= 3e-14*max(1., abs(value), abs(saved))
    pins.append(rec(owner_path))
    all_input_pins.extend(pins)
    rows.append({'J': j, 'kind': kind, 'inputs': pins,
                 'comparison_rows': actual,
                 'maximum_scaled': max(v['scaled'] for v in actual.values()),
                 'threshold': 2e-8,
                 'all_saved_owner_numbers_match': True,
                 'passed': all(v['scaled'] <= 2e-8 for v in actual.values()),
                 'assembler_identity': 'Same source for radial pair. Angular pair changes only the independently reviewed einsum-to-BLAS contraction and captured-source basename.'})
assert all(sha(Path(r['path'])) == r['sha256'] for r in all_input_pins)
result = {'status': 'PASS' if all(r['passed'] for r in rows) else 'FAIL_preserved',
          'source': rec(Path(__file__)), 'comparisons': rows,
          'BLAS_static_and_complete_saved_array_review': rec(review),
          'versions': {'python': sys.version, 'numpy': np.__version__},
          'scientific_kernel_or_assembler_run': False,
          'generator_spectrum_or_propagation': False,
          'scope': 'Saved quadrature/angular sensitivity for fixed J1/J2,N8,rb.98 trial spaces and the named energy-Galerkin/SAT problem. Neither degree/PDE convergence nor ghost-only attribution follows.'}
OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({'status': result['status'], 'result': rec(OUT),
                  'maxima': {f"J{r['J']}-{r['kind']}": r['maximum_scaled'] for r in rows}}, indent=2))
