"""Independent saved-array quadrature/angle comparisons; no kernel calls."""
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
P = ROOT / 'build-layer-research/boundary/total-j-finite-rb-control-20261009'
OUT = HERE / 'quadrature-angular-readback.json'
assert not OUT.exists()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rec(path):
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def fro(a):
    return float(np.sqrt(np.sum(a*a)))


def error(a, b):
    return {'scaled': fro(a-b)/max(1., fro(a), fro(b)),
            'absolute': fro(a-b), 'maxabs': float(np.max(np.abs(a-b)))}


names = (
    ('segmented-radial-32-64',
     'J0-N8-rb.98-segmentedQ32-a12x24-refinement001',
     '1eef30acecef90bb7b8f852524562b4273765b7f6713a99bb1bd9b5caead15dc',
     'J0-N8-rb.98-segmentedQ64-a12x24-refinement001',
     '2ed0da45a995669f7e3e2fedba231eeda0dcb06f43577125d4a194974c4f4742',
     'J0-segmented-quadrature-comparison.json'),
    ('angular-12x24-16x32',
     'J0-N8-rb.98-segmentedQ64-a12x24-refinement001',
     '2ed0da45a995669f7e3e2fedba231eeda0dcb06f43577125d4a194974c4f4742',
     'J0-N8-rb.98-segmentedQ64-a16x32-angular001',
     '8a5e9be63e323ddd08862eedad96400fdee21ebee873440c6ade47b4bc848117',
     'J0-angular-comparison.json'),
)
keys = ('E', 'Kweak', 'Kstrong', 'Gvolume', 'Fboundary', 'SATload',
        'Jbulk', 'Jsat', 'manufactured_load', 'manufactured_boundary_load')
rows = []
for label, a_name, a_pin, b_name, b_pin, old_name in names:
    dirs = [P/a_name, P/b_name]
    arrays = []
    reports = []
    pins = []
    for folder, pin in zip(dirs, (a_pin, b_pin)):
        assert sha(folder/'operator.npz') == pin
        report = json.loads((folder/'report.json').read_text())
        assert report['operator_sha256'] == pin
        assert sha(folder/'assemble_segmented.py') == report['source_sha256']
        assert sha(folder/'energy_coefficients.py') == report['coefficient_source_sha256']
        assert sha(folder/'canceled_basis_complex.py') == report['canceled_source_sha256']
        with np.load(folder/'operator.npz', allow_pickle=False) as z:
            arrays.append({k: z[k] for k in keys})
        reports.append(report)
        pins.append({'matrix': rec(folder/'operator.npz'),
                     'report': rec(folder/'report.json'),
                     'assembler_source': rec(folder/'assemble_segmented.py'),
                     'energy_source': rec(folder/'energy_coefficients.py'),
                     'canceled_source': rec(folder/'canceled_basis_complex.py')})
    for key in ('J', 'N', 'rb', 'source_sha256', 'coefficient_source_sha256',
                'canceled_source_sha256', 'executable_sha256'):
        assert reports[0][key] == reports[1][key], key
    if label.startswith('segmented'):
        assert [r['Q_per_panel'] for r in reports] == [32, 64]
        assert reports[0]['angular_rule'] == reports[1]['angular_rule'] == [12, 24]
    else:
        assert [r['angular_rule'] for r in reports] == [[12, 24], [16, 32]]
        assert reports[0]['Q_per_panel'] == reports[1]['Q_per_panel'] == 64
    assert reports[0]['quadrature_panels_r'] == reports[1]['quadrature_panels_r']
    actual = {k: error(arrays[0][k], arrays[1][k]) for k in keys}
    old = json.loads((P/old_name).read_text())
    assert old['passed'] and old['threshold'] == 2e-8
    for key, values in old['rows'].items():
        for metric in ('scaled', 'absolute', 'maxabs'):
            tolerance = 3e-14*max(1., abs(values[metric]), abs(actual[key][metric]))
            assert abs(actual[key][metric]-values[metric]) <= tolerance
    rows.append({'name': label, 'inputs': pins,
                 'source_execution_identity': 'Same captured assembler, energy coefficients, canceled basis and historical actual-kernel executable hashes.',
                 'owner_comparison': rec(P/old_name),
                 'all_owner_saved_comparison_numbers_match': True,
                 'expanded_comparison_rows': actual,
                 'maximum_scaled': max(v['scaled'] for v in actual.values()),
                 'threshold': 2e-8,
                 'passed': all(v['scaled'] <= 2e-8 for v in actual.values())})
result = {'status': 'PASS' if all(r['passed'] for r in rows) else 'FAIL_preserved',
          'source': rec(Path(__file__)), 'comparisons': rows,
          'versions': {'python': sys.version, 'numpy': np.__version__},
          'scientific_kernel_or_assembler_run': False,
          'generator_spectrum_or_propagation': False,
          'scope': 'Saved integration-rule sensitivity for fixed J0,N8,rb.98 and the named unchanged energy-Galerkin/SAT problem; not degree/PDE convergence or a ghost-closure attribution.'}
OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({'status': result['status'], 'result': rec(OUT),
                  'maxima': {r['name']: r['maximum_scaled'] for r in rows}}, indent=2))
