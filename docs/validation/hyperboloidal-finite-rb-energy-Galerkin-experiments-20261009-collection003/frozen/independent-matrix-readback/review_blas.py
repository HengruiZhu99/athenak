"""Static contraction-only source diff and saved complete-array readback."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import warnings

import numpy as np

warnings.simplefilter('error')
np.seterr(all='raise')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P = ROOT/'build-layer-research/boundary/total-j-finite-rb-control-20261009'
OUT = HERE/'blas-contraction-review-001'
assert not OUT.exists()
OUT.mkdir()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rec = lambda p: {'path': str(p.resolve()), 'sha256': sha(p), 'bytes': p.stat().st_size}
fro = lambda a: float(np.sqrt(np.sum(a*a)))
old = P/'J0-N8-rb.98-segmentedQ64-a12x24-refinement001'
new = P/'J0-N8-rb.98-segmentedQ64-a12x24-BLAS-equivalence001'
old_source, new_source = old/'assemble_segmented.py', new/'assemble_blas.py'
assert sha(old_source) == 'd28f652a641293f525c4f6a091dfd17edb97831bdc72d066f8368c78bfd99e0d'
assert sha(new_source) == 'a3b140d7ef41bd5d83ffd9e03cba373f1c10f9486b07c815fd5d80c061926fd7'
before = {p: sha(p) for p in (old_source, new_source, P/'BLAS-equivalence-report.json',
                              P/'BLAS-equivalence-plan.json', P/'check_blas_equivalence.py')}


class Filename(ast.NodeTransformer):
    def visit_Constant(self, node):
        if node.value == 'assemble_segmented.py':
            node.value = 'assemble_blas.py'
        return node


old_tree = Filename().visit(ast.parse(old_source.read_text()))
new_tree = ast.parse(new_source.read_text())
for tree in (old_tree, new_tree):
    angle = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == 'angle_work']
    assert len(angle) == 1
    tree.body.remove(angle[0])
assert ast.dump(old_tree, include_attributes=False) == ast.dump(new_tree, include_attributes=False)
diff = ''.join(difflib.unified_diff(old_source.read_text().splitlines(True),
                                  new_source.read_text().splitlines(True),
                                  fromfile=str(old_source), tofile=str(new_source)))
(OUT/'source.diff').write_text(diff)
saved = json.loads((P/'BLAS-equivalence-report.json').read_text())
assert saved['passed'] and saved['finite_complete_arrays'] and saved['rank_equal']
assert saved['threshold'] == 2e-9
assert sha(P/'check_blas_equivalence.py') == saved['source_sha256']
expected_npz = ('2ed0da45a995669f7e3e2fedba231eeda0dcb06f43577125d4a194974c4f4742',
                '1a420c90e12c8071613d9217271e3af6c1f8d2f9760c4c4518899da7f0d82282')
data = []
reports = []
for folder, expected in zip((old, new), expected_npz):
    assert sha(folder/'operator.npz') == expected
    report = json.loads((folder/'report.json').read_text())
    assert report['operator_sha256'] == expected and report['passed_single_quadrature_algebra']
    assert report['observed_incoming_rank'] == 4
    with np.load(folder/'operator.npz', allow_pickle=False) as z:
        data.append({k: z[k] for k in z.files})
    reports.append(report)
assert data[0].keys() == data[1].keys() == saved['rows'].keys()
for key in ('J', 'N', 'rb', 'Q_per_panel', 'quadrature_panels_r', 'angular_rule',
            'coefficient_source_sha256', 'canceled_source_sha256', 'executable_sha256'):
    assert reports[0][key] == reports[1][key], key
rows = {}
for key in data[0]:
    a, b = data[0][key], data[1][key]
    assert a.dtype == b.dtype == np.float64 and a.shape == b.shape
    assert np.isfinite(a).all() and np.isfinite(b).all()
    delta = a-b
    actual = {'scaled': fro(delta)/max(1., fro(a), fro(b)),
              'absolute': fro(delta), 'maxabs': float(np.max(np.abs(delta)))}
    for metric, value in actual.items():
        assert abs(value-saved['rows'][key][metric]) <= 3e-14*max(1., abs(value))
    assert actual['scaled'] <= 2e-9
    rows[key] = actual
assert data[0]['B'].tobytes() == data[1]['B'].tobytes()
assert data[0]['incoming_singular_values'].tobytes() == data[1]['incoming_singular_values'].tobytes()
assert all(sha(path) == digest for path, digest in before.items())
result = {
    'status': 'PASS_static_contraction_diff_and_complete_saved_arrays',
    'source': rec(Path(__file__)),
    'input_pins': [rec(p) for p in before]+[rec(d/'operator.npz') for d in (old, new)]
                  +[rec(d/'report.json') for d in (old, new)],
    'AST_diff_scope': 'Only angle_work body and source-copy basename differ; all other AST statements are identical.',
    'contraction_math': 'Flattened (angle,component) dot gives sum_a,f A[a,f,i]*w[a]*B[a,f,j], exactly the original afi,a,afj einsum over real arrays. All integrands, cutoffs, fits, trial functions, quadrature, weak/G/SAT/forcing expressions and thresholds are unchanged.',
    'complete_array_count': len(rows), 'rows': rows,
    'maximum_scaled_difference': max(v['scaled'] for v in rows.values()),
    'B_and_incoming_singular_values_bitwise_unchanged': True,
    'saved_full_equivalence_report_numbers_match': True,
    'threshold': 2e-9,
    'scientific_assembler_or_point_kernel_run': False,
    'generator_spectrum_or_propagation': False,
    'scope': 'Contraction-only numerical equivalence at fixed J0,N8,rb.98 and the saved rule. No generic platform equivalence, new PDE/CPBC/stability or higher-J matrix admission follows.'}
(OUT/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({'status': result['status'], 'result': rec(OUT/'result.json'),
                  'complete_arrays': len(rows), 'maximum_scaled': result['maximum_scaled_difference']}, indent=2))
