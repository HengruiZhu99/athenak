"""Cheap exact saved-preimage/shear readback; no rank or actual-kernel rerun."""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GATE = ROOT/'build-layer-research/continuum/q-curvature-tensor-freedoms/' \
    'immutable-Q-curvature-tensor-freedoms-20261009'
CUBIC = ROOT/'build-layer-research/continuum/q-null-curvature-ideal/' \
    'immutable-Q-Einstein-cubic-null-curvature-timejet-20261009'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rational(value):
    return F(str(value)).limit_denominator(1000000)


def targets(vector, a):
    get = lambda k: vector.get(k, F(0))
    h = [get(7)/2+get(10), get(11)]
    q1 = [-2*a*h[0]-a*get(27)/2-a*get(30),
          -2*a*h[1]-a*get(31)]
    A = [get(12)/2+get(15), get(16)]
    S = [(get(48)-get(69))/2, (get(49)+get(68))/2]
    return h, q1, A, S


assert sha(GATE/'index.json') == (
    'c16dbe6eb947a9c5f8710c297c042868c3dd74581742df9cf81786875316380d')
index = json.loads((GATE/'index.json').read_text())
for record in index['files']:
    path = GATE/record['path']
    assert sha(path) == record['sha256']
    assert path.stat().st_size == record['bytes']
saved = json.loads((GATE/'receipt.json').read_text())
assert saved['passed_finite_jet_tensor_freedom_probe']
assert saved['inputs_before'] == saved['inputs_after']
assert sha(CUBIC/'index.json') == (
    '0302a9da72f1a36f5fd6bf7dc9311fccc08c97ac1eace3282423a2ee38517f13')
assert sha(CUBIC/'actual-release.json') == (
    '1e01f0e8f4244daa7c0a0a40f91fe30673200e3d6c0d2637d182616012c1cd20')
actual = json.loads((CUBIC/'actual-release.json').read_text())
labels = {name: k for k, name in enumerate(actual['labels'])}
checked_preimages = checked_shear_columns = 0
for row, data in zip(saved['rows'], actual['radii']):
    a = rational(row['a'])
    assert a == rational(data['a'])
    assert row['compatible_frame_rank'] == 128
    assert row['compatible_frame_nullity'] == 272
    assert row['two_TF_first_normal_cut_metric_image_rank'] == 2
    assert row['clean_representative_extra_row_rank'] == 132
    columns = data['orientations'][0]['columns']
    for k, column in enumerate(columns):
        h, q1, A, S = targets({k: F(1)}, a)
        shear = [
            (rational(column['E'][labels['R0_12[0,0,0]']])
             + 2*rational(column['E'][labels['R0_15[0,0,0]']]))/2,
            rational(column['E'][labels['R0_16[0,0,0]']])]
        assert all(A[j]-q1[j]/(2*a)-2*h[j]-S[j] == -a*a*shear[j]/2
                   for j in range(2))
        checked_shear_columns += 1
    for family in ['preimages', 'clean_representatives']:
        for target, preimage in enumerate(row[family]):
            v = {k: F(value) for k, value in preimage['nonzero_coefficients']}
            for condition in range(246):
                assert sum(value*rational(columns[k]['E'][condition])
                           for k, value in v.items()) == 0
            assert sum(value*rational(columns[k]['delta_Rq'])
                       for k, value in v.items()) == 0
            h, q1, A, S = targets(v, a)
            assert q1 == [F(int(j == target)) for j in range(2)]
            assert A == list(map(F, preimage['A_boundary_TF']))
            if family == 'clean_representatives':
                assert h == S == [F(0), F(0)]
                assert A == [value/(2*a) for value in q1]
            else:
                assert h == list(map(F, preimage['h0_TF']))
                assert S == list(map(F, preimage['normal_frame_gradient_TF']))
            checked_preimages += 1

receipt = {
    'passed_independent_tensor_freedom_source_math_and_saved_result_review': True,
    'HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                    cwd=ROOT, text=True).strip(),
    'index_sha256': sha(GATE/'index.json'),
    'indexed_files_verified': len(index['files']),
    'reviewed_source_sha256': sha(GATE/'check_tensor.py'),
    'reviewed_report_sha256': sha(GATE/'REPORT.md'),
    'review_script_sha256': sha(Path(__file__)),
    'parent_cubic_index_sha256': sha(CUBIC/'index.json'),
    'parent_cubic_actual_sha256': sha(CUBIC/'actual-release.json'),
    'cheap_exact_readback': {
        'parameter_values': [.5, .75, 1., 2.],
        'shear_identity_columns': checked_shear_columns,
        'saved_preimages': checked_preimages,
        'all_selected_reconstructed_E_and_curvature_kernel_residuals_exact_zero': True,
        'targets_exact_identity_and_shear_relations_exact': True,
        'rank_proof': 'Two exact saved preimages of independent two-component '
        'targets establish image rank2 directly; no full rank/nullspace rerun.',
        'arithmetic': 'Python Fraction, same denominator<=1000000 '
        'reconstruction as the frozen source.'},
    'q1_definition': 'At fixed angular labels q_AB=r^2 bargamma_AB. '
    'With r^2=1-2a Omega, the linear first-normal coefficient is '
    'q1_TF=(bargamma_AB,1-2a bargamma_AB,0)^TF, using the fixed reference '
    'unit-cut metric at the base point.',
    'TF_scope': 'This is reference-TF of the coefficient. Linearizing '
    'a TF projection of q1 with respect to the perturbed live q0 instead '
    'gives q1_TF+2a h0_TF. No identification with physical radiation is made.',
    'explicit_S_components': {
        'plus': '(partial_y h_ny-partial_z h_nz)/2',
        'cross': '(partial_y h_nz+partial_z h_ny)/2',
        'tensor': '(partial_(A h_nB))^TF with unit-weight symmetrization'},
    'shear_identity': 'A0_TF-q1_TF/(2a)-2h0_TF-S=-a^2 R0_A_TF/2',
    'representative_scope': 'h0_TF=S=0 adds four independent rows only to '
    'choose transparent preimages; neither an extra necessary condition '
    'nor a boundary prescription or falloff. Their A0_TF=q1_TF/(2a) '
    'follows from the imposed shear identity.',
    'scientific_corrections': [],
    'notation_clarification': 'Frozen receipt wording .5(sym derivative) '
    'is ambiguous; the explicit S components above state the actual code '
    'and the usual unit-weight convention. Frozen bytes remain unchanged.',
    'scope': 'Two necessary finite Taylor-jet tensor freedoms only; no '
    'genuine radiative Weyl data, exact Einstein germ, hierarchy preservation, '
    'native/global evolution or source admission.',
    'kernel_or_evolution_rerun': False, 'frozen_mutation': False}
(HERE/'independent-tensor-review.json').write_text(
    json.dumps(receipt, indent=2, allow_nan=False)+'\n')
print(json.dumps({'passed': True, 'receipt_sha256':
                  sha(HERE/'independent-tensor-review.json')}, indent=2))
