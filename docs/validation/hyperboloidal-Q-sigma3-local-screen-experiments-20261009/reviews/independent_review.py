"""Read-only independent sigma3 source/draft and saved-result review.

No kernel execution, eigenvalue recomputation, native/global run or frozen edit.
"""
import hashlib
import json
import math
import pathlib
import subprocess
import numpy as np

ROOT = pathlib.Path('/Users/hz0693/research/hyperboloidal')
HERE = ROOT / 'build-layer-research/q-sigma3-stage'
BASES = [
    ROOT / 'build-layer-research/continuum/q-sigma3-frozen-fourier/immutable-Q-sigma3-raw22-intrinsic20-frozen-Fourier-20261009',
    ROOT / 'build-layer-research/continuum/q-sigma3-frozen-fourier-deterministic/immutable-Q-sigma3-deterministic-reanalysis-20261009',
]
INDEX_SHA = [
    '97eca136515520bb7a551b32013ce4faa7bafde0d3a079cee2620f8367e95aa3',
    '943db2465bb361802fdd237f7658f11f66f142ecff6d147920d0c8e9393a4077',
]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def finite(v):
    if isinstance(v, float):
        assert math.isfinite(v)
    elif isinstance(v, dict):
        for item in v.values():
            finite(item)
    elif isinstance(v, list):
        for item in v:
            finite(item)


def record(p):
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}


counts = []
receipts = []
for base, expected in zip(BASES, INDEX_SHA):
    assert sha(base / 'index.json') == expected
    index = json.loads((base / 'index.json').read_text())
    for name, row in index['files'].items():
        p = base / name
        assert sha(p) == row['sha256'], name
        assert p.stat().st_size == row['bytes'], name
        if p.suffix == '.json':
            finite(json.loads(p.read_text()))
    assert len(index['files']) == index['file_count']
    assert sum(v['bytes'] for v in index['files'].values()) == index['bytes']
    receipt = json.loads((base / 'receipt.json').read_text())
    assert receipt['source_before'] == receipt['source_after']
    for name, digest in receipt['source_before'].items():
        assert sha(ROOT / name) == digest, name
    receipts.append(receipt)
    counts.append({'index': record(base / 'index.json'),
                   'files': len(index['files']), 'bytes': index['bytes'],
                   'unchanged_source_inputs': len(receipt['source_before'])})
assert [d['files'] for d in counts] == [46, 13]
assert [d['unchanged_source_inputs'] for d in counts] == [381, 385]
assert [len(r['commands']) for r in receipts] == [5, 1]
assert all(c['returncode'] == 0 for r in receipts for c in r['commands'])
assert all((BASES[0] / c['stderr']).stat().st_size == 0
           for c in receipts[0]['commands'][:4])
assert (BASES[0] / receipts[0]['commands'][4]['stderr']).stat().st_size == 2433
assert (BASES[1] / receipts[1]['commands'][0]['stderr']).stat().st_size == 0
assert sha(BASES[0] / 'matrices-release.bin') == sha(BASES[0] / 'matrices-debug.bin')
assert (BASES[0] / 'metadata-release.json').read_bytes() == (
    BASES[0] / 'metadata-debug.json').read_bytes()

report = json.loads((BASES[1] / 'check-report.json').read_text())
summary = json.loads((HERE / 'summary.json').read_text())
finite(summary)
common = set(summary).intersection(report)
for key in common:
    assert summary[key] == report[key], key
assert summary['target_a0p5_intrinsic20_by_k'] == [
    row for row in report['worst_by_a_form_kind_k']
    if row['a'] == .5 and row['kind'] == 'intrinsic20'
    and row['form'] in [2, 3, 4, 5, 6]]
assert summary['target_a0p5_new_forms_sampled_bands'] == [
    row for row in report['sampled_bands_below_scalar_coordinate_Nyquist']
    if row['a'] == .5 and row['kind'] == 'intrinsic20'
    and row['form'] in [5, 6]]

# Saved eigenvalues only: no call to an eigensolver or matrix construction.
positive_counts = {}
roots_total = 0
original_roots = np.load(BASES[0] / 'roots.npz')
accepted_roots = np.load(BASES[1] / 'roots.npz')
for kind, dimension in [('raw22', 22), ('intrinsic20', 20),
                        ('value_only_RJB20', 20)]:
    z = accepted_roots[kind]
    assert z.shape == (1960, dimension) and np.isfinite(z).all()
    roots_total += z.size
    positive_counts[kind] = int(np.count_nonzero(np.max(z.real, axis=1) > 1e-8))
    assert np.array_equal(np.count_nonzero(z.real > 1e-8, axis=1),
                          np.count_nonzero(original_roots[kind].real > 1e-8, axis=1))
    if kind != 'value_only_RJB20':
        assert np.array_equal(z, original_roots[kind])
    for row in report['worst_by_a_form_kind_k']:
        if row['kind'] != kind:
            continue
        meta = json.loads((BASES[0] / 'metadata-release.json').read_text())['rows']
        indices = [i for i, d in enumerate(meta)
                   if all(d[k] == row[k] for k in ['a', 'form', 'k'])]
        assert float(np.max(z[indices].real)) == row['max_real']
assert positive_counts == report['positive_primitive_matrix_counts']
assert roots_total == 121520
meta = json.loads((BASES[0] / 'metadata-release.json').read_text())
assert len(meta['rows']) == 1960
assert all(d['Vlate'] == 0 for d in meta['lifts'] if d['r'] == .85)
assert all(d['W'] == 0 for d in meta['lifts'] if d['r'] == .45)
for form in [2, 5]:
    row = next(d for d in report['worst_by_a_form_kind']
               if d['kind'] == 'intrinsic20' and d['a'] == .5 and d['form'] == form)
    assert (row['r'], row['k'], row['dir'], row['max_real']) == (
        .85, 4, 0, 22.467615596732056)

review = {
    'status': 'PASS_independent_read_only_source_math_draft_saved_result_review',
    'reviewer': '/root/literature_gauge',
    'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                    text=True).strip(),
    'method': 'Independent source/formula review, hash verification and saved scalar/eigenvalue aggregate readback only.',
    'scientific_kernel_rerun': False,
    'eigensolver_reanalysis': False,
    'native_global_or_BH_evolution': False,
    'frozen_files_mutated': False,
    'verified_bundles': counts,
    'summary_common_fields_exact': len(common),
    'target_per_k_and_band_rows_exact': True,
    'saved_root_count': roots_total,
    'saved_primitive_positive_matrix_counts': positive_counts,
    'saved_raw22_intrinsic20_roots_byte_values_identical': True,
    'saved_positive_counts_identical_each_parameter_row': True,
    'reviewed_stage': {name: record(HERE / name) for name in
                       ['audit-draft.md', 'archive-README.md', 'summary.json',
                        'ROOT-NOTATION-ADDENDUM.md']},
    'source_findings': [
        'Actual C0 geometric source is called with 10/alpha and kappa2=0; P storage/evolution is untouched. Q variants use xi=1/a and preferred-source helper; physical-P source-off spatial-norm control is separately bound with rho=1.5.',
        'Raw22 independently seeds all primitive components. Intrinsic20 completes det(g)=1 and trace_g(A)=0 in every consumed jet via exact quotient jets; values-only R J22 B(x0) omits the coefficient-lift derivatives and is a distinct comparator.',
        'Real cosine and imaginary sine seeds reproduce exp(+ik n.x) with d=ikn and dd=-k^2 n n. Coordinate k is unweighted; Penrose k and physical spatial Omega*k_Penrose are separately recorded.',
        'The independently reconstructed sigma3-minus-sigma5 source is -2 w alpha^2/(Omega |dOmega|^2) times outer(dOmega,dNraw). It changes beta RHS value rows with alpha/chi/metric/beta value inputs; off-diagonal symmetric metric inputs carry factor two. It adds no principal derivatives within either sigma pair.',
        'There is no claim that Q/physical-P/global/physical-inner gauge forms all have equal principal matrices. Single beta-pole assembly and phase FD binding directly test the actual assembled sources.',
        'Late feedback vanishes at r=.85, leaving the target a=.5 k4 radial maximum exactly sigma-independent. Earlier sigma3 reduces some finite-k maxima but has worse sampled k0 and retains positive roots.',
        'The original C++ commands have empty stderr; the original fifth checker warnings and false clean-stderr prose are preserved. The additive warning-free checker uses unchanged actual payloads and thresholds; raw/chart roots are identical and values-only RJB roots are compared numerically without ordering or multiplicity claims.',
        'The exact compact summary and both frozen indexes support the draft counts, equations, timings, error scales and negative comparison. Scalar pi/h bands are selections, not a native discrete-symbol/accuracy or oblique Nyquist certificate.'
    ],
    'preserved_prose_clarifications': [
        {'location': 'original frozen DERIVATION.md line 29',
         'original': 'Only beta value columns change',
         'correction': 'Only beta RHS rows change, depending on alpha/chi/metric/beta value columns. Frozen source, equations and checker are correct and unchanged; ROOT-NOTATION-ADDENDUM.md records this.'},
        {'location': 'reviewed audit-draft.md source/control gate paragraph',
         'original': 'physical-P core identities',
         'correction': 'These sampled identities are at r=.45, where W_gauge=0 inside the geometric transition, not the exact Cauchy core r<=.05. Public copy should say W_gauge=0 inner-gauge branch identities; original reviewed draft may remain byte-exact.'}
    ],
    'blocking_scientific_corrections': [],
    'scope': 'Local finite-Omega frozen primitive negative comparison only. No subsidiary-mode classification, continuum/global/native instability theorem, uniform energy, finite-Q amplitude blowup, full nonlinear/Einstein-germ/radiative hierarchy, or stable pulse/BH admission follows. Later BH still requires wormhole-to-trumpet transition with the Minkowski hyperboloidal reference.',
    'source': record(pathlib.Path(__file__)),
}
out = HERE / 'independent-draft-review.json'
assert not out.exists(), 'Use a fresh additive review rather than overwrite accepted bytes.'
out.write_text(json.dumps(review, indent=2, allow_nan=False) + '\n')
print(json.dumps({'status': review['status'], 'receipt': record(out),
                  'counts': counts, 'roots': roots_total,
                  'positive_counts': positive_counts}, indent=2))
