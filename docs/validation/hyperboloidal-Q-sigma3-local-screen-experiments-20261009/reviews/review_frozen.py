"""Read-only root check of completed sigma3 artifacts and exact draft summary."""
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def finite(x):
    if isinstance(x, float):
        assert math.isfinite(x)
    elif isinstance(x, dict):
        for v in x.values():
            finite(v)
    elif isinstance(x, list):
        for v in x:
            finite(v)


bases = [
    ROOT/'build-layer-research/continuum/q-sigma3-frozen-fourier/immutable-Q-sigma3-raw22-intrinsic20-frozen-Fourier-20261009',
    ROOT/'build-layer-research/continuum/q-sigma3-frozen-fourier-deterministic/immutable-Q-sigma3-deterministic-reanalysis-20261009',
]
expected = [
    '97eca136515520bb7a551b32013ce4faa7bafde0d3a079cee2620f8367e95aa3',
    '943db2465bb361802fdd237f7658f11f66f142ecff6d147920d0c8e9393a4077',
]
counts = []
for base, digest in zip(bases, expected):
    assert sha(base/'index.json') == digest
    index = json.loads((base/'index.json').read_text())
    finite(index)
    for name, spec in index['files'].items():
        p = base/name
        assert sha(p) == spec['sha256']
        assert p.stat().st_size == spec['bytes']
        if p.suffix == '.json':
            finite(json.loads(p.read_text()))
    receipt = json.loads((base/'receipt.json').read_text())
    assert receipt['source_before'] == receipt['source_after']
    for name, digest in receipt['source_before'].items():
        assert sha(ROOT/name) == digest, name
    counts.append({'frozen_files': len(index['files']),
                   'unchanged_source_inputs': len(receipt['source_before'])})

original = json.loads((bases[0]/'receipt.json').read_text())
accepted = json.loads((bases[1]/'receipt.json').read_text())
assert len(original['commands']) == 5 and len(accepted['commands']) == 1
assert all(c['returncode'] == 0 for c in original['commands']+accepted['commands'])
assert all((bases[0]/c['stderr']).stat().st_size == 0 for c in original['commands'][:4])
assert (bases[0]/original['commands'][4]['stderr']).stat().st_size == 2433
assert (bases[1]/accepted['commands'][0]['stderr']).stat().st_size == 0
assert accepted['positive_root_counts_identical_every_parameter_row']

report = json.loads((bases[1]/'check-report.json').read_text())
summary = json.loads((HERE/'summary.json').read_text())
finite(summary)
common = set(summary)&set(report)
for k in common:
    assert summary[k] == report[k], k
target = [d for d in report['worst_by_a_form_kind_k']
          if d['a'] == .5 and d['kind'] == 'intrinsic20' and d['form'] in [2,3,4,5,6]]
assert summary['target_a0p5_intrinsic20_by_k'] == target
band = [d for d in report['sampled_bands_below_scalar_coordinate_Nyquist']
        if d['a'] == .5 and d['kind'] == 'intrinsic20' and d['form'] in [5,6]]
assert summary['target_a0p5_new_forms_sampled_bands'] == band
for form in [2,5]:
    row = next(d for d in report['worst_by_a_form_kind']
               if d['a'] == .5 and d['kind'] == 'intrinsic20' and d['form'] == form)
    assert row['r'] == .85 and row['k'] == 4
    assert row['max_real'] == 22.467615596732056
meta = json.loads((bases[0]/'metadata-release.json').read_text())
assert all(d['Vlate'] == 0 for d in meta['lifts'] if d['r'] == .85)

review = {
    'kind': 'root-read-only-actual-sigma3-source-and-summary-review',
    'frozen_index_sha256': expected,
    'verified_inputs': counts,
    'common_summary_fields_exact': len(common),
    'selected_per_k_and_band_rows_exact': True,
    'original_warning_and_additive_correction_verified': True,
    'mathematics_checked': [
        'Independent delta N and factor-two symmetric metric columns; sigma3-minus5 beta value feedback rank one with factor -2.',
        'Actual raw22 source binding and all-consumed-jet det/trace intrinsic20 completion; values-only RJB is a distinct comparator.',
        'Complex phase sign from real/imaginary seeds, actual physical-P geometry and kappa1=10/alpha normalization.',
        'Single beta-pole assembly, inherited early/late helpers and no new principal derivatives from changing sigma.',
        'Late r=.85 feedback zero, positive primitive roots retained without constraint-growth or stability inference.',
    ],
    'required_corrections': [],
    'scientific_rerun_performed': False,
    'draft_sha256': sha(HERE/'audit-draft.md'),
    'summary_sha256': sha(HERE/'summary.json'),
    'scope': 'Local negative continuum screen; no native/global sigma3 or stable pulse/BH acceptance.',
}
out = HERE/'root-review.json'
assert not out.exists()
out.write_text(json.dumps(review, indent=2, allow_nan=False)+'\n')
print(json.dumps({'passed': True, 'review_sha256': sha(out), 'counts': counts}))
