"""Compare original constraint histories at t=.2; never interpolate fields."""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CONTROLS = HERE.parent
FAMILY = ROOT/'build-layer-research/continuum/preferred/native-overlay/spatial-norm-family'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def endpoint(run):
    result = json.loads((run/'results.json').read_text())
    case, = result['cases']
    assert case['exit_status'] == 0
    path = run/case['name']/'hyp.z4c.user.hst'
    data = np.atleast_2d(np.loadtxt(path))
    assert data.shape[1] == 15 and np.isfinite(data).all()
    assert (np.diff(data[:, 0]) > 0).all()
    t = .2
    assert data[0, 0] <= t <= data[-1, 0]
    right = int(np.searchsorted(data[:, 0], t))
    exact = data[right, 0] == t
    left = right if exact else right-1
    fraction = 0 if exact else (t-data[left, 0])/(data[right, 0]-data[left, 0])
    values = {}
    for name, index in [('H', 2), ('M', 3), ('Z', 4), ('Theta', 5)]:
        values[name] = float(np.interp(t, data[:, 0], data[:, index]))
    return {'run': str(run.relative_to(ROOT)), 'case': case['name'],
            'requested_comparison_time': t, 'exact_history_row': bool(exact),
            'history_bracket_times': [float(data[left, 0]), float(data[right, 0])],
            'right_interpolation_weight': float(fraction), 'constraints': values,
            'history_sha256': sha(path), 'result_sha256': sha(run/'results.json')}


runs = {
    'composed_N24': CONTROLS/'composed-N24-t2',
    'composed_N36': CONTROLS/'composed-N36-t0.2',
    'baseline_N24': FAMILY/'native-long',
    'baseline_N36': ROOT/'build-layer-research/spatial-norm-native-controls/N36-t0.2',
}
rows = {name: endpoint(path) for name, path in runs.items()}
comparisons = {}
for family in ['composed', 'baseline']:
    comparisons[family+'_N36_over_N24'] = {
        key: rows[family+'_N36']['constraints'][key]/rows[family+'_N24']['constraints'][key]
        for key in ['H', 'M', 'Z', 'Theta']}
for resolution in ['N24', 'N36']:
    comparisons[resolution+'_composed_over_baseline'] = {
        key: rows['composed_'+resolution]['constraints'][key]/rows['baseline_'+resolution]['constraints'][key]
        for key in ['H', 'M', 'Z', 'Theta']}
out = {'status': 'PASS',
       'scope': ('Original HST RMS comparison at coordinate t=.2. N24 histories '
                 'are linearly interpolated between their actual cadence times; '
                 'N36 has an exact final row. No exact N24 field snapshot at t.2, '
                 'no field interpolation/comparison across resolutions, no measured '
                 'convergence order and no stability acceptance.'),
       'sources_sha256': {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
       'rows': rows, 'ratios': comparisons}
(HERE/'early-t0.2-comparison.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
