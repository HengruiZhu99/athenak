"""Audit completed N36 native control and compare histories at physical time .2."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FAMILY = ROOT/'build-layer-research/continuum/preferred/native-overlay/spatial-norm-family'
spec = importlib.util.spec_from_file_location('reader', ROOT/'vis/python/bin_convert.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
spec = importlib.util.spec_from_file_location(
    'budget', ROOT/'tst/hyperboloidal/analyze_native_constraints.py')
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


launch = json.loads((HERE/'N36-launch.json').read_text())
assert sha(ROOT/launch['build_receipt']) == launch['build_receipt_sha256']
build = json.loads((ROOT/launch['build_receipt']).read_text())
assert all(sha(ROOT/k) == v for k, v in build['source_sha256'].items())
assert all(sha(FAMILY/k) == v for k, v in build['overlay_sha256'].items())
result_path = HERE/'N36-t0.2/results.json'
result = json.loads(result_path.read_text())
assert result['sha256'] == launch['native_executable_sha256']
case = result['cases'][0]
assert case['exit_status'] == 0 and case['diagnostics']['time'] == .2
directory = result_path.parent/case['name']
assert sha(directory/'layer.athinput') == case['input_sha256']
history = np.atleast_2d(np.loadtxt(directory/'hyp.z4c.user.hst'))
assert history.shape[1] == 15 and np.isfinite(history).all()
assert (np.diff(history[:, 0]) > 0).all()
snapshots = []
for path in sorted((directory/'bin').glob('*.z4c.*.bin')):
    data = reader.read_binary(str(path))
    fields = {k: np.asarray(v) for k, v in data['mb_data'].items()}
    mask = fields['z4c_active'].astype(bool)
    assert np.isin(fields['z4c_active'], [0, 1]).all()
    assert all(np.isfinite(v[mask]).all() for v in fields.values())
    alpha, chi = (float(fields[k][mask].min()) for k in ('z4c_alpha', 'z4c_chi'))
    assert alpha > 0 and chi > 0
    metric = np.zeros((int(mask.sum()), 3, 3))
    for i, j, suffix in [(0, 0, 'xx'), (0, 1, 'xy'), (0, 2, 'xz'),
                         (1, 1, 'yy'), (1, 2, 'yz'), (2, 2, 'zz')]:
        metric[:, i, j] = metric[:, j, i] = fields['z4c_g'+suffix][mask]/fields['z4c_chi'][mask]
    ev = np.linalg.eigvalsh(metric)
    assert ev.min() > 0
    snapshots.append({'time': data['time'], 'cycle': data['cycle'], 'sha256': sha(path),
                      'path': str(path.relative_to(ROOT)), 'active_cells': int(mask.sum()),
                      'all_active_fields_finite': True, 'alpha_min': alpha, 'chi_min': chi,
                      'physical_metric_eigen_min': float(ev.min()),
                      'physical_metric_eigen_max': float(ev.max())})
budgets = [budget.analyze(p, [0, .05, .45, .85, .9, .95, 1])
           for p in sorted((directory/'bin').glob('*.con.*.bin'))]
dt = [float(v) for v in re.findall(r'\bdt=([0-9.eE+-]+)', (directory/'run.log').read_text())]
assert dt and min(dt) > 0
compare = {}
baseline_dir = ROOT/'build-layer-research/clean-wide-kappa10-long'
baseline = json.loads((baseline_dir/'results.json').read_text())['cases'][0]
for label, path in [('N36-spatialnorm', directory/'hyp.z4c.user.hst'),
                    ('N24-spatialnorm', FAMILY/'native-long/finite-angular-long-N24/'
                     'hyp.z4c.user.hst'),
                    ('N24-production', baseline_dir/baseline['name']/'hyp.z4c.user.hst')]:
    values = np.atleast_2d(np.loadtxt(path))
    assert values[0, 0] <= .2 <= values[-1, 0] and np.isfinite(values).all()
    compare[label] = {'history_sha256': sha(path),
                      'H': float(np.interp(.2, values[:, 0], values[:, 2])),
                      'M': float(np.interp(.2, values[:, 0], values[:, 3])),
                      'Z': float(np.interp(.2, values[:, 0], values[:, 4]))}
out = {'scope': ('Independent completed native N36 early combined spatial/time refinement; '
                 'same pole.03 but actual dt differs. No pure spatial-order/stability claim.'),
       'launch': launch, 'result_sha256': sha(result_path), 'case': case,
       'source_and_overlay_hashes_still_match_build': True,
       'history_sha256': sha(directory/'hyp.z4c.user.hst'),
       'history_rows': history.tolist(), 'first_dt': float(history[0, 1]),
       'history_dt_range': [float(history[:, 1].min()), float(history[:, 1].max())],
       'rounded_log_dt_range': [min(dt), max(dt)],
       'audit_source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [
           Path(__file__), ROOT/'vis/python/bin_convert.py',
           ROOT/'tst/hyperboloidal/analyze_native_constraints.py']},
       'native_snapshots': snapshots, 'radial_constraint_budgets': budgets,
       'matched_physical_time_0.2': compare,
       'all_retained_files': {str(p.relative_to(ROOT)): {
           'sha256': sha(p), 'bytes': p.stat().st_size}
                              for p in sorted(result_path.parent.rglob('*')) if p.is_file()}}
(HERE/'N36-audit.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
print('PASS', len(snapshots), 'positive/SPD native snapshots; actual dt', history[0, 1])
print(json.dumps(compare, indent=2))
