"""Read-only portable hash, guard and observed-order checks; no native runs."""
from pathlib import Path
import hashlib
import json
import math
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

inventory = read(HERE/'inventory.json')
for item in inventory['files']:
    path = HERE/item['path']
    assert sha(path) == item['sha256'] and path.stat().st_size == item['bytes'], item['path']
binding_path = HERE/'compile-evidence/source.json'
binding = read(binding_path)
for file, suffix in [('build-source.cpp', '/tst/hyperbolic_shift_3d/scri_sampled_accuracy.cpp'),
                     ('build-driver.py', '/tst/hyperbolic_shift_3d/scri_sampled_accuracy.py')]:
    matches = [value for key, value in binding['inputs'].items() if key.endswith(suffix)]
    assert matches == [sha(HERE/'compile-evidence'/file)]
header = binding['retained_native_header']
assert sha(HERE/'compile-evidence'/header['path']) == header['sha256'] == binding['inputs'][header['source']]
headers = read(HERE/'frozen-header-bindings.json')['headers']
for item in headers:
    raw = subprocess.check_output(['git', 'show', item['frozen_git_commit']+':'+item['path']], cwd=ROOT)
    assert hashlib.sha256(raw).hexdigest() == item['sha256']

limits = dict(weight_moment_error=1e-12, weight_diagonal_error=1e-14,
              analytic_first_jet_fd_error=1e-7, analytic_second_jet_fd_error=1e-6,
              analytic_mixed_hessian_fd_error=1e-6, native_pole_algebra_error=2e-12,
              analytic_AD_vs_five_point_normal_error=2e-8, null_residual=2e-14,
              surface_determinant_error=2e-14, surface_A_trace=2e-14,
              volume_determinant_error=2e-14, volume_A_trace=2e-14)
reports = {}
for n in (8, 12, 16, 24):
    folder = HERE/'results'/f'n{n}'
    execution, report = read(folder/'execution.json'), read(folder/'accuracy.json')
    assert execution['status'] == report['status'] == 'completed' and execution['returncode'] == 0
    assert execution['report_sha256'] == sha(folder/'accuracy.json')
    assert execution['binary_sha256'] == binding['binary_sha256']
    assert execution['source_binding_sha256'] == sha(binding_path)
    assert report['n'] == execution['n'] == n and report['h'] == .25/n
    assert execution['seconds'] < execution['timeout_seconds'] <= 390
    assert [f['fixture'] for f in report['fixtures']] == ['radial_p1', 'angular_p1', 'radial_p2', 'angular_p2']
    for fixture in report['fixtures']:
        for key, limit in limits.items():
            assert math.isfinite(fixture['controls'][key]) and 0 <= fixture['controls'][key] <= limit
        if fixture['vanishing_power'] == 2:
            assert fixture['exact_boundary_regular']
            assert fixture['controls']['exact_endpoint_pole_max'] <= 2e-12
            assert fixture['controls']['exact_regularity_max'] <= 2e-12
        for method in fixture['errors'].values():
            for region in method['regions'].values():
                assert region['points'] > 0 and region['area'] > 0
                assert math.isfinite(region['l2']) and region['l2'] >= 0
                assert math.isfinite(region['linf_component']) and region['linf_component'] >= 0
            assert abs(method['regions']['all']['area']-4*math.pi) < 5e-13
    reports[n] = report

summary = read(HERE/'summary.json')
assert summary['expected_order_imposed'] is False
for index, fixture in enumerate(summary['fixtures']):
    for name, rows in fixture['methods'].items():
        previous = None
        for row in rows:
            values = reports[row['n']]['fixtures'][index]['errors'][name]['regions']['all']
            for norm in ('l2', 'linf_component'):
                assert row[norm] == values[norm]
                if previous is not None:
                    expected = math.log(previous[norm]/row[norm])/math.log(previous['h']/row['h'])
                    assert abs(row['observed_order_'+norm]-expected) < 1e-14
            previous = row
print(json.dumps(dict(status='verified', artifacts=len(inventory['files']), frozen_headers=len(headers),
                     resolutions=4, fixtures=16, native_evaluations_performed=False,
                     evolution_or_stability_admitted=False), indent=2))
