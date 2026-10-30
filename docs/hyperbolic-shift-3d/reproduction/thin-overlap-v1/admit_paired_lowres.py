#!/usr/bin/env python3
"""Admit only the frozen n16 L1/kappa1 sparse scheme for bounded stability tests.

The first batch's failed decimal-clock comparison remains failed. This selects
its verified component/equilibrium evidence and the later independently
verified matched-clock and puncture comparisons; it never admits that batch.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]/'tst/hyperbolic_shift_3d/aurora'))
import control

p = argparse.ArgumentParser(description=__doc__)
for name in ('root', 'core-audit', 'clock-audit', 'puncture-audit', 'output'):
    p.add_argument('--'+name, type=Path, required=True)
a = p.parse_args()
read = lambda path: json.loads(path.read_text())
core, clock, puncture = [read(path) for path in (a.core_audit, a.clock_audit, a.puncture_audit)]
assert core['status'] == 'verified_recorded_failure'
assert clock['status'] == puncture['status'] == 'passed'
assert set(clock['comparisons']) == {'direct-communication', 'resumed-communication', 'dyadic-restart'}
assert set(puncture['comparisons']) == {'puncture-equivalence'}
for result in [*clock['comparisons'].values(), *puncture['comparisons'].values()]:
    assert result['bitwise_payload_equal'] and result['restart_equivalence']
    assert result['maximum_absolute_error'] == 0 and result['compared_doubles'] == 9000416
directories = [a.root/'campaigns'/('preflight-thin-'+name+'-v1') for name in ('core', 'clock', 'puncture')]
bindings = [read(d/'binding.json') for d in directories]
source = bindings[0]['source_manifest_sha256']
assert all(b['source_manifest_sha256'] == source for b in bindings)
assert clock['source_manifest_sha256'] == puncture['source_manifest_sha256'] == source
assert control.digest(a.root/'source-thin-v1-manifest.json') == source
for name in ('paired-device', 'minkowski-l1-k1'):
    path = directories[0]/'output'/(name+'.report.json')
    assert control.digest(path) == core['verified_artifacts'][str(path.relative_to(a.root))]['sha256']
device = read(directories[0]/'output/paired-device.report.json')
assert device['gpu_cpu_parity'] and device['placement_passed'] and len(device['ranks']) == 12
mink = read(directories[0]/'output/minkowski-l1-k1.report.json')
assert mink['status'] == 'completed' and mink['max_state_deviation'] < 1e-10 and mink['H_L2'] < 1e-10
command = next(c for c in read(directories[2]/'plan.json')['commands'] if c['name'] == 'puncture-sparse')
config = control.normal_derivative_config(command)
assert config['core_evolved_layers'] == 1 and config['trace_kappa'] == 1 and config['shifted_overlap_donors']
for key, value in config.items():
    if key != 'executable':
        assert mink[key] == value, key
binary = command['executable']; digest = bindings[2]['binaries'][binary]
assert all(b['binaries'][binary] == digest for b in bindings)
build = read(a.root/'build-logs/thin-v1/build-status.json')
assert build['status'] == 'passed' and build['binaries'][binary] == digest
evidence = {path.name: control.digest(path) for path in (a.core_audit, a.clock_audit, a.puncture_audit)}
result = dict(passed=True, phase='preflight', source_manifest_sha256=source,
              low_resolution_n=16, binaries={binary: digest}, normal_derivative_configs=[config],
              gpu_cpu_parity=True, restart_equivalence=True, evidence=evidence, evidence_relative_to_admission=True,
              scope='Only frozen n16 L1/kappa1 sparse V12 for bounded stability work. GPU/CPU parity covers the added paired operator; complete-state GPU communication and restart equality are separately verified. Not a full trajectory CPU/GPU parity, stability theorem, or convergence admission. Original decimal restart failure remains recorded.')
with a.output.open('x') as stream:
    json.dump(result, stream, indent=2); stream.write('\n')
print(json.dumps(result, indent=2))
