#!/usr/bin/env python3
"""Reapply the original native monitor while retaining every archived failure."""
import argparse
import hashlib
import json
from pathlib import Path
from run_finite_radius_long_stage import monitor

p = argparse.ArgumentParser()
p.add_argument('manifest', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
result = {}
for name, spec in json.loads(a.manifest.read_text()).items():
    prior = json.loads(Path(spec['prior']).read_text()) if spec.get('prior') else None
    errors = set((prior or {}).get('errors', []))
    for directory in spec['directories']:
        case = Path(directory)
        receipt = json.loads((case/'receipt.json').read_text())
        prior = monitor(case, receipt, prior)
        errors.update(prior['errors'])
    result[name] = dict(latest_monitor=prior, historical_errors=sorted(errors),
                        historical_monitor_passed=not errors, provenance=spec)
    result[name]['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result[name]['monitor_script_sha256'] = hashlib.sha256(
        Path(__file__).with_name('run_finite_radius_long_stage.py').read_bytes()).hexdigest()
    result[name]['manifest_sha256'] = hashlib.sha256(a.manifest.read_bytes()).hexdigest()
a.output.write_text(json.dumps(result, indent=2) + '\n')
print({name: dict(latest_passed=x['latest_monitor']['passed'],
                  historical_failures=len(x['historical_errors'])) for name, x in result.items()})
