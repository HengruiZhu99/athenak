#!/usr/bin/env python3
"""Apply the unchanged tight saved-checkpoint gate to historical Aurora norms.

This is separate from the pointwise native-versus-legacy CPU oracle. It retains
failures rather than substituting a CPU-recomputed reference for recorded data.
"""
import argparse
import hashlib
import json
import pathlib
import re

parser = argparse.ArgumentParser()
parser.add_argument('--reference', type=pathlib.Path, required=True)
parser.add_argument('--runs', type=pathlib.Path, nargs='+', required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
args = parser.parse_args()
reference = json.loads(args.reference.read_text())
expected = reference['constraint_RMS'] + reference['outer_constraint_RMS']
result = {'passed': True, 'gate': 'abs(delta)<=5e-11*(1+abs(expected))',
    'reference': str(args.reference.resolve()),
    'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
    'reference_step': reference['steps'], 'reference_time': reference['t'], 'runs': []}
for run in args.runs:
    log = run/'run.log'
    values = {}
    for m in re.finditer(r'^norm scenario=2 region=(\d) field=(\d) actual=(\S+) expected=(\S+)$',
                         log.read_text(), re.M):
        values[4*int(m[1])+int(m[2])-1] = float(m[3])
    assert set(values) == set(range(8)), 'saved full/fixed-outer norm coverage missing'
    checks = []
    for i, want in enumerate(expected):
        value = values[i]
        tolerance = 5e-11*(1+abs(want))
        delta = abs(value-want)
        checks.append({'region':'full' if i<4 else 'fixed_outer', 'field':['H','M','Theta','Z'][i%4],
            'actual':value, 'reference':want, 'absolute_difference':delta,
            'tolerance':tolerance, 'tolerance_ratio':delta/tolerance, 'passed':delta<=tolerance})
    passed = all(c['passed'] for c in checks)
    result['passed'] &= passed
    result['runs'].append({'directory':str(run.resolve()), 'passed':passed, 'checks':checks,
        'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()})
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
raise SystemExit(0 if result['passed'] else 1)
