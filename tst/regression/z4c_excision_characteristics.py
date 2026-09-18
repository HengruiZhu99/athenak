#!/usr/bin/env python3
"""Check Schwarzschild light/lapse/shift excision placement from a built executable."""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / 'inputs/z4c_ks_background.athinput'
    base = source.read_text()
    cases = [
        ('default', {}, 0.7083761391893204, False, False),
        ('weaker_shift', {'z4c/shift_Gamma': 0.25}, 1.0898667730579938, False, False),
        ('weaker_lapse', {'z4c/residual_lapse_f': 0.5}, 0.7083761391893204, False, False),
        ('explicit_radii', {}, 0.7083761391893204, True, False),
        ('spin_needs_radii', {'problem/bh_spin': 0.5, 'coord/a': 0.5}, None, False, True),
        ('spin_explicit', {'problem/bh_spin': 0.5, 'coord/a': 0.5}, None, True, False),
        ('advection_needs_radii', {'z4c/lapse_advect': 0.5,
                                 'z4c/boundary_rhs': 'sommerfeld'}, None, False, True),
    ]
    results = {}
    for name, overrides, expected, explicit, reject in cases:
        run = args.output.resolve() / name
        run.mkdir(parents=True, exist_ok=False)
        text = base
        if not explicit:
            text = re.sub(r'^excision_(freeze|ramp)_radius\s*=.*\n', '', text, flags=re.M)
        (run / 'input.athinput').write_text(text)
        command = [str(args.exe.resolve()), '-i', 'input.athinput', 'time/nlim=0',
                   'z4c/debug_balance=false', 'z4c/debug_balance_profiles=false']
        command += ['{}={}'.format(k, v) for k, v in overrides.items()]
        with (run / 'run.log').open('w') as log:
            status = subprocess.run(command, cwd=str(run), stdout=log,
                                    stderr=subprocess.STDOUT).returncode
        log = (run / 'run.log').read_text()
        if reject:
            assert status != 0 and 'all-ingoing bound is unavailable' in log, name
            results[name] = {'unsupported_default_rejected': True}
            continue
        assert status == 0, name
        row = re.search(r'^EXCISION_SETUP (.*)$', log, re.M)
        assert row, name
        fields = dict(part.split('=', 1) for part in row.group(1).split())
        radius = float(fields['allingoing'])
        if expected is None:
            assert math.isnan(radius), name
        else:
            assert abs(radius - expected) < 5e-6, (name, radius, expected)
        freeze = float(fields['freeze'])
        if explicit:
            assert freeze == 1.0 and float(fields['ramp']) == 1.4, fields
        else:
            assert abs(freeze - 0.95*expected) < 5e-6, fields
        results[name] = {'bound': radius if math.isfinite(radius) else None,
                         'freeze': freeze, 'ramp': float(fields['ramp'])}
    (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
