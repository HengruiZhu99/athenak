#!/usr/bin/env python3
"""Check frozen numerical sources and effective command overrides across segments."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source_index', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('evidence', nargs='+', type=Path)
    a = p.parse_args()
    expected = json.loads(a.source_index.read_text())
    binary = None; physics = {}; records = []; inputs = {}
    runtime = {'cubed_sphere/conformal_checkpoint', 'cubed_sphere/conformal_restart',
               'cubed_sphere/conformal_wall_seconds', 'cubed_sphere/conformal_max_cycles'}
    for directory in a.evidence:
        review = json.loads((directory/'chain-review.json').read_text())
        for segment in review['segments']:
            base = directory/segment['job_id'].split('.')[0]
            path = base/'launch.json'
            launch = json.loads(path.read_text())
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            if binary is None: binary = launch['binary_sha256']
            assert launch['binary_sha256'] == binary
            assert all(launch['source_sha256'].get(k) == v for k, v in expected.items())
            for case in sorted(base.iterdir()):
                path = case/'receipt.json'
                if not path.is_file(): continue
                receipt = json.loads(path.read_text())
                inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                # Athena consumes repeated overrides in order: the last wins.
                effective = {}
                for arg in receipt['command']:
                    if '=' in arg:
                        key, value = arg.split('=', 1)
                        if '/' in key: effective[key] = value
                assert effective['cubed_sphere/conformal_checkpoint_tag'] == binary
                physics_now = {k: v for k, v in effective.items() if k not in runtime}
                if case.name in physics: assert physics[case.name] == physics_now
                else: physics[case.name] = physics_now
                assert receipt['reference_unchanged']
                records.append({'job': segment['job_id'], 'case': case.name,
                                'input_sha256': receipt['input_sha256'],
                                'runtime_overrides': {k: v for k, v in effective.items() if k in runtime}})
    assert len({row['input_sha256'] for row in records}) == 1
    baseline = physics['degree2']
    varied = {'cubed_sphere/conformal_outer_degree', 'cubed_sphere/conformal_cfl',
              'cubed_sphere/conformal_pole_cfl'}
    for name, values in physics.items():
        assert {k:v for k,v in values.items() if k not in varied} == {
            k:v for k,v in baseline.items() if k not in varied}
        assert int(values['cubed_sphere/conformal_outer_degree']) == int(name[-1])
        factor = .5 if name.startswith('half-') else 1.
        for key in ['cubed_sphere/conformal_cfl', 'cubed_sphere/conformal_pole_cfl']:
            assert float(values[key]) == factor*float(baseline[key])
    a.output.write_text(json.dumps({'pass': True, 'numerical_source_entries': len(expected),
        'binary_sha256': binary, 'effective_physics_by_case': physics, 'records': records,
        'input_sha256': inputs,
        'scope': 'Frozen numerical-source and effective command continuity. Runtime paths, wall limits and maximum-cycle guards may differ. Hash manifests are provenance, not proof of scientific correctness.'}, indent=2)+'\n')
    print('Verified', len(records), 'case segments;', len(expected), 'numerical source entries')


if __name__ == '__main__': main()
