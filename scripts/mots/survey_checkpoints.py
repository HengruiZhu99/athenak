#!/usr/bin/env python3
"""Search an ordered amplitude inventory, newest saved slices first.

Nondetection means only nondetection on the enumerated saved slices. A failed
executable or an incomplete frozen-state check aborts, never classifies a case.
Reruns resume completed immutable searches after checking their manifests.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', required=True, type=Path)
    p.add_argument('--athena', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--launcher', default='')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = json.loads(args.inventory.read_text())
    state = dict(started=time.time(), criterion='angular_candidate',
                 scope='existence on enumerated saved slices through t=50', cases=[])
    save(args.output/'state.json', state)
    script = Path(__file__).with_name('search_checkpoint.py')
    for case in inventory:
        checkpoints = sorted(Path(case['directory']).glob('rst/*.rst'), reverse=True)
        if not checkpoints:
            raise RuntimeError('No checkpoints for '+case['label'])
        record = dict(case, classification='searching', slices=[])
        state['cases'].append(record)
        for checkpoint in checkpoints:
            out = args.output/case['label']/checkpoint.stem
            if not out.exists():
                command = [sys.executable, str(script), '--athena', str(args.athena),
                           '--checkpoint', str(checkpoint), '--output', str(out),
                           '--launcher', args.launcher, '--lmax', '128', '--l-start', '8',
                           '--radii', '4', '--iterations', '500', '--profile-points', '1061']
                subprocess.run(command, check=True)
            manifest = json.loads((out/'manifest.json').read_text())
            result = json.loads((out/'search/frozen_mots.json').read_text())
            if manifest.get('returncode') != 0 or not manifest.get('checkpoint_unchanged'):
                raise RuntimeError('Unsuccessful frozen search '+str(out))
            if not result.get('active_state_unchanged') or not result.get('mesh_unchanged'):
                raise RuntimeError('Frozen invariant failure '+str(out))
            if result['time'] > 50.0000001:
                raise RuntimeError('Slice exceeds prescribed time window '+str(out))
            rows = list(csv.DictReader((out/'search/mots.mots_candidates.csv').open()))
            accepted = [r for r in rows if r['policy_accepted'] == '1']
            best = min(rows, key=lambda r: float(r['epsilon2'])) if rows else None
            item = dict(checkpoint=str(checkpoint), output=str(out), **result,
                        best=best, accepted=accepted,
                        elapsed=manifest['finished']-manifest['started'])
            record['slices'].append(item)
            print(case['label'], checkpoint.name, result['time'],
                  result['candidate_detected'], best and best['epsilon2'], flush=True)
            save(args.output/'state.json', state)
            if result['candidate_detected']:
                record['classification'] = 'candidate_detected'
                break
        else:
            record['classification'] = 'no_candidate_on_saved_slices'
        record['available_checkpoints'] = len(checkpoints)
        save(args.output/'state.json', state)
    state['finished'] = time.time()
    state['status'] = 'complete'
    save(args.output/'state.json', state)


if __name__ == '__main__':
    main()
