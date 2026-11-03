#!/usr/bin/env python3
"""Apply the user's post-only expansion tolerance without overwriting evidence."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


def classify(row, tolerance):
    rms = float(row['horizon_trial_expansion_rms'])
    radius = float(row['horizon_trial_radius'])
    passed = (row['horizon_complete_surface'] == '1'
              and math.isfinite(rms) and 0 <= rms <= tolerance
              and math.isfinite(radius) and radius > 0)
    mass = float(row['horizon_mass'])
    return dict(post_expansion_tolerance=tolerance,
                post_expansion_pass=int(passed),
                post_mass_available=int(passed and math.isfinite(mass) and mass > 0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    policy_path = Path(__file__).with_name('horizon-post-policy.json')
    policy = json.loads(policy_path.read_text())
    data = a.history.read_bytes()
    rows = list(csv.DictReader(io.StringIO(data.decode())))
    enriched = [dict(row, **classify(row, policy['expansion_rms_tolerance'])) for row in rows]
    assert enriched, 'Empty history'
    a.output.mkdir(parents=True, exist_ok=False)
    with (a.output/'history-post.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(enriched[0]))
        writer.writeheader(); writer.writerows(enriched)
    summary = dict(policy=policy, history=str(a.history.resolve()),
                   history_sha256=hashlib.sha256(data).hexdigest(),
                   policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
                   rows=len(rows), online_accepted_rows=sum(r['horizon_found']=='1' for r in rows),
                   post_expansion_pass_rows=sum(r['post_expansion_pass'] for r in enriched),
                   post_mass_available_rows=sum(r['post_mass_available'] for r in enriched),
                   final_time_M=float(rows[-1]['t_over_M']),
                   final_expansion_rms=float(rows[-1]['horizon_trial_expansion_rms']),
                   final_post_expansion_pass=bool(enriched[-1]['post_expansion_pass']))
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
