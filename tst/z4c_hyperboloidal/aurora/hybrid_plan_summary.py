#!/usr/bin/env python3
"""Compare cached sparse communication strategies from initialization receipts."""
import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('log', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    records = []
    for line in a.log.read_text().splitlines():
        if 'AGREEMENT_PLAN rank=' not in line:
            continue
        values = dict(re.findall(r'(\w+)=([\deE.+-]+)', line))
        records.append({k: float(v) if k in ['kappa', 'rate_bound', 'dt', 'build_seconds']
                        else int(v) for k, v in values.items()})
    assert len(records) == 24 and {r['rank'] for r in records} == set(range(24))
    assert len({r['global_samples'] for r in records}) == 1
    total = lambda k: sum(r[k] for r in records)
    assert total('send_donors') == total('receive_donors')
    assert total('owner_once_rows') == records[0]['global_samples']
    active = [r['active_points'] for r in records]
    result = {
        'scope': 'Initialization geometry; owner-compute alternative is a sparse-volume model, not a measured implementation. No message aggregation or overlap timing is inferred.',
        'log_sha256': hashlib.sha256(a.log.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'fields': 26, 'bytes_per_field': 8, 'ranks': records,
        'duplicated_row_factor': total('local_rows') / total('owner_once_rows'),
        'duplicated_entry_factor': total('entries') / total('owner_once_entries'),
        'chosen': {'sequential_exchange_phases': 1,
                   'global_remote_bytes_per_stage': total('receive_donors') * 26 * 8,
                   'row_evaluations_per_stage': total('local_rows')},
        'owner_compute_model': {'sequential_exchange_phases': 2,
                               'gather_remote_bytes_per_stage': total('owner_once_remote_gather') * 26 * 8,
                               'return_remote_bytes_per_stage': total('owner_once_remote_return') * 26 * 8,
                               'row_evaluations_per_stage': total('owner_once_rows')},
        'active_points': {'sum': sum(active), 'min': min(active), 'max': max(active),
                          'max_over_mean': max(active) / statistics.mean(active)},
        'max_plan_build_seconds': max(r['build_seconds'] for r in records),
        'max_dt_times_bound': max(r['dt'] * r['rate_bound'] for r in records)}
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'ranks'}, indent=2))


if __name__ == '__main__':
    main()
