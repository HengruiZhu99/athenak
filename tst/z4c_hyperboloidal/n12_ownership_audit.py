#!/usr/bin/env python3
"""Count actual N12 GPU trace ownership; no evolution or performance measurement."""
import argparse
import json
from pathlib import Path
import struct

import numpy as np

from native_stage_trace_compare import MAGIC, metadata, require, sha


def equal_cost_owners(ranks):
    """Replay Mesh::LoadBalance's float arithmetic for 106 unit-cost blocks."""
    owners = np.empty(106, dtype=int)
    rank, total, current = ranks-1, np.float32(106), np.float32(0)
    target = total/np.float32(ranks)
    for gid in range(105, -1, -1):
        current += np.float32(1)
        owners[gid] = rank
        if current >= target and rank > 0:
            rank -= 1
            total -= current
            current = np.float32(0)
            target = total/np.float32(rank+1)
    return owners


def audit(folder, mapping):
    info = metadata(folder)
    ranks = info['ranks']
    require(info['backend'] == 'SYCL' and ranks in (24, 96), 'not a qualified GPU layout')
    predicted = equal_cost_owners(ranks)
    rows = [dict(rank=rank, blocks=[], bulk_points=0, scri_points=0) for rank in range(ranks)]
    hashes = {str(folder/'results.json'): sha(folder/'results.json')}
    for region, total in (('bulk', 229392), ('scri', 864)):
        gids = mapping['surface_keys' if region == 'scri' else 'bulk_keys'][:, 0]
        expected = np.bincount(gids, minlength=106)
        actual = np.zeros(106, dtype=int)
        ordinals = []
        for rank in range(ranks):
            path = folder/f'geometry-{region}-rank{rank}.bin'
            raw = path.read_bytes()
            require(raw[:16] == MAGIC, 'invalid trace magic')
            count, nk, nv, kind, version = struct.unpack_from('<QIIII', raw, 16)
            require((nk, nv, kind, version) == (6, 3, int(region == 'scri'), 1),
                    'unexpected geometry trace schema')
            require(len(raw) == 40+48*count, 'incomplete geometry trace')
            data = np.frombuffer(raw, dtype=np.dtype([('key', '<i4', 6), ('xyz', '<f8', 3)]),
                                 count=count, offset=40)
            keys = data['key']
            require(bool(np.all((keys[:, 1] >= 0) & (keys[:, 1] < 106))), 'invalid block ID')
            require(bool(np.all(predicted[keys[:, 1]] == rank)),
                    'observed ownership differs from native unit-cost load balancing')
            actual += np.bincount(keys[:, 1], minlength=106)
            ordinals.extend(keys[:, 0].tolist())
            rows[rank][region+'_points'] = int(count)
            if region == 'bulk':
                rows[rank]['blocks'] = np.unique(keys[:, 1]).tolist()
            hashes[str(path)] = sha(path)
        require(np.array_equal(actual, expected), 'block point counts differ from bound mapping')
        require(sorted(ordinals) == list(range(total)), 'missing or duplicate canonical points')
    bulk = np.array([row['bulk_points'] for row in rows])
    return dict(ranks=ranks, total_bulk_points=int(bulk.sum()), total_scri_points=864,
        blocks_per_rank_min=min(len(row['blocks']) for row in rows),
        blocks_per_rank_max=max(len(row['blocks']) for row in rows),
        bulk_per_rank_min=int(bulk.min()), bulk_per_rank_mean=float(bulk.mean()),
        bulk_per_rank_max=int(bulk.max()), largest_bulk_ranks=np.flatnonzero(bulk == bulk.max()).tolist(),
        mean_to_max_bulk_count=float(bulk.mean()/bulk.max()),
        surface_owners=[row['rank'] for row in rows if row['scri_points']],
        native_unit_cost_assignment_exact=True, per_rank=rows, input_sha256=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--two-node-trace', type=Path, required=True)
    parser.add_argument('--eight-node-trace', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    mapping_path = args.mapping/'mapping.npz'
    manifest = json.loads((args.mapping/'manifest.json').read_text())
    require(sha(mapping_path) == manifest['mapping_sha256'], 'mapping hash changed')
    with np.load(mapping_path) as mapping:
        two = audit(args.two_node_trace.resolve(), mapping)
        eight = audit(args.eight_node_trace.resolve(), mapping)
    require(two['ranks'] == 24 and eight['ranks'] == 96, 'swapped layouts')
    source = Path(__file__).resolve()
    load_balance = source.parents[2]/'src/mesh/load_balance.cpp'
    speedup = two['bulk_per_rank_max']/eight['bulk_per_rank_max']
    report = dict(schema='native-n12-ownership-audit-v1', passed=True,
        scope='Observed ownership counts only; no measured performance or GPU utilization.',
        layouts={'24': two, '96': eight},
        uniform_bulk_point_cost_model=dict(speedup=speedup, efficiency=speedup/4,
            assumptions='Equal time per bulk point, concurrent ranks, no communication, surface, launch or diagnostic costs. This is not an overall scaling bound.'),
        input_sha256={str(mapping_path.resolve()): sha(mapping_path), str(load_balance): sha(load_balance)},
        source_sha256={str(source): sha(source)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in ('passed', 'scope', 'uniform_bulk_point_cost_model')}, indent=2))


if __name__ == '__main__':
    main()
