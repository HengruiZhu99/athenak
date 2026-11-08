#!/usr/bin/env python3
"""Validate a metadata-only capture and project its plan onto verified owners."""
import argparse
import csv
import json
from pathlib import Path
import struct

import numpy as np

from native_bvals_route_audit import check_inputs, require, sha
from n12_ownership_audit import equal_cost_owners


REASONS = ('eligible', 'not_single', 'nonunit_weight', 'different_chart',
           'different_layer', 'different_level', 'different_node_offset',
           'not_adjacent', 'missing_native_neighbor', 'native_neighbor_metadata',
           'outside_native_receive_box', 'outside_native_send_box', 'index_translation')


def table(path):
    with path.open() as stream:
        return [{key: int(value) for key, value in row.items()}
                for row in csv.DictReader(stream, delimiter='\t')]


def layout(data, blocks, routes, owners):
    ranks = int(owners.max()) + 1
    receive = np.zeros((ranks, ranks), dtype=np.int64)
    remaining = np.zeros_like(receive)
    local = []
    targets = owners[data[:, 0]]
    for rank in range(ranks):
        selected = data[targets == rank]
        points, inverse = np.unique(selected[:, 1:5], axis=0, return_inverse=True)
        flags = np.zeros(len(points), dtype=np.uint8)
        np.bitwise_or.at(flags, inverse, selected[:, 5].astype(np.uint8))
        donor_ranks = owners[points[:, 0]]
        remote = donor_ranks != rank
        for source in range(ranks):
            if source == rank:
                continue
            receive[rank, source] = np.count_nonzero(remote & (donor_ranks == source))
            remaining[rank, source] = np.count_nonzero(remote & (donor_ranks == source)
                                                       & ((flags & 2) != 0))
        own_blocks = [row for row in blocks if owners[row['gid']] == rank]
        own_routes = [row for row in routes if owners[row['target_gid']] == rank]
        remote_routes = [row for row in own_routes if owners[row['donor_gid']] != rank]
        reason = {key: sum(row[key] for row in own_blocks) for key in REASONS}
        category = lambda mask: dict(
            unique=int(np.count_nonzero(mask)),
            eligible_exclusive=int(np.count_nonzero(mask & (flags == 1))),
            shared=int(np.count_nonzero(mask & (flags == 3))),
            remaining_only=int(np.count_nonzero(mask & (flags == 2))))
        local.append(dict(rank=rank, blocks=[row['gid'] for row in own_blocks],
            rows=sum(row['rows'] for row in own_blocks), reasons=reason,
            unique_donors=len(points), local_donors=category(~remote),
            remote_donors=category(remote),
            eligible_remote_rows=sum(row['eligible_rows'] for row in remote_routes),
            eligible_local_rows=sum(row['eligible_rows'] for row in own_routes)
                                - sum(row['eligible_rows'] for row in remote_routes),
            remote_native_routes=len(remote_routes),
            remote_slab_points=sum(row['slab_points'] for row in remote_routes),
            remote_slab_noneligible_plan_points=sum(row['plan_rows_in_slab']-row['eligible_rows']
                                                    for row in remote_routes),
            remote_slab_evolved_points=sum(row['evolved_in_slab'] for row in remote_routes),
            remote_slab_unused_points=sum(row['slab_points']-row['plan_rows_in_slab']
                -row['evolved_in_slab'] for row in remote_routes)))
    for rank, row in enumerate(local):
        row['send_donors'] = int(receive[:, rank].sum())
        row['receive_peers'] = int(np.count_nonzero(receive[rank]))
        row['send_peers'] = int(np.count_nonzero(receive[:, rank]))
        row['mapped_receive_messages_per_stage'] = 2*row['receive_peers']
        row['mapped_send_messages_per_stage'] = 2*row['send_peers']
        row['mapped_receive_bytes_per_stage'] = int(receive[rank].sum())*57*8
        row['mapped_send_bytes_per_stage'] = row['send_donors']*57*8
        row['remaining_sparse_receive_bytes_per_stage'] = int(remaining[rank].sum())*57*8
        row['removable_sparse_receive_bytes_per_stage'] = (
            row['mapped_receive_bytes_per_stage']-row['remaining_sparse_receive_bytes_per_stage'])
        row['native_slab_62_field_bytes_per_stage'] = row['remote_slab_points']*62*8
        row['native_slab_hypothetical_57_field_bytes_per_stage'] = row['remote_slab_points']*57*8
        require(row['remote_donors']['unique'] == int(receive[rank].sum()), 'receive count mismatch')
        require(row['remote_donors']['eligible_exclusive'] ==
                int(receive[rank].sum()-remaining[rank].sum()), 'exclusive count mismatch')
    aggregate_keys = ('rows', 'unique_donors', 'eligible_remote_rows', 'eligible_local_rows',
        'remote_native_routes', 'remote_slab_points', 'remote_slab_noneligible_plan_points',
        'remote_slab_evolved_points', 'remote_slab_unused_points',
        'mapped_receive_messages_per_stage', 'mapped_receive_bytes_per_stage',
        'remaining_sparse_receive_bytes_per_stage', 'removable_sparse_receive_bytes_per_stage',
        'native_slab_62_field_bytes_per_stage', 'native_slab_hypothetical_57_field_bytes_per_stage')
    totals = {key: sum(row[key] for row in local) for key in aggregate_keys}
    totals['remote_donors'] = {key: sum(row['remote_donors'][key] for row in local)
                              for key in ('unique', 'eligible_exclusive', 'shared', 'remaining_only')}
    totals['reasons'] = {key: sum(row['reasons'][key] for row in local) for key in REASONS}
    totals['hypothetical_remaining_sparse_plus_full_native_62_bytes_per_stage'] = (
        totals['remaining_sparse_receive_bytes_per_stage'] + totals['native_slab_62_field_bytes_per_stage'])
    totals['hypothetical_remaining_sparse_plus_full_native_57_bytes_per_stage'] = (
        totals['remaining_sparse_receive_bytes_per_stage'] + totals['native_slab_hypothetical_57_field_bytes_per_stage'])
    totals['maximum_rank_receive_bytes_per_stage'] = max(row['mapped_receive_bytes_per_stage'] for row in local)
    totals['maximum_rank_send_bytes_per_stage'] = max(row['mapped_send_bytes_per_stage'] for row in local)
    require(sum(row['mapped_send_bytes_per_stage'] for row in local)
            == totals['mapped_receive_bytes_per_stage'], 'global send/receive bytes differ')
    return dict(ranks=ranks, totals=totals, per_rank=local,
        peer_traffic=[dict(receiver=int(r), sender=int(s), unique_donors=int(receive[r, s]),
                          remaining_donors=int(remaining[r, s]))
                      for r, s in zip(*np.nonzero(receive))])


def compare(args):
    capture = args.capture.resolve()
    proof = json.loads(args.build_proof.read_text())
    require(proof['schema'] == 'native-bvals-route-build-v1' and proof['status'] == 'built'
            and proof['runtime_inputs_unchanged'] is True, 'invalid isolated build proof')
    check_inputs(proof)
    require(sha(proof['executable']) == proof['executable_sha256'], 'helper executable changed')
    execution = json.loads((capture/'execution.json').read_text())
    require(execution['exit_code'] == 0 and execution['ranks'] == 2
            and execution['executable_sha256'] == proof['executable_sha256'], 'invalid CPU2 execution')
    require(execution['build_proof_sha256'] == sha(args.build_proof)
            and execution['command'] == ['mpiexec', '-n', '2', proof['executable'], execution['input']],
            'execution not bound to exact helper invocation/build')
    require(sha(execution['input']) == execution['input_sha256'], 'capture input changed')
    require((capture/'run.log').read_text().count('PASS metadata-only native bvals route capture') == 1,
            'missing terminal metadata PASS')
    hashes = {str(args.build_proof.resolve()): sha(args.build_proof),
              str(capture/'execution.json'): sha(capture/'execution.json'),
              str(capture/'run.log'): sha(capture/'run.log'),
              str(Path(execution['input']).resolve()): execution['input_sha256']}
    stats, blocks, routes, arrays = [], [], [], []
    for rank in range(2):
        suffix = f'-rank{rank}'
        files = [capture/f'stats{suffix}.json', capture/f'blocks{suffix}.tsv',
                 capture/f'routes{suffix}.tsv', capture/f'uses{suffix}.bin']
        hashes.update({str(path): sha(path) for path in files})
        st = json.loads(files[0].read_text())
        require(st['schema'] == 'native-bvals-route-capture-v1' and st['passed']
                and st['rank'] == rank and st['ranks'] == 2 and st['real_bytes'] == 8
                and st['origin'] == 7 and st['cells'] == 12
                and st['backend'] != 'SYCL' and st['public_exchange_statistics_match']
                and not st['field_transport_called'], 'invalid CPU metadata')
        require((st['residual_fields'], st['physical_fields']) == (26, 31), 'field counts changed')
        br, rr = table(files[1]), table(files[2])
        require(len(br) == st['blocks'] and all(row['owner'] == rank for row in br), 'block ownership mismatch')
        require(all(sum(row[key] for key in REASONS) == row['rows'] for row in br), 'reason coverage mismatch')
        raw = files[3].read_bytes()
        require(raw[:16] == b'BVALS_USES_V1'.ljust(16, b'\0'), 'invalid donor-use magic')
        count, words, version = struct.unpack_from('<QII', raw, 16)
        require((words, version) == (6, 1) and count == st['use_records']
                and len(raw) == 32+24*count, 'invalid donor-use shape')
        data = np.frombuffer(raw, dtype='<i4', offset=32).reshape(-1, 6)
        require(np.all(np.isin(data[:, 0], [row['gid'] for row in br]))
                and np.all((data[:, 1] >= 0) & (data[:, 1] < 106))
                and np.all((data[:, 2:5] >= 0) & (data[:, 2:5] < 26))
                and np.all((data[:, 5] >= 1) & (data[:, 5] <= 3)), 'invalid donor-use keys')
        require(len(np.unique(data[:, :5], axis=0)) == len(data), 'duplicate donor-use record')
        require(sum(row['eligible_rows'] for row in rr) == sum(row['eligible'] for row in br),
                'eligible route coverage mismatch')
        require(all(row['slab_points'] >= row['plan_rows_in_slab']+row['evolved_in_slab']
                    and row['plan_rows_in_slab'] >= row['eligible_rows'] > 0 for row in rr),
                'invalid full-slab accounting')
        stats.append(st); blocks.extend(br); routes.extend(rr); arrays.append(data)
    require(sorted(row['gid'] for row in blocks) == list(range(106)), 'incomplete native GID coverage')
    require(stats[0]['backend'] == stats[1]['backend'], 'backend mismatch')
    data = np.concatenate(arrays)
    actual_owners = np.empty(106, dtype=np.int32)
    for row in blocks:
        actual_owners[row['gid']] = row['owner']
    actual = layout(data, blocks, routes, actual_owners)
    for row, st in zip(actual['per_rank'], stats):
        for key in ('unique_donors', 'send_donors', 'receive_peers', 'send_peers'):
            require(row[key] == st[key], 'offline/public exchange '+key+' mismatch')
        require(row['remote_donors']['unique'] == st['remote_donors'], 'offline remote donor mismatch')
    ownership = json.loads(args.ownership_audit.read_text())
    require(ownership['schema'] == 'native-n12-ownership-audit-v1' and ownership['passed'],
            'ownership evidence not admitted')
    hashes[str(args.ownership_audit.resolve())] = sha(args.ownership_audit)
    projected = {}
    for ranks in (24, 96):
        owner_report = ownership['layouts'][str(ranks)]
        require(owner_report['native_unit_cost_assignment_exact'], 'unverified ownership model')
        owners = np.full(106, -1, dtype=np.int32)
        seen = []
        for row in owner_report['per_rank']:
            owners[row['blocks']] = row['rank']; seen.extend(row['blocks'])
        require(sorted(seen) == list(range(106)) and np.array_equal(owners, equal_cost_owners(ranks)),
                'observed owners differ from exact native float32 unit-cost assignment')
        projected[str(ranks)] = layout(data, blocks, routes, owners)
    result = dict(schema='native-bvals-route-audit-v1', passed=True,
        scope='CPU host-generated plan metadata;24/96 traffic is an offline projection onto previously verified Aurora GID ownership. No field transport, evolution, transport replacement, numerical equivalence or performance claim.',
        backend=stats[0]['backend'], capture_ranks=2, total_blocks=106,
        total_rows=sum(row['rows'] for row in blocks),
        total_donor_entries=sum(row['entries'] for row in blocks),
        distinct_target_block_donor_pairs=len(data),
        exterior_surface_dependencies=sum(row['exterior'] for row in blocks),
        first_failure_reasons={key: sum(row[key] for row in blocks) for key in REASONS},
        diagnostic_single_nonunit_near_one=sum(row['single_nonunit_near_one'] for row in blocks),
        actual_decomposition=actual, projected_decompositions=projected,
        notes=['First-failure buckets are mutually exclusive; near-unit diagnostic does not change eligibility.',
            'Counts include bulk mapped halos only; surface exchanges remain outside this audit.',
            'Bytes count payload once via receive totals, per stage; no headers, startup metadata or timing included.',
            'Full native slabs may include noneligible/unused/evolved cells and are not an implemented replacement.',
            'Generic CC uses62 fields for the two existing31-slot arrays;57 fields assumes a future compact residual pack.',
            'Unit-weight coalescing is compiler/host dependent; this CPU capture is not an actual Aurora donor-plan measurement.'],
        input_sha256=hashes, source_sha256={str(Path(__file__).resolve()): sha(__file__),
            str(Path(__file__).with_name('n12_ownership_audit.py').resolve()):
                sha(Path(__file__).with_name('n12_ownership_audit.py'))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('passed', 'total_rows', 'first_failure_reasons')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--build-proof', type=Path, required=True)
    parser.add_argument('--ownership-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    compare(parser.parse_args())
