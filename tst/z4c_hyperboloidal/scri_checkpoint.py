#!/usr/bin/env python3
"""Check a distributed native scri record against its accepted checkpoint.

Every retained bulk/edge/surface value is compared bitwise. Independently
recompute the saved derivative rows and normal lifts from native support.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'vis/python'))
import scri_reader as reader
from scri_geometry import endpoint_jets
from restart_roundtrip import parse, payload


def verify(checkpoint, record, tolerance=2e-11):
    info = parse(checkpoint)
    data = payload(info)
    first = reader.read(record)
    ranks = int(first['record'][4])
    paths = [record.with_name(record.name.replace('r00000000.scri', f'r{rank:08}.scri'))
             for rank in range(ranks)]
    if 'r00000000.scri' not in record.name:
        raise ValueError('supply the rank-zero record')
    parts = reader.load_record(paths)
    if int(first['record'][0]) != info['cycle'] or float(first['time'][0]) != info['time']:
        raise ValueError('checkpoint and scri record must have identical cycle and time')
    checked = 0
    errors = dict(sampled_residual_jets=0., normal_lift_jets=0., endpoint_jets=0.)
    for state, epoch in parts:
        n, origin, overlap = map(int, epoch['configuration'][:3])
        base, width = origin-overlap, n+2*overlap
        if data.shape[1] != width**3+n*n:
            raise ValueError('checkpoint layout does not match archive configuration')
        for name in ('bulk', 'edge', 'surface'):
            ids = epoch[name+'_ids'].astype(np.int64)
            if name == 'bulk':
                gid, k, j, i = ids[:, :4].T
                slot = ((k-base)*width+j-base)*width+i-base
            elif name == 'edge':
                gid, k, j = ids[:, :3].T
                slot = width**3+k*n+j
            else:
                gid, _, k, j = ids[:, :4].T
                slot = width**3+k*n+j
            expected = np.asarray(data[gid, slot, :31], dtype='=f8')
            actual = np.asarray(state[name+'_state'], dtype='=f8')
            if not np.array_equal(actual.view('=u8'), expected.view('=u8')):
                raise AssertionError(f'{name} accepted values differ on rank {state["record"][3]}')
            checked += actual.size
        independent = {
            'sampled_residual_jets': reader.reconstruct_sampled_jets(state, epoch),
            'normal_lift_jets': reader.normal_lifts(state, epoch),
            'endpoint_jets': reader.restored_endpoint_jets(state, epoch),
        }
        for name, expected in independent.items():
            actual = endpoint_jets(state, epoch) if name == 'endpoint_jets' else state[name]
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise AssertionError('nonfinite '+name)
            error = float(np.max(abs(actual-expected), initial=0))
            errors[name] = max(errors[name], error)
            if error > tolerance:
                raise AssertionError(f'{name} reconstruction error {error} exceeds {tolerance}')
    return dict(checkpoint=str(checkpoint), record=str(record), cycle=info['cycle'],
                time=info['time'], ranks=ranks, accepted_scalars_bitwise_equal=checked,
                reconstruction_max_abs_errors=errors, tolerance=tolerance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.checkpoint.resolve(), args.record.resolve())
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
