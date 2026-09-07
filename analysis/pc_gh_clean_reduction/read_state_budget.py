#!/usr/bin/env python3
"""Read lossless signed state budgets; ghost validity is not assumed."""
import argparse
import json
from pathlib import Path

import numpy as np

FIELDS = ('cycle stage operation event rank block level nvar nk nj ni '
          'is ie js je ks ke before_ghost_valid after_ghost_valid endian_marker').split()


def records(path):
    with path.open('rb') as file:
        while True:
            magic = file.read(8)
            if not magic:
                return
            if magic != b'PCGHBUD1':
                raise ValueError('unknown or truncated state-budget record')
            raw = file.read(160)
            if len(raw) != 160:
                raise ValueError('truncated state-budget header')
            endian = '<' if int.from_bytes(raw[-8:], 'little') == 0x0102030405060708 else '>'
            header = dict(zip(FIELDS, np.frombuffer(raw, dtype=endian+'i8').tolist()))
            if header['endian_marker'] != 0x0102030405060708:
                raise ValueError('bad state-budget endian marker')
            geometry = np.frombuffer(file.read(64), dtype=endian+'f8')
            if len(geometry) != 8:
                raise ValueError('truncated geometry')
            shape = tuple(header[k] for k in ['nvar', 'nk', 'nj', 'ni'])+(3,)
            if min(shape) < 1 or np.prod(shape) > 2**31:
                raise ValueError('invalid or excessive record dimensions')
            raw = file.read(8*int(np.prod(shape)))
            if len(raw) != 8*np.prod(shape):
                raise ValueError('truncated state-budget payload')
            data = np.frombuffer(raw, dtype=endian+'f8').reshape(shape)
            yield header, geometry, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path)
    parser.add_argument('--oracle', action='store_true')
    args = parser.parse_args()
    summaries = []
    for header, geometry, data in records(args.file):
        before, after, delta = (data[..., i] for i in range(3))
        finite = np.isfinite(data).all()
        if finite and not np.array_equal(after-before, delta):
            raise AssertionError('stored signed increment does not match same-cell subtraction')
        if args.oracle:
            n, k, j, i = np.indices(before.shape)
            seed = .01*(n+1)+.001*i
            correction = .1*((i+2*j+3*k+5*n)%7-3)
            if header['operation'] == 1001:
                np.testing.assert_allclose(before, seed, rtol=0, atol=2e-15)
                np.testing.assert_allclose(delta, correction, rtol=0, atol=2e-15)
            elif header['operation'] == 1002:
                np.testing.assert_allclose(before, seed+correction, rtol=0, atol=2e-15)
                assert np.count_nonzero(delta) == 0
            elif header['operation'] in [0, 7, 8, 1, 2] and header['event'] < 5:
                # AthenaK initializes the uniform Minkowski ghost state through
                # these actual task wrappers before calling the final adapter.
                assert np.count_nonzero(delta) == 0
            else:
                raise AssertionError('unexpected test record')
        summaries.append(dict(header=header, geometry=geometry.tolist(), finite=bool(finite),
                              delta_linf=float(abs(delta).max()),
                              scalar_max_difference=float(after.max()-before.max()),
                              signed_sum=float(delta.sum()),
                              note='state components including ghosts; not constraint norms'))
    if args.oracle:
        assert [r['header']['operation'] for r in summaries] == [0, 7, 8, 1, 2, 1001, 1002]
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
