#!/usr/bin/env python3
"""Exact no-KO periodic TT spatial operator; no full-system or AMR claim."""
import argparse
import json
from pathlib import Path

import sympy as s


def check(block_n):
    n = 2*block_n
    radius = 3
    derivative, correction = s.zeros(n), s.zeros(n)
    offsets = list(range(-radius, radius+1))
    coefficients = s.finite_diff_weights(1, offsets, 0)[1][-1]
    for i in range(n):
        for j, coefficient in zip(offsets, coefficients):
            derivative[i, (i+j)%n] += coefficient*n
    for block in range(2):
        origin = block*block_n
        for i in range(block_n):
            for j, coefficient in zip(offsets, coefficients):
                ghost = i+j
                if 0 <= ghost < block_n:
                    continue
                start = max(-4, min(ghost-radius, block_n+4-2*radius-1))
                nodes = list(range(start, start+2*radius+1))
                weights = s.finite_diff_weights(1, nodes, ghost)[1][-1]
                row = s.zeros(1, n)
                for node, value in zip(nodes, weights):
                    row[0, (origin+node)%n] += value*n
                correction[origin+i, :] += coefficient*n*(
                    row-derivative[(origin+ghost)%n, :])
    spatial = derivative*derivative+correction
    polynomial = spatial.charpoly().as_poly()
    intervals = polynomial.intervals(eps=s.Rational(1, 10**12))
    real_count = sum(m for interval, m in intervals)
    positive_count = sum(m for (lo, hi), m in intervals if lo > 0)
    row = dict(block_n=block_n, polynomial=str(s.factor(polynomial.as_expr())),
               degree=polynomial.degree(), real_roots_with_multiplicity=real_count,
               positive_real_roots=positive_count,
               max_real_root_interval=str(intervals[-1]),
               constant_kernel=spatial*s.ones(n, 1) == s.zeros(n, 1),
               square_free_spatial_polynomial=s.gcd(polynomial, polynomial.diff()).degree() == 0)
    assert real_count == n and positive_count == 0 and intervals[-1][0] == (0, 0)
    assert row['square_free_spatial_polynomial'] and row['constant_kernel']
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [check(n) for n in [8, 16]]
    args.output.write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
