#!/usr/bin/env python3
"""Falsify a shifted-stencil shortcut for derivative ghosts in a periodic TT model.

This is a necessary subsystem check, not the full Einstein/interface operator.
It tests using available four primary ghosts with locally shifted FD stencils.
The full-width control uses the global centered derivative at every ghost.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import sympy as sp


def weights(nodes, point):
    return [float(x) for x in sp.finite_diff_weights(1, nodes, point)[1][-1]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for order in [2, 4, 6]:
        radius = order//2
        for block_n in [8, 16]:
            n = 2*block_n
            h = 1/n
            offsets = list(range(-radius, radius+1))
            coeff = weights(offsets, 0)
            derivative = np.zeros((n, n))
            for i in range(n):
                for j, c in zip(offsets, coeff):
                    derivative[i, (i+j)%n] += c/h
            correction = np.zeros((n, n))
            for block in range(2):
                origin = block*block_n
                for i in range(block_n):
                    for off, c in zip(offsets, coeff):
                        ghost = i+off
                        if 0 <= ghost < block_n:
                            continue
                        start = max(-4, min(ghost-radius, block_n+4-order-1))
                        nodes = list(range(start, start+order+1))
                        row = np.zeros(n)
                        for node, value in zip(nodes, weights(nodes, ghost)):
                            row[(origin+node)%n] += value/h
                        correction[origin+i] += c/h*(row-derivative[(origin+ghost)%n])
            identity = np.eye(n)
            for shortcut in [False, True]:
                k = correction if shortcut else np.zeros_like(correction)
                # h_t=v, v_t=D q+K h, q_t=D v-lambda(q-D h).
                # K is the actual change from reconstruction at ghost consumers.
                generator = np.block([[0*identity, identity, 0*identity],
                                      [k, 0*identity, derivative],
                                      [derivative, derivative, -identity]])
                for epsilon in [0., .3]:
                    ko = np.zeros((n, n))
                    m = radius+1
                    for i in range(n):
                        for j in range(-m, m+1):
                            ko[i, (i+j)%n] -= epsilon/h*(-1.)**j*math.comb(2*m, m+j)/4**m
                    a = generator+np.kron(np.eye(3), ko)
                    eig = np.linalg.eigvals(a)
                    rate = float(eig.real.max())
                    normalized = rate/(1+np.linalg.norm(a, ord=2))
                    rows.append(dict(order=order, block_n=block_n, shortcut=shortcut,
                                     ko=epsilon, max_real_eigenvalue=rate,
                                     normalized_positive_rate=normalized,
                                     eigenvalue_gate='FAIL' if normalized>1e-10 else 'PASS',
                                     scope='necessary periodic TT subsystem test only'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
