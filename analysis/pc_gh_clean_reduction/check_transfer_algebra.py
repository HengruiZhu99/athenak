#!/usr/bin/env python3
"""Periodic tensor-factor transfer model; not a finite-block AMR qualification."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[2]


def along(matrix, array, axis):
    return np.moveaxis(np.tensordot(matrix, array, axes=(1, axis)), 0, axis)


def transfer(matrix, array, dim):
    for axis in range(dim):
        array = along(matrix, array, axis)
    return array


def derivative(array, axis, length, order):
    offsets = list(range(-order//2, order//2+1))
    weights = sp.finite_diff_weights(1, offsets, 0)[1][-1]
    return sum(float(w)*np.roll(array, -i, axis=axis)
               for i, w in zip(offsets, weights))*array.shape[axis]/length


def prolong(n, order, source):
    ng = 2 if order == 2 else 4
    raw = re.search(rf'const Real wght{ng}\[{ng+1}\] = \{{([^}}]+)\}}', source)[1]
    exact = [sp.Rational(v.strip()) for v in raw.split(',')]
    offsets = np.arange(ng+1)-ng//2
    for power in range(ng+1):
        assert sum(w*int(i)**power for w, i in zip(exact, offsets)) == sp.Rational(-1, 4)**power
    matrix = np.zeros((2*n, n))
    for i in range(n):
        matrix[2*i, (i+offsets)%n] = np.array(exact, float)
        matrix[2*i+1, (i+offsets)%n] = np.array(exact[::-1], float)
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = (ROOT/'src/mesh/mesh_refinement.cpp').read_text()
    # Compile the production-location algebra primitive, independently exercised
    # below on dense rows, then use tensor contractions for multidimensional grids.
    library = args.output/'transfer_probe.dylib'
    command = ['clang++', '-std=c++17', '-O2', '-shared', '-fPIC',
               '-I'+str(ROOT/'src'), str(Path(__file__).with_name('transfer_probe.cpp')),
               '-o', str(library)]
    subprocess.run(command, check=True)
    lib = ctypes.CDLL(str(library.resolve()))
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    lib.transfer_probe.argtypes = [ctypes.c_int, ctypes.c_int]+[pointer]*5
    rng = np.random.default_rng(20260907)
    compiled_error = 0.0
    for rows, cols in [(8, 8), (16, 8), (8, 16)]:
        weights = rng.normal(size=(rows, cols))
        auxiliary, target = rng.normal(size=(2, cols))
        destination = rng.normal(size=rows)
        actual = np.empty(rows)
        lib.transfer_probe(rows, cols, weights, auxiliary, target, destination, actual)
        expected = destination+weights@(auxiliary-target)
        compiled_error = max(compiled_error, float(np.max(abs(actual-expected))/(1+np.max(abs(expected)))))
    assert compiled_error < 2e-12
    results = []
    for order in [2, 4, 6]:
        negative = False
        for dim in [2, 3]:
            lengths = [1., 1.3, 1.7][:dim]
            for n in [8, 16, 32]:
                matrix = prolong(n, order, source)
                coords = np.meshgrid(*[(np.arange(n)+.5)/n for _ in range(dim)], indexing='ij')
                phase = sum(2*np.pi*x for x in coords)
                w = .7+.08*np.sin(phase)
                rho = 1.1+.07*np.cos(phase+.3)
                # Five arbitrary smooth chart potentials give determinant-one
                # non-diagonal geometry when reconstructed; no component g transfer.
                potentials = [w, rho*w]+[.03*np.sin((j+1)*phase+.2*j) for j in range(5)]
                potentials += [.02*np.cos(phase+.3*j) for j in range(3)]
                fine = [transfer(matrix, v, dim) for v in potentials]
                fine[1] = transfer(matrix, rho, dim)*transfer(matrix, w, dim)
                seed = .001*np.cos(phase+.7)
                worst = injection = curl = 0.
                for family, (coarse_u, fine_u) in enumerate(zip(potentials, fine)):
                    coarse_g = [derivative(coarse_u, a, lengths[a], order) for a in range(dim)]
                    fine_target = [derivative(fine_u, a, lengths[a], order) for a in range(dim)]
                    fine_g = []
                    for a in range(dim):
                        error = seed*(a+1)*(family+1)
                        old = transfer(matrix, coarse_g[a]+error, dim)
                        residual = transfer(matrix, error, dim)
                        new = fine_target[a]+transfer(matrix, coarse_g[a]+error-coarse_g[a], dim)
                        scale = 1+np.max(abs(fine_target[a]))+np.max(abs(residual))
                        worst = max(worst, float(np.max(abs(new-fine_target[a]-residual))/scale))
                        injection = max(injection, float(np.max(abs(old-fine_target[a]-residual))))
                        fine_g.append(new-residual)
                    for a in range(dim):
                        for b in range(a):
                            value = derivative(fine_g[b], a, lengths[a], order)-derivative(fine_g[a], b, lengths[b], order)
                            curl = max(curl, float(np.max(abs(value))))
                assert worst < 2e-12
                negative |= injection > 1e-8
                results.append(dict(order=order, dimension=dim, n=n,
                    normalized_residual_identity_error=worst,
                    independent_transfer_injection_max=injection,
                    reconstructed_gradient_curl_max=curl))
        assert negative, order
    result = dict(status='PASS', scope=__doc__, command=command,
                  compiled_normalized_error=compiled_error,
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(), rows=results,
                  exclusions=['finite-block restriction', 'MPI', 'CUDA', 'halo support',
                              'coupled amplification', 'evolution', 'production communication linkage'])
    (args.output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
