"""Compare full production affine probes with an independently transformed symbol."""
import argparse
from pathlib import Path

import numpy as np
import sympy as s

from symbol import T, SELECT, baseline, coordinate_damping_symbol, full_regular_symbol, old

_transforms = {}


def transform(matrix):
    """u_new=U*u_old under a constant, unimodular Cartesian coordinate map."""
    key = tuple(matrix)
    if key in _transforms:
        return _transforms[key]
    inverse = matrix.inv()
    columns = []
    for col in range(55):
        state = s.eye(55)[:, col]
        out = s.zeros(55, 1)
        for i in (0, 7, 17, 18):
            out[i] = state[i]
        for indices in (old.G, old.AT):
            tensor = s.Matrix(3, 3, lambda i, j: state[indices[old.sym_index(i, j)]])
            tensor = matrix.T*tensor*matrix
            for i in range(3):
                for j in range(i, 3):
                    out[indices[old.sym_index(i, j)]] = tensor[i, j]
        for indices in ((14, 15, 16), (19, 20, 21)):
            vec = inverse*s.Matrix([state[i] for i in indices])
            for i, index in enumerate(indices):
                out[index] = vec[i]
        for indices in ((22, 23, 24), (43, 44, 45)):
            vec = matrix.T*s.Matrix([state[i] for i in indices])
            for i, index in enumerate(indices):
                out[index] = vec[i]
        tensor = s.Matrix(3, 3, lambda i, j: state[old.b_index(i, j)])
        tensor = matrix.T*tensor*inverse.T
        for i in range(3):
            for j in range(3):
                out[old.b_index(i, j)] = tensor[i, j]
        for k in range(3):
            for i in range(3):
                for j in range(i, 3):
                    out[old.q_index(k, i, j)] = sum(
                        matrix[a, k]*matrix[b, i]*matrix[c, j]
                        *state[old.q_index(a, b, c)]
                        for a in range(3) for b in range(3) for c in range(3))
        columns.append(out)
    result = s.Matrix.hstack(*columns)
    _transforms[key] = result
    return result


def cartesian_principal(w, rho, switch, rate, normal):
    p = baseline(w, rho, switch)+coordinate_damping_symbol(w, rho, rate)
    result = s.zeros(50)
    for d in range(3):
        permutation = s.zeros(3)
        for a in range(3):
            permutation[a, (a+d) % 3] = 1
        u = SELECT*transform(permutation)*T
        result += normal[d]*u*p*u.inv()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    data = np.genfromtxt(args.run/"principal.csv", delimiter=",", names=True)
    cases = np.genfromtxt(args.run/"cases.csv", delimiter=",", names=True)
    states = np.genfromtxt(args.run/"states.csv", delimiter=",", names=True)
    maxima = []
    for case in cases:
        c = int(case["case"])
        measured = np.zeros((55, 55))
        rows = data[data["case"] == c]
        measured[rows["row"].astype(int), rows["col"].astype(int)] = rows["value"]
        w, rho, rate = [s.Rational(str(case[key])) for key in ("w", "rho", "rate")]
        normal = s.Matrix([s.Rational(str(case[key])) for key in ("nx", "ny", "nz")])
        metric_map = s.eye(3)
        if case["curved"]:
            metric_map[1, 0] = s.Rational(1, 4)
        canonical_normal = metric_map.inv().T*normal
        zeta = rho*w**3
        fraction = (zeta-s.Rational(1, 10))/s.Rational(2, 5)
        switch = 0 if zeta <= s.Rational(1, 10) else (
            1 if zeta >= s.Rational(1, 2) else fraction**2*(3-2*fraction))
        p = cartesian_principal(w, rho, switch, rate, canonical_normal)
        shift = s.Matrix([s.Rational(1, 4), -s.Rational(3, 8), s.Rational(1, 8)])
        p += shift.dot(normal)*s.eye(50)
        tangent = transform(metric_map)*T
        if c >= 8:
            seed = [s.Rational(str(row["value"])) for row in states[states["case"] == c]]
            at = s.Matrix(3, 3, lambda i, j: seed[old.AT[old.sym_index(i, j)]])
            qs = [s.Matrix(3, 3, lambda i, j: seed[old.q_index(k, i, j)]) for k in range(3)]
            tangent = T.copy()
            for col in range(50):
                dg = s.Matrix(3, 3, lambda i, j: T[old.G[old.sym_index(i, j)], col])
                tangent[13, col] += s.trace(dg*at)
                for k in range(3):
                    tangent[old.q_index(k, 2, 2), col] += s.trace(dg*qs[k])
            full = shift.dot(normal)*s.eye(55)
            for axis in range(3):
                rotation = s.zeros(3)
                for a in range(3):
                    rotation[a, (a+axis) % 3] = 1
                u = transform(rotation)
                rotated_at = rotation.inv().T*at*rotation.inv()
                full += normal[axis]*u*full_regular_symbol(w, rho, switch, rate, rotated_at)*u.inv()
            p = SELECT*full*tangent
            assert all(s.cancel(value) == 0 for value in full*tangent-tangent*p)
        assert np.isfinite(measured).all()
        error = np.einsum('ij,jk->ik', measured, np.asarray(tangent, dtype=float),
                          optimize=False)-np.asarray(tangent*p, dtype=float)
        maximum = float(np.max(np.abs(error)))
        maxima.append((c, maximum))
        if maximum > 2.e-11:
            index = np.unravel_index(np.argmax(np.abs(error)), error.shape)
            raise AssertionError(f"case {c}, component {index}: {error[index]}")
        print(f"PASS: production case {c}, full tangent matrix max error {maximum:.3e}")
    np.savetxt(args.run/"principal-errors.csv", maxima, delimiter=",",
               header="case,max_error", comments="", fmt=["%d", "%.17e"])
    print("PASS: axes, oblique covector, nonzero shift, and non-diagonal SPD metric")


if __name__ == "__main__":
    main()
