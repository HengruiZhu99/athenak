"""Exact constraint identities and numerical full-system spectra from production probes."""
import argparse
from pathlib import Path

import numpy as np
import sympy as s

from symbol import SELECT, T, old


def matrix_csv(path, case=None):
    data = np.genfromtxt(path, delimiter=",", names=True)
    if case is not None:
        data = data[data["case"] == case]
    result = s.zeros(55)
    for row in data:
        # These Minkowski coefficients are exact dyadic rationals. Recovering
        # them must stay within the measured binary64 tolerance.
        value = s.Rational(float(row["value"])).limit_denominator(65536)
        assert abs(float(value)-row["value"]) < 2.e-12
        result[int(row["row"]), int(row["col"])] = value
    return SELECT*result*T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    p = matrix_csv(args.run/"principal.csv", 0)
    j = matrix_csv(args.run/"source-jacobian.csv")
    case = np.genfromtxt(args.run/"cases.csv", delimiter=",", names=True)[0]
    rate = s.Rational(str(case["rate"]))
    k = s.symbols("k", real=True)
    constraints = []
    for d in range(3):
        row = s.zeros(1, 55)
        row[22+d] = 1
        if d == 0:
            row[0] = -s.I*k
        constraints.append(row)
        row = s.zeros(1, 55)
        row[43+d] = 1
        if d == 0:
            row[0] = row[18] = -2*s.I*k
        constraints.append(row)
        for a in range(3):
            row = s.zeros(1, 55)
            row[old.b_index(d, a)] = 1
            if d == 0:
                row[19+a] = -s.I*k
            constraints.append(row)
        for a, b in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]:
            row = s.zeros(1, 55)
            row[old.q_index(d, a, b)] = 1
            if d == 0:
                row[old.G[old.sym_index(a, b)]] = -s.I*k
            constraints.append(row)
    c = s.Matrix.vstack(*constraints)*T
    residual = c*(j+s.I*k*p)-(-rate+s.I*k/4)*c
    assert all(s.expand(value) == 0 for value in residual)
    assert c.rank() == 30
    print("PASS: all 30 independent reductions satisfy exact advective decay for every Fourier k")
    # Construct the exact reduction-manifold embedding from the configuration
    # derivatives, rather than inferring the physical spectrum from eigenvalues.
    embedding = s.zeros(55, 20)
    embedding[:22, :] = T[:22, :20]
    embedding[22, :] = s.I*k*embedding[0, :]
    embedding[43, :] = 2*s.I*k*(embedding[0, :]+embedding[18, :])
    for a in range(3):
        embedding[old.b_index(0, a), :] = s.I*k*embedding[19+a, :]
        for b in range(a, 3):
            embedding[old.q_index(0, a, b), :] = s.I*k*embedding[old.G[old.sym_index(a, b)], :]
    embedding = SELECT*embedding
    operator = j+s.I*k*p
    physical = operator[:20, :]*embedding
    assert all(s.expand(value) == 0 for value in operator*embedding-embedding*physical)
    shifted = physical-s.I*k*s.eye(20)/4
    z = s.symbols("z")
    expected = (z*z+2*k*k)*(z*z+k*k)**5*(z*z+z+k*k)**3*(z*z+2*z+k*k)
    assert s.factor(shifted.charpoly(z).as_expr()-expected) == 0
    print("PASS: exact all-k physical/GH polynomial: "
          "(z^2+2k^2)(z^2+k^2)^5(z^2+z+k^2)^3(z^2+2z+k^2)")
    print("PASS: full spectrum is non-growing for every real k; reduction factor=(z+lambda)^30")
    records = []
    for wave in (0.0, 0.01, 0.1, 1.0, 10.0, 100.0):
        operator = np.asarray(j+s.I*wave*p, dtype=complex)
        eigenvalues = np.linalg.eigvals(operator)
        maximum = float(np.max(eigenvalues.real))
        assert maximum < 2.e-8, (wave, maximum)
        records.append((wave, maximum))
        print(f"PASS: full production Fourier spectrum k={wave:g}; max real part {maximum:.3e}")
    np.savetxt(args.run/"fourier-spectrum.csv", records, delimiter=",",
               header="k,max_real_part", comments="")
    print("SCOPE: shifted flat linearization with kappa=1 and eta=0; no puncture stability claim")


if __name__ == "__main__":
    main()
