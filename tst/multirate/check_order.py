#!/usr/bin/env python3
"""Exact colored-tree order certificate for the actual dyadic component method.

Source: Sarshar, Roberts & Sandu, arXiv:1804.07716v4, Appendix A.10.
We omit the zero-weight fifth RHS stage (all fifth coupling columns vanish).
The lambda>=2 matrix mislabeled slow<-fast in the HTML is fast<-slow, as
confirmed by internal-consistency row sums and the complete order conditions.

For coupling only between adjacent time levels, any colored tree through order
four spans at most four consecutive levels. A shifted window of the hierarchy
is repeated composition of the same verified local tableau. This supplies the
arbitrary-depth argument; a fifth level is checked as an additional regression.
This certifies smooth reduced ODE updates at common macro endpoints, not
intermediate dense output, projected/floored PDE stability, or nonlinear scri
accuracy. No third-party Python packages are required.
"""

import argparse
from fractions import Fraction as Q
from itertools import product
import json
import math
import subprocess


def zeros(n, m):
    return [[Q(0) for _ in range(m)] for _ in range(n)]


def matrix(rows):
    return [[Q(value) for value in row] for row in rows]


def mv(a, v):
    return [sum(x * y for x, y in zip(row, v)) for row in a]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def had(a, b):
    return [x * y for x, y in zip(a, b)]


A = matrix([[0, 0, 0, 0], [Q(2, 5), 0, 0, 0],
            [-Q(3, 20), Q(3, 4), 0, 0],
            [Q(19, 44), -Q(15, 44), Q(10, 11), 0]])
B = [Q(11, 72), Q(25, 72), Q(25, 72), Q(11, 72)]
S1 = matrix([[0, 0, 0, 0], [Q(4, 5), 0, 0, 0], [-Q(9, 5), 3, 0, 0],
             [Q(53, 11), -Q(95, 11), Q(20, 11), 4]])
F1 = matrix([[0, 0, 0, 0], [Q(1, 5), 0, 0, 0],
             [Q(9, 80), Q(3, 16), 0, 0], [0, Q(87, 88), -Q(43, 88), 0]])
F2 = matrix([[Q(11, 144), Q(25, 144), Q(25, 144), Q(11, 144)],
             [-Q(53, 450), Q(184, 225), 0, 0],
             [Q(391, 600), Q(89, 600), 0, 0],
             [0, -Q(93, 88), Q(3379, 1584), -Q(11, 144)]])


def tableau(depth):
    steps = [2 ** q for q in range(depth)]
    d, b, c = {}, [], []
    for q, n in enumerate(steps):
        b.append([v / n for _ in range(n) for v in B])
        block = zeros(4 * n, 4 * n)
        for k, i, j in product(range(n), range(4), range(4)):
            block[4 * k + i][4 * k + j] = A[i][j] / n
            for p in range(k):
                block[4 * k + i][4 * p + j] = B[j] / n
        d[q, q] = block
        c.append(mv(block, [1] * (4 * n)))
    for q in range(depth - 1):
        n = steps[q]
        sf, fs = zeros(4 * n, 8 * n), zeros(8 * n, 4 * n)
        for k, i, j in product(range(n), range(4), range(4)):
            for p in range(2 * k):
                sf[4 * k + i][4 * p + j] = B[j] / (2 * n)
            sf[4 * k + i][8 * k + j] = S1[i][j] / (2 * n)
            # S2 is zero for the four active rows.
            for p in range(k):
                fs[8 * k + i][4 * p + j] = B[j] / n
                fs[8 * k + 4 + i][4 * p + j] = B[j] / n
            fs[8 * k + i][4 * k + j] = F1[i][j] / n
            fs[8 * k + 4 + i][4 * k + j] = F2[i][j] / n
        d[q, q + 1], d[q + 1, q] = sf, fs
    return steps, d, b, c


def certify(depth, executable=None):
    steps, d, b, c = tableau(depth)
    cache, conditions = {}, []

    def ac(q, r):
        if (q, r) not in cache:
            cache[q, r] = mv(d[q, r], [1] * len(b[r]))
        return cache[q, r]

    for q, r in d:
        assert ac(q, r) == c[q], ("internal consistency", q, r)
    for q in range(depth):
        conditions.append(("leaf", (q,), dot(b[q], [1] * len(b[q])), Q(1)))
        neighbors = [r for r in range(depth) if (q, r) in d]
        for r in neighbors:
            conditions.append(("chain2", (q, r), dot(b[q], ac(q, r)), Q(1, 2)))
            for s in range(depth):
                if (r, s) not in d:
                    continue
                ars = ac(r, s)
                v = mv(d[q, r], ars)
                conditions.append(("chain3", (q, r, s), dot(b[q], v), Q(1, 6)))
                for t in range(depth):
                    if (s, t) in d:
                        value = dot(b[q], mv(d[q, r], mv(d[r, s], ac(s, t))))
                        conditions.append(("chain4", (q, r, s, t), value, Q(1, 24)))
                    if (r, t) in d:
                        value = dot(b[q], mv(d[q, r], had(ars, ac(r, t))))
                        conditions.append(("fork4", (q, r, s, t), value, Q(1, 12)))
                    if (q, t) in d:
                        value = dot(b[q], had(v, ac(q, t)))
                        conditions.append(("mixed4", (q, r, s, t), value, Q(1, 8)))
            for s in neighbors:
                value = dot(b[q], had(ac(q, r), ac(q, s)))
                conditions.append(("bush3", (q, r, s), value, Q(1, 3)))
                for t in neighbors:
                    value = dot(b[q], had(had(ac(q, r), ac(q, s)), ac(q, t)))
                    conditions.append(("bush4", (q, r, s, t), value, Q(1, 4)))
    for name, colors, value, expected in conditions:
        assert value == expected, (name, colors, value, expected)

    offsets = [sum(4 * n for n in steps[:q]) for q in range(depth)]
    count = sum(4 * n for n in steps)
    predecessors = [set() for _ in range(count)]
    for (q, r), block in d.items():
        for i, row in enumerate(block):
            predecessors[offsets[q] + i].update(offsets[r] + j for j, v in enumerate(row) if v)
    pending, done, waves = set(range(count)), set(), 0
    while pending:
        ready = {i for i in pending if predecessors[i] <= done}
        assert ready, ("dependency cycle", pending)
        done |= ready
        pending -= ready
        waves += 1

    if executable:
        actual = json.loads(subprocess.check_output([executable, "--tableau", str(depth)], text=True))
        seen = set()
        for row in actual["rows"]:
            target, donor = row["target"], row["donor"]
            q = max(q for q, offset in enumerate(offsets) if offset <= target)
            i = target - offsets[q]
            expected = d[q, donor][i]
            assert len(row["values"]) == len(expected)
            assert all(math.isclose(x, float(y), rel_tol=3e-15, abs_tol=3e-16)
                       for x, y in zip(row["values"], expected)), ("C++ tableau differs", q, i, donor)
            assert (q, i, donor) not in seen
            seen.add((q, i, donor))
        assert len(seen) == sum(len(block) for block in d.values())
    print(f"levels={depth}: {len(conditions)} exact colored-tree conditions; "
          f"{count} stages, {waves} dependency waves; all passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable")
    args = parser.parse_args()
    for number_of_levels in range(1, 6):
        certify(number_of_levels, args.executable)
