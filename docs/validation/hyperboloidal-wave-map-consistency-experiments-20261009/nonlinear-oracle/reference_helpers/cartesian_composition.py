"""Independent radial tensor composition check; no owner Cartesian API."""
import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import mpmath as mp

from radial_oracle import component, exact, radial, sha, side_at, textnum

HERE = Path(__file__).resolve().parent


def indices(n):
    return [m for m in itertools.product(range(n+1), repeat=3) if sum(m) <= n]


def delta(i, j):
    return int(i == j)


def scalar_cart(jet, xyz, m):
    r = mp.sqrt(sum(x*x for x in xyz))
    n = [x/r for x in xyz]
    axes = [i for i, count in enumerate(m) for _ in range(count)]
    order = len(axes)
    if order == 0:
        return jet[0]
    if order == 1:
        return jet[1]*n[axes[0]]
    if order == 2:
        i, j = axes
        return (jet[2]-jet[1]/r)*n[i]*n[j]+jet[1]/r*delta(i, j)
    if order == 3:
        i, j, k = axes
        return ((jet[3]-3*jet[2]/r+3*jet[1]/r**2)*n[i]*n[j]*n[k]
                +(jet[2]/r-jet[1]/r**2)
                *(delta(i, j)*n[k]+delta(i, k)*n[j]+delta(j, k)*n[i]))
    assert order == 4
    c4 = jet[4]-6*jet[3]/r+15*jet[2]/r**2-15*jet[1]/r**3
    c2 = jet[3]/r-3*jet[2]/r**2+3*jet[1]/r**3
    c0 = jet[2]/r**2-jet[1]/r**3
    pairs = 0
    for a, b in itertools.combinations(range(4), 2):
        rest = [k for k in range(4) if k not in (a, b)]
        pairs += delta(axes[a], axes[b])*n[axes[rest[0]]]*n[axes[rest[1]]]
    i, j, k, ell = axes
    return (c4*n[i]*n[j]*n[k]*n[ell]+c2*pairs+c0
            *(delta(i, j)*delta(k, ell)+delta(i, k)*delta(j, ell)
              +delta(i, ell)*delta(j, k)))


def inverse_power_jet(jet, r, power):
    out = []
    for n in range(len(jet)):
        value = 0
        for k in range(n+1):
            value += (math.comb(n, k)*jet[n-k]*(-1)**k
                      *mp.rf(power, k)*r**(-power-k))
        out.append(value)
    return out


def sub(m, *axes):
    counts = list(m)
    for i in axes:
        counts[i] -= 1
    return tuple(counts)


def vector_cart(jet, xyz, m, i):
    r = mp.sqrt(sum(x*x for x in xyz))
    divided = inverse_power_jet(jet, r, 1)
    value = xyz[i]*scalar_cart(divided, xyz, m)
    if m[i]:
        value += m[i]*scalar_cart(divided, xyz, sub(m, i))
    return value


def tensor_cart(jetr, jett, xyz, m, i, j):
    r = mp.sqrt(sum(x*x for x in xyz))
    c = inverse_power_jet([a-b for a, b in zip(jetr, jett)], r, 2)
    value = delta(i, j)*scalar_cart(jett, xyz, m)+xyz[i]*xyz[j]*scalar_cart(c, xyz, m)
    if m[i]:
        value += m[i]*xyz[j]*scalar_cart(c, xyz, sub(m, i))
    if m[j]:
        value += m[j]*xyz[i]*scalar_cart(c, xyz, sub(m, j))
    coefficient = m[i]*(m[j]-delta(i, j))
    if coefficient:
        value += coefficient*scalar_cart(c, xyz, sub(m, i, j))
    return value


def value(r, side, name):
    return component(r, side, name, False)+component(r, side, name, True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    recipe = json.loads((HERE/'cartesian-recipe.json').read_text())
    for key in ('radial_oracle.py', 'cartesian_composition.py'):
        assert sha(HERE/key) == recipe['sources'][key]
    all_rows = []
    precision_rows = []
    for digits in (100, 130):
        saved = []
        with mp.workdps(digits):
            for radius, *direction in recipe['points']:
                norm = mp.sqrt(sum(x*x for x in direction))
                xyz = [exact(radius)*x/norm for x in direction]
                r = mp.sqrt(sum(x*x for x in xyz))
                side = side_at(r)
                fields = {'omega': 4, 'alpha': 3, 'chi': 3, 'P': 3,
                          'beta': 3, 'lambda': 2, 'g_radial': 3,
                          'A_radial': 3, 'A_tangent': 3}
                jets = {name: [radial(r, name, n, side) for n in range(order+1)]
                        for name, order in fields.items()}
                def add(kind, name, m, expected, function, components=()):
                    actual = mp.diff(function, tuple(xyz), m)
                    err = abs(actual-expected)/max(1, abs(actual), abs(expected))
                    row = dict(digits=digits, radius=radius, direction=direction,
                               kind=kind, field=name, components=components,
                               multiindex=m, analytic=textnum(expected, digits),
                               direct=textnum(actual, digits), scaled=float(err),
                               passed=bool(err <= mp.mpf('1e-70')))
                    all_rows.append(row)
                    saved.append(row)
                for name in ('omega', 'alpha', 'chi', 'P'):
                    for m in indices(fields[name]):
                        add('scalar', name, m, scalar_cart(jets[name], xyz, m),
                            lambda x, y, z: value(mp.sqrt(x*x+y*y+z*z), side, name))
                for name in ('beta', 'lambda'):
                    for i in range(3):
                        def fn(x, y, z):
                            xx = (x, y, z)
                            rr = mp.sqrt(sum(t*t for t in xx))
                            return value(rr, side, name)*xx[i]/rr
                        for m in indices(fields[name]):
                            add('vector', name, m, vector_cart(jets[name], xyz, m, i), fn, (i,))
                for name, tangential in (('g_radial', 'chi'), ('A_radial', 'A_tangent')):
                    for i in range(3):
                        for j in range(i, 3):
                            def fn(x, y, z):
                                xx = (x, y, z)
                                rr = mp.sqrt(sum(t*t for t in xx))
                                t = value(rr, side, tangential)
                                return t*delta(i, j)+(value(rr, side, name)-t)*xx[i]*xx[j]/rr**2
                            for m in indices(3):
                                add('tensor', name, m, tensor_cart(jets[name], jets[tangential], xyz, m, i, j), fn, (i, j))
        precision_rows.append(saved)
    with mp.workdps(150):
        precision_max = max(float(abs(mp.mpf(lo['direct'])-mp.mpf(hi['direct']))
                                  /max(1, abs(mp.mpf(hi['direct']))))
                            for lo, hi in zip(*precision_rows))
    (out/'comparisons.json').write_text(json.dumps(all_rows, indent=2)+'\n')
    receipt = dict(kind='Independent Cartesian composition consistency only',
                   source_sha256=sha(__file__), recipe_sha256=sha(HERE/'cartesian-recipe.json'),
                   seconds=time.monotonic()-started, cases=len(all_rows),
                   scaled_max=max(row['scaled'] for row in all_rows),
                   precision_scaled_max=precision_max,
                   passed=all(row['passed'] for row in all_rows) and precision_max <= 1e-70,
                   origin_scope='Exact core profiles are constant, beta/A/Lambda zero; origin does not consume radial divisions.',
                   scope=recipe['scope'])
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))
    if not receipt['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
