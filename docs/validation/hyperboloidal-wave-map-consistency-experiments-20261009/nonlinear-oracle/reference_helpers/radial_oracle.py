"""Independent direct-function/mp.diff oracle; no Taylor backend import."""
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import mpmath as mp

ROOT = Path(__file__).resolve().parents[3]
OWNER = ROOT / 'build-layer-research/boundary/einstein-coordinate-gauge-local-20261009'
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(value):
    n, d = float(value).as_integer_ratio()
    return mp.mpf(n) / d


def profiles(r, side):
    """Return independent background and small defect functions at r.

    Defects use expm1/log1p and factored differences. Derivatives of a tiny
    defect are evaluated separately from its O(1) background by mp.diff.
    """
    a = mp.mpf(1) / 2
    r0, r1, width = exact(.05), exact(.95), exact(.9)
    z, one = mp.mpf(0), mp.mpf(1)
    o, op = one-r*r, -2*r
    b0, l0 = r/a, one+r*r
    bg = dict(A_radial=z, A_tangent=z, L=one, P=z, alpha=one,
              b=z, beta=z, chi=one, complement=one, g_radial=one,
              **{'lambda': z}, omega=one, weight=z)
    de = {key: z for key in bg}
    if side == 'core':
        return bg, de
    if side == 'outer':
        bg.update(L=l0, P=-3/a, alpha=l0, b=b0, beta=-b0,
                  complement=z, omega=o, weight=one)
        return bg, de

    s, t = (r-r0)/width, (r1-r)/width
    g = -one/s+one/t
    gp = (one/s**2+one/t**2)/width
    e = mp.exp(g if side == 'left' else -g)
    small = e/(one+e)
    wp = e*gp/(one+e)**2
    if side == 'left':
        w = small
        do = w*(o-one)
        dl = w*(o-one-r*op)-r*wp*(o-one)
        da2 = 2*w*(o-one)+w*w*((o-one)**2+b0*b0)
        da = da2/(mp.sqrt(one+da2)+one)
        alpha, ell, omega = one+da, one+dl, one+do
        b = b0*w
        delta = da-dl
        dchi = mp.expm1(mp.mpf(2)/3*mp.log1p(delta/ell))
        dgr = mp.expm1(mp.mpf(4)/3*mp.log1p(-delta/alpha))
        de.update(L=dl, alpha=da, b=b, beta=-b*alpha/ell,
                  chi=dchi, complement=-w, g_radial=dgr,
                  omega=do, weight=w)
        de['P'] = -(3*w+r*omega*wp/ell)/a
        defect_d = da2-2*dl-dl*dl
    else:
        q = small
        w = one-q
        do = q*(one-o)
        db = -b0*q
        dl = q*(one-o+r*op)+r*wp*(one-o)
        da2 = q*(2*o*(one-o)-2*b0*b0)+q*q*((one-o)**2+b0*b0)
        da = da2/(mp.sqrt(l0*l0+da2)+l0)
        alpha, ell, omega = l0+da, l0+dl, o+do
        b = b0+db
        delta = da-dl
        dchi = mp.expm1(mp.mpf(2)/3*mp.log1p(delta/ell))
        dgr = mp.expm1(mp.mpf(4)/3*mp.log1p(-delta/alpha))
        bg.update(L=l0, P=-3/a, alpha=l0, b=b0, beta=-b0,
                  complement=z, omega=o, weight=one)
        de.update(L=dl, alpha=da, b=db, beta=-db-b*delta/ell,
                  chi=dchi, complement=q, g_radial=dgr,
                  omega=do, weight=-q)
        de['P'] = 3*q/a-r*omega*wp/(a*ell)
        defect_d = da2-2*l0*dl-dl*dl
    chi, gr = one+dchi, one+dgr
    de['A_radial'] = -2*gr*r*wp/(3*a*ell)
    de['A_tangent'] = chi*r*wp/(3*a*ell)
    # D=alpha^2-L^2 is differentiated independently below. Keeping the
    # numerator tiny avoids subtracting rounded O(1) logarithmic derivatives.
    de['_D'] = defect_d
    de['_ell'] = ell
    de['_alpha'] = alpha
    de['_chi'] = chi
    de['_gr'] = gr
    return bg, de


def side_at(r):
    if r <= exact(.05):
        return 'core'
    if r >= exact(.95):
        return 'outer'
    s = (r-exact(.05))/exact(.9)
    t = (exact(.95)-r)/exact(.9)
    return 'left' if -1/s+1/t <= 0 else 'right'


def component(r, side, name, defect):
    bg, de = profiles(r, side)
    if name != 'lambda' or side in ('core', 'outer'):
        return (de if defect else bg)[name]
    if not defect:
        return mp.mpf(0)
    ell, alpha, chi, gr = (de[k] for k in ('_ell', '_alpha', '_chi', '_gr'))
    lp = mp.diff(lambda x: profiles(x, side)[1]['_ell'], r)
    dp = mp.diff(lambda x: profiles(x, side)[1]['_D'], r)
    return (mp.mpf(4)/3*(lp*de['_D']-ell*dp/2)/(ell*alpha**2*gr)
            + 2*(de['g_radial']-de['chi'])/(r*chi*gr))


def radial(r, name, order, side=None):
    side = side_at(r) if side is None else side
    return sum(mp.diff(lambda x: component(x, side, name, defect), r, order)
               for defect in (False, True))


def textnum(x, digits):
    return mp.nstr(x, digits, strip_zeros=False)


def run_radial(recipe, out):
    outputs = []
    for digits in (100, 130):
        rows = []
        with mp.workdps(digits):
            for radius in recipe['radii']:
                r = exact(radius)
                row = {'radius_hex': float(radius).hex(), 'side': side_at(r), 'fields': {}}
                for field in recipe['radial_fields_in_output_order']:
                    name = field['name']
                    row['fields'][name] = [textnum(radial(r, name, n), digits)
                                           for n in field['ordinary_derivatives']]
                rows.append(row)
        path = out / ('oracle-%d.json' % digits)
        path.write_text(json.dumps(rows, indent=2)+'\n')
        outputs.append(rows)
    # Validate original binary64 output against both independently differentiated
    # precisions, retaining every error and tail decision rather than just maxima.
    raw = (OWNER / 'reference-gate-attempt001/reference-release.stdout').read_text()
    native = [[float(v) for v in line.split()] for line in raw.splitlines()]
    assert len(native) == 38 and all(len(row) == 57 for row in native)
    comparisons = []
    with mp.workdps(150):
        for i, radius in enumerate(recipe['radii']):
            assert native[i][0].hex() == float(radius).hex()
            col = 3
            for field in recipe['radial_fields_in_output_order']:
                name = field['name']
                for n in field['ordinary_derivatives']:
                    lo = mp.mpf(outputs[0][i]['fields'][name][n])
                    hi = mp.mpf(outputs[1][i]['fields'][name][n])
                    actual = exact(native[i][col])
                    precision_scaled = abs(lo-hi)/max(1, abs(hi))
                    scaled = abs(actual-hi)/max(1, abs(hi))
                    cutoff_small = min(mp.mpf(outputs[1][i]['fields'][key][0])
                                       for key in ('weight', 'complement'))
                    # Endpoint tails are identified by the cutoff VALUE, not a
                    # cancellation-small derivative (e.g. even orders at r=.5).
                    tail = (name in ('weight', 'complement')
                            and cutoff_small <= mp.mpf('.25')
                            and (n > 0 or abs(hi) <= mp.mpf('.25')))
                    relative = abs(actual-hi)/abs(hi) if hi else mp.mpf(0)
                    precision_relative = abs(lo-hi)/abs(hi) if hi else mp.mpf(0)
                    relative_required = tail and abs(hi) >= mp.mpf('1e-300')
                    nonzero_required = tail and abs(hi) >= mp.mpf('1e-320')
                    ok = (scaled <= mp.mpf('2e-10') and precision_scaled <= mp.mpf('1e-80')
                          and (not relative_required or relative <= mp.mpf('2e-8'))
                          and (not relative_required or precision_relative <= mp.mpf('1e-80'))
                          and (not nonzero_required or actual != 0))
                    comparisons.append(dict(radius=radius, field=name, derivative=n,
                                            native=native[i][col], oracle=textnum(hi, 132),
                                            scaled=float(scaled), precision_scaled=float(precision_scaled),
                                            relative=float(relative),
                                            precision_relative=float(precision_relative),
                                            tail_relative_required=relative_required,
                                            tail_nonzero_required=nonzero_required, passed=bool(ok)))
                    col += 1
            assert col == 57
    (out/'radial-comparisons.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    return dict(cases=len(comparisons), passed=all(x['passed'] for x in comparisons),
                scaled_max=max(x['scaled'] for x in comparisons),
                precision_scaled_max=max(x['precision_scaled'] for x in comparisons),
                tail_relative_max=max(x['relative'] for x in comparisons if x['tail_relative_required']),
                tail_relative_cases=sum(x['tail_relative_required'] for x in comparisons),
                tail_nonzero_cases=sum(x['tail_nonzero_required'] for x in comparisons),
                failed=[x for x in comparisons if not x['passed']])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    plan = json.loads((HERE/'recipe.json').read_text())
    for row in plan['inputs']:
        assert sha(ROOT/row['path']) == row['sha256'], row
    assert sha(__file__) == plan['source_sha256']
    recipe = json.loads((OWNER/'reference-local-recipe.json').read_text())
    summary = run_radial(recipe, out)
    receipt = dict(kind='Independent high-order radial reference-jet oracle',
                   source_sha256=sha(__file__), recipe_sha256=sha(HERE/'recipe.json'),
                   python=platform.python_version(), mpmath=mp.__version__,
                   seconds=time.monotonic()-started, radial=summary,
                   passed=summary['passed'], scope=plan['scope'])
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))
    if not receipt['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
