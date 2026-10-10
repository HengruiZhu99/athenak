"""Bounded exact arithmetic/remainder controls; never a domain certificate."""
import argparse
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    from admission import check
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    _, recipe, _, index_sha, _ = check(args.recipe, args.authorization, args.output, "units")
    # No mathematical imports before the complete exact admission above.
    from fractions import Fraction as Q
    from math import factorial
    from interval import Context, directed, floorlog2, pow2, encode, decode
    from producer_bounds import remainder_constants, quad_lower, regular

    out = Path(args.output)
    if (out / "report.json").exists():
        raise RuntimeError("unit report already exists")
    rows = []

    def require(condition, name):
        rows.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise RuntimeError("unit failed: " + name)

    ctx = Context(recipe["bits"])
    for x in [Q(0), Q(1, 3), Q(-1, 3), pow2(-2000), -pow2(2000), Q(7, 8)]:
        lo, hi = directed(x, ctx.bits, False), directed(x, ctx.bits, True)
        require(lo <= x <= hi, "directed-inclusion:" + str(x))
        require(decode(encode(lo)) == lo and decode(encode(hi)) == hi, "dyadic-roundtrip:" + str(x))
        if x:
            step = pow2(floorlog2(abs(x)) - ctx.bits + 1)
            require(hi - lo <= step, "directed-one-step:" + str(x))
    for a, b in [(Q(-3), Q(2)), (Q(1, 3), Q(2, 3)), (Q(-5), Q(-2))]:
        x = ctx.enclose(a, b)
        sq = x.square()
        values = [a * a, b * b] + ([Q(0)] if a <= 0 <= b else [])
        require(sq.lo <= min(values) <= max(values) <= sq.hi, "square:" + str((a, b)))
        y = ctx.enclose(Q(3, 7), Q(5, 4))
        prod = x * y
        targets = [a * Q(3, 7), a * Q(5, 4), b * Q(3, 7), b * Q(5, 4)]
        require(prod.lo <= min(targets) and prod.hi >= max(targets), "product:" + str((a, b)))
    try:
        ctx.point(1) / ctx.enclose(-1, 1)
    except ZeroDivisionError:
        require(True, "zero-denominator-rejection")
    else:
        require(False, "zero-denominator-rejection")
    for x in [Q(0), Q(1), Q(2), Q(7, 13), pow2(-2001), pow2(2001)]:
        s = ctx.sqrt(ctx.point(x))
        require(s.lo >= 0 and s.lo * s.lo <= x <= s.hi * s.hi, "sqrt-square:" + str(x))
    for y in [Q(0), Q(1, 16), Q(1, 32), Q(3, 2), Q(600)]:
        actual = ctx.exp_neg(ctx.point(y))
        # Independent exact rational alternating bounds through degree200/201,
        # after the same proved range reduction, with no outward arithmetic.
        m, z = 0, y
        while z > Q(1, 16):
            m, z = m + 1, z / 2
        low = sum(((-z) ** n) / factorial(n) for n in range(202))
        high = sum(((-z) ** n) / factorial(n) for n in range(201))
        low, high = directed(low, 512, False), directed(high, 512, True)
        for _ in range(m):
            low, high = directed(low * low, 512, False), directed(high * high, 512, True)
        require(actual.lo <= low <= high <= actual.hi, "exp-independent-alternating:" + str(y))
    K = recipe["series_order"]
    bounds = remainder_constants(K)
    for j, bound in enumerate(bounds):
        require(bound < pow2(-100), "fixed-K32-scaled-remainder:" + str(j))
    for j in range(K + 1):
        moment = Q(16, (2 * j + 1) * (2 * j + 3) * (2 * j + 5))
        independent = 2 * (Q(1, 2 * j + 1) - Q(2, 2 * j + 3) + Q(1, 2 * j + 5))
        require(moment == independent, "even-polynomial-moment:" + str(j))
        b = Q(8 * (j + 1) * (j + 2), factorial(2 * j + 5))
        require(b == moment / (8 * factorial(2 * j)), "series-coefficient:" + str(j))
        if j:
            require(j * b == moment / (16 * factorial(2 * j - 1)), "rho-derivative-coefficient:" + str(j))
    for sigma in [Q(7, 20), Q(1, 2)]:
        C, CT, Crho = regular(ctx, ctx.point(0), ctx.point(0), Q(0), sigma, K)
        require(C.lo == 0 == C.hi and Crho.lo == 0 == Crho.hi, "exact-origin-parity:" + str(sigma))
        target = 2 / (sigma * sigma)
        require(CT.lo <= target <= CT.hi, "exact-origin-CT:" + str(sigma))
    for q, L, expected in [(Q(-2), Q(3), Q(-2)), (Q(2), Q(1), Q(-1, 8)),
                            (Q(1), Q(2), Q(-3, 4)), (Q(0), Q(0), Q(0))]:
        require(quad_lower(q, L) == expected, "quadratic-branch:" + str((q, L)))
    if len(rows) != recipe["expected_unit_count"]:
        raise RuntimeError("unit registry count mismatch")
    from cache_units import run_cache_units
    cache_report = run_cache_units()
    if cache_report["case_count"] != recipe["expected_cache_unit_count"]:
        raise RuntimeError("cache-unit recipe count mismatch")
    (out / "cache-report.json").write_text(json.dumps(cache_report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    report = {"passed": True, "stage": "units", "source_index_sha256": index_sha,
              "case_count": len(rows), "cases": rows,
              "cache_unit_count": cache_report["case_count"], "cache_units_passed": True,
              "cache_report": "cache-report.json", "combined_unit_count": len(rows)+cache_report["case_count"],
              "global_slicing_certificate_passed": False,
              "domain_boxes_evaluated": 0, "mpmath_oracle_used": False}
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
