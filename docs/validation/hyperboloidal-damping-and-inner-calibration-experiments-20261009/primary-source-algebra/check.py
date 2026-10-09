"""Exact Eq19 determinant/Routh check; no variable-coefficient stability claim."""
import hashlib
import json
from pathlib import Path

import sympy as sp


def main():
    s, k, w, rho = sp.symbols("s k w rho", real=True)
    matrix = sp.Matrix([
        [-s*s-w*w-k*(2+rho)*s, k*sp.I*w],
        [-k*rho*sp.I*w, -s*s-w*w-k*s]])
    polynomial = sp.Poly(sp.expand(matrix.det()), s)
    expected = (s*s+k*(2+rho)*s+w*w)*(s*s+k*s+w*w)-k*k*rho*w*w
    assert sp.simplify(polynomial.as_expr()-expected) == 0
    one, a3, a2, a1, a0 = polynomial.all_coeffs()
    assert one == 1
    delta2 = sp.factor(a3*a2-a1)
    delta3 = sp.factor(a3*a2*a1-a1*a1-a3*a3*a0)
    assert sp.simplify(delta2-k*(3+rho)*(w*w+k*k*(2+rho))) == 0
    assert sp.simplify(delta3-2*k**4*(3+rho)**2*w*w*(1+rho)) == 0
    positive_rho = {k: 1, rho: sp.Rational(1, 10), w: sp.Rational(1, 10)}
    counter = polynomial.as_expr().subs(positive_rho)
    assert counter.subs(s, 0) < 0
    roots = sp.nroots(counter, n=80, maxsteps=200)
    positive = [root for root in roots if abs(sp.im(root)) < sp.Float("1e-70")
                and sp.re(root) > 0]
    assert len(positive) == 1
    safe = []
    for r in [sp.Rational(-1, 5), sp.Rational(-1, 10), sp.Integer(0)]:
        for frequency in [sp.Rational(1, 100), sp.Rational(1, 10), 1, 10, 100]:
            substitutions = {k: 1, rho: r, w: frequency}
            assert all(q.subs(substitutions) > 0 for q in [a3, a2, a1, a0, delta2, delta3])
            safe.append({"rho": str(r), "omega_over_kappa": str(frequency), "Hurwitz": True})
    out = {
        "status": "PASS", "paper": "Gundlach, Calabrese, Hinder, Martin-Garcia, Constraint damping in the Z4 formulation and harmonic gauge",
        "primary_url": "https://arxiv.org/pdf/gr-qc/0504114v2",
        "version": "arXiv:gr-qc/0504114v2, 14 July 2005",
        "equation_locations": "Eq2 and Eqs6-8: PDF page2; Eq19 and prose after Eq23: PDF page3",
        "mapping": "rho=kappa2, kappa=physical kappa1; K sign and future normal agree",
        "polynomial": str(polynomial.as_expr()),
        "delta2": str(delta2), "delta3": str(delta3),
        "all_nonzero_frequency_flat_Hurwitz_range": "kappa>0 and -1<rho<=0",
        "rho_positive_restriction": "For rho>0, stability additionally requires omega²>kappa²rho; below that threshold at least one positive real root exists",
        "positive_counterexample": {"kappa": 1, "rho": ".1", "omega": ".1",
                                    "polynomial": str(counter), "constant": str(counter.subs(s, 0)),
                                    "positive_root_80digits": str(positive[0])},
        "safe_parameter_checks": safe,
        "printed_equation_vs_prose": "Printed Eq19 gives the restriction above; the later broader rho>-1 prose is not a sufficient all-frequency claim. No author-intent inference.",
        "scope": "Constant-coefficient inertial flat frozen calculation only; variable damping/normal gradients, curved lower-order terms, scri limits, nonlinear closure and native RK remain separate gates",
        "profile_condition": "For kappa_input(1+kappa2)=2S/a²+d*Omega, all-frequency flat interval requires 0<2S/a²+d*Omega<=kappa_input at each point",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sympy": sp.__version__,
    }
    Path(__file__).with_name("receipt.json").write_text(json.dumps(out, indent=2, allow_nan=False)+"\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
