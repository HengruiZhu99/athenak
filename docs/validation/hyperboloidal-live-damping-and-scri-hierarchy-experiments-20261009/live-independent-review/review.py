"""Independent exact algebra and 100-digit initial witness; no evolution."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import mpmath as mp
import sympy as sy


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MATH = HERE.parent / "live-damping-assessment/immutable-live-math-20261009"
HELPER = HERE.parent / "live-damping-control/live_damping_profile.hpp"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(MATH / "index.json") == "7f253e0ff6053543514831c3efb5b55cc1910aa52fdca875e6f9e9df3d40bef3"
    index = json.loads((MATH / "index.json").read_text())
    for name, digest in index["files"].items():
        assert sha(MATH / name) == digest
    assert sha(HELPER) == "69bbbc137486eb3398f94ed50a8b04583c375da049372b82b19330d8432fc153"
    O, kap, V, bO, th, K = sy.symbols("Omega kap V betaDotOmega Theta K", real=True)
    Oi, Vi, bOi, thi = sy.symbols("Omega_i V_i partial_i_betaDotOmega Theta_i", real=True)
    raw = O-1+2*bO/kap
    m = kap*V*raw
    mi = sy.diff(m, O)*Oi+sy.diff(m, V)*Vi+sy.diff(m, bO)*bOi
    full = sy.expand(Vi*(2*bO+kap*(O-1))+V*(2*bOi+kap*Oi))
    assert sy.simplify(mi-full) == 0
    base = (2*bO-kap)/O
    blended = (2*bO-kap-m)/O
    assert sy.simplify(blended-((1-V)*base-V*kap)) == 0
    assert sy.simplify(blended.subs(V, 1)+kap) == 0
    delta = -m*th/O
    assert sy.simplify(sy.diff(delta, bO)+2*V*th/O) == 0
    # delta Kij=-f gammaij, delta K=-3f gives the signs below.
    f = m*th/O
    delta_H = 2*K*(-3*f)-2*(-K*f)
    assert sy.simplify(delta_H+4*K*f) == 0
    delta_M = 2*(mi*th/O+m*thi/O-m*th*Oi/O**2)
    assert sy.simplify(delta_M-2*sy.diff(f, th)*thi
                       -2*sy.diff(f, O)*Oi-2*sy.diff(f, V)*Vi
                       -2*sy.diff(f, bO)*bOi) == 0
    S, a = sy.symbols("S a", positive=True)
    outer_bO = S/a**2-2*O/a
    eff = sy.expand(2*outer_bO+kap*O)
    fixed = 2*S/a**2+(kap-2*S/a**2)*O
    assert sy.simplify(eff-fixed-2*(S-2*a)*O/a**2) == 0
    assert sy.simplify((eff-fixed).subs(a, S/2)) == 0
    mp.mp.dps = 100
    r0, r1, width = mp.mpf('.05'), mp.mpf('.95'), mp.mpf('.9')

    def excess(r):
        w = 1/(1+mp.exp(width/(r-r0)-width/(r1-r)))
        wp = w*(1-w)*width*((r-r0)**-2+(r1-r)**-2)
        omega = 1-w*r*r
        op = -wp*r*r-2*w*r
        boost = 2*r*w
        alpha = mp.sqrt(omega*omega+boost*boost)
        L = omega-r*op
        beta = -boost*alpha/L
        shape = (1-r*r)**4*mp.exp(-4*r*r)
        return 2*(beta-mp.mpf('.02')*shape)*op+10*omega-10

    witness = mp.findroot(lambda r: mp.diff(excess, r), (mp.mpf('.103'), mp.mpf('.108')))
    value = excess(witness)
    assert value > mp.mpf('4.2e-9')
    frozen_result = json.loads((MATH / "result.json").read_text())
    expected = frozen_result["strict_bound_counterexample"]
    assert abs(witness-mp.mpf(expected["r_max_excess"])) < mp.mpf('1e-65')
    assert abs(value-mp.mpf(expected["kappa_eff_excess_above_10"])) < mp.mpf('1e-65')
    report = {
        "status": "PASS", "scope": "Independent algebra/helper inspection and100-digit initial strict-bound witness only; actual compiled gate review pending",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sources": {str(path.relative_to(ROOT)): sha(path) for path in
                    (Path(__file__), MATH / "index.json", HELPER)},
        "algebra": {"m": str(m), "m_i": str(full), "sigma_blended": str(blended),
                    "sigma_where_V1": "-kap", "delta_H": str(sy.factor(delta_H)),
                    "delta_M_i": str(delta_M),
                    "delta_P_Theta_beta_j": "-2V*Theta*Omega_j/Omega",
                    "outer_eff_reference": str(eff),
                    "difference_from_previous_fixed_profile": str(sy.factor(eff-fixed))},
        "witness": {"radial_negative_x_point": mp.nstr(witness, 90),
                    "effective_kappa_excess": mp.nstr(value, 90),
                    "positive_raw_kappa2": mp.nstr(value/10, 90)},
        "reviewed_bound_scope": "50-digit interval assessment uses actual S1,a.5,kappa10 initial angular shift .02,width.5; no extension to other a/S or evolution",
        "reviewed_derivative_scope": "Helper retains V_i,beta_i and Omega_ij; V is prescribed and kappa_input is spatially constant positive",
        "no_principal_derivative_addition": True,
        "limitations": ["No new tensor/native execution in this review.",
                        "At finite Theta the added beta value columns are required; the autonomous eight-constraint reference linearization does not establish nonlinear closure.",
                        "No invariant0<effective_kappa<=kappa_input region, energy, scri falloff, stability or BH claim.",
                        "V=.15-.3 is distinct from geometry cutoff .05-.95 and gauge cutoff .45-.85 for S1."]}
    (HERE / "math-review.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("PASS independent live-damping exact algebra/helper and100-digit strict-bound witness")
    print(json.dumps(report["witness"], indent=2))
    print("receipt SHA256", sha(HERE / "math-review.json"))


if __name__ == "__main__":
    main()
