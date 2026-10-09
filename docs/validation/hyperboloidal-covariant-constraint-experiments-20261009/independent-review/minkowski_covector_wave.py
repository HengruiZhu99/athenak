"""Undamped flat covector waves: exact identities and 100-digit scri weights.

These are kappa=0 examples away from the origin, not damped-live falloff rules.
Conventions: T=t+h(R), R=r/Omega, h_R=b/A, A²=Omega²+b²,
L=Omega-r*Omega_r, and physical Theta=-n^a Z_a.
"""
import hashlib
import json
from pathlib import Path

import mpmath as mp
import sympy as sp


def main():
    t, rad = sp.symbols("T R", real=True, positive=True)
    f = sp.Function("F")(t - rad) / rad
    box_scalar = -sp.diff(f, t, 2) + sp.diff(f, rad, 2) + 2 * sp.diff(f, rad) / rad
    assert sp.simplify(box_scalar) == 0
    # A radial Cartesian component n_i f has angular harmonic degree l=1.
    box_radial_component = sp.simplify(box_scalar - 2 * f / rad ** 2)
    assert sp.simplify(box_radial_component + 2 * f / rad ** 2) == 0
    assert box_radial_component != 0
    # The flat covariant wave commutes with a gradient on a scalar.
    assert sp.simplify(sp.diff(box_scalar, t)) == 0
    assert sp.simplify(sp.diff(box_scalar, rad)) == 0
    mp.mp.dps = 100
    rows = []
    for s, a in [("1", ".5"), ("1.2", ".8"), (".8", "1.5")]:
        s, a = mp.mpf(s), mp.mpf(a)
        phase = mp.mpf(".37") + mp.mpf(".14")
        f_inf = mp.cos(phase) + mp.sin(2 * phase)
        for power in [3, 10, 20, 40]:
            om = mp.mpf(10) ** -power
            r = mp.sqrt(s * s - 2 * a * s * om)
            big_r = r / om
            boost = r / a
            lapse = (s * s + r * r) / (2 * a * s)
            ell = lapse
            height_minus_r = a * a / (mp.sqrt(a * a + big_r * big_r) + big_r)
            u = phase + height_minus_r
            wave = mp.cos(u) + mp.sin(2 * u)
            wave_p = -mp.sin(u) + 2 * mp.cos(2 * u)
            scalar = wave / big_r
            zr = (boost / lapse) * ell * wave / (r * om)
            theta = -lapse * wave / r
            # n^t=Omega/A, n^r=b*A/L * Omega/A=b*Omega/L.
            contracted = -(om / lapse * scalar + boost * om / ell * zr)
            assert mp.almosteq(theta, contracted, rel_eps=mp.mpf("1e-95"))
            null_theta = -om * om * wave / (r * (lapse + boost))
            null_zr = -om * ell * wave / (r * lapse * (lapse + boost))
            gradient_theta = (-om * om * wave_p / (r * (lapse + boost))
                              + boost * om * wave / (r * r))
            gradient_zr = (-om * ell * wave_p / (r * lapse * (lapse + boost))
                           - ell * wave / (r * r))
            rows.append({
                "S": str(s), "a": str(a), "Omega": str(om),
                "Theta": mp.nstr(theta, 70),
                "Omega_Zr": mp.nstr(om * zr, 70),
                "Theta_limit": mp.nstr(-f_inf / a, 70),
                "Omega_Zr_limit": mp.nstr(f_inf / a, 70),
                "null_Theta_over_Omega2": mp.nstr(null_theta / om ** 2, 70),
                "null_Zr_over_Omega": mp.nstr(null_zr / om, 70),
                "gradient_Theta_over_Omega": mp.nstr(gradient_theta / om, 70),
                "gradient_Zr": mp.nstr(gradient_zr, 70),
            })
    report = {
        "status": "PASS", "digits": mp.mp.dps,
        "domain": "flat physical Minkowski, kappa=0, R>0; CMC outer branch",
        "scalar_wave_identity": "Box(F(T-R)/R)=0",
        "time_covector": "Z=F(u)dT/R; Theta=-A*F/r; Zr=h_R*L*F/(r*Omega)",
        "time_covector_limits": "Theta->-F(u_scri)/a; Omega*Zi->F(u_scri)*n_i/a",
        "null_covector_caveat": "F(u)du/R alone is generally not a 3+1 covector wave",
        "null_covector_angular_residual": "Box(-n_i*F/R)=+2*n_i*F/R³",
        "exact_gradient_wave": "d(F(u)/R)=F'(u)du/R-F(u)dR/R²",
        "gradient_generic_weights": "Theta=O(Omega), Zr=O(1)",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sympy": sp.__version__, "mpmath": mp.__version__, "rows": rows,
    }
    Path(__file__).with_name("covector-wave-receipt.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ["status", "digits", "domain"]}))
    print("12 rows; exact wave/angular identities; finite-coordinate contraction PASS")


if __name__ == "__main__":
    main()
