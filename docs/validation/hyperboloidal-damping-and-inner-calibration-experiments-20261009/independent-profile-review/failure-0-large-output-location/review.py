"""Independent read-only profile source, algebra and finite-Omega matrix review."""
import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sp


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
GATE = ROOT / "build-layer-research/continuum/damping-profile-control/immutable-C0-profile-local-v2-20261009"
EXPECTED = "c9180b5bbedb2a0069a54853768a96f8b0f69736c30a8bc9c277b77f11fc39ea"
V1 = "5f12b22de8c9fb75761cf6c723d55938f4dfdc144505d3368a45b93f94d5ec4d"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix(value):
    a = np.asarray(value)
    assert np.isfinite(a).all()
    return a[:, :, 0] + 1j*a[:, :, 1]


def verify_index(path, digest):
    assert sha(path / "index.json") == digest
    index = json.loads((path / "index.json").read_text())
    for name, value in index["files"].items():
        assert sha(path / name) == value
    for name, value in index["large_outputs_outside_snapshot"].items():
        p = GATE.parent / name
        assert sha(p) == value["sha256"] and p.stat().st_size == value["bytes"]
    return index


def main():
    np.seterr(all="raise")
    index = verify_index(GATE, EXPECTED)
    old = verify_index(GATE.parent / "immutable-C0-profile-local-20261009", V1)
    shared = set(old["files"]) & set(index["files"])
    changed = {name for name in shared if old["files"][name] != index["files"][name]}
    assert changed == {"REPORT.md"}
    receipt = json.loads((GATE / "receipt.json").read_text())
    assert receipt["passed_profile_local_gates"] and not receipt["global_native_or_scri_stability_accepted"]
    assert receipt["sources_unchanged"]
    assert receipt["source_before"] == receipt["source_after"]
    assert len(receipt["source_before"]) == 378 and len(receipt["commands"]) == 12
    assert all(row["returncode"] == 0 and row["stderr"] == "" for row in receipt["commands"])
    assert all(sha(ROOT / name) == digest for name, digest in receipt["source_after"].items())
    for row in receipt["commands"]:
        if "stdout_file" in row:
            assert sha(GATE / row["stdout_file"]) == row["stdout_sha256"]
    assert sha(GATE / "damping_profile.hpp") == "64ba382f509e81fe347b96e915d3933ee7188c97fd6a054524b1aa187a891532"
    helper = (GATE / "profile_helpers.hpp").read_text()
    assert "ConformalRHS(u,o,D(kappa)/u.alpha.value" in helper
    assert "TensorC1Additions" not in helper and "ResearchAddC1" not in helper
    # Independent exact tensor-invariant and coefficient-gradient normalization.
    o, S, a, kap, theta, theta_i, o_i, K = sp.symbols("Omega S a kappa Theta Theta_i Omega_i K", positive=True)
    crit = 2*S/a**2
    d = kap-crit
    m = -d*(1-o)
    k2 = m/kap
    eff = sp.factor(kap*(1+k2))
    assert sp.simplify(eff-crit-d*o) == 0 and k2.subs(o, 1) == 0
    isolated = 2*sp.diff(m, o)*o_i*theta/o
    total = 2*sp.diff(m/o, o)*o_i*theta
    assert sp.simplify(isolated-2*d*o_i*theta/o) == 0
    assert sp.simplify(total-2*d*o_i*theta/o**2) == 0
    # Delta Kij=-X gammaij gives Delta H=2K(-3X)-2K(-X).
    X = m*theta/o
    deltaH = sp.expand(2*K*(-3*X)-2*K*(-X))
    assert sp.simplify(deltaH+4*K*m*theta/o) == 0
    alpha = S/a-o
    alpha_w = -S/a**2+2*o/a
    sigma0 = sp.cancel((-2*alpha_w-eff)/o)
    sigma1 = sp.cancel((2*alpha/a-eff)/o)
    assert sp.simplify(sigma0+4/a+d) == sp.simplify(sigma1+2/a+d) == 0
    assert sp.diff(sigma0, o) == sp.diff(sigma1, o) == 0
    # Nonlinear actual tensor checks include Release and ASan/UBSan endpoints.
    tensor = json.loads((GATE / "tensor-gate.json").read_text())
    assert tensor == json.loads((GATE / "tensor-gate-debug.json").read_text())
    assert tensor["rows"] == 4004 and tensor["Einstein_addition_max"] == 0
    assert tensor["nonlinear_parts_identity_error"] < 1e-14
    assert tensor["exact_core_kappa2_zero"] and tensor["no_new_double_pole"]
    principal = json.loads((GATE / "principal.log").read_text())
    assert principal["passed_kernel_cases"] == 360 and principal["max_kernel_symbol_error"] < 1e-12
    # Independently re-evaluate sampled matrix identity, roots and native-step RK3.
    full = json.loads((GATE.parent / "full20.json").read_text())
    assert len(full) == 760
    groups, roots, rk = {}, [], []
    for row in full:
        L = matrix(row["L"])
        key = tuple(row[name] for name in ("a", "perturb", "r", "Omega", "k", "oblique"))
        groups.setdefault(key, {})[row["profile"]] = L
        weights = np.array([1.]*12+[1/max(row["k"], 1.)]*8)
        B = weights[:, None]*L/weights[None, :]
        ev = np.linalg.eigvals(B)
        roots.append({name: row[name] for name in ("a", "profile", "perturb", "r", "Omega", "k", "oblique")} | {"max_real": float(ev.real.max())})
        if row["a"] == .5 and row["perturb"] == 0:
            for omega in (.0142578125, .0038368055555556, .003251953125):
                if abs(row["Omega"]-omega) > 1e-10:
                    continue
                dt = .03*omega
                z = dt*ev[ev.real <= 0]
                excess = max(0., float(abs(1+z+z*z/2+z*z*z/6).max())-1)
                assert excess < 1e-12
                rk.append({"profile": row["profile"], "Omega": omega, "k": row["k"], "oblique": row["oblique"], "excess": excess})
    identity = 0.
    for key, pair in groups.items():
        aa, perturb, r, omega, k, oblique = key
        expected = np.zeros((20, 20))
        mm = (2/aa**2-10)*(1-omega)
        expected[2, 3] = expected[3, 3] = -mm/omega
        identity = max(identity, float(abs(pair[1]-pair[0]-expected).max()/(1+abs(expected).max())))
    assert identity < 1e-12 and len(rk) == 96
    worst = [max((x for x in roots if x["a"] == .5 and x["profile"] == mode and x["perturb"] == 0), key=lambda x:x["max_real"]) for mode in (0, 1)]
    assert all(row["max_real"] > 3 for row in worst)
    # Reconstructed analytic pole check is exact only for that rational matrix.
    poles = json.loads((GATE / "leading-pole.json").read_text())
    assert len(poles) == 8
    pole_results = []
    lam = sp.Symbol("lambda")
    for row in poles:
        P = matrix(row["P"])
        assert abs(P.imag).max() == 0
        R = sp.Matrix([[sp.Rational(float(x)).limit_denominator(1000000) for x in line] for line in P.real])
        reconstruction = float(abs(np.asarray(R, dtype=float)-P.real).max())
        assert reconstruction < 1e-11
        assert 20-R.rank() == 20-(R*R).rank() == 5
        factors = sp.factor_list(R.charpoly(lam).as_expr())[1]
        for factor, mult in factors:
            if factor == lam:
                assert mult == 5
                continue
            coeff = sp.Poly(factor, lam).all_coeffs()
            if coeff[0] < 0:
                coeff = [-x for x in coeff]
            deg = len(coeff)-1
            hurwitz = sp.Matrix(deg, deg, lambda i,j: coeff[2*j-i+1] if 0 <= 2*j-i+1 <= deg else 0)
            assert all(hurwitz[:i, :i].det() > 0 for i in range(1, deg+1))
        pole_results.append({"a": row["a"], "profile": row["profile"], "reconstruction_max": reconstruction, "zero_nullity": 5, "square_zero_nullity": 5})
    chain = json.loads((GATE.parent / "constraint-profile.json").read_text())
    assert len(chain) == 1000
    fine, local_roots, omitted, gauge = [], [], [], []
    for row in chain:
        Q, D, pred = [matrix(row[name]) for name in ("Q", "D", "subsidiary")]
        assert abs(Q[:, [0, 4, 5, 6]]).max() == 0
        if row["level"] != 4:
            continue
        fine.append(float(abs(D-pred).max()/(1+abs(D).max()+abs(pred).max())))
        ng = matrix(row["without_dkappa2"])
        omitted.append(float(abs(D-ng).max()/(1+abs(D).max()+abs(ng).max())))
        G = matrix(row["constraint_generator"])[:, :8]
        weights = np.array([1.]*4+[max(row["k"], 1.)]*4)
        local_roots.append(float(np.linalg.eigvals(weights[:, None]*G/weights[None, :]).real.max()))
        gauge.append(float(abs(D[:, [0, 4, 5, 6]]).max()))
    assert len(fine) == len(local_roots) == 200
    assert max(fine) < 1e-7 and max(local_roots) < 0 and max(omitted) > .008
    report = json.loads((GATE / "check-report.json").read_text())
    assert abs(max(fine)-report["subsidiary"]["max_finest_relative_error"]) < 1e-15
    assert abs(max(local_roots)-report["subsidiary"]["worst_local_root"]["max_real"]) < 1e-10
    result = {"status": "PASS", "final_gate_index_sha256": EXPECTED, "preserved_v1_index_sha256": V1,
              "v2_shared_file_changes": sorted(changed), "unchanged_source_paths": 378, "commands_exit0": 12,
              "helper_sha256": sha(GATE / "damping_profile.hpp"),
              "exact_general_profile": {"m": str(m), "effective": str(eff), "Delta_Hdot": str(sp.factor(deltaH)),
                  "isolated_dkappa2_M_value": str(isolated), "total_M_value": str(sp.factor(total)),
                  "full_Delta_Mdot": "2m Theta_i/Omega + 2(kappa_input-2S/a^2)Omega_i Theta/Omega^2",
                  "sigma_C0_reference": str(sigma0), "sigma_C1_reference_only": str(sigma1),
                  "flat_interval_restriction": "kappa_input>=2S/a^2>0 with0<=Omega<=1; rho=kappa2",
                  "dimensions": "Omega,kappa2 dimensionless; S,a length; kappa_input,Kcrit,d,m inverse length"},
              "tensor": tensor, "principal": principal, "finite_full20_rows": len(full),
              "only_two_Theta_column_difference_error": identity, "reference_primitive_worst": worst,
              "negative_root_scalar_RK3_cases": len(rk), "maximum_RK3_excess": max(x["excess"] for x in rk),
              "reconstructed_analytic_poles": pole_results, "constraint_chain_rows": len(chain),
              "finest_constraint_rows": len(fine), "max_finest_relative_closure_error": max(fine),
              "maximum_local8_real_root": max(local_roots), "omitted_dkappa2_negative_control": max(omitted),
              "finest_gauge_D_absolute_max": max(gauge),
              "source_notes": "C0 only; no C1/covector repair or new double pole; direct spatial-Z damping, physical-P storage and gauge unchanged",
              "admission": "Fixed finite-Omega exploratory global/native reference/short gates only",
              "limitations": "Reference sigma cancellation is not live closure; positive finite primitive roots remain; sampled scalar RK admission is not matrix contractivity; no global energy, nonlinear scri, stable finite pulse or BH acceptance",
              "source_sha256": sha(Path(__file__)), "numpy": np.__version__, "sympy": sp.__version__}
    (HERE / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
