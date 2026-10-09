"""Read-only independent review of the actual full20 finite-Omega C1 matrices.

The balanced norm is Omega-dependent and degenerates on Atilde/Lambda at scri.
This review neither imposes a falloff nor proves a uniform raw-norm estimate.
"""
import hashlib
import json
import platform
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = ROOT / "build-layer-research/continuum/covariant-z4-candidate"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix(row):
    raw = np.asarray(row["L"], dtype=float)
    assert raw.shape == (20, 20, 2) and np.isfinite(raw).all()
    return raw[:, :, 0] + 1j * raw[:, :, 1]


def rk3(z):
    return np.eye(20) + z + z @ z / 2 + z @ z @ z / 6


def expm(z):
    """Taylor scaling/squaring with small one-norm argument, no eigensolver."""
    norm = np.linalg.norm(z, 1)
    scale = max(0, int(np.ceil(np.log2(max(norm / .125, 1)))))
    y = z / 2 ** scale
    term = np.eye(20, dtype=complex)
    value = term.copy()
    for k in range(1, 101):
        term = term @ y / k
        value += term
        if np.linalg.norm(term, 1) < 1e-17:
            break
    else:
        raise AssertionError("Taylor series did not converge")
    for _ in range(scale):
        value = value @ value
    return value


def main():
    source = CANDIDATE / "full20.cpp"
    inputs = [source, CANDIDATE / "small-Omega.json", CANDIDATE / "Fourier.json"]
    rows = json.loads(inputs[1].read_text())
    assert len(rows) == 384
    result = []
    coefficient_error = 0.0
    similarity_error = 0.0
    for row in rows:
        om = row["Omega"]
        assert 0 < om < 1 and row["k"] == 0 and not row["oblique"]
        assert row["form"] in (0, 1, 2)
        m = matrix(row)
        # y=T u, with Omega on curvature and connection variables only.
        # Omega is coordinate-time independent, hence y_t=T L T^-1 y.
        weights = np.ones(20)
        weights[12:] = om
        balanced = om * weights[:, None] * m / weights[None, :]
        z = .03 * om * m
        raw_rk = rk3(z)
        balanced_rk = rk3(.03 * balanced)
        direct = weights[:, None] * raw_rk / weights[None, :]
        relative = np.linalg.norm(direct - balanced_rk, 2) / max(
            1, np.linalg.norm(balanced_rk, 2))
        similarity_error = max(similarity_error, float(relative))
        assert relative < 1e-12
        eigenvalues = np.linalg.eigvals(m)
        pole = float((om * om * m[17, 3]).real)
        if not row["perturb"]:
            expected = -2 * row["r"] / row["a"] ** 2 if row["form"] else 0
            coefficient_error = max(coefficient_error, abs(pole - expected))
            assert abs(pole - expected) < 2e-12
        record = {key: row[key] for key in (
            "a", "kappa", "form", "norm_gauge", "perturb", "Omega", "r")}
        record.update({
            "Omega2_LambdaTheta": pole,
            "balanced_generator_norm2": float(np.linalg.norm(balanced, 2)),
            "raw_RK3_norm2": float(np.linalg.norm(raw_rk, 2)),
            "Omega_times_raw_RK3_norm2": float(om * np.linalg.norm(raw_rk, 2)),
            "balanced_RK3_norm2": float(np.linalg.norm(balanced_rk, 2)),
            "balanced_exact_norm2": float(np.linalg.norm(expm(.03 * balanced), 2)),
            "scaled_spectral_radius": float(om * max(abs(eigenvalues))),
            "max_real_eigenvalue": float(max(eigenvalues.real)),
            "RK3_spectral_radius": float(max(abs(np.linalg.eigvals(raw_rk)))),
        })
        result.append(record)
    groups = []
    for a in (.5, .75, 1, 2):
        for kappa in (5, 10):
            for form in (0, 1, 2):
                for norm in (0, 1):
                    for perturb in (0, .01):
                        group = [r for r in result if (
                            r["a"], r["kappa"], r["form"], r["norm_gauge"],
                            r["perturb"]) == (a, kappa, form, norm, perturb)]
                        assert len(group) == 4
                        group.sort(key=lambda r: r["Omega"], reverse=True)
                        smallest, previous = group[-1], group[-2]
                        ratio = (smallest["raw_RK3_norm2"] /
                                 previous["raw_RK3_norm2"])
                        groups.append({
                            "a": a, "kappa": kappa, "form": form,
                            "norm_gauge": norm, "perturb": perturb,
                            "smallest_Omega_row": smallest,
                            "raw_RK_norm_smallest_over_previous": ratio,
                            "max_balanced_generator_norm2": max(
                                r["balanced_generator_norm2"] for r in group),
                        })
    reference = [r for r in result if r["a"] == .5 and r["kappa"] == 10
                 and not r["perturb"] and r["form"] == 2
                 and r["norm_gauge"] == 1]
    reference.sort(key=lambda r: r["Omega"], reverse=True)
    output = {
        "status": "PASS",
        "scope": "Independent finite-Omega source/matrix/propagator review only",
        "rows": len(rows),
        "form_definition": {
            "0": "production C0 geometry",
            "1": "mechanical Appendix-B C1 additions",
            "2": "C1 plus separate covector connection completion",
        },
        "normalization": "S=1; physical Theta; y=T u; T[12:20]=Omega",
        "weighted_norm_caveat": "T degenerates at scri; no raw-norm uniform bound",
        "coefficient_identity": "reference: Omega² L[Lambda_r,Theta]=-2r/a²",
        "coefficient_max_absolute_error": coefficient_error,
        "RK3_similarity_max_relative_error": similarity_error,
        "dt": ".03*Omega; local frozen operator only",
        "all_rows_finite": True,
        "groups": groups,
        "a05_kappa10_covector_norm_reference": reference,
        "versions": {"python": platform.python_version(),
                     "numpy": np.__version__},
        "exponential": "Independent Taylor scaling/squaring; scaled 1-norm<=.125",
        "inputs": {str(p.relative_to(ROOT)): {"sha256": sha(p),
                   "bytes": p.stat().st_size} for p in inputs},
        "review_source_sha256": sha(Path(__file__)),
    }
    out = Path(__file__).with_name("receipt.json")
    out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: output[key] for key in (
        "status", "rows", "coefficient_max_absolute_error",
        "RK3_similarity_max_relative_error")}))
    print(json.dumps({"a05_kappa10_covector_norm_reference": reference}))


if __name__ == "__main__":
    main()
