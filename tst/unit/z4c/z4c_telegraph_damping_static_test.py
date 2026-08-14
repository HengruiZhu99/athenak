#!/usr/bin/env python3
"""Fail-closed source contract for local telegrapher damping prescriptions."""

from __future__ import annotations

import argparse
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_dir.resolve()
    helper = (root / "src/z4c/telegraph_damping.hpp").read_text()
    rhs = (root / "src/z4c/z4c_calcrhs.cpp").read_text()
    constructor = (root / "src/z4c/z4c.cpp").read_text()
    history = (root / "src/outputs/history.cpp").read_text()
    output = (root / "src/outputs/basetype_output.cpp").read_text()

    modes = (
        "max_domain_abs_K",
        "local_abs_K",
        "local_extrinsic_curvature_norm",
        "local_chi_gradient_norm",
    )
    for mode in modes:
        require(mode in helper and mode in constructor and mode in rhs,
                f"missing end-to-end damping mode {mode}")

    require("return {mu / tau, kappa / tau};" in helper,
            "tau/kappa max-K scaling must cancel to mu/tau and kappa/tau")
    require("Q=mu/max|K|" in helper and "tau_eff=tau/max|K|" in helper and
            "kappa_eff=kappa/max|K|" in helper,
            "scale-invariant parameterization is not documented in code")
    require("coefficients.damping * z4c.vB_d" in rhs and
            "coefficients.gradient * dalpha_d" in rhs,
            "production B_i RHS does not use the reviewed coefficients")
    require("local_mu * max_abs_K" not in rhs and
            "max_abs_K * local_mu" not in rhs,
            "local mu acquired a forbidden second max-K factor")
    require("LocalExtrinsicCurvatureNormTelegraphMu" in rhs and
            "LocalChiGradientNormTelegraphMu" in rhs,
            "physical local contractions are not routed into production")
    require("Kokkos::pow(chi, -4.0 / chi_psi_power)" in helper,
            "chi-gradient norm does not use the physical inverse metric factor")
    for forbidden in ("sqrt(chi) *", "alpha * local_mu", "floor(local_mu",
                      "fmax(local_mu"):
        require(forbidden not in helper + rhs,
                f"forbidden local damping modification present: {forbidden}")

    require('variable.compare("z4c_telegraph_mu")' in output,
            "telegraph mu spatial output is not registered")
    require('pdata->label[telegraph_mu_min_index] = "muMin";' in history and
            'pdata->label[telegraph_mu_max_index] = "muMax";' in history,
            "telegraph mu history extrema are not registered")
    require("RoundoffSafeNonnegativeSqrt" in helper and
            "kRoundoffMultiplier = 64.0" in helper,
            "minimal roundoff-only contraction safeguard is absent")
    print("Z4c telegraph damping static contract passed")


if __name__ == "__main__":
    main()
