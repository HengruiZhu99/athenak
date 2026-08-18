#!/usr/bin/env python3
"""Fail-closed source contract for the default-off chi pre-update audit."""

import argparse
from pathlib import Path


def require(text: str, token: str) -> None:
    if token not in text:
        raise SystemExit(f"missing required diagnostic contract: {token}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_dir
    tasks = (source / "src/z4c/z4c_tasks.cpp").read_text()
    boundary = (source / "src/z4c/z4c_Sbc.cpp").read_text()
    update = (source / "src/z4c/z4c_update.cpp").read_text()
    rhs = (source / "src/z4c/z4c_calcrhs.cpp").read_text()
    config = (source / "src/z4c/chi_parent_provenance.cpp").read_text()

    require(config, 'GetOrAddBoolean("z4c", "chi_parent_provenance_diagnostic", false)')
    require(tasks, "RecordBeforeCopy(pdrive, stage)")
    require(tasks, "RecordAfterCopy(pdrive, stage)")
    require(boundary, "AnalyzePreUpdate(pdriver, stage)")
    if boundary.index("AnalyzePreUpdate(pdriver, stage)") > boundary.index(
            "return status;", boundary.index("AnalyzePreUpdate(pdriver, stage)")):
        raise SystemExit("pre-update diagnostic is not before the boundary task return")
    require(update, "gam0*u0(m,n,k,j,i) + gam1*u1(m,n,k,j,i) +")
    require(config, "EvaluateChiRKCandidate(")
    require(rhs, "DirectionalScalarAdvective(")
    require(rhs, "Lchi = use_o2_shift_advection")
    require(rhs, "derivatives.ScalarAdvective(z4c.beta_u, z4c.chi)")
    require(rhs, "derivatives.ScalarAdvectiveO2(z4c.beta_u, z4c.chi)")
    require(rhs, "chi_adv_total_production")
    require(rhs, "chi_curvature_source")
    require(rhs, "chi_rhs_before_ko")
    require(rhs, "chi_rhs_after_ko")
    require(config, "pre-update chi candidate is nonpositive or nonfinite")
    diagnostic_branch = rhs[rhs.index("if (collect_chi_provenance) {"):
                            rhs.index("LTheta = use_o2_shift_advection")]
    if "Lchi += contribution" in diagnostic_branch:
        raise SystemExit("diagnostic reconstructs the production advective term")
    require(rhs, "const Real contribution = u_rhs(m,n,k,j,i) - rhs_before;")
    for forbidden in ("chi_min_floor =", "fabs(chi", "abs(chi", "clip("):
        if forbidden in config:
            raise SystemExit(f"diagnostic introduced forbidden state repair: {forbidden}")
    print("CHI_PREUPDATE_DIAGNOSTIC_STATIC_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
