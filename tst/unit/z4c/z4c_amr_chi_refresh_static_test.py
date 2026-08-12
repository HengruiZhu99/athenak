#!/usr/bin/env python3
"""Static ownership and ordering checks for the narrow Z4c chi AMR repair."""

import argparse
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_dir.resolve()
    prolongation = (root / "src/mesh/prolongation.hpp").read_text(encoding="utf-8")
    restriction = (root / "src/mesh/restriction.hpp").read_text(encoding="utf-8")
    refinement = (root / "src/mesh/mesh_refinement.cpp").read_text(encoding="utf-8")
    boundary = (root / "src/bvals/prolongation.cpp").read_text(encoding="utf-8")
    tasks = (root / "src/z4c/z4c_tasks.cpp").read_text(encoding="utf-8")

    helper = prolongation[prolongation.index("ChiProlongationStatus ProlongPositiveChiCC") :]
    high_order = helper.index("HighOrderProlongCC<NGHOST>")
    parent = helper.index("ProlongationParentStencilFinitePositive", high_order)
    candidate = helper.index("ProlongationSiblingGroupFinitePositive", parent)
    limited = helper.index("ProlongCC(", candidate)
    limited_check = helper.index("ProlongationSiblingGroupFinitePositive", limited)
    require(high_order < parent < candidate < limited < limited_check,
            "chi candidate/parent/fallback ordering changed")
    require("Kokkos::isfinite(parent) || !(parent > 0.0)" in prolongation,
            "full parent-stencil finite/positive gate disappeared")
    require("Kokkos::isfinite(child) || !(child > 0.0)" in prolongation,
            "complete sibling finite/positive gate disappeared")
    for forbidden in ("chi_min_floor", "fmax(", "max(parent", "epsilon"):
        require(forbidden not in helper,
                f"forbidden chi flooring/clipping token entered helper: {forbidden}")

    dispatch = refinement[refinement.index("void MeshRefinement::RefineCC") :]
    require("v == z4c::Z4c::I_Z4C_CHI" in dispatch,
            "positivity fallback is not isolated to the chi component")
    require("AMR_Z4C_CHI_PROLONGATION" in dispatch and
            "local_fallback_groups=" in dispatch and
            "global_fallback_groups=" in dispatch,
            "deterministic fallback evidence record disappeared")
    require("MPI_Allreduce(local_counts, global_counts" in dispatch and
            "MPI_Gather(&local_counts[1]" in dispatch,
            "local/global fallback accounting is not collective")

    boundary_dispatch = boundary[
        boundary.index("void MeshBoundaryValuesCC::ProlongateCC") :
        boundary.index("void MeshBoundaryValuesFC::FillCoarseInBndryFC")
    ]
    chi_branch = boundary_dispatch.index("v == z4c::Z4c::I_Z4C_CHI")
    generic_branch = boundary_dispatch.index("} else {", chi_branch)
    require("ProlongPositiveChiCC<2>" in boundary_dispatch[chi_branch:generic_branch] and
            "ProlongPositiveChiCC<3>" in boundary_dispatch[chi_branch:generic_branch] and
            "ProlongPositiveChiCC<4>" in boundary_dispatch[chi_branch:generic_branch],
            "boundary chi does not use the positive sibling-group helper for every order")
    require("HighOrderProlongCC" not in boundary_dispatch[chi_branch:generic_branch],
            "boundary chi branch bypasses the positive helper")
    require("HighOrderProlongCC<4>" in boundary_dispatch[generic_branch:],
            "generic Z4c high-order boundary path disappeared")
    require("BOUNDARY_Z4C_CHI_PROLONGATION" in boundary_dispatch and
            "MPI_Allreduce(local_counts, global_counts" in boundary_dispatch and
            "invalid_parent_stencils=" in boundary_dispatch and
            "invalid_limited_groups=" in boundary_dispatch,
            "boundary chi aggregate fail-closed evidence disappeared")
    for forbidden in ("chi_min_floor", "fmax(", "epsilon"):
        require(forbidden not in boundary_dispatch,
                f"forbidden boundary chi repair token entered source: {forbidden}")

    z4c_prolongate = tasks[
        tasks.index("TaskStatus Z4c::Prolongate") :
        tasks.index("void Z4c::FillBuiltInPhysicalBoundaryGhosts")
    ]
    refresh = z4c_prolongate.index(
        "pbval_u->FillCoarseInBndryCC(u0, coarse_u0, true)")
    prolong = z4c_prolongate.index("pbval_u->ProlongateCC(u0, coarse_u0, true)")
    require(refresh < prolong,
            "Z4c same-level coarse corners are not refreshed before prolongation")

    fill_coarse = boundary[
        boundary.index("void MeshBoundaryValuesCC::FillCoarseInBndryCC") :
        boundary.index("void MeshBoundaryValuesCC::ProlongateCC")
    ]
    for order in (2, 3, 4):
        require(f"case {order}:" in fill_coarse and
                f"RestrictInterpolation<{order}>" in fill_coarse,
                f"3D Z4c coarse refresh is missing NGHOST={order}")
    require("CompleteFinePairCoarseRange" in fill_coarse and
            "fine_n1 = a.extent_int(4)" in fill_coarse and
            "fine_n2 = a.extent_int(3)" in fill_coarse and
            "fine_n3 = a.extent_int(2)" in fill_coarse,
            "same-level coarse refresh no longer clamps to stored fine pairs")
    require("if (NGHOST == 3)" in restriction and
            "constexpr Real weight[4]" in restriction,
            "NGHOST=3 restriction implementation is missing")
    require("fi == 0" in restriction and "fi == outer_i" in restriction and
            "restrict_4th_edge.d_view" in restriction,
            "outer stored NGHOST=4 pairs no longer use oriented edge weights")

    restrict_cc = refinement[
        refinement.index("void MeshRefinement::RestrictCC") :
        refinement.index("void MeshRefinement::RestrictFC")
    ]
    for order in (2, 3, 4):
        require(f"case {order}:" in restrict_cc and
                f"RestrictInterpolation<{order}>" in restrict_cc,
                f"full Z4c restriction is missing NGHOST={order}")

    adaptive = refinement[
        refinement.index("void MeshRefinement::AdaptiveMeshRefinement") :
        refinement.index("void MeshRefinement::CheckForRefinement")
    ]
    redist = adaptive.index("RedistAndRefineMeshBlocks")
    boundary = adaptive.index("InitBoundaryValuesAndPrimitives", redist)
    algebraic = adaptive.index("EnforceAlgConstr", boundary)
    adm = adaptive.index("ConvertZ4cToADM", algebraic)
    constraints = adaptive.index("ADMConstraints_", adm)
    timestep = adaptive.index("pz4c->NewTimeStep", constraints)
    require(redist < boundary < algebraic < adm < constraints < timestep,
            "post-regrid finalized Z4c/ADM/constraint ordering changed")

    print("Z4c chi AMR refresh static checks passed")


if __name__ == "__main__":
    main()
