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
    ownership = (root / "src/bvals/coarse_cache_ownership.hpp").read_text(
        encoding="utf-8")
    tasks = (root / "src/z4c/z4c_tasks.cpp").read_text(encoding="utf-8")
    z4c_header = (root / "src/z4c/z4c.hpp").read_text(encoding="utf-8")
    z4c_source = (root / "src/z4c/z4c.cpp").read_text(encoding="utf-8")

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
    limited_helper = prolongation[
        prolongation.index("ChiProlongationStatus ProlongLimitedPositiveChiCC") :
        prolongation.index("ChiProlongationStatus ProlongPositiveChiCC")
    ]
    require("LimitedProlongationParentNeighborhoodFinitePositive" in limited_helper and
            "ProlongCC(" in limited_helper and
            "ProlongationSiblingGroupFinitePositive" in limited_helper,
            "limited-O2 chi helper lost parent/child positivity gates")
    for forbidden in ("chi_min_floor", "fmax(", "epsilon"):
        require(forbidden not in limited_helper,
                f"forbidden limited-O2 chi token entered source: {forbidden}")

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
    require("ProlongPositiveChiCC<2>" in boundary_dispatch and
            "ProlongPositiveChiCC<3>" in boundary_dispatch and
            "ProlongPositiveChiCC<4>" in boundary_dispatch,
            "boundary chi does not use the positive sibling-group helper for every order")
    require("HighOrderProlongCC<4>" in boundary_dispatch,
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
            "Z4c coarse-cache ownership gate is not called before prolongation")

    fill_coarse = boundary[
        boundary.index("void MeshBoundaryValuesCC::FillCoarseInBndryCC") :
        boundary.index("void MeshBoundaryValuesCC::ProlongateCC")
    ]
    ownership_gate = fill_coarse.index(
        "if (!ShouldLocallyRefreshSameLevelCoarseCache(is_z4c)) return;")
    kernel = fill_coarse.index('Kokkos::parallel_for("ProlCCSame"')
    require(ownership_gate < kernel,
            "Z4c owner-authoritative cache gate no longer precedes local writes")
    require("return !is_z4c;" in ownership and
            "isame_z4c" in ownership,
            "coarse-cache ownership policy no longer preserves Z4c receives")
    for order in (2, 3, 4):
        require(f"case {order}:" in fill_coarse and
                f"RestrictInterpolation<{order}>" in fill_coarse,
                f"Z4c coarse refresh is missing NGHOST={order}")
    fill_2d = fill_coarse[
        fill_coarse.index("// restrict in 2D") :
        fill_coarse.index("// restrict in 3D")
    ]
    require("if (!is_z4c || limited_o2)" in fill_2d,
            "2D coarse refresh no longer separates generic and Z4c restriction")
    require("Z4cAMRTransfer::limited_o2" in fill_coarse,
            "2D/3D coarse refresh is not controlled by the Z4c transfer option")
    for order in (2, 3, 4):
        require(f"RestrictInterpolation<{order}>" in fill_2d,
                f"2D Z4c coarse refresh bypasses NGHOST={order} high-order restriction")
    require("CompleteFinePairCoarseRange" in fill_coarse and
            "fine_n1 = a.extent_int(4)" in fill_coarse and
            "fine_n2 = a.extent_int(3)" in fill_coarse and
            "fine_n3 = a.extent_int(2)" in fill_coarse,
            "same-level coarse refresh no longer clamps to stored fine pairs")
    require("O4RestrictionStencil1D" in restriction and
            "SelectO4RestrictionStencil" in restriction and
            "5.0 / 16.0, 15.0 / 16.0, -5.0 / 16.0, 1.0 / 16.0" in restriction and
            "1.0 / 16.0, -5.0 / 16.0, 15.0 / 16.0, 5.0 / 16.0" in restriction,
            "mirror-paired active-only O4 edge restriction is missing")
    require("(a.extent_int(4) - nx1) / 2" in restriction and
            "(a.extent_int(3) - nx2) / 2" in restriction,
            "O4 restriction does not distinguish allocated nghost=4 from fd_stencil=3")
    require("if (nx3 == 1)" in restriction and
            "a(m, v, fk, refj + jj, refi + ii)" in restriction,
            "collapsed-x3 tensor restriction implementation is missing")
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
    restrict_2d = restrict_cc[
        restrict_cc.index("// restrict in 2D") :
        restrict_cc.index("// restrict in 3D")
    ]
    require("if (!is_z4c || limited_o2)" in restrict_2d,
            "2D full restriction no longer separates generic and Z4c restriction")
    require("Z4cAMRTransfer::limited_o2" in restrict_cc,
            "full Z4c restriction is not controlled by the transfer option")
    for order in (2, 3, 4):
        require(f"RestrictInterpolation<{order}>" in restrict_2d,
                f"2D Z4c full restriction bypasses NGHOST={order} high-order restriction")

    require("Z4cAMRTransfer" in z4c_header and
            "amr_transfer" in z4c_header and
            'GetOrAddString("z4c", "amr_transfer", "high_order")' in z4c_source and
            'amr_transfer == "limited_o2"' in z4c_source,
            "Z4c AMR transfer option is absent or does not default high-order")
    require("ProlongLimitedPositiveChiCC" in boundary_dispatch and
            "ProlongLimitedPositiveChiCC" in dispatch,
            "limited-O2 chi is not wired into both prolongation call sites")
    require(boundary_dispatch.count("ProlongCC(") >= 2 and
            dispatch.count("ProlongCC(") >= 2,
            "limited-O2 non-chi Z4c prolongation is not wired at both sites")

    adaptive = refinement[
        refinement.index("void MeshRefinement::AdaptiveMeshRefinement") :
        refinement.index("void MeshRefinement::CheckForRefinement")
    ]
    redist = adaptive.index("RedistAndRefineMeshBlocks")
    algebraic = adaptive.index("EnforceAlgConstr", redist)
    boundary = adaptive.index("InitBoundaryValuesAndPrimitives", algebraic)
    adm = adaptive.index("ConvertZ4cToADM", boundary)
    constraints = adaptive.index("ADMConstraints_", adm)
    timestep = adaptive.index("pz4c->NewTimeStep", constraints)
    require(redist < algebraic < boundary < adm < constraints < timestep,
            "post-regrid finalized Z4c/ADM/constraint ordering changed")

    queue = tasks[tasks.index("void Z4c::QueueZ4cTasks"):
                  tasks.index("TaskStatus Z4c::InitRecv")]
    common_order = [queue.index(marker) for marker in
                    ("Z4c_ExplRK", "Z4c_RestU", "Z4c_SendU", "Z4c_RecvU",
                     "Z4c_BCS", "Z4c_Prolong", "Z4c_AxisGhostsPost",
                     "Z4c_Z4c2ADM")]
    require(common_order == sorted(common_order),
            "accepted Z4c cache/ghost/ADM task order changed")
    cc_branch = queue[queue.index("  } else {", queue.index("if (vertex_centered)")):
                      queue.index("  }\n  pnr->QueueTask(&Z4c::SendU")]
    require("{Z4c_ChiFloor}" in cc_branch and "{Z4c_ExplRK}" in cc_branch and
            "Task_Run, {Z4c_AlgC}" in cc_branch,
            "cell-centered projection no longer precedes restriction")
    require("Task_Run, {Z4c_AxisGhostsPost}" in queue and
            "Task_Run, {Z4c_VCFinalize}" in queue,
            "native-VC accepted-state finalizer is not between ghosts and ADM")
    vc_finalize = tasks[tasks.index("TaskStatus Z4c::FinalizeVertexAcceptedState"):
                        tasks.index("TaskStatus Z4c::ConvertZ4cToADM")]
    vc_order = [vc_finalize.index(marker) for marker in
                ("AlgConstr(", "ApplyVertexAxisRegularity(",
                 "SynchronizeSharedNodes(", "RestrictVC(", "InitRecv(",
                 "PackAndSendVC(", "RecvAndUnpackVC(",
                 "FillBuiltInPhysicalBoundaryGhosts(", "ProlongateVC(",
                 "ReconstructAxisParityGhosts(", "CheckStateAdmissibility(")]
    require(vc_order == sorted(vc_order),
            "native-VC accepted projection/restriction/ghost rebuild order changed")
    require(queue.count("&Z4c::EnforceAlgConstr") == 2,
            "expected floor/no-floor dependency branches for one algebraic task id")

    print("Z4c chi AMR refresh static checks passed")


if __name__ == "__main__":
    main()
