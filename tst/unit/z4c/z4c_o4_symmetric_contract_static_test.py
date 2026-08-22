#!/usr/bin/env python3
"""Fail-closed source contract for the production nghost=4 / O4 Z4c path."""

import argparse
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.source_dir.resolve()

    prolong = (root / "src/mesh/prolongation.hpp").read_text(encoding="utf-8")
    restrict = (root / "src/mesh/restriction.hpp").read_text(encoding="utf-8")
    buffers = (root / "src/bvals/buffs_cc.cpp").read_text(encoding="utf-8")
    copies = (root / "src/bvals/bvals_cc.cpp").read_text(encoding="utf-8")
    tasks = (root / "src/z4c/z4c_tasks.cpp").read_text(encoding="utf-8")
    refinement = (root / "src/mesh/mesh_refinement.cpp").read_text(encoding="utf-8")
    z4c = (root / "src/z4c/z4c.cpp").read_text(encoding="utf-8")
    history = (root / "src/outputs/history.cpp").read_text(encoding="utf-8")
    production_test = (
        root / "tst/unit/z4c/cartoon_production_kernel_test.cpp"
    ).read_text(encoding="utf-8")

    for weights in (
        "-5.0/128.0, 35.0/128.0, 105.0/128.0, -7.0/128.0",
        "-7.0/128.0, 105.0/128.0, 35.0/128.0, -5.0/128.0",
    ):
        require(weights in prolong, f"missing O4 prolongation weights {weights}")
    require("NGHOST == 3 && !offseti" in prolong and
            "NGHOST == 3 && !offsetj" in prolong,
            "O4 left child does not shift to the reflected parent stencil")
    require("const int lower = NGHOST == 3 ? -2" in prolong and
            "const int upper = NGHOST == 3 ? 2" in prolong,
            "O4 chi gate does not cover the complete five-parent union")

    for weights in (
        "5.0 / 16.0, 15.0 / 16.0, -5.0 / 16.0, 1.0 / 16.0",
        "1.0 / 16.0, -5.0 / 16.0, 15.0 / 16.0, 5.0 / 16.0",
        "-1.0 / 16.0, 9.0 / 16.0, 9.0 / 16.0, -1.0 / 16.0",
    ):
        require(weights in restrict, f"missing O4 restriction weights {weights}")
    require("fine_pair_start == active_start" in restrict and
            "fine_pair_start == active_start + active_extent - 2" in restrict,
            "O4 edge restriction is not keyed to active sibling pairs")
    require("(a.extent_int(4) - nx1) / 2" in restrict and
            "(a.extent_int(3) - nx2) / 2" in restrict,
            "O4 restriction confuses allocated ghosts with fd_stencil")

    # Same-level fine ghosts use ng-wide exact pack/unpack ranges.  The values
    # are direct assignments at both same-rank and MPI paths; no interpolator
    # is present in either production copy kernel.
    require("int ng  = mb_indcs.ng;" in buffers and
            "mb_indcs.ie - ng1" in buffers and
            "mb_indcs.ie + ng" in buffers and
            "mb_indcs.is - ng" in buffers,
            "same-level fine buffer no longer spans allocated nghost")
    require("a(m,v,k,j,i) = rbuf[n].vars" in copies and
            "rbuf[dn].vars" in copies and "= a(m,v,k,j,i);" in copies,
            "same-level fine state is no longer copied exactly")
    same_level_body = copies[copies.index("TaskStatus MeshBoundaryValuesCC::PackAndSendCC"):
                             copies.index("TaskStatus MeshBoundaryValuesCC::RecvAndUnpackCC")]
    require("ProlongInterpolation" not in same_level_body and
            "HighOrderProlongCC" not in same_level_body,
            "same-level pack path interpolates authoritative fine state")

    # nghost=4 gives cn=2 coarse groups; each HighOrderProlongCC call writes
    # two children, so the complete four allocated fine ghost layers are filled.
    require("int cn = mb_indcs.ng/2" in buffers and
            "iprol.bie = mb_indcs.cie + cn" in buffers and
            "iprol.bis = mb_indcs.cis - cn" in buffers,
            "coarse-fine prolongation inventory does not span ng/2 parent groups")
    require("fi+1" in prolong and "fj+1" in prolong,
            "coarse-fine parent group no longer writes both children")

    queue = tasks[tasks.index("void Z4c::QueueZ4cTasks"):
                  tasks.index("TaskStatus Z4c::InitRecv")]
    markers = ("Z4c_ExplRK", "Z4c_RestU", "Z4c_SendU", "Z4c_RecvU",
               "Z4c_BCS", "Z4c_Prolong", "Z4c_AxisGhostsPost",
               "Z4c_Z4c2ADM")
    positions = [queue.index(marker) for marker in markers]
    require(positions == sorted(positions),
            "accepted-state cache/ghost/ADM order is not authoritative")
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
    vc_markers = ("AlgConstr(", "ApplyVertexAxisRegularity(",
                  "SynchronizeSharedNodes(", "RestrictVC(", "InitRecv(",
                  "PackAndSendVC(", "RecvAndUnpackVC(",
                  "FillBuiltInPhysicalBoundaryGhosts(", "ProlongateVC(",
                  "ReconstructAxisParityGhosts(", "CheckStateAdmissibility(")
    vc_positions = [vc_finalize.index(marker) for marker in vc_markers]
    require(vc_positions == sorted(vc_positions),
            "native-VC accepted projection/restriction/ghost rebuild order changed")
    require("stage == pdrive->nexp_stages" in tasks,
            "vacuum algebraic projection lost final-stage-only policy")

    adaptive = refinement[refinement.index("RedistAndRefineMeshBlocks"):
                          refinement.index("nmb_created += nnew")]
    amr_markers = ("EnforceAlgConstr", "pdriver->InitBoundaryValuesAndPrimitives",
                   "ConvertZ4cToADM", "ADMConstraints_", "pz4c->NewTimeStep")
    amr_positions = [adaptive.index(marker) for marker in amr_markers]
    require(amr_positions == sorted(amr_positions),
            "regrid cache/ghost state is not rebuilt from projected active state")

    require("history_kretschmann && opt.fd_stencil != 4" not in z4c,
            "obsolete O6-only history Kretschmann gate remains")
    for selector in (2, 3, 4):
        require(f"case {selector}:" in history and
                f"Z4cHistoryMaxKretschmann<z4c::CartoonSO2, {selector}>" in history,
                f"history Kretschmann dispatch missing stencil {selector}")
    require("const int allocated_ghosts = stencil == 3 ? 4 : stencil;" in
            production_test,
            "production kernel test does not exercise fd_stencil=3 with nghost=4")

    print("O4 symmetric production source contract passed")


if __name__ == "__main__":
    main()
