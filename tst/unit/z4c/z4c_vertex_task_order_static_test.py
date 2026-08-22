#!/usr/bin/env python3
"""Static contract for native-VC accepted-state ordering."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TASKS = (ROOT / "src/z4c/z4c_tasks.cpp").read_text(encoding="utf-8")
REFINEMENT = (ROOT / "src/mesh/mesh_refinement_vc.cpp").read_text(encoding="utf-8")
BVALS_VC = (ROOT / "src/bvals/bvals_vc.cpp").read_text(encoding="utf-8")
BUFFS_VC = (ROOT / "src/bvals/buffs_vc.cpp").read_text(encoding="utf-8")


def require(fragment: str, message: str) -> None:
    if fragment not in TASKS:
        raise RuntimeError(message)


require("vertex_topology_plan->SynchronizeSharedNodes(u0);\n"
        "      pmy_pack->pmesh->pmr->RestrictVC(u0, coarse_u0);",
        "VC restriction must consume canonical shared vertices")
require("AlgConstr(pmy_pack, pdrive, stage);\n"
        "  ApplyVertexAxisRegularity(u0, stage, \"post_accepted_projection\");\n"
        "  vertex_topology_plan->SynchronizeSharedNodes(u0);",
        "accepted VC projection must follow the first boundary synchronization")
for fragment in (
    "pbval_u_vc->InitRecv(nz4c)",
    "pbval_u_vc->PackAndSendVC(u0, coarse_u0)",
    "pbval_u_vc->RecvAndUnpackVC(u0, coarse_u0)",
    "pbval_u_vc->ProlongateVC(u0, coarse_u0, opt.spatial_order, I_Z4C_CHI)",
    "ReconstructAxisParityGhosts()",
):
    require(fragment, f"accepted VC rebuild is missing {fragment}")
require("Task_Run, {Z4c_VCFinalize}",
        "ADM conversion must depend on accepted VC finalization")
if ("const auto flags = refine_flag.d_view;" not in REFINEMENT or
        "const auto new_to_old_device = new_to_old.d_view;" not in REFINEMENT):
    raise RuntimeError("native VC refinement does not capture explicit device views")
if "flags.d_view(new_to_old.d_view" in REFINEMENT:
    raise RuntimeError("native VC refinement captures DualView wrappers in a kernel")
if "const auto coarse = recvbuf[n].iprol[0];" in BVALS_VC:
    raise RuntimeError("VC prolongation dereferences host boundary metadata in a device kernel")
if "auto iprol = prolongation_bounds.d_view;" not in BVALS_VC or \
        "const auto coarse = iprol(n);" not in BVALS_VC:
    raise RuntimeError("VC prolongation must capture device-resident prolongation bounds")
for fragment in (
    "MeshBoundaryValues::InitializeBuffers(nvar);",
    "prolongation_bounds.h_view(n) = recvbuf[n].iprol[0];",
    "prolongation_bounds.template sync<DevExeSpace>();",
):
    if fragment not in BUFFS_VC:
        raise RuntimeError(f"VC boundary bounds initialization is missing {fragment}")

print("PASS: native VC accepted-state task ordering")
