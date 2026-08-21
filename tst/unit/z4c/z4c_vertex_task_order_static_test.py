#!/usr/bin/env python3
"""Static contract for native-VC accepted-state ordering."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TASKS = (ROOT / "src/z4c/z4c_tasks.cpp").read_text(encoding="utf-8")


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

print("PASS: native VC accepted-state task ordering")
