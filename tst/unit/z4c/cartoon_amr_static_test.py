#!/usr/bin/env python3
"""Focused source and coefficient checks for the half-plane Cartoon AMR slice."""

from fractions import Fraction
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
PROLONGATION = (ROOT / "src/mesh/prolongation.hpp").read_text(encoding="utf-8")
REFINEMENT = (ROOT / "src/mesh/mesh_refinement.cpp").read_text(encoding="utf-8")
BVALS = (ROOT / "src/bvals/prolongation.cpp").read_text(encoding="utf-8")
Z4C_AMR = (ROOT / "src/z4c/z4c_amr.cpp").read_text(encoding="utf-8")
KERR = (ROOT / "src/pgen/z4c/kerr_puncture.cpp").read_text(encoding="utf-8")
IRISK = (ROOT / "src/pgen/z4c_irisk_xcts.cpp").read_text(encoding="utf-8")
FASTFLOW = (ROOT / "src/z4c/fastflow.cpp").read_text(encoding="utf-8")
M0_FASTFLOW = (ROOT / "src/z4c/cartoon_m0_fastflow.cpp").read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_cubic_weights() -> None:
    nodes = (-1, 0, 1, 2)
    rules = (
        (Fraction(-1, 4), (15, 135, -27, 5)),
        (Fraction(1, 4), (-7, 105, 35, -5)),
    )
    for target, numerators in rules:
        weights = tuple(Fraction(value, 128) for value in numerators)
        for degree in range(4):
            interpolated = sum(
                weight * Fraction(node) ** degree
                for weight, node in zip(weights, nodes)
            )
            require(
                interpolated == target**degree,
                f"nghost=3 rule misses degree {degree} at {target}",
            )


def check_dispatch_and_collapsed_storage() -> None:
    for source, name in ((REFINEMENT, "mesh refinement"), (BVALS, "boundary")):
        require(
            "case 3: HighOrderProlongCC<3>" in source,
            f"{name} prolongation omits nghost=3",
        )
    require("const bool collapsed_x3 = (nx3 == 1);" in PROLONGATION,
            "collapsed storage guard is missing")
    require("const int nk = collapsed_x3 ? 1 : NGHOST+1;" in PROLONGATION,
            "collapsed storage still traverses a transverse stencil")
    require("const int ck = collapsed_x3 ? k : k-NGHOST/2+kk;" in PROLONGATION,
            "collapsed storage still offsets the transverse read")
    require("if (nx3 > 1) {\n    a(m,v,fk+1" in PROLONGATION,
            "fine transverse writes are not guarded")
    require("weights.d_view(wghtk,wghtj,wghti)" in PROLONGATION,
            "established Cartesian coefficient-table path disappeared")


def check_mirror_reconciliation_order() -> None:
    gather = REFINEMENT.index("MPI_Allgatherv(MPI_IN_PLACE")
    reconcile = REFINEMENT.index("ReconcileCartoonRefinementFlags(pmy_mesh", gather)
    sync = REFINEMENT.index("refine_flag.template modify<HostMemSpace>();", reconcile)
    update = REFINEMENT.index("void MeshRefinement::UpdateMeshBlockTree")
    require(gather < reconcile < sync < update,
            "mirror flags are not reconciled after gather and before tree mutation")
    require("std::max(refine_flag.h_view(gid), refine_flag.h_view(mirror_gid))"
            in REFINEMENT, "refine > hold > derefine precedence changed")
    require("mirror.lx1 = nradial - 1 - loc.lx1;" in REFINEMENT,
            "signed-rho logical mirror map changed")
    require(
        "z4c::Z4cCoordinateMap::signed_rho_z_suppressed_y_v1" in REFINEMENT,
        "legacy mirror reconciliation is not explicitly map-gated",
    )
    require(
        "half-plane tree is physical storage and must never" in REFINEMENT,
        "half-plane AMR non-mirroring contract is missing",
    )
    require("std::sort(cllderef, cllderef + ctnd, Mesh::GreaterLevel);"
            in REFINEMENT,
            "derefine parent sort does not cover the complete half-open range")
    require("std::sort(cllderef, &(cllderef[ctnd-1])" not in REFINEMENT,
            "legacy derefine sort still excludes the final parent")


def check_bounded_hierarchy_control() -> None:
    adaptive = REFINEMENT.index("void MeshRefinement::AdaptiveMeshRefinement")
    criterion = REFINEMENT.index("CheckForRefinement(pmy_mesh->pmb_pack);", adaptive)
    control = REFINEMENT.index("ApplyAMRJumpHierarchyControl", criterion)
    transaction = REFINEMENT.index("BeginTransaction(*this)", control)
    tree = REFINEMENT.index("UpdateMeshBlockTree(nnew, ndel)", transaction)
    require(criterion < control < transaction < tree,
            "hierarchy control is not applied after criteria and before tree mutation")
    require("std::abs(location.lx1 - seed.lx1) <= 1" in REFINEMENT and
            "std::abs(location.lx2 - seed.lx2) <= 1" in REFINEMENT,
            "buffered target does not add a full same-level Chebyshev ring")
    require("if (location.level != target_absolute_level) continue;" in REFINEMENT,
            "buffered target is not restricted to the target parent level")
    require("if (diagnostic.ShouldFreezeHierarchy())" in REFINEMENT and
            "flags.h_view(gid) = 0" in REFINEMENT,
            "frozen hierarchy does not suppress both refinement flag signs")


def check_collapsed_dchi() -> None:
    match = re.search(
        r"if \(nx3 > 1\) \{\s*"
        r"d2 \+= SQR\(u0\(m,I_Z4C_CHI,k\+1,j,i\) - "
        r"u0\(m,I_Z4C_CHI,k-1,j,i\)\);\s*\}",
        Z4C_AMR,
    )
    require(match is not None, "collapsed dchi still reads k+/-1")


def check_configured_stencil_dispatch() -> None:
    # AMR requires an even allocated ghost width, so O4 legitimately has
    # nghost=4 but fd_stencil=3.  Mathematical dispatch must follow the latter.
    require(KERR.count("switch (pack->pz4c->opt.fd_stencil)") == 2,
            "Kerr ADM conversion/constraint dispatch is not configuration-based")
    require(IRISK.count("switch (pmbp->pz4c->opt.fd_stencil)") == 2,
            "Iris ADM conversion/constraint dispatch is not configuration-based")
    for source, name in ((BVALS, "boundary"), (REFINEMENT, "AMR")):
        require(source.count("const int z4c_stencil =") == 2,
                f"{name} does not bind both mathematical stencil dispatches")
        require(source.count("switch (z4c_stencil)") >= 2,
                f"{name} still dispatches Z4c operators from allocation width")
        require("z4c_stencil > indcs.ng" in source,
                f"{name} does not fail closed when stencil exceeds storage")
    require("const int stencil = pmbp->z4c_symmetry.stencil_width;" in FASTFLOW,
            "general FastFlow interpolation ignores configured Z4c stencil")
    require("const int fd_stencil = pack_->z4c_symmetry.stencil_width;" in M0_FASTFLOW,
            "m=0 FastFlow interpolation ignores configured Z4c stencil")
    require("if (fd_stencil == 2)" in M0_FASTFLOW and
            "else if (fd_stencil == 3)" in M0_FASTFLOW,
            "m=0 FastFlow derivative dispatch is not configuration-based")


check_cubic_weights()
check_dispatch_and_collapsed_storage()
check_mirror_reconciliation_order()
check_bounded_hierarchy_control()
check_collapsed_dchi()
check_configured_stencil_dispatch()
print("cartoon AMR static checks passed")
