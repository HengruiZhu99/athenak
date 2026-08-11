#!/usr/bin/env python3
"""Focused source and coefficient checks for the signed-rho Cartoon AMR slice."""

from fractions import Fraction
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
PROLONGATION = (ROOT / "src/mesh/prolongation.hpp").read_text(encoding="utf-8")
REFINEMENT = (ROOT / "src/mesh/mesh_refinement.cpp").read_text(encoding="utf-8")
BVALS = (ROOT / "src/bvals/prolongation.cpp").read_text(encoding="utf-8")
Z4C_AMR = (ROOT / "src/z4c/z4c_amr.cpp").read_text(encoding="utf-8")


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


def check_collapsed_dchi() -> None:
    match = re.search(
        r"if \(nx3 > 1\) \{\s*"
        r"d2 \+= SQR\(u0\(m,I_Z4C_CHI,k\+1,j,i\) - "
        r"u0\(m,I_Z4C_CHI,k-1,j,i\)\);\s*\}",
        Z4C_AMR,
    )
    require(match is not None, "collapsed dchi still reads k+/-1")


check_cubic_weights()
check_dispatch_and_collapsed_storage()
check_mirror_reconciliation_order()
check_collapsed_dchi()
print("cartoon AMR static checks passed")
