#!/usr/bin/env python3
"""Fail-closed source contract for half-plane Kerr initialization."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PGEN = (ROOT / "src/pgen/z4c/kerr_puncture.cpp").read_text(encoding="utf-8")
POINT = (ROOT / "src/pgen/z4c/kerr_puncture.hpp").read_text(encoding="utf-8")
TASKS = (ROOT / "src/z4c/z4c_tasks.cpp").read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


require(
    "CoordinateMap::half_rho_z_suppressed_y_v2" in POINT,
    "Kerr point evaluator omits the half-plane coordinate map",
)
require(
    "signed_rho_z_suppressed_y_v1" not in POINT + PGEN,
    "Kerr initializer still exposes the independently stored signed-rho map",
)
capture = PGEN.index("const bool is_axis_ghost =")
guard = PGEN.index("i < is", capture)
compile_time_branch = PGEN.index("if constexpr (Map ==", guard)
guard_use = PGEN.index("if (is_axis_ghost)", compile_time_branch)
evaluate = PGEN.index("kerr_puncture::Evaluate<Map, Gauge>", guard)
require(
    capture < guard < compile_time_branch < guard_use < evaluate,
    "axis ghosts are not excluded with CUDA-safe captures before analytic evaluation",
)
require(
    "mb_bcs.d_view" not in PGEN[compile_time_branch:guard_use],
    "a device view is first captured inside the Kerr if-constexpr branch",
)

adm_fill = PGEN.index("derive Kerr puncture ADM axis ghosts")
gauge_fill = PGEN.index("derive Kerr puncture gauge axis ghosts", adm_fill)
convert = PGEN.index("pack->pz4c->ADMToZ4c<2>", gauge_fill)
require(
    adm_fill < gauge_fill < convert,
    "ADM and gauge parity ghosts are not reconstructed before ADM-to-Z4c conversion",
)
reconstruct = PGEN.index("pack->pz4c->ReconstructAxisParityGhosts();", convert)
to_adm = PGEN.index("pack->pz4c->Z4cToADM(pack);", reconstruct)
constraints = PGEN.index("pack->pz4c->ADMConstraints<2>", to_adm)
require(
    convert < reconstruct < to_adm < constraints,
    "derived Z4c parity ghosts are not reconstructed before ADM constraints",
)

require(
    "void Z4c::ReconstructAxisParityGhosts()" in TASKS,
    "the pgen and evolution tasks do not share one parity reconstruction path",
)
task_call = TASKS.index("ReconstructAxisParityGhosts();")
implementation = TASKS.index("void Z4c::ReconstructAxisParityGhosts()", task_call)
require(task_call < implementation, "task wrapper does not call the shared implementation")

print("Kerr half-plane initialization static checks passed")
