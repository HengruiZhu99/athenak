#!/usr/bin/env python3
"""Fail-closed source contract for universal Z4c admissibility checks."""

from __future__ import annotations

import argparse
from pathlib import Path


def require(text: str, token: str, source: Path) -> None:
    if token not in text:
        raise SystemExit(f"missing {token!r} in {source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    z4c = (args.source_dir / "src/z4c/z4c.cpp").read_text()
    tasks = (args.source_dir / "src/z4c/z4c_tasks.cpp").read_text()
    update = (args.source_dir / "src/z4c/z4c_update.cpp").read_text()
    rhs = (args.source_dir / "src/z4c/z4c_calcrhs.cpp").read_text()
    header = (args.source_dir / "src/z4c/state_admissibility.hpp").read_text()
    if "detg = detg >" in z4c:
        raise SystemExit("legacy determinant substitution remains in AlgConstr")
    for token in (
        "EvaluateZ4cState",
        "ProjectAdmissibleConformalState",
        "nonpositive_metric_pivot_0",
        "nonpositive_metric_pivot_1",
        "nonpositive_metric_pivot_2",
        "MPI_Allreduce",
    ):
        require(header if token != "MPI_Allreduce" else z4c, token,
                args.source_dir / ("src/z4c/state_admissibility.hpp" if token != "MPI_Allreduce" else "src/z4c/z4c.cpp"))
    require(z4c, 'Kokkos::View<Real *> packed_values',
            args.source_dir / "src/z4c/z4c.cpp")
    require(z4c, 'packed_values(variable) = state(m, variable, k, j, i)',
            args.source_dir / "src/z4c/z4c.cpp")
    require(z4c, r'\"logical_location\"', args.source_dir / "src/z4c/z4c.cpp")
    require(z4c, 'output << "\\\"nan\\\""', args.source_dir / "src/z4c/z4c.cpp")
    if "Kokkos::subview(state, m, Kokkos::ALL(), k, j, i)" in z4c:
        raise SystemExit("noncontiguous state-point host copy remains")
    require(rhs, "Z4cStateCheckpoint::pre_rhs", args.source_dir / "src/z4c/z4c_calcrhs.cpp")
    require(update, "Z4cStateCheckpoint::post_rk_update", args.source_dir / "src/z4c/z4c_update.cpp")
    for checkpoint in (
        "post_restriction",
        "post_receive",
        "post_physical_bc",
        "post_prolongation",
        "post_amr_transfer",
    ):
        require(tasks, f"Z4cStateCheckpoint::{checkpoint}", args.source_dir / "src/z4c/z4c_tasks.cpp")
    require(z4c, "Z4cStateCheckpoint::pre_algconstr", args.source_dir / "src/z4c/z4c.cpp")
    require(z4c, "Z4cStateCheckpoint::post_algconstr", args.source_dir / "src/z4c/z4c.cpp")
    print("Z4C_STATE_ADMISSIBILITY_STATIC_PASS")


if __name__ == "__main__":
    main()
