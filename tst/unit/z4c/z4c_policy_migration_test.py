#!/usr/bin/env python3
"""Static ownership and host-dispatch contract for the shared Z4c policy migration."""

from __future__ import annotations

import argparse
import math
import pathlib
import re
import sys


RAW = re.compile(r"\b(Dx|Dxx|Dxy|Lx|Diss)\s*<")


def strip_comments_preserving_lines(source: str) -> str:
    def blank(match: re.Match[str]) -> str:
        return "".join("\n" if character == "\n" else " " for character in match.group())

    source = re.sub(r"/\*.*?\*/", blank, source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", blank, source)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def exercise_physical_coarse_fine_corner_composition() -> None:
    """Model the production NGHOST=4 outer-x2/coarse-x1 overlap on both sides."""
    ng = 4
    active_lo = ng
    active_hi = ng + 32 - 1
    extent = 32 + 2 * ng
    components = ("chi", "gxx", "gxy", "gxz", "gyy", "gyz", "gzz")

    def new_field() -> dict[str, list[list[float]]]:
        return {name: [[0.0 for _ in range(extent)] for _ in range(extent)]
                for name in components}

    def sentinel(name: str, j: int, rho_sign: float) -> float:
        z_slope = 1.0e-4 * (j - active_lo)
        values = {
            "chi": 0.85 + z_slope,
            "gxx": 1.10 + z_slope,
            "gxy": 0.01,
            "gxz": rho_sign * (0.02 + z_slope),
            "gyy": 1.00 + z_slope,
            "gyz": rho_sign * 0.015,
            "gzz": 0.95 + z_slope,
        }
        return values[name]

    def fill_active(field: dict[str, list[list[float]]], rho_sign: float) -> None:
        for name in components:
            for j in range(active_lo, active_hi + 1):
                value = sentinel(name, j, rho_sign)
                for i in range(active_lo, active_hi + 1):
                    field[name][j][i] = value

    def outer_x2_builtin(field: dict[str, list[list[float]]]) -> None:
        # This is the order-2 Extrapolate expression used by the campaign input.
        for values in field.values():
            for i in range(extent):
                f0 = values[active_hi][i]
                f1 = values[active_hi - 1][i]
                for layer in range(1, ng + 1):
                    values[active_hi + layer][i] = f0 + layer * (f0 - f1)

    def prolong_x1_side(field: dict[str, list[list[float]]], outer: bool,
                        rho_sign: float) -> None:
        indices = range(active_hi + 1, extent) if outer else range(0, active_lo)
        for name in components:
            for j in range(active_lo, active_hi + 1):
                value = sentinel(name, j, rho_sign)
                for i in indices:
                    field[name][j][i] = value

    def determinant(field: dict[str, list[list[float]]], j: int, i: int) -> float:
        xx = field["gxx"][j][i]
        xy = field["gxy"][j][i]
        xz = field["gxz"][j][i]
        yy = field["gyy"][j][i]
        yz = field["gyz"][j][i]
        zz = field["gzz"][j][i]
        return (xx * yy * zz + 2.0 * xy * xz * yz - xx * yz * yz -
                yy * xz * xz - zz * xy * xy)

    results: dict[str, dict[str, list[list[float]]]] = {}
    for label, outer, rho_sign in (("negative", False, -1.0),
                                   ("positive", True, 1.0)):
        field = new_field()
        fill_active(field, rho_sign)
        outer_x2_builtin(field)
        prolong_x1_side(field, outer, rho_sign)
        corner_i = range(active_hi + 1, extent) if outer else range(0, active_lo)
        corner_j = range(active_hi + 1, extent)
        require(all(field[name][j][i] == 0.0 for name in components
                    for j in corner_j for i in corner_i),
                f"{label} first-pass/prolong unexpectedly filled physical corner")
        outer_x2_builtin(field)
        for j in corner_j:
            for i in corner_i:
                require(all(math.isfinite(field[name][j][i]) for name in components),
                        f"{label} second physical pass left nonfinite corner")
                require(field["chi"][j][i] > 0.0 and determinant(field, j, i) > 0.0,
                        f"{label} second physical pass left invalid chi/metric")
        results[label] = field

    for j in range(active_hi + 1, extent):
        left = results["negative"]
        right = results["positive"]
        require(left["gxz"][j][0] == -right["gxz"][j][extent - 1] and
                left["gyz"][j][0] == -right["gyz"][j][extent - 1],
                "signed-rho odd tensor parity changed in corner composition")
        for name in ("chi", "gxx", "gxy", "gyy", "gzz"):
            require(left[name][j][0] == right[name][j][extent - 1],
                    f"signed-rho even tensor parity changed for {name}")


def kokkos_lambda_bodies(source: str) -> list[str]:
    """Return balanced-brace bodies for the simple KOKKOS_LAMBDA forms used here."""
    bodies: list[str] = []
    search_from = 0
    while True:
        start = source.find("KOKKOS_LAMBDA", search_from)
        if start < 0:
            return bodies
        opening = source.find("{", start)
        require(opening >= 0, "KOKKOS_LAMBDA without a body")
        depth = 1
        cursor = opening + 1
        while cursor < len(source) and depth:
            if source[cursor] == "{":
                depth += 1
            elif source[cursor] == "}":
                depth -= 1
            cursor += 1
        require(depth == 0, "unbalanced KOKKOS_LAMBDA body")
        bodies.append(source[opening:cursor])
        search_from = cursor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    source_dir = arguments.source_dir.resolve()
    exercise_physical_coarse_fine_corner_composition()

    residual: list[tuple[pathlib.Path, int, str]] = []
    for path in sorted((source_dir / "src").rglob("*")):
        if path.suffix not in {".cpp", ".hpp"}:
            continue
        relative = path.relative_to(source_dir)
        if relative in {
            pathlib.Path("src/utils/finite_diff.hpp"),
            pathlib.Path("src/z4c/cartoon_derivatives.hpp"),
        }:
            continue
        text = strip_comments_preserving_lines(path.read_text(encoding="utf-8"))
        for match in RAW.finditer(text):
            residual.append((relative, text.count("\n", 0, match.start()) + 1,
                             match.group(1)))

    require(len(residual) == 3, f"expected three deferred raw FD calls, got {residual}")
    require(all(path == pathlib.Path("src/dyn_grmhd/dyn_grmhd.cpp")
                for path, _, _ in residual),
            f"raw FD call escaped shared Z4c provider: {residual}")
    require([operation for _, _, operation in residual] == ["Dx", "Dx", "Dx"],
            f"unexpected deferred operation set: {residual}")

    required_markers = {
        "src/z4c/z4c_calcrhs.cpp": [
            "CalcRHSImpl<CellCenteredZ4c, CartoonSO2, 2>",
            "CalcRHSImpl<VertexCenteredZ4c, Cartesian3D, 4>",
            "MakeZ4cDerivativeProvider<Centering, Symmetry, NGHOST>",
        ],
        "src/z4c/z4c_adm.cpp": [
            "ADMConstraintsImpl<CellCenteredZ4c, CartoonSO2, NGHOST>",
            "MakeZ4cDerivativeProvider<Centering, Symmetry, FD_STENCIL>",
            "TensorVariance::all_lower",
        ],
        "src/z4c/z4c_Sbc.cpp": [
            "MakeZ4cDerivativeProvider<Centering, Symmetry, 2>",
            "Z4cBoundaryRHSImpl<CellCenteredZ4c, CartoonSO2>",
        ],
        "src/z4c/z4c_calculate_weyl_scalars.cpp": [
            "Z4cWeylImpl<CellCenteredZ4c, CartoonSO2, 2>",
            "TensorVariance::all_lower",
            "WeylX3Coordinate<Centering, Symmetry>",
            "InitializeWeylTetradSeed<Symmetry>",
        ],
        "src/outputs/derived_variables.cpp": [
            "DispatchZ4cDerivedDiagnostics",
            "ComputeZ4cCurvatureDiagnostics<NGHOST>",
        ],
        "src/outputs/history.cpp": [
            "DispatchZ4cHistoryMaxKretschmann",
            "ComputeZ4cCurvatureDiagnostics<NGHOST, false>",
        ],
    }
    for relative, markers in required_markers.items():
        text = (source_dir / relative).read_text(encoding="utf-8")
        for marker in markers:
            require(marker in text, f"missing {marker!r} in {relative}")

    # nvcc forbids first-capturing z4c/g3u inside an if-constexpr in its extended
    # device lambda.  Keep that compile-time choice inside the named derivative
    # factory and make all field captures unconditional in the Gamma kernel.
    adm_source = strip_comments_preserving_lines(
        (source_dir / "src/z4c/z4c_adm.cpp").read_text(encoding="utf-8"))
    gamma_launch = adm_source.index('par_for("initialize Gamma"')
    gamma_body = next(
        body for body in kokkos_lambda_bodies(adm_source[gamma_launch:])
        if "MakeZ4cDerivativeProvider" in body)
    require("if constexpr" not in gamma_body,
            "ADM-to-Z4c Gamma device lambda reintroduced constexpr first-captures")

    # Keep the symmetry-dependent x3 coordinate choice in a named device helper.
    # nvcc rejects first-capturing k inside an if-constexpr in this extended lambda.
    weyl_source = strip_comments_preserving_lines(
        (source_dir / "src/z4c/z4c_calculate_weyl_scalars.cpp").read_text(
            encoding="utf-8"))
    weyl_launch = weyl_source.index('par_for("z4c_weyl_scalar"')
    weyl_body = next(
        body for body in kokkos_lambda_bodies(weyl_source[weyl_launch:])
        if "WeylX3Coordinate<Centering, Symmetry>" in body)
    require("if constexpr" not in weyl_body,
            "Weyl device lambda reintroduced constexpr first-captures")

    # Curvature consumers formerly hard-coded stencil four.  Freeze the intentional
    # semantic correction: every consumer now dispatches the configured 2/3/4 stencil.
    for relative in [
        "src/z4c/curvature_diagnostics.cpp",
        "src/outputs/derived_variables.cpp",
        "src/outputs/history.cpp",
    ]:
        text = (source_dir / relative).read_text(encoding="utf-8")
        for stencil in (2, 3, 4):
            require(f"CartoonSO2, {stencil}" in text and
                    f"Cartesian3D, {stencil}" in text,
                    f"{relative} does not dispatch actual stencil {stencil}")

    # The runtime mode is consumed only by host wrappers.  Device lambdas receive a
    # concrete empty policy tag and cannot accidentally capture the enum or host config.
    for relative in required_markers:
        if not relative.startswith("src/z4c/") and not relative.startswith("src/outputs/"):
            continue
        text = strip_comments_preserving_lines(
            (source_dir / relative).read_text(encoding="utf-8"))
        for body in kokkos_lambda_bodies(text):
            require("z4c_symmetry" not in body and "Z4cSymmetryMode" not in body,
                    f"host symmetry state captured by a device lambda in {relative}")

    sbc = (source_dir / "src/z4c/z4c_Sbc.cpp").read_text(encoding="utf-8")
    cartoon_return = (
        "if constexpr (std::is_same_v<Symmetry, CartoonSO2>) {\n"
        "    return TaskStatus::complete;"
    )
    require(cartoon_return in sbc,
            "Cartoon Sommerfeld path lacks an unconditional compile-time x3 return")
    require(sbc.index(cartoon_return) <
            sbc.index('"z4crhs_bc_x3"'),
            "Cartoon Sommerfeld path does not return before suppressed x3 faces")
    physical = (source_dir / "src/bvals/physics/z4c_bcs.cpp").read_text(encoding="utf-8")
    require(physical.index("if (pm->two_d) return;") < physical.index('"z4cbc_x3"'),
            "collapsed physical boundary path does not stop before x3 faces")

    # The persisted central scalar reconstructs production con.C from fixed canonical
    # active-cell slots.  A mixed 2:1 quadrant is natively restricted from four fine
    # cells; curvature is never evaluated at a ghost-backed or virtual site.
    sampler = (source_dir / "src/z4c/cartoon_meridional_sampler.hpp").read_text(
        encoding="utf-8")
    central_start = sampler.index(
        "inline CartoonCentralSample SampleCartoonCentralDiagnostics")
    central_end = sampler.index("inline const char *CartoonCentralSampleStatusMessage")
    central = strip_comments_preserving_lines(sampler[central_start:central_end])
    require("Kokkos::RangePolicy<DevExeSpace>(0, kCartoonCentralMaxSources)" in central and
            "const CartoonCentralSupport point = supports.point[s]" in central and
            "if (!point.expected || point.local_block < 0) return;" in central,
            "central sampler does not evaluate canonical active physical supports")
    require("constraints(point.local_block, 0, point.k, point.j, point.i)" in central,
            "central sampler does not consume production con.C at active supports")
    require("ReconstructCartoonCentralSupportValues" in central and
            "Z4cAggregateConstraintNorm(constraint_sum)" in sampler,
            "central sampler does not apply the aggregate constraint norm")
    require("CartoonCentralActiveCellHasStoredDerivativeHalo" in sampler and
            "ValidateCartoonCentralSupportSet<NGHOST>" in central,
            "central curvature supports lack an explicit active-halo contract")
    require("point.j, point.i" in central and "stencil.i0" not in central and
            "stencil.j0" not in central,
            "central curvature can still be evaluated at a ghost interpolation site")
    require("flags[s] != 1" in central and
            "MPI_Allreduce(MPI_IN_PLACE, values, 3 * kCartoonCentralMaxSources" in central and
            "MPI_Allreduce(MPI_IN_PLACE, flags, 2 * kCartoonCentralMaxSources" in central,
            "central physical supports do not fail closed on per-support ownership")
    require("point.restriction_weight = refined ? 0.25 : 1.0" in sampler and
            "point.final_weight = half_plane" in sampler and
            "refined ? 0.125 : 0.5" in sampler and
            "refined ? 0.0625 : 0.25" in sampler and
            "common_level = std::min(common_level, quadrant_level[quadrant])" in sampler,
            "central mixed-level restriction lacks fixed reviewed geometry/weights")
    for term in ("SQR(con.H", "con.M(m,k,j,i)", "SQR(z4c.vTheta", "4.0*con.Z"):
        require(term in adm_source,
                f"production con.C omitted full constraint-inventory term {term!r}")
    history = (source_dir / "src/outputs/history.cpp").read_text(encoding="utf-8")
    constraint_maximum = history[
        history.index("ConstraintMaximum CartoonConstraintMaximum("):
        history.index("template <typename Symmetry, int NGHOST>")]
    max_abs_k = history[
        history.index('Kokkos::parallel_reduce(\n      "Z4cHistoryMaxAbsK"'):
        history.index("pdata->hdata[9] = max_abs_K;")]
    require(history.count("Z4cDiagnosticCellMeasure(") == 2 and
            "Z4cDiagnosticCellMeasure(" not in constraint_maximum and
            "Z4cDiagnosticCellMeasure(" not in max_abs_k and
            "Kokkos::RangePolicy<DevExeSpace>(0, nmkji)" in
            constraint_maximum and
            "Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji)" in max_abs_k,
            "Cartoon volume policy leaked into full-plane extrema")

    # Initial sampling follows initialized ADM/constraints, while accepted-step
    # sampling follows the after-timeintegrator task list (whose Z4c tail refreshes
    # ADM and constraints) and the authoritative time/cycle increment.
    driver = (source_dir / "src/driver/driver.cpp").read_text(encoding="utf-8")
    initialize = driver[driver.index("void Driver::Initialize(Mesh"):
                        driver.index("void Driver::Execute(Mesh")]
    require(initialize.index("InitBoundaryValuesAndPrimitives(pmesh, res_flag)") <
            initialize.index("pz4c->ConvertZ4cToADM(this, nexp_stages)") <
            initialize.index("pz4c->ADMConstraints_(this, nexp_stages)") <
            initialize.index("UpdateCartoonCentralState(pmesh, res_flag)") <
            initialize.index("for (auto &out : pout->pout_list)"),
            "central initialization does not follow fresh ADM/constraints")
    driver_header = (source_dir / "src/driver/driver.hpp").read_text(encoding="utf-8")
    boundary_init = driver[driver.index("void Driver::InitBoundaryValuesAndPrimitives"):
                           driver.index("// Initialize HYDRO")]
    refinement = (source_dir / "src/mesh/mesh_refinement.cpp").read_text(
        encoding="utf-8")
    require("bool preserve_restored_z4c = false" in driver_header and
            "bool preserve_restored_z4c)" in boundary_init,
            "Z4c restart-preservation flag is not explicit and default-false")
    require("if (pz4c != nullptr && !preserve_restored_z4c)" in boundary_init,
            "restart preservation does not guard exactly the Z4c initialization branch")
    require("if (pz4c != nullptr)" in boundary_init and
            "pz4c->FillAxisParityGhosts(this, 0)" in boundary_init,
            "fresh/restart initialization does not regenerate derived axis ghosts")
    for required in ("RestrictU", "InitRecv", "SendU", "ClearSend", "ClearRecv",
                     "RecvU", "Z4cBoundaryRHS", "ApplyPhysicalBCs", "Prolongate",
                     "FillBuiltInPhysicalBoundaryGhosts"):
        require(f"pz4c->{required}" in boundary_init,
                f"fresh/AMR Z4c initialization lost {required}")
    initialization_order = [boundary_init.index(f"pz4c->{required}") for required in
                            ("RestrictU", "InitRecv", "SendU", "ClearSend",
                             "ClearRecv", "RecvU", "Z4cBoundaryRHS",
                             "ApplyPhysicalBCs", "Prolongate",
                             "FillBuiltInPhysicalBoundaryGhosts")]
    require(initialization_order == sorted(initialization_order),
            "fresh/AMR Z4c boundary initialization order changed")
    require("InitBoundaryValuesAndPrimitives(pmy_mesh);" in refinement and
            "InitBoundaryValuesAndPrimitives(pmy_mesh," not in refinement,
            "AMR-created blocks do not use default-false Z4c initialization")
    amr_finalize = refinement[
        refinement.index("RedistAndRefineMeshBlocks"):
        refinement.index("nmb_created += nnew")]
    amr_finalize_order = [amr_finalize.index(marker) for marker in
                          ("EnforceAlgConstr", "pdriver->InitBoundaryValuesAndPrimitives",
                           "ConvertZ4cToADM", "ADMConstraints_", "pz4c->NewTimeStep")]
    require(amr_finalize_order == sorted(amr_finalize_order),
            "post-AMR projected active state does not precede cache/ghost reconstruction")
    execute = driver[driver.index("void Driver::Execute(Mesh"):
                     driver.index("void Driver::Finalize(Mesh")]
    normal_sample = execute.index("UpdateCartoonCentralState(pmesh, false)")
    stop_check = execute.index("if (!step_stop_reason.empty())")
    stop_output = execute.index("write_scheduled_outputs();", stop_check)
    stop_break = execute.index("break;", stop_output)
    amr = execute.index("pmesh->pmr->AdaptiveMeshRefinement(this, pin)")
    refresh_guard = execute.index(
        "if (topology_changed && pmesh->pmb_pack->pz4c != nullptr)")
    refresh_sample = execute.index("UpdateCartoonCentralState(pmesh, true)",
                                   refresh_guard)
    ordinary_output = execute.index("write_scheduled_outputs();", refresh_sample)
    capacity_stop = execute.index("if (user_stop) {break;}", ordinary_output)
    require(execute.index('ExecuteTaskList(pmesh, "after_stagen", stage)') <
            execute.index('ExecuteTaskList(pmesh, "after_timeintegrator", 1)') <
            execute.index("pmesh->time = pmesh->time + pmesh->dt") <
            execute.index("pmesh->ncycle++") < normal_sample < stop_check <
            stop_output < stop_break < amr < refresh_guard < refresh_sample <
            ordinary_output < capacity_stop,
            "accepted-step/stop/AMR/refresh/output ordering changed")
    require(execute.count("UpdateCartoonCentralState(pmesh, false)") == 1 and
            execute.count("UpdateCartoonCentralState(pmesh, true)") == 1,
            "driver duplicates accepted-step or post-AMR central collectives")
    require("pmesh->pmr->nmb_created != created_before" in execute and
            "pmesh->pmr->nmb_deleted != deleted_before" in execute,
            "post-AMR central refresh is not guarded by an actual topology change")
    require(execute.count("for (auto &out : pout->pout_list)") == 1 and
            execute.index("auto write_scheduled_outputs") <
            execute.index("while ((pmesh->time < tlim)"),
            "scheduled output policy is duplicated inside the evolution loop")
    require(initialize.index("for (auto &out : pout->pout_list)") >
            initialize.index("UpdateCartoonCentralState(pmesh, res_flag)"),
            "cycle-zero initialization checkpoint no longer follows central sampling")
    tasks = (source_dir / "src/z4c/z4c_tasks.cpp").read_text(encoding="utf-8")
    built_in = tasks[tasks.index("void Z4c::FillBuiltInPhysicalBoundaryGhosts"):
                     tasks.index("TaskStatus Z4c::ApplyPhysicalBCs")]
    public_bc = tasks[tasks.index("TaskStatus Z4c::ApplyPhysicalBCs"):
                      tasks.index("TaskStatus Z4c::TrackCompactObjects")]
    require(built_in.count("pbval_u->Z4cBCs") == 1 and
            "user_bcs" not in built_in,
            "built-in-only corner pass invokes the wrong boundary policy")
    require(public_bc.index("FillBuiltInPhysicalBoundaryGhosts") <
            public_bc.index("user_bcs_func") and
            public_bc.count("user_bcs_func") == 1 and
            tasks.count("user_bcs_func") == 1,
            "public Z4c boundary task no longer invokes each user callback exactly once")
    require(boundary_init.count("FillBuiltInPhysicalBoundaryGhosts") == 1,
            "fresh/AMR initialization lost or duplicated the corner completion pass")
    queue = tasks[tasks.index("void Z4c::QueueZ4cTasks"):
                  tasks.index("TaskStatus Z4c::CopyU")]
    require(queue.index('"Z4c_CopyU"') < queue.index('"Z4c_AxisGhosts"') <
            queue.index('"Z4c_CalcRHS"') and
            queue.count("Task_Run, {Z4c_AxisGhosts}") == 3,
            "each O2/O4/O6 RHS is not ordered after the explicit axis parity fill")
    require(queue.index('"Z4c_ExplRK"') < queue.index('"Z4c_AlgC"') <
            queue.index('"Z4c_RestU"') < queue.index('"Z4c_Prolong"') <
            queue.index('"Z4c_AxisGhostsPost"') <
            queue.index('"Z4c_Z4c2ADM"'),
            "projected accepted state does not own the next cache/ghost/ADM state")
    axis_fill = tasks[tasks.index("TaskStatus Z4c::FillAxisParityGhosts"):
                      tasks.index("TaskStatus Z4c::SendU")]
    require("half_rho_z_suppressed_y_v2" in axis_fill and
            "kHalfPlaneCartoonSchema" in axis_fill and
            "BoundaryFlag::axis" in axis_fill and
            "FillZ4cAxisGhostLine" in axis_fill and
            "Z4cBCs" not in axis_fill and "user_bcs" not in axis_fill,
            "pre-RHS axis task is not a half-plane-only exact parity fill")
    require(queue.index("Z4c_BCS") < queue.index("Z4c_Prolong"),
            "normal RK physical/prolongation task ordering changed")
    task_order = [tasks.index(marker) for marker in
                  ('"Z4c_AlgC"', '"Z4c_RestU"', '"Z4c_SendU"', '"Z4c_RecvU"',
                   '"Z4c_BCS"', '"Z4c_Prolong"', '"Z4c_AxisGhostsPost"',
                   '"Z4c_Z4c2ADM"')]
    require(task_order == sorted(task_order),
            "normal Z4c boundary/projection/ADM task order changed")
    require(tasks.index("&Z4c::ConvertZ4cToADM") <
            tasks.index("&Z4c::ADMConstraints_") and
            '"Z4c_Z4c2ADM",\n                 Task_Run' in tasks and
            '"Z4c_ADMC", Task_End' in tasks,
            "Z4c post-step task tail does not initialize ADM/constraints before completion")

    validator = (source_dir / "src/mesh/meshblock_pack.cpp").read_text(encoding="utf-8")
    require('input.problem_generator == "z4c_cartoon_derivatives"' in
            (source_dir / "src/z4c/z4c_symmetry.cpp").read_text(encoding="utf-8"),
            "Cartoon pgen gate is not restricted to the derivative MMS")
    require('pin->GetString("problem", "check_only")' in validator and
            "cartoon_derivative_check_only_valid" in validator,
            "Cartoon derivative MMS lacks strict check_only parsing")

    print("Z4c policy migration ownership and host-dispatch contract passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as error:
        print(error, file=sys.stderr)
        sys.exit(1)
