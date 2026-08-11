#!/usr/bin/env python3
"""Static ownership and host-dispatch contract for the shared Z4c policy migration."""

from __future__ import annotations

import argparse
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
            "CalcRHSImpl<CartoonSO2, 2>",
            "CalcRHSImpl<Cartesian3D, 4>",
            "MakeCellCenteredDerivativeProvider<Symmetry, NGHOST>",
        ],
        "src/z4c/z4c_adm.cpp": [
            "ADMConstraintsImpl<CartoonSO2, NGHOST>",
            "MakeCellCenteredDerivativeProvider<Symmetry, FD_STENCIL>",
            "TensorVariance::all_lower",
        ],
        "src/z4c/z4c_Sbc.cpp": [
            "MakeCellCenteredDerivativeProvider<Symmetry, 2>",
            "Z4cBoundaryRHSImpl<CartoonSO2>",
        ],
        "src/z4c/z4c_calculate_weyl_scalars.cpp": [
            "Z4cWeylImpl<CartoonSO2, 2>",
            "TensorVariance::all_lower",
            "WeylX3Coordinate<Symmetry>",
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
        if "MakeCellCenteredDerivativeProvider" in body)
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
        if "WeylX3Coordinate<Symmetry>" in body)
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

    # The persisted central scalar is the square root of the production aggregate
    # con.C.  Freeze both sides of that contract so it cannot silently become a
    # hand-picked subset of constraint components.
    sampler = (source_dir / "src/z4c/cartoon_meridional_sampler.hpp").read_text(
        encoding="utf-8")
    require("SampleCartoonMeridionalScalar(constraints, 0, stencil)" in sampler,
            "central sampler does not consume the production con.C aggregate")
    require("Z4cAggregateConstraintNorm(c)" in sampler,
            "central sampler does not apply the aggregate constraint norm")
    for term in ("SQR(con.H", "con.M(m,k,j,i)", "SQR(z4c.vTheta", "4.0*con.Z"):
        require(term in adm_source,
                f"production con.C omitted full constraint-inventory term {term!r}")
    history = (source_dir / "src/outputs/history.cpp").read_text(encoding="utf-8")
    require(history.count("Z4cDiagnosticCellMeasure(") == 1 and
            'Kokkos::parallel_reduce(\n      "Z4cHistoryMaxAbsK"' in history and
            "Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji)" in history,
            "Cartoon volume policy leaked into full-plane extrema")

    # Initial sampling follows initialized ADM/constraints, while accepted-step
    # sampling follows the after-timeintegrator task list (whose Z4c tail refreshes
    # ADM and constraints) and the authoritative time/cycle increment.
    driver = (source_dir / "src/driver/driver.cpp").read_text(encoding="utf-8")
    initialize = driver[driver.index("void Driver::Initialize(Mesh"):
                        driver.index("void Driver::Execute(Mesh")]
    require(initialize.index("InitBoundaryValuesAndPrimitives(pmesh)") <
            initialize.index("pz4c->ConvertZ4cToADM(this, nexp_stages)") <
            initialize.index("pz4c->ADMConstraints_(this, nexp_stages)") <
            initialize.index("UpdateCartoonCentralState(pmesh, res_flag)") <
            initialize.index("for (auto &out : pout->pout_list)"),
            "central initialization does not follow fresh ADM/constraints")
    execute = driver[driver.index("void Driver::Execute(Mesh"):
                     driver.index("void Driver::Finalize(Mesh")]
    require(execute.index('ExecuteTaskList(pmesh, "after_stagen", stage)') <
            execute.index('ExecuteTaskList(pmesh, "after_timeintegrator", 1)') <
            execute.index("pmesh->time = pmesh->time + pmesh->dt") <
            execute.index("pmesh->ncycle++") <
            execute.index("UpdateCartoonCentralState(pmesh, false)") <
            execute.index("for (auto &out : pout->pout_list)"),
            "accepted-step central sampling is not post-task/post-time and pre-output")
    tasks = (source_dir / "src/z4c/z4c_tasks.cpp").read_text(encoding="utf-8")
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
