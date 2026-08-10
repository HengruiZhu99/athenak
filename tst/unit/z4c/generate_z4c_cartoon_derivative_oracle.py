#!/usr/bin/env python3
"""Generate the explicit C++ and high-precision Cartoon MMS oracle artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import sympy as sp

if sp.__version__ != "1.14.0":
    raise RuntimeError(f"Cartoon MMS oracle requires SymPy 1.14.0, got {sp.__version__}")


COMPONENTS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
# Conventional Cartesian points (x,y,z).  The meridional production plane is y=0;
# these cover signed generic points, the true axis, every fitted half-cell family,
# first raw layers, and the frozen +/-0.5 physical-radius targets.
POINTS = (
    (sp.Rational(-5, 4), sp.S.Zero, sp.Rational(-2, 5)),
    (sp.Rational(-9, 2), sp.S.Zero, sp.Rational(3, 10)),
    (sp.Rational(-7, 2), sp.S.Zero, sp.Rational(-3, 10)),
    (sp.Rational(-5, 2), sp.S.Zero, sp.Rational(1, 5)),
    (sp.Rational(-3, 2), sp.S.Zero, sp.Rational(-1, 5)),
    (sp.Rational(-1, 2), sp.S.Zero, sp.Rational(2, 5)),
    (sp.S.Zero, sp.S.Zero, sp.Rational(1, 5)),
    (sp.Rational(1, 2), sp.S.Zero, sp.Rational(-2, 5)),
    (sp.Rational(3, 2), sp.S.Zero, sp.Rational(1, 5)),
    (sp.Rational(5, 2), sp.S.Zero, sp.Rational(-1, 5)),
    (sp.Rational(7, 2), sp.S.Zero, sp.Rational(3, 10)),
    (sp.Rational(9, 2), sp.S.Zero, sp.Rational(-3, 10)),
    (sp.Rational(5, 4), sp.S.Zero, sp.Rational(2, 5)),
)


def expressions() -> tuple[tuple[sp.Symbol, ...], list[tuple[str, sp.Expr]]]:
    x, y, z = sp.symbols("x y z", real=True)
    s = x * x + y * y
    radius2 = s + z * z
    envelope = sp.exp(-sp.Rational(1, 5) * radius2)
    scalar = envelope * (1 + sp.Rational(17, 100) * z +
                         sp.Rational(11, 100) * s -
                         sp.Rational(7, 100) * s * z +
                         sp.Rational(3, 100) * z**3)
    a = envelope * (sp.Rational(70, 100) + sp.Rational(13, 100) * z +
                    sp.Rational(9, 100) * s - sp.Rational(4, 100) * z**2)
    b = envelope * (-sp.Rational(20, 100) + sp.Rational(19, 100) * z -
                    sp.Rational(6, 100) * s + sp.Rational(5, 100) * s * z)
    c = envelope * (sp.Rational(30, 100) - sp.Rational(8, 100) * z +
                    sp.Rational(7, 100) * s + sp.Rational(2, 100) * z**2)
    radial = sp.Matrix((x, y, 0))
    azimuthal = sp.Matrix((-y, x, 0))
    axial = sp.Matrix((0, 0, 1))
    vector = a * radial + c * azimuthal + b * axial

    p = envelope * (sp.Rational(110, 100) + sp.Rational(12, 100) * z +
                    sp.Rational(5, 100) * s)
    q = envelope * (-sp.Rational(30, 100) + sp.Rational(7, 100) * z -
                    sp.Rational(4, 100) * s + sp.Rational(2, 100) * z**2)
    h = envelope * (sp.Rational(21, 100) - sp.Rational(5, 100) * z +
                    sp.Rational(3, 100) * s)
    u = envelope * (-sp.Rational(17, 100) + sp.Rational(9, 100) * z +
                    sp.Rational(2, 100) * s)
    w = envelope * (sp.Rational(14, 100) + sp.Rational(4, 100) * z -
                    sp.Rational(3, 100) * s)
    zz = envelope * (sp.Rational(90, 100) - sp.Rational(11, 100) * z +
                     sp.Rational(8, 100) * s - sp.Rational(2, 100) * s * z)
    identity_perp = sp.diag(1, 1, 0)
    tensor = (p * identity_perp + q * radial * radial.T +
              h * (radial * azimuthal.T + azimuthal * radial.T) +
              u * (radial * axial.T + axial * radial.T) +
              w * (azimuthal * axial.T + axial * azimuthal.T) +
              zz * axial * axial.T)

    # Explicit conventional-to-provider permutation: (x,z,Y)=(rho,z,suppressed).
    coordinates = (x, z, y)
    values: list[tuple[str, sp.Expr]] = [("scalar", scalar)]
    values += [(f"scalar_first[{d}]", sp.diff(scalar, coordinates[d]))
               for d in range(3)]
    values += [(f"scalar_second[{a}][{b}]",
                sp.diff(scalar, coordinates[a], coordinates[b]))
               for a in range(3) for b in range(3)]
    mapped_vector = (vector[0], vector[2], vector[1])
    plane = {y: sp.S.Zero}
    required_vector_plane = (x * a, b, x * c)
    assert all(sp.simplify(mapped_vector[index].subs(plane) -
                           required_vector_plane[index].subs(plane)) == 0
               for index in range(3))
    for component, value in enumerate(mapped_vector):
        values.append((f"vector[{component}]", value))
        values += [(f"vector_first[{component}][{d}]", sp.diff(value, coordinates[d]))
                   for d in range(3)]
        values += [(f"vector_second[{component}][{a}][{b}]",
                    sp.diff(value, coordinates[a], coordinates[b]))
                   for a in range(3) for b in range(3)]
    map_index = (0, 2, 1)
    mapped_tensor = tuple(tensor[map_index[first], map_index[second]]
                          for first, second in COMPONENTS)
    required_tensor_plane = (p + q * x**2, u * x, h * x**2,
                             zz, w * x, p)
    assert all(sp.simplify(mapped_tensor[index].subs(plane) -
                           required_tensor_plane[index].subs(plane)) == 0
               for index in range(6))
    for component, (first, second) in enumerate(COMPONENTS):
        value = mapped_tensor[component]
        values.append((f"tensor[{component}]", value))
        values += [(f"tensor_first[{component}][{d}]", sp.diff(value, coordinates[d]))
                   for d in range(3)]
        values += [(f"tensor_second[{component}][{a}][{b}]",
                    sp.diff(value, coordinates[a], coordinates[b]))
                   for a in range(3) for b in range(3)]
    return (x, y, z), values


def render_header(symbols: tuple[sp.Symbol, ...], values: list[tuple[str, sp.Expr]]) -> str:
    def emit_function(function: str, struct: str,
                      selected: list[tuple[str, sp.Expr]], prefix: str = "") -> list[str]:
        replacements, reduced = sp.cse([value for _, value in selected],
                                       symbols=sp.numbered_symbols("t"))
        generated = ["KOKKOS_INLINE_FUNCTION",
                     f"void {function}(const Real x, const Real y, const Real z,",
                     f"                {struct} &oracle) {{"]
        generated += [f"  const Real {temporary} = {sp.ccode(expression)};"
                      for temporary, expression in replacements]
        for (target, _), expression in zip(selected, reduced):
            target = target[len(prefix):] if prefix and target.startswith(prefix) else target
            generated.append(f"  oracle.{target} = {sp.ccode(expression)};")
        generated += ["}", ""]
        return generated

    replacements, reduced = sp.cse([value for _, value in values], symbols=sp.numbered_symbols("t"))
    lines = [
        "// Generated by tst/unit/z4c/generate_z4c_cartoon_derivative_oracle.py.",
        "// Do not edit by hand.",
        "#ifndef PGEN_UNIT_TESTS_Z4C_CARTOON_DERIVATIVES_ORACLE_HPP_",
        "#define PGEN_UNIT_TESTS_Z4C_CARTOON_DERIVATIVES_ORACLE_HPP_",
        "",
        '#include "athena.hpp"',
        "",
        "namespace z4c_mms {",
        "",
        "struct AnalyticOracle {",
        "  Real scalar;",
        "  Real scalar_first[3];",
        "  Real scalar_second[3][3];",
        "  Real vector[3];",
        "  Real vector_first[3][3];",
        "  Real vector_second[3][3][3];",
        "  Real tensor[6];",
        "  Real tensor_first[6][3];",
        "  Real tensor_second[6][3][3];",
        "};",
        "",
        "struct FieldValues { Real scalar; Real vector[3]; Real tensor[6]; };",
        "struct ScalarOracle { Real first[3]; Real second[3][3]; };",
        "struct VectorOracle { Real first[3]; Real second[3][3]; };",
        "struct TensorOracle { Real first[3]; Real second[3][3]; };",
        "",
    ]
    field_values = [(name, value) for name, value in values
                    if name == "scalar" or
                    re.fullmatch(r"vector\[\d\]", name) or
                    re.fullmatch(r"tensor\[\d\]", name)]
    lines += emit_function("EvaluateFieldValues", "FieldValues", field_values)
    scalar_values = [(name.replace("scalar_", ""), value) for name, value in values
                     if name.startswith("scalar_first") or
                     name.startswith("scalar_second")]
    lines += emit_function("EvaluateScalarOracle", "ScalarOracle", scalar_values)
    for component in range(3):
        prefix = f"vector_"
        selected = []
        for name, value in values:
            match = re.fullmatch(rf"vector_(first|second)\[{component}\](.*)", name)
            if match:
                selected.append((match.group(1) + match.group(2), value))
        lines += emit_function(f"EvaluateVectorOracle{component}", "VectorOracle", selected)
    lines += ["KOKKOS_INLINE_FUNCTION",
              "void EvaluateVectorOracle(const int component, const Real x, const Real y,",
              "                          const Real z, VectorOracle &oracle) {",
              "  if (component == 0) EvaluateVectorOracle0(x, y, z, oracle);",
              "  if (component == 1) EvaluateVectorOracle1(x, y, z, oracle);",
              "  if (component == 2) EvaluateVectorOracle2(x, y, z, oracle);",
              "}", ""]
    for component in range(6):
        selected = []
        for name, value in values:
            match = re.fullmatch(rf"tensor_(first|second)\[{component}\](.*)", name)
            if match:
                selected.append((match.group(1) + match.group(2), value))
        lines += emit_function(f"EvaluateTensorOracle{component}", "TensorOracle", selected)
    lines += ["KOKKOS_INLINE_FUNCTION",
              "void EvaluateTensorOracle(const int component, const Real x, const Real y,",
              "                          const Real z, TensorOracle &oracle) {",
    ]
    for component in range(6):
        lines.append(f"  if (component == {component}) EvaluateTensorOracle{component}(x, y, z, oracle);")
    lines += ["}", "",
        "KOKKOS_INLINE_FUNCTION",
        "void EvaluateAnalyticOracle(const Real x, const Real y, const Real z,",
        "                            AnalyticOracle &oracle) {",
    ]
    for temporary, expression in replacements:
        lines.append(f"  const Real {temporary} = {sp.ccode(expression)};")
    for (target, _), expression in zip(values, reduced):
        lines.append(f"  oracle.{target} = {sp.ccode(expression)};")
    lines += ["}", "", "}  // namespace z4c_mms", "",
              "#endif  // PGEN_UNIT_TESTS_Z4C_CARTOON_DERIVATIVES_ORACLE_HPP_", ""]
    return "\n".join(lines)


def render_reference(symbols: tuple[sp.Symbol, ...],
                     values: list[tuple[str, sp.Expr]]) -> str:
    records = []
    for point in POINTS:
        substitutions = dict(zip(symbols, point))
        records.append({
            "point_xyz": [str(coordinate) for coordinate in point],
            "values": {name: str(sp.N(value.subs(substitutions), 90))
                       for name, value in values},
        })
    return json.dumps({"precision_digits": 90, "records": records},
                      indent=2, sort_keys=True) + "\n"


def render_series(symbols: tuple[sp.Symbol, ...],
                  values: list[tuple[str, sp.Expr]]) -> str:
    _, y, _ = symbols
    lookup = dict(values)
    series = []
    for direction in range(3):
        series.append((f"scalar.first.{direction}", lookup[f"scalar_first[{direction}]"]))
    for first, second in COMPONENTS:
        series.append((f"scalar.second.{first}.{second}",
                       lookup[f"scalar_second[{first}][{second}]"]))
    scalar_advection = sum(lookup[f"vector[{direction}]"] *
                           lookup[f"scalar_first[{direction}]"] for direction in range(3))
    series.append(("scalar.advective", scalar_advection))
    for component in range(3):
        for direction in range(3):
            series.append((f"vector.{component}.first.{direction}",
                           lookup[f"vector_first[{component}][{direction}]"]))
    for component in range(3):
        for first, second in COMPONENTS:
            series.append((f"vector.{component}.second.{first}.{second}",
                           lookup[f"vector_second[{component}][{first}][{second}]"]))
    divergence = sum(lookup[f"vector_first[{component}][{component}]"]
                     for component in range(3))
    series.append(("vector.divergence", divergence))
    for component in range(3):
        advection = sum(lookup[f"vector[{direction}]"] *
                        lookup[f"vector_first[{component}][{direction}]"]
                        for direction in range(3))
        series.append((f"vector.{component}.advective", advection))
    for variance in ("lower", "upper"):
        for component, (component_first, component_second) in enumerate(COMPONENTS):
            prefix = f"tensor.{variance}.{component_first}.{component_second}"
            for direction in range(3):
                series.append((f"{prefix}.first.{direction}",
                               lookup[f"tensor_first[{component}][{direction}]"]))
            for first, second in COMPONENTS:
                series.append((f"{prefix}.second.{first}.{second}",
                               lookup[f"tensor_second[{component}][{first}][{second}]"]))
            advection = sum(lookup[f"vector[{direction}]"] *
                            lookup[f"tensor_first[{component}][{direction}]"]
                            for direction in range(3))
            series.append((f"{prefix}.advective", advection))
    for component in range(10):
        series.append((f"state.{component}.dissipation", None))
    assert len(series) == 171 and len({name for name, _ in series}) == 171
    records = []
    for index, (name, expression) in enumerate(series):
        if expression is None:
            classification = "exact_discrete"
        elif sp.simplify(expression.subs(y, 0)) == 0:
            classification = "exact_identity"
        else:
            classification = "truncating"
        independent = name.startswith((
            "tensor.lower.0.0.", "tensor.lower.0.2.", "tensor.lower.2.2.",
            "tensor.upper.0.0.", "tensor.upper.0.2.", "tensor.upper.2.2."))
        noise_lanes = ["shared"] + (["independent"] if independent else [])
        convergence_lanes = (["clean", "shared"] +
                             (["independent"] if independent else [])
                             if classification == "truncating" else ["clean"])
        records.append({"index": index, "name": name,
                        "classification": classification,
                        "noise_lanes": noise_lanes,
                        "convergence_lanes": convergence_lanes})
    return json.dumps({"schema": "athenak_z4c_cartoon_mms_series_v1",
                       "count": 171, "series": records},
                      indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    header = root / "src/pgen/unit_tests/z4c_cartoon_derivatives_oracle.hpp"
    reference = root / "tst/unit/z4c/z4c_cartoon_derivatives_reference.json"
    series = root / "tst/unit/z4c/z4c_cartoon_derivatives_series.json"
    symbols, values = expressions()
    products = {header: render_header(symbols, values),
                reference: render_reference(symbols, values),
                series: render_series(symbols, values)}
    if args.check:
        stale = [str(path.relative_to(root)) for path, content in products.items()
                 if not path.exists() or path.read_text(encoding="utf-8") != content]
        if stale:
            raise SystemExit("stale generated Cartoon MMS artifacts: " + ", ".join(stale))
    else:
        for path, content in products.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    for path, content in products.items():
        print(hashlib.sha256(content.encode()).hexdigest(), path.relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
