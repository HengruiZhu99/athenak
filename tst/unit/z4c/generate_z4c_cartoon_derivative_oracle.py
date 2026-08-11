#!/usr/bin/env python3
"""Generate the explicit C++ and high-precision Cartoon MMS oracle artifacts."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re

import sympy as sp

if sp.__version__ != "1.14.0":
    raise RuntimeError(f"Cartoon MMS oracle requires SymPy 1.14.0, got {sp.__version__}")


COMPONENTS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
EXACT_PLANE_ALGEBRAIC = {
    "vector.0.first.2", "vector.2.first.2",
    "tensor.lower.0.0.first.2", "tensor.lower.0.1.first.2",
    "tensor.lower.0.2.first.2", "tensor.lower.1.2.first.2",
    "tensor.lower.2.2.first.2", "tensor.upper.0.0.first.2",
    "tensor.upper.0.1.first.2", "tensor.upper.0.2.first.2",
    "tensor.upper.1.2.first.2", "tensor.upper.2.2.first.2",
}
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

FIELD_MAXIMA = {
    "scalar": sp.Rational(520, 100),
    "vector.0": sp.Rational(678, 100),
    "vector.1": sp.Rational(266, 100),
    "vector.2": sp.Rational(405, 100),
    "tensor.0.0": sp.Rational(1136, 100),
    "tensor.0.1": sp.Rational(186, 100),
    "tensor.0.2": sp.Rational(567, 100),
    "tensor.1.1": sp.Rational(249, 100),
    "tensor.1.2": sp.Rational(159, 100),
    "tensor.2.2": sp.Rational(191, 100),
}
COEFFICIENTS = {
    "2": {"C1": sp.S.One, "C2": sp.Rational(4), "CM": sp.S.One,
          "CU": sp.Rational(4), "CKO": sp.Rational(16), "CE": sp.S.One,
          "CO": sp.Rational(4, 3), "CQ": sp.Rational(20, 9)},
    "3": {"C1": sp.Rational(3, 2), "C2": sp.Rational(16, 3),
          "CM": sp.Rational(9, 4), "CU": sp.Rational(19, 6),
          "CKO": sp.Rational(64), "CE": sp.Rational(3, 2),
          "CO": sp.Rational(28, 15), "CQ": sp.Rational(226, 75)},
    "4": {"C1": sp.Rational(11, 6), "C2": sp.Rational(272, 45),
          "CM": sp.Rational(121, 36), "CU": sp.Rational(3),
          "CKO": sp.Rational(256), "CE": sp.Rational(5, 2),
          "CO": sp.Rational(76, 35), "CQ": sp.Rational(12598, 3675)},
}
OPERATION_CAPS = {"field_fill": 128, "active": 128, "raw": 192,
                  "fit": 384, "divergence": 192, "advection": 384,
                  "oracle": 256}


def upward_hex(value: sp.Rational) -> str:
    exact = Fraction(int(value.p), int(value.q))
    rounded = float(exact)
    if Fraction.from_float(rounded) < exact:
        rounded = math.nextafter(rounded, math.inf)
    return rounded.hex()


def expression_operation_count(expression: sp.Expr) -> int:
    return int(sp.count_ops(expression)) + 7 * sum(
        1 for node in sp.preorder_traversal(expression) if node.func == sp.exp)


def cse_operation_count(selected: list[tuple[str, sp.Expr]]) -> int:
    replacements, reduced = sp.cse([value for _, value in selected],
                                   symbols=sp.numbered_symbols("t"))
    return sum(expression_operation_count(value) for _, value in replacements) + \
        sum(expression_operation_count(value) for value in reduced)


def split_operation_counts(values: list[tuple[str, sp.Expr]]) -> dict[str, int]:
    groups: dict[str, list[tuple[str, sp.Expr]]] = {
        "field.scalar": [item for item in values if item[0] == "scalar"],
        "field.vector": [item for item in values
                         if re.fullmatch(r"vector\[\d\]", item[0])],
        "field.tensor_a": [item for item in values
                           if re.fullmatch(r"tensor\[[0-2]\]", item[0])],
        "field.tensor_b": [item for item in values
                           if re.fullmatch(r"tensor\[[3-5]\]", item[0])],
        "scalar.first": [item for item in values if item[0].startswith("scalar_first")],
        "scalar.second": [item for item in values if item[0].startswith("scalar_second")],
    }
    for family, count in (("vector", 3), ("tensor", 6)):
        for component in range(count):
            for derivative in ("first", "second"):
                groups[f"{family}.{component}.{derivative}"] = [
                    item for item in values if re.fullmatch(
                        rf"{family}_{derivative}\[{component}\].*", item[0])]
    counts = {name: cse_operation_count(selected) for name, selected in groups.items()}
    assert all(count <= (OPERATION_CAPS["field_fill"] if name.startswith("field.")
                         else OPERATION_CAPS["oracle"])
               for name, count in counts.items())
    return counts


def plane_triangle_bound(expression: sp.Expr, symbols: tuple[sp.Symbol, ...]) -> sp.Rational:
    x, y, z = symbols
    plane = expression.subs(y, 0)
    envelope_free = plane.xreplace({atom: sp.S.One for atom in plane.atoms(sp.exp)})
    polynomial = sp.Poly(sp.expand(envelope_free), x, z)
    bound = sum(abs(coefficient) * 3 ** (monomial[0] + monomial[1])
                for monomial, coefficient in polynomial.terms())
    assert bound.is_Rational and bound >= 0
    return bound


def block(kind: str, components: list[str], scale: int = 1,
          direction: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {"kind": kind, "components": components,
                                "scale": str(scale)}
    if direction is not None:
        result["direction"] = direction
    return result


def vector_suppressed_first(component: int) -> dict[str, list[dict[str, object]]]:
    if component == 1:
        return {"fitted": [], "raw": []}
    rotated = 2 if component == 0 else 0
    return {"fitted": [block("odd_value", [f"vector.{rotated}"])],
            "raw": [block("value_over_r", [f"vector.{rotated}"])]}


def tensor_suppressed_first(first: int, second: int) -> dict[str, list[dict[str, object]]]:
    pair = (first, second)
    if pair in ((0, 0), (2, 2)):
        return {"fitted": [block("rho_quad_value", ["tensor.0.2"], 2)],
                "raw": [block("value_over_r", ["tensor.0.2"], 2)]}
    if pair == (0, 2):
        difference = ["tensor.0.0", "tensor.2.2"]
        return {"fitted": [block("rho_quad_value", difference)],
                "raw": [block("value_over_r", difference)]}
    if pair == (0, 1):
        return {"fitted": [block("odd_value", ["tensor.1.2"])],
                "raw": [block("value_over_r", ["tensor.1.2"])]}
    if pair == (1, 2):
        return {"fitted": [block("odd_value", ["tensor.0.1"])],
                "raw": [block("value_over_r", ["tensor.0.1"])]}
    return {"fitted": [], "raw": []}


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
                      selected: list[tuple[str, sp.Expr]], prefix: str = "",
                      cap: int = 256) -> list[str]:
        count = cse_operation_count(selected)
        if count > cap:
            raise RuntimeError(f"generated {function} has {count} operations, cap={cap}")
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
    field_groups = (
        ("Scalar", [item for item in field_values if item[0] == "scalar"]),
        ("Vector", [item for item in field_values if item[0].startswith("vector")]),
        ("TensorA", [item for item in field_values if re.fullmatch(r"tensor\[[0-2]\]", item[0])]),
        ("TensorB", [item for item in field_values if re.fullmatch(r"tensor\[[3-5]\]", item[0])]),
    )
    for suffix, selected in field_groups:
        lines += emit_function(f"EvaluateFieldValues{suffix}", "FieldValues", selected,
                               cap=OPERATION_CAPS["field_fill"])
    lines += ["KOKKOS_INLINE_FUNCTION",
              "void EvaluateFieldValues(const Real x, const Real y, const Real z,",
              "                         FieldValues &oracle) {",
              "  EvaluateFieldValuesScalar(x, y, z, oracle);",
              "  EvaluateFieldValuesVector(x, y, z, oracle);",
              "  EvaluateFieldValuesTensorA(x, y, z, oracle);",
              "  EvaluateFieldValuesTensorB(x, y, z, oracle);",
              "}", ""]
    scalar_values = [(name.replace("scalar_", ""), value) for name, value in values
                     if name.startswith("scalar_first") or
                     name.startswith("scalar_second")]
    for derivative in ("first", "second"):
        selected = [item for item in scalar_values if item[0].startswith(derivative)]
        lines += emit_function(f"EvaluateScalarOracle{derivative.title()}",
                               "ScalarOracle", selected,
                               cap=OPERATION_CAPS["oracle"])
    lines += ["KOKKOS_INLINE_FUNCTION",
              "void EvaluateScalarOracle(const Real x, const Real y, const Real z,",
              "                          ScalarOracle &oracle) {",
              "  EvaluateScalarOracleFirst(x, y, z, oracle);",
              "  EvaluateScalarOracleSecond(x, y, z, oracle);",
              "}", ""]
    for component in range(3):
        for derivative in ("first", "second"):
            selected = []
            for name, value in values:
                match = re.fullmatch(
                    rf"vector_{derivative}\[{component}\](.*)", name)
                if match:
                    selected.append((derivative + match.group(1), value))
            lines += emit_function(
                f"EvaluateVectorOracle{component}{derivative.title()}",
                "VectorOracle", selected, cap=OPERATION_CAPS["oracle"])
        lines += ["KOKKOS_INLINE_FUNCTION",
                  f"void EvaluateVectorOracle{component}(const Real x, const Real y, const Real z,",
                  "                           VectorOracle &oracle) {",
                  f"  EvaluateVectorOracle{component}First(x, y, z, oracle);",
                  f"  EvaluateVectorOracle{component}Second(x, y, z, oracle);",
                  "}", ""]
    lines += ["KOKKOS_INLINE_FUNCTION",
              "void EvaluateVectorOracle(const int component, const Real x, const Real y,",
              "                          const Real z, VectorOracle &oracle) {",
              "  if (component == 0) EvaluateVectorOracle0(x, y, z, oracle);",
              "  if (component == 1) EvaluateVectorOracle1(x, y, z, oracle);",
              "  if (component == 2) EvaluateVectorOracle2(x, y, z, oracle);",
              "}", ""]
    for component in range(6):
        for derivative in ("first", "second"):
            selected = []
            for name, value in values:
                match = re.fullmatch(
                    rf"tensor_{derivative}\[{component}\](.*)", name)
                if match:
                    selected.append((derivative + match.group(1), value))
            lines += emit_function(
                f"EvaluateTensorOracle{component}{derivative.title()}",
                "TensorOracle", selected, cap=OPERATION_CAPS["oracle"])
        lines += ["KOKKOS_INLINE_FUNCTION",
                  f"void EvaluateTensorOracle{component}(const Real x, const Real y, const Real z,",
                  "                           TensorOracle &oracle) {",
                  f"  EvaluateTensorOracle{component}First(x, y, z, oracle);",
                  f"  EvaluateTensorOracle{component}Second(x, y, z, oracle);",
                  "}", ""]
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


def roundoff_branches(name: str) -> dict[str, object]:
    """Return the finite source-family row used by the binary64 floor audit."""
    active: list[dict[str, object]] = []
    fitted: list[dict[str, object]] = []
    raw: list[dict[str, object]] = []
    source_branch = ""
    if name.startswith("scalar."):
        component = "scalar"
        match = re.fullmatch(r"scalar\.first\.(\d)", name)
        if match:
            direction = int(match.group(1))
            source_branch = "scalar_first"
            if direction < 2:
                active = [block("dx" if direction == 0 else "dz", [component])]
        match = re.fullmatch(r"scalar\.second\.(\d)\.(\d)", name)
        if match:
            first, second = map(int, match.groups())
            source_branch = "scalar_second"
            if second < 2:
                kind = "dxx" if first == second == 0 else \
                    ("dzz" if first == second == 1 else "dxz")
                active = [block(kind, [component])]
            elif first == second == 2:
                fitted = [block("even_derivative", [component], 2)]
                raw = [block("dx_over_r", [component])]
        if name == "scalar.advective":
            source_branch = "scalar_advection"
            active = [block("up", [f"vector.{direction}", component],
                            direction=direction) for direction in (0, 1)]
    elif name == "vector.divergence":
        source_branch = "vector_divergence"
        active = [block("div_dx", ["vector.0"]), block("div_dz", ["vector.1"])]
        fitted = [block("div_odd_value", ["vector.0"])]
        raw = [block("div_value_over_r", ["vector.0"])]
    elif name.startswith("vector."):
        match = re.fullmatch(r"vector\.(\d)\.first\.(\d)", name)
        if match:
            component_index, direction = map(int, match.groups())
            component = f"vector.{component_index}"
            source_branch = "vector_first"
            if direction < 2:
                active = [block("dx" if direction == 0 else "dz", [component])]
            else:
                branches = vector_suppressed_first(component_index)
                fitted, raw = branches["fitted"], branches["raw"]
        match = re.fullmatch(r"vector\.(\d)\.second\.(\d)\.(\d)", name)
        if match:
            component_index, first, second = map(int, match.groups())
            component = f"vector.{component_index}"
            source_branch = "vector_second"
            if second < 2:
                kind = "dxx" if first == second == 0 else \
                    ("dzz" if first == second == 1 else "dxz")
                active = [block(kind, [component])]
            elif first == second == 2:
                if component_index == 1:
                    fitted = [block("even_derivative", [component], 2)]
                    raw = [block("dx_over_r", [component])]
                else:
                    fitted = [block("rho_odd_derivative", [component], 2)]
                    raw = [block("dx_over_r", [component]),
                           block("value_over_r2", [component])]
            elif component_index != 1:
                active_direction = first if first != 2 else second
                rotated = 2 if component_index == 0 else 0
                rotated_component = f"vector.{rotated}"
                if active_direction == 0:
                    fitted = [block("rho_odd_derivative", [rotated_component], 2)]
                    raw = [block("dx_over_r", [rotated_component]),
                           block("value_over_r2", [rotated_component])]
                else:
                    fitted = [block("odd_active_value", [rotated_component],
                                    direction=1)]
                    raw = [block("dz_over_r", [rotated_component])]
        match = re.fullmatch(r"vector\.(\d)\.advective", name)
        if match:
            component_index = int(match.group(1))
            component = f"vector.{component_index}"
            source_branch = "vector_advection"
            active = [block("up", [f"vector.{direction}", component],
                            direction=direction) for direction in (0, 1)]
            branches = vector_suppressed_first(component_index)
            fitted = [{**item, "kind": "product_" + str(item["kind"]),
                       "components": ["vector.2", *item["components"]]}
                      for item in branches["fitted"]]
            raw = [{**item, "kind": "product_" + str(item["kind"]),
                    "components": ["vector.2", *item["components"]]}
                   for item in branches["raw"]]
    elif name.startswith("tensor."):
        match = re.fullmatch(
            r"tensor\.(lower|upper)\.(\d)\.(\d)\.(first|second)\.(\d)(?:\.(\d))?",
            name)
        if match:
            _, first_component, second_component, derivative, first, second = match.groups()
            a, b, first = int(first_component), int(second_component), int(first)
            second_value = None if second is None else int(second)
            component = f"tensor.{a}.{b}"
            source_branch = f"tensor_{derivative}"
            if derivative == "first":
                if first < 2:
                    active = [block("dx" if first == 0 else "dz", [component])]
                else:
                    branches = tensor_suppressed_first(a, b)
                    fitted, raw = branches["fitted"], branches["raw"]
            elif second_value is not None and second_value < 2:
                kind = "dxx" if first == second_value == 0 else \
                    ("dzz" if first == second_value == 1 else "dxz")
                active = [block(kind, [component])]
            elif first == second_value == 2:
                difference = ["tensor.0.0", "tensor.2.2"]
                if (a, b) in ((0, 0), (2, 2)):
                    fitted = [block("even_derivative", [component], 2),
                              block("quad_value", difference, 2)]
                    raw = [block("dx_over_r", [component]),
                           block("value_over_r2", difference, 2)]
                elif (a, b) == (0, 2):
                    fitted = [block("rho2_quad_derivative", [component], 2),
                              block("quad_value", [component], 2)]
                    raw = [block("dx_over_r", [component]),
                           block("value_over_r2", [component], 4)]
                elif (a, b) in ((0, 1), (1, 2)):
                    fitted = [block("rho_odd_derivative", [component], 2)]
                    raw = [block("dx_over_r", [component]),
                           block("value_over_r2", [component])]
                else:
                    fitted = [block("even_derivative", [component], 2)]
                    raw = [block("dx_over_r", [component])]
            elif second_value is not None:
                active_direction = first if first != 2 else second_value
                difference = ["tensor.0.0", "tensor.2.2"]
                if (a, b) in ((0, 0), (2, 2)):
                    if active_direction == 0:
                        fitted = [block("quad_value", ["tensor.0.2"], 2),
                                  block("rho2_quad_derivative", ["tensor.0.2"], 4)]
                    else:
                        fitted = [block("rho_quad_active_value", ["tensor.0.2"], 2,
                                        direction=1)]
                    raw = [block("active_over_r", ["tensor.0.2"], 2,
                                 direction=active_direction)]
                    if active_direction == 0:
                        raw.append(block("value_over_r2", ["tensor.0.2"], 2))
                elif (a, b) == (0, 2):
                    if active_direction == 0:
                        fitted = [block("quad_value", difference),
                                  block("rho2_quad_derivative", difference, 2)]
                    else:
                        fitted = [block("rho_quad_active_value", difference,
                                        direction=1)]
                    raw = [block("active_over_r", difference,
                                 direction=active_direction)]
                    if active_direction == 0:
                        raw.append(block("value_over_r2", difference))
                elif (a, b) in ((0, 1), (1, 2)):
                    rotated = "tensor.1.2" if (a, b) == (0, 1) else "tensor.0.1"
                    if active_direction == 0:
                        fitted = [block("rho_odd_derivative", [rotated], 2)]
                    else:
                        fitted = [block("odd_active_value", [rotated], direction=1)]
                    raw = [block("active_over_r", [rotated],
                                 direction=active_direction)]
                    if active_direction == 0:
                        raw.append(block("value_over_r2", [rotated]))
        match = re.fullmatch(r"tensor\.(lower|upper)\.(\d)\.(\d)\.advective", name)
        if match:
            _, first_component, second_component = match.groups()
            a, b = int(first_component), int(second_component)
            component = f"tensor.{a}.{b}"
            source_branch = "tensor_advection"
            active = [block("up", [f"vector.{direction}", component],
                            direction=direction) for direction in (0, 1)]
            branches = tensor_suppressed_first(a, b)
            fitted = [{**item, "kind": "product_" + str(item["kind"]),
                       "components": ["vector.2", *item["components"]]}
                      for item in branches["fitted"]]
            raw = [{**item, "kind": "product_" + str(item["kind"]),
                    "components": ["vector.2", *item["components"]]}
                   for item in branches["raw"]]
    elif name.startswith("state."):
        source_branch = "dissipation"
        component_index = int(name.split(".")[1])
        component = (["scalar"] + [f"vector.{value}" for value in range(3)] +
                     [f"tensor.{a}.{b}" for a, b in COMPONENTS])[component_index]
        active = [block("ko", [component], direction=direction)
                  for direction in (0, 1)]
    if not source_branch:
        raise RuntimeError(f"missing roundoff source-family mapping for {name}")
    return {"source_branch": source_branch, "fitted_t": "runtime_target_layer",
            "raw_class": "outside_regularized_half_cell_layers",
            "active": active, "fitted": fitted, "raw": raw}


def verify_fit_tables() -> list[dict[str, object]]:
    fixtures = []
    derivative_distinguished = False
    tolerance = sp.Rational(1, 10**10)
    for nghost in (2, 3, 4):
        nodes = [(sp.Rational(layer) + sp.Rational(1, 2)) ** 2
                 for layer in range(nghost)]
        target = sp.symbols("target", real=True)
        basis = []
        derivatives = []
        for point, point_s in enumerate(nodes):
            polynomial = sp.prod((target - nodes[other]) / (point_s - nodes[other])
                                 for other in range(nghost) if other != point)
            basis.append(polynomial)
            derivatives.append(sp.diff(polynomial, target))
        for layer in range(nghost):
            half_index = 2 * layer + 1
            for sign in (-1, 1):
                for endpoint in (-1, 1):
                    signed_half = sign * (sp.Rational(half_index) + endpoint * tolerance)
                    evaluated = (signed_half / 2) ** 2
                    for power, derivative_name in ((0, "CE"), (1, "CO"), (2, "CQ")):
                        value_norm = sum(abs(sp.N(polynomial.subs(target, evaluated), 50)) /
                                         (sp.Rational(point) + sp.Rational(1, 2)) ** power
                                         for point, polynomial in enumerate(basis))
                        derivative_norm = sum(
                            abs(sp.N(derivative.subs(target, evaluated), 50)) /
                            (sp.Rational(point) + sp.Rational(1, 2)) ** power
                            for point, derivative in enumerate(derivatives))
                        construction_norm = sum(
                            abs(sp.N(
                                (sp.S.One / (nodes[point] - nodes[differentiated])) *
                                sp.prod((evaluated - nodes[other]) /
                                        (nodes[point] - nodes[other])
                                        for other in range(nghost)
                                        if other not in (point, differentiated)), 50)) /
                            (sp.Rational(point) + sp.Rational(1, 2)) ** power
                            for point in range(nghost)
                            for differentiated in range(nghost)
                            if differentiated != point)
                        derivative_distinguished |= derivative_norm != value_norm
                        assert value_norm <= sp.Rational(5, 4) * 2 ** power
                        assert derivative_norm <= (sp.Rational(5, 4) *
                                                   COEFFICIENTS[str(nghost)][derivative_name])
                        assert construction_norm <= (sp.Rational(5, 4) *
                                                     COEFFICIENTS[str(nghost)]
                                                     [derivative_name])
                        fixtures.append({"nghost": nghost, "layer": layer,
                                         "sign": sign, "band_endpoint": endpoint,
                                         "power": power,
                                         "construction_condition_checked": True})
    assert derivative_distinguished
    return fixtures


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
        elif name in EXACT_PLANE_ALGEBRAIC:
            classification = "exact_plane_algebraic"
        else:
            classification = "truncating"
        independent = name.startswith((
            "tensor.lower.0.0.", "tensor.lower.0.2.", "tensor.lower.2.2.",
            "tensor.upper.0.0.", "tensor.upper.0.2.", "tensor.upper.2.2."))
        noise_lanes = ["shared"] + (["independent"] if independent else [])
        convergence_lanes = (["clean", "shared"] +
                             (["independent"] if independent else [])
                             if classification == "truncating" else ["clean"])
        oracle_bound = sp.S.Zero if expression is None else \
            plane_triangle_bound(expression, symbols)
        records.append({"index": index, "name": name,
                        "classification": classification,
                        "oracle_bound_rational": str(oracle_bound),
                        "oracle_bound_hex": upward_hex(oracle_bound),
                        "oracle_operation_count": 0 if expression is None else
                        OPERATION_CAPS["oracle"],
                        "roundoff_family": roundoff_branches(name),
                        "noise_lanes": noise_lanes,
                        "convergence_lanes": convergence_lanes})
    assert {record["name"] for record in records
            if record["classification"] == "exact_plane_algebraic"} == \
        EXACT_PLANE_ALGEBRAIC
    field_lookup = dict(values)
    component_expressions = {
        "scalar": field_lookup["scalar"],
        **{f"vector.{component}": field_lookup[f"vector[{component}]"]
           for component in range(3)},
        **{f"tensor.{first}.{second}": field_lookup[f"tensor[{component}]"]
           for component, (first, second) in enumerate(COMPONENTS)},
    }
    assert all(plane_triangle_bound(expression, symbols) <= FIELD_MAXIMA[name]
               for name, expression in component_expressions.items())
    fit_fixtures = verify_fit_tables()
    return json.dumps({"schema": "athenak_z4c_cartoon_mms_series_v1",
                       "count": 171,
                       "contract_errata": {
                           "scalar.second.0.0":
                           "active radial Dxx; scalar.second.2.2 owns 2 EvenP"},
                       "roundoff_policy": {
                           "binary64_epsilon_hex": float.fromhex("0x1.0000000000000p-52").hex(),
                           "operation_caps": OPERATION_CAPS,
                           "split_function_operation_counts":
                           split_operation_counts(values),
                           "field_maxima": {
                               name: {"rational": str(value), "hex": upward_hex(value)}
                               for name, value in FIELD_MAXIMA.items()},
                           "coefficients": {
                               nghost: {name: {"rational": str(value),
                                               "hex": upward_hex(value)}
                                        for name, value in row.items()}
                               for nghost, row in COEFFICIENTS.items()},
                           "fit_safety_rational": "5/4",
                           "fit_safety_hex": upward_hex(sp.Rational(5, 4)),
                           "global_slack_rational": "2",
                           "global_slack_hex": upward_hex(sp.Rational(2)),
                           "fit_fixture_count": len(fit_fixtures),
                           "fit_construction_condition_checked": True,
                           "fit_fixture_sha256": hashlib.sha256(
                               json.dumps(fit_fixtures, sort_keys=True,
                                          separators=(",", ":")).encode()).hexdigest(),
                       },
                       "series": records},
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
