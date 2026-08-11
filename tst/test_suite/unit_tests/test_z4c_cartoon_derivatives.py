#!/usr/bin/env python3
"""Immutable CPU/CUDA MPI campaign driver for the input-selected Cartoon MMS.

This driver intentionally has no configure or build capability.  One already-built
Athena executable is checksum-bound and reused for the complete backend matrix.
"""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid


SCHEMA = "athenak_z4c_cartoon_derivative_mms_campaign_v1"
REDUCTION_TOLERANCE_FACTOR = 4096.0
SATURATION_FACTOR = 4096.0
DIAGNOSTIC_RESOLUTIONS = (32, 64, 128, 256)
QUALIFICATION_DOMAIN = (-2.0, 2.0, -2.0, 2.0)
CERTIFIED_COORDINATE_LIMIT = 3.0
EXPECTED_CASES = 3 * 4 * 8
PRESERVED_JOB_56586376_BYTES = 715_807_842
PRESERVED_JOB_56586376_CONVERGENCE_SHA256 = \
    "fdb4222c246b49d4df3c8ef40688dafacfd7983d5f090a2fe148d051538778a0"
PRESERVED_JOB_56586376_EVIDENCE_SHA256 = \
    "347b210b251e6100413ffef5f691edb684b79a2b41fcb6c8757b8a6233fb1869"
PRESERVED_JOB_56586376_LOG_SHA256 = \
    "47522f4f70ad34d15c994e969f84c6e395dd434bcd5e238fff5f719ef2dd43b1"
PRESERVED_JOB_56586376_AUDIT_COUNTS = (63_880, 18_508, 4_512, 3_696)
CASE_ENVIRONMENT_KEYS = (
    "CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "KOKKOS_NUM_DEVICES",
    "KOKKOS_NUM_THREADS", "OMPI_COMM_WORLD_SIZE", "PMIX_NAMESPACE",
    "PMI_SIZE", "SLURM_CPU_BIND", "SLURM_GPU_BIND", "SLURM_GPUS",
    "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS", "SLURM_JOB_ID",
    "SLURM_JOB_NODELIST", "SLURM_NNODES", "SLURM_NTASKS",
    "SLURM_NTASKS_PER_NODE", "SLURM_STEP_GPUS", "SLURM_STEP_ID",
)
CASE_FIXED_FILES = {
    "input.athinput", "stdout.txt", "stderr.txt", "cartoon_mms.mms.json",
    "cartoon_mms.mms.csv", "cartoon_mms.mms.probes.csv", "result.json",
}
NORM_FIELDS = {
    "operator", "mask", "count", "nonfinite", "l1", "l2", "linfinity",
    "cyl_count", "cylindrical_applicable", "cyl_l1", "cyl_l2",
    "cyl_linfinity", "shared_l1", "shared_l2", "shared_linfinity",
    "shared_delta_l1", "shared_delta_l2", "shared_delta_linfinity",
    "independent_l1", "independent_l2", "independent_linfinity",
    "independent_delta_l1", "independent_delta_l2",
    "independent_delta_linfinity", "rotation_linfinity", "target_abs_rho",
    "radius_applicable", "actual_abs_rho", "mask_xor",
}
PROBE_FIELDS = {
    "operator", "mask", "side", "layer_index", "classification",
    "target_rho_applicable", "target_rho", "actual_rho", "target_z", "actual_z",
    "global_cell_id", "raw_error",
}
EXPECTED_COEFFICIENT_RATIONALS = {
    "2": {"C1": "1", "C2": "4", "CM": "1", "CU": "4", "CKO": "16",
          "CE": "1", "CO": "4/3", "CQ": "20/9"},
    "3": {"C1": "3/2", "C2": "16/3", "CM": "9/4", "CU": "19/6",
          "CKO": "64", "CE": "3/2", "CO": "28/15", "CQ": "226/75"},
    "4": {"C1": "11/6", "C2": "272/45", "CM": "121/36", "CU": "3",
          "CKO": "256", "CE": "5/2", "CO": "76/35", "CQ": "12598/3675"},
}
FAMILY_MAPPING_SHA256 = "16b31c7da2defe5c55e221eefde44cb8936f8caf92fb46cef15a7703b7848667"
BLOCK_POWERS = {
    "dx": (1, 0, 0), "div_dx": (1, 0, 0), "dz": (0, 1, 0),
    "div_dz": (0, 1, 0), "dxx": (2, 0, 0), "dzz": (0, 2, 0),
    "dxz": (1, 1, 0), "value_over_r": (0, 0, -1),
    "div_value_over_r": (0, 0, -1), "value_over_r2": (0, 0, -2),
    "dx_over_r": (1, 0, -1), "dz_over_r": (0, 1, -1),
    "active_over_r": (1, 0, -1), "even_derivative": (2, 0, 0),
    "odd_value": (1, 0, 0), "div_odd_value": (1, 0, 0),
    "rho_odd_derivative": (3, 0, 1), "odd_active_value": (2, 0, 0),
    "quad_value": (2, 0, 0), "rho_quad_value": (2, 0, 1),
    "rho2_quad_derivative": (4, 0, 2), "rho_quad_active_value": (3, 0, 1),
    "ko": (1, 0, 0), "up": (1, 0, 0),
}
MASK_WEIGHT_POLICY = {"uniform": "one", "cylindrical": "positive_rho"}


def json_integer(record: dict[str, object], field: str, minimum: int = 0) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"JSON field {field} is not an integer >= {minimum}")
    return value


def json_number(record: dict[str, object], field: str,
                minimum: float | None = None) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"JSON field {field} is not a finite number")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise RuntimeError(f"JSON field {field} is outside its finite range")
    return number


def validate_result_numbers(result: dict[str, object]) -> None:
    for field in ("spatial_order", "nghost", "nx1", "nx2", "nx3", "mpi_ranks",
                  "initial_cycle", "pgen_final_cycle", "owned_cells",
                  "operator_count", "diagnostic_axis_operator_count",
                  "diagnostic_axis_nonfinite", "nonfinite_count"):
        json_integer(result, field)
    for field in ("initial_time", "pgen_final_time", "noise_amplitude",
                  "maximum_error", "maximum_noise_delta", "noise_delta_bound",
                  "maximum_rotation_residual", "rotation_residual_bound",
                  "diagnostic_axis_linf", "diagnostic_axis_tolerance"):
        json_number(result, field, 0.0)


def validate_certified_domain(domain: tuple[float, float, float, float],
                              resolutions_by_order: dict[int, tuple[int, ...]],
                              qualification: bool) -> dict[str, object]:
    if (len(domain) != 4 or any(not math.isfinite(value) for value in domain) or
            not (domain[0] < domain[1] and domain[2] < domain[3]) or
            abs(domain[0] + domain[1]) > 32 * sys.float_info.epsilon *
            max(1.0, abs(domain[0]), abs(domain[1]))):
        raise RuntimeError("domain must be finite, ordered, and signed-rho symmetric")
    if qualification and tuple(domain) != QUALIFICATION_DOMAIN:
        raise RuntimeError("qualification domain is frozen to [-2,2]x[-2,2]")
    reaches = []
    for order, resolutions in resolutions_by_order.items():
        nghost = order // 2 + 1
        for resolution in resolutions:
            if resolution <= 0:
                raise RuntimeError("resolution pool contains a nonpositive entry")
            hx = (domain[1] - domain[0]) / resolution
            hz = (domain[3] - domain[2]) / resolution
            rho_reach = max(abs(domain[0] - (nghost - 0.5) * hx),
                            abs(domain[1] + (nghost - 0.5) * hx))
            z_reach = max(abs(domain[2] - (nghost - 0.5) * hz),
                          abs(domain[3] + (nghost - 0.5) * hz))
            if rho_reach >= CERTIFIED_COORDINATE_LIMIT or \
               z_reach >= CERTIFIED_COORDINATE_LIMIT:
                raise RuntimeError("active plus maximum ghost reach leaves the certified "
                                   "|rho|,|z|<3 manufactured-field envelope")
            reaches.append({"order": order, "resolution": resolution,
                            "rho_reach": rho_reach, "z_reach": z_reach})
    return {"domain": list(domain), "strict_coordinate_limit":
            CERTIFIED_COORDINATE_LIMIT,
            "maximum_rho_reach": max(item["rho_reach"] for item in reaches),
            "maximum_z_reach": max(item["z_reach"] for item in reaches)}


def up(value: float) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError("coefficient floor arithmetic became invalid")
    return math.nextafter(value, math.inf) if value > 0.0 else 0.0


def up_add(first: float, second: float) -> float:
    return up(first + second)


def up_mul(first: float, second: float) -> float:
    return up(first * second)


def up_div(first: float, second: float) -> float:
    if second <= 0.0:
        raise RuntimeError("coefficient floor division requires a positive denominator")
    return up(first / second)


def gamma(operation_count: int, epsilon: float) -> float:
    product = up_mul(float(operation_count), epsilon)
    if product >= 1.0:
        raise RuntimeError("coefficient floor gamma is not representable")
    denominator = math.nextafter(1.0 - product, -math.inf)
    return up_div(product, denominator)


def sum_up(values: list[float]) -> float:
    result = 0.0
    for value in values:
        result = up_add(result, value)
    return result


def policy_number(record: dict[str, object]) -> float:
    value = float.fromhex(str(record["hex"]))
    if value < float(Fraction(str(record["rational"]))):
        raise RuntimeError("roundoff policy constant was not rounded upward")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_strict(path: Path) -> object:
    def reject_constant(token: str) -> None:
        raise ValueError(f"nonfinite JSON token {token}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def finite_value(row: dict[str, str], field: str) -> float:
    if field not in row or row[field] == "":
        raise RuntimeError(f"missing applicable numeric field {field}")
    value = float(row[field])
    if not math.isfinite(value):
        raise RuntimeError(f"nonfinite numeric field {field}")
    return value


def integer_value(row: dict[str, str], field: str, minimum: int = 0) -> int:
    if not re.fullmatch(r"[+-]?\d+", row.get(field, "")):
        raise RuntimeError(f"malformed integer field {field}")
    value = int(row[field])
    if value < minimum:
        raise RuntimeError(f"out-of-range integer field {field}")
    return value


def boolean_value(row: dict[str, str], field: str) -> bool:
    if row.get(field) not in {"true", "false"}:
        raise RuntimeError(f"malformed applicability field {field}")
    return row[field] == "true"


def validate_norm_row(row: dict[str, str]) -> None:
    if set(row) != NORM_FIELDS or not row.get("operator") or not row.get("mask"):
        raise RuntimeError("norm CSV schema differs from the frozen exact field set")
    count = integer_value(row, "count", 1)
    if integer_value(row, "nonfinite") != 0:
        raise RuntimeError("norm CSV reports a nonfinite computation")
    cyl_count = integer_value(row, "cyl_count")
    if cyl_count > count:
        raise RuntimeError("cylindrical count exceeds uniform count")
    for field in ("l1", "l2", "linfinity", "shared_l1",
                  "shared_l2", "shared_linfinity", "shared_delta_l1",
                  "shared_delta_l2", "shared_delta_linfinity", "independent_l1",
                  "independent_l2", "independent_linfinity", "independent_delta_l1",
                  "independent_delta_l2", "independent_delta_linfinity",
                  "rotation_linfinity"):
        finite_value(row, field)
    cylindrical = boolean_value(row, "cylindrical_applicable")
    if cylindrical != (cyl_count > 0):
        raise RuntimeError("cylindrical applicability/count disagreement")
    for field in ("cyl_l1", "cyl_l2", "cyl_linfinity"):
        if cylindrical:
            finite_value(row, field)
        elif row.get(field) != "":
            raise RuntimeError("inapplicable cylindrical norm must be blank")
    radius = boolean_value(row, "radius_applicable")
    expected_radius = row["mask"].startswith("fixed_rho_")
    if radius != expected_radius:
        raise RuntimeError("radius applicability/mask disagreement")
    for field in ("target_abs_rho", "actual_abs_rho"):
        if radius:
            finite_value(row, field)
        elif row.get(field) != "":
            raise RuntimeError("inapplicable radius metadata must be blank")
    if not re.fullmatch(r"[0-9a-f]+", row.get("mask_xor", "")):
        raise RuntimeError("malformed mask_xor")


def expected_probe_metadata(mask: str) -> tuple[str, str, bool]:
    if mask == "diagnostic_axis":
        return "axis", "diagnostic_axis", True
    side = "negative" if mask.endswith("negative") or "negative_" in mask \
        else "positive"
    if mask.startswith("fitted_layer_"):
        return side, "fitted", False
    if mask.startswith("raw_transition_"):
        return side, "raw_transition", False
    if mask.startswith("fixed_rho_"):
        return side, "fixed_radius", True
    if mask.startswith("regular_"):
        return side, "regular", True
    raise RuntimeError(f"unknown probe mask {mask}")


def validate_probe_row(row: dict[str, str]) -> None:
    if set(row) != PROBE_FIELDS or not row.get("operator"):
        raise RuntimeError("probe CSV schema differs from the frozen exact field set")
    side, classification, target_applicable = expected_probe_metadata(row["mask"])
    if row["side"] != side or row["classification"] != classification:
        raise RuntimeError("probe side/classification disagrees with mask")
    integer_value(row, "layer_index")
    global_id = integer_value(row, "global_cell_id", -1)
    if (classification == "diagnostic_axis") != (global_id == -1):
        raise RuntimeError("probe global-cell applicability disagreement")
    if boolean_value(row, "target_rho_applicable") != target_applicable:
        raise RuntimeError("probe target-rho applicability disagreement")
    if target_applicable:
        target = finite_value(row, "target_rho")
        expected_target = (0.0 if classification == "diagnostic_axis" else
                           (-1.0 if side == "negative" else 1.0))
        if classification == "fixed_radius":
            expected_target *= 0.5
        if target != expected_target:
            raise RuntimeError("probe target_rho differs from its semantic mask")
    elif row.get("target_rho") != "":
        raise RuntimeError("inapplicable target_rho must be blank")
    for field in ("actual_rho", "target_z", "actual_z", "raw_error"):
        finite_value(row, field)


def validate_probe_geometry(row: dict[str, str], order: int, resolution: int,
                            domain: tuple[float, float, float, float]) -> None:
    validate_probe_row(row)
    classification = row["classification"]
    layer = integer_value(row, "layer_index")
    if classification == "diagnostic_axis":
        if (layer != 0 or finite_value(row, "actual_rho") != 0.0 or
                finite_value(row, "target_z") != 0.0 or
                finite_value(row, "actual_z") != 0.0):
            raise RuntimeError("diagnostic-axis probe is not at the true axis/center")
        return
    h = (domain[1] - domain[0]) / resolution
    hz = (domain[3] - domain[2]) / resolution
    actual_rho = finite_value(row, "actual_rho")
    actual_z = finite_value(row, "actual_z")
    expected_layer = math.floor(abs(actual_rho / h))
    if (layer != expected_layer or (row["side"] == "positive") != (actual_rho > 0.0)):
        raise RuntimeError("probe layer/side differs from physical geometry")
    expected_actual_z = domain[2] + (resolution // 2 + 0.5) * hz
    if finite_value(row, "target_z") != 0.0 or actual_z != expected_actual_z:
        raise RuntimeError("probe z target/actual differs from cell-center geometry")
    nghost = order // 2 + 1
    if classification == "fitted":
        match = re.fullmatch(r"fitted_layer_(\d+)_(negative|positive)", row["mask"])
        if match is None or int(match.group(1)) != layer or layer >= nghost:
            raise RuntimeError("fitted probe mask/layer differs from provider geometry")
    elif classification == "raw_transition":
        if layer != nghost:
            raise RuntimeError("raw-transition probe is not at NGHOST")
    elif classification == "fixed_radius":
        if layer != math.floor(0.5 / h):
            raise RuntimeError("fixed-radius probe layer differs from rho=0.5 target")
    elif classification == "regular":
        if layer != math.floor(1.0 / h):
            raise RuntimeError("regular probe layer differs from rho=1 target")
    else:
        raise RuntimeError("unknown probe geometry classification")
    expected_rho = (-1.0 if row["side"] == "negative" else 1.0) * (layer + 0.5) * h
    if actual_rho != expected_rho:
        raise RuntimeError("probe actual_rho differs from selected cell center")


def expected_masks(order: int) -> list[str]:
    nghost = order // 2 + 1
    masks = ["full_signed_plane", "regular_negative", "regular_positive"]
    masks += [f"fitted_layer_{layer}_{side}" for layer in range(nghost)
              for side in ("negative", "positive")]
    masks += ["raw_transition_negative", "raw_transition_positive",
              "fixed_rho_negative_0.5", "fixed_rho_positive_0.5"]
    return masks


def probe_series_identity(row: dict[str, str]) -> str:
    fields = [row["operator"], row["mask"], row["side"]]
    if row["classification"] in {"fitted", "raw_transition"}:
        fields.append(row["layer_index"])
    return "|".join(fields)


def validate_case_inventory(order: int, operator_names: list[str],
                            rows: list[dict[str, str]],
                            probes: list[dict[str, str]]) -> None:
    masks = expected_masks(order)
    expected_norm = {(operator, mask) for operator in operator_names for mask in masks}
    actual_norm = [(row["operator"], row["mask"]) for row in rows]
    if len(actual_norm) != len(set(actual_norm)) or set(actual_norm) != expected_norm:
        raise RuntimeError("case norm rows differ from the exact operator/mask inventory")
    nonaxis_masks = [mask for mask in masks if mask != "full_signed_plane"]
    expected_nonaxis = {(operator, mask) for operator in operator_names
                        for mask in nonaxis_masks}
    actual_nonaxis = [(row["operator"], row["mask"]) for row in probes
                      if row["mask"] != "diagnostic_axis"]
    if (len(actual_nonaxis) != len(set(actual_nonaxis)) or
            set(actual_nonaxis) != expected_nonaxis):
        raise RuntimeError("case probes differ from the exact non-axis inventory")
    axis = [row["operator"] for row in probes if row["mask"] == "diagnostic_axis"]
    if axis != operator_names[:161]:
        raise RuntimeError("case probes differ from the ordered 161-axis inventory")


def validate_rank_binding(binding: object) -> None:
    fields = {"rank", "local_rank", "hostname", "cuda_visible_devices",
              "visible_device_token", "selected_uuid", "gpu_name",
              "binding_verified"}
    if (not isinstance(binding, dict) or set(binding) != fields or
            not isinstance(binding["rank"], int) or binding["rank"] < 0 or
            not isinstance(binding["local_rank"], int) or binding["local_rank"] < 0 or
            not isinstance(binding["hostname"], str) or not binding["hostname"] or
            not isinstance(binding["binding_verified"], bool)):
        raise RuntimeError("rank binding differs from the exact evidence schema")
    for field in ("cuda_visible_devices", "visible_device_token",
                  "selected_uuid", "gpu_name"):
        if binding[field] is not None and not isinstance(binding[field], str):
            raise RuntimeError(f"rank binding field {field} is neither string nor null")


def write_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict[str, object]],
                     delimiter: str = ",") -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter=delimiter,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def replace_parameter(text: str, block: str, key: str, value: object) -> str:
    pattern = re.compile(rf"(?ms)(^<{re.escape(block)}>\s*$)(.*?)(?=^<|\Z)")
    match = pattern.search(text)
    if not match:
        raise RuntimeError(f"input lacks <{block}>")
    body = match.group(2)
    key_pattern = re.compile(rf"(?m)^(\s*{re.escape(key)}\s*=\s*).*$")
    if key_pattern.search(body):
        body = key_pattern.sub(rf"\g<1>{value}", body)
    else:
        body += f"{key} = {value}\n"
    return text[:match.start(2)] + body + text[match.end(2):]


def render_input(base: Path, order: int, resolution: int, phase: int,
                 basename: str, domain: tuple[float, float, float, float]) -> str:
    text = base.read_text(encoding="utf-8")
    nghost = order // 2 + 1
    for block, key, value in (
        ("job", "basename", basename), ("mesh", "nghost", nghost),
        ("mesh", "nx1", resolution), ("mesh", "nx2", resolution),
        ("z4c", "spatial_order", order), ("problem", "noise_phase", phase),
        ("problem", "output_directory", "."),
        ("mesh", "x1min", domain[0]), ("mesh", "x1max", domain[1]),
        ("mesh", "x2min", domain[2]), ("mesh", "x2max", domain[3]),
    ):
        text = replace_parameter(text, block, key, value)
    return text


def rate_policy(order: int, mask: str, lane: str, norm: str,
                probe_classification: str | None = None) -> tuple[float, float]:
    expected = float(order)
    margin = 0.15 if lane == "clean" else 0.5
    if probe_classification == "raw_transition" or mask.startswith("raw_transition"):
        expected = float(order - 1)
        margin = 0.25 if lane == "clean" else 0.5
    elif probe_classification == "fitted" or mask.startswith("fitted_layer_"):
        margin = 0.25 if lane == "clean" else 0.5
    elif mask == "full_signed_plane":
        if norm in ("l1", "cyl_l1", "cyl_l2"):
            expected = float(order)
        elif norm == "l2":
            expected = order - 0.5
        else:
            expected = float(order - 1)
        margin = 0.25 if lane == "clean" else 0.5
    return expected, margin


def linear_block_value(kind: str, components: list[str], scale: float,
                       values: dict[str, float], coefficients: dict[str, float],
                       h: float, hz: float, radius: float,
                       direction: int | None) -> float:
    component = sum_up([values[name] for name in components])
    c1, c2, cm = (coefficients[name] for name in ("C1", "C2", "CM"))
    fit = coefficients["fit_safety"]
    if kind in {"dx", "div_dx"}:
        value = up_div(up_mul(c1, component), h)
    elif kind in {"dz", "div_dz"}:
        value = up_div(up_mul(c1, component), hz)
    elif kind == "dxx":
        value = up_div(up_div(up_mul(c2, component), h), h)
    elif kind == "dzz":
        value = up_div(up_div(up_mul(c2, component), hz), hz)
    elif kind == "dxz":
        value = up_div(up_div(up_mul(cm, component), h), hz)
    elif kind in {"value_over_r", "div_value_over_r"}:
        value = up_div(component, radius)
    elif kind == "value_over_r2":
        value = up_div(up_div(component, radius), radius)
    elif kind == "dx_over_r":
        value = up_div(up_div(up_mul(c1, component), h), radius)
    elif kind == "dz_over_r":
        value = up_div(up_div(up_mul(c1, component), hz), radius)
    elif kind == "active_over_r":
        spacing = h if direction == 0 else hz
        value = up_div(up_div(up_mul(c1, component), spacing), radius)
    elif kind == "even_derivative":
        value = up_div(up_div(up_mul(up_mul(fit, coefficients["CE"]), component), h), h)
    elif kind in {"odd_value", "div_odd_value"}:
        value = up_div(up_mul(up_mul(fit, 2.0), component), h)
    elif kind == "rho_odd_derivative":
        numerator = up_mul(up_mul(fit, coefficients["CO"]), component)
        value = up_mul(radius, up_div(up_div(up_div(numerator, h), h), h))
    elif kind == "odd_active_value":
        spacing = h if direction == 0 else hz
        derivative = up_div(up_mul(c1, component), spacing)
        value = up_div(up_mul(up_mul(fit, 2.0), derivative), h)
    elif kind == "quad_value":
        value = up_div(up_div(up_mul(up_mul(fit, 4.0), component), h), h)
    elif kind == "rho_quad_value":
        quad = up_div(up_div(up_mul(up_mul(fit, 4.0), component), h), h)
        value = up_mul(radius, quad)
    elif kind == "rho2_quad_derivative":
        derivative = up_mul(up_mul(fit, coefficients["CQ"]), component)
        for _ in range(4):
            derivative = up_div(derivative, h)
        value = up_mul(up_mul(radius, radius), derivative)
    elif kind == "rho_quad_active_value":
        spacing = h if direction == 0 else hz
        derivative = up_div(up_mul(c1, component), spacing)
        quad = up_div(up_div(up_mul(up_mul(fit, 4.0), derivative), h), h)
        value = up_mul(radius, quad)
    elif kind == "ko":
        spacing = h if direction == 0 else hz
        value = up_div(up_mul(coefficients["CKO"], component), spacing)
    else:
        raise RuntimeError(f"unknown linear coefficient-floor block {kind}")
    return up_mul(scale, value)


def block_bounds(block_record: dict[str, object], maxima: dict[str, float],
                 errors: dict[str, float], coefficients: dict[str, float],
                 h: float, hz: float, radius: float) -> tuple[float, float, int]:
    kind = str(block_record["kind"])
    components = [str(value) for value in block_record["components"]]
    scale = float(Fraction(str(block_record["scale"])))
    direction = block_record.get("direction")
    direction_value = None if direction is None else int(direction)
    if kind == "up":
        velocity, field = components
        spacing = h if direction_value == 0 else hz
        derivative_b = up_div(up_mul(coefficients["CU"], maxima[field]), spacing)
        derivative_e = up_div(up_mul(coefficients["CU"], errors[field]), spacing)
        bound = up_mul(scale, up_mul(maxima[velocity], derivative_b))
        propagated = up_mul(scale, sum_up([
            up_mul(errors[velocity], derivative_b),
            up_mul(maxima[velocity], derivative_e),
            up_mul(errors[velocity], derivative_e)]))
        return bound, propagated, 384
    product = kind.startswith("product_")
    base_kind = kind[len("product_"):] if product else kind
    if product:
        velocity, *field_components = components
        field_b = linear_block_value(base_kind, field_components, scale, maxima,
                                     coefficients, h, hz, radius, direction_value)
        field_e = linear_block_value(base_kind, field_components, scale, errors,
                                     coefficients, h, hz, radius, direction_value)
        bound = up_mul(maxima[velocity], field_b)
        propagated = sum_up([up_mul(errors[velocity], field_b),
                             up_mul(maxima[velocity], field_e),
                             up_mul(errors[velocity], field_e)])
        return bound, propagated, 384
    bound = linear_block_value(base_kind, components, scale, maxima, coefficients,
                               h, hz, radius, direction_value)
    propagated = linear_block_value(base_kind, components, scale, errors, coefficients,
                                    h, hz, radius, direction_value)
    if base_kind.startswith("div_"):
        operations = 192
    elif ("over_r" in base_kind or base_kind.startswith("value_over") or
          base_kind.startswith("active_over")):
        operations = 192
    elif base_kind in {"even_derivative", "odd_value", "rho_odd_derivative",
                       "odd_active_value", "quad_value", "rho_quad_value",
                       "rho2_quad_derivative", "rho_quad_active_value"}:
        operations = 384
    else:
        operations = 128
    return bound, propagated, operations


def cell_clean_floor(item: dict[str, object], policy: dict[str, object],
                     order: int, h: float, hz: float, radius: float,
                     fitted: bool) -> dict[str, object]:
    epsilon = float.fromhex(str(policy["binary64_epsilon_hex"]))
    maxima = {name: policy_number(value) for name, value in
              dict(policy["field_maxima"]).items()}
    field_gamma = gamma(int(dict(policy["operation_caps"])["field_fill"]), epsilon)
    errors = {name: up_mul(field_gamma, value) for name, value in maxima.items()}
    coefficient_row = dict(dict(policy["coefficients"])[str(order // 2 + 1)])
    coefficients = {name: policy_number(value) for name, value in
                    coefficient_row.items()}
    coefficients["fit_safety"] = float.fromhex(str(policy["fit_safety_hex"]))
    family = dict(item["roundoff_family"])
    blocks = list(family["active"]) + list(family["fitted" if fitted else "raw"])
    propagated_terms = []
    arithmetic_terms = []
    magnitudes = []
    for block_record in blocks:
        bound, propagated, operations = block_bounds(
            dict(block_record), maxima, errors, coefficients, h, hz, radius)
        magnitudes.append(bound)
        propagated_terms.append(propagated)
        arithmetic_terms.append(up_mul(gamma(operations, epsilon),
                                       up_add(bound, propagated)))
    propagated_sum = sum_up(propagated_terms)
    arithmetic_sum = sum_up(arithmetic_terms)
    if len(blocks) > 1:
        arithmetic_sum = up_add(
            arithmetic_sum, up_mul(gamma(len(blocks) - 1, epsilon),
                                   sum_up(magnitudes + propagated_terms +
                                          arithmetic_terms)))
    production_sum = up_add(propagated_sum, arithmetic_sum)
    oracle_bound = float.fromhex(str(item["oracle_bound_hex"]))
    oracle_roundoff = up_mul(gamma(256, epsilon), oracle_bound)
    final_magnitude = sum_up([sum_up(magnitudes), oracle_bound,
                              production_sum, oracle_roundoff])
    subtraction = up_mul(gamma(1, epsilon), final_magnitude)
    clean_floor = up_mul(float.fromhex(str(policy["global_slack_hex"])),
                         sum_up([production_sum, oracle_roundoff, subtraction]))
    return {"family_sha256": canonical_digest(family),
            "coefficient_row_sha256": canonical_digest(coefficient_row),
            "fit_fixture_sha256": policy["fit_fixture_sha256"],
            "branch": "fitted" if fitted else "raw",
            "h": h, "hz": hz, "abs_rho": radius,
            "block_count": len(blocks), "production_magnitude": sum_up(magnitudes),
            "propagated_input": propagated_sum,
            "production_roundoff": arithmetic_sum,
            "oracle_bound": oracle_bound, "oracle_roundoff": oracle_roundoff,
            "subtraction_roundoff": subtraction, "clean_floor": clean_floor}


def mask_radial_indices(mask: str, order: int, resolution: int,
                        domain: tuple[float, float, float, float]) -> \
        list[tuple[float, bool, bool]]:
    h = (domain[1] - domain[0]) / resolution
    nghost = order // 2 + 1
    fixed_layer = math.floor(0.5 / h)
    selected = []
    for index in range(resolution):
        radius = domain[0] + (index + 0.5) * h
        layer = math.floor(abs(radius / h))
        side = "positive" if radius > 0.0 else "negative"
        include = mask == "full_signed_plane"
        include |= mask == f"fitted_layer_{layer}_{side}" and layer < nghost
        include |= mask == f"raw_transition_{side}" and layer == nghost
        include |= mask == f"regular_{side}" and layer > nghost and abs(radius) >= 0.75
        include |= mask == f"fixed_rho_{side}_0.5" and layer == fixed_layer
        if include:
            selected.append((abs(radius), layer < nghost, radius > 0.0))
    if not selected:
        raise RuntimeError(f"coefficient-floor mask is empty: {mask}")
    return selected


def aggregate_clean_floor(item: dict[str, object], policy: dict[str, object],
                          order: int, resolution: int, mask: str, norm: str,
                          domain: tuple[float, float, float, float]) -> dict[str, object]:
    h = (domain[1] - domain[0]) / resolution
    hz = (domain[3] - domain[2]) / resolution
    radial = mask_radial_indices(mask, order, resolution, domain)
    cylindrical = norm.startswith("cyl_")
    if cylindrical:
        radial = [entry for entry in radial if entry[2]]
    cells = [cell_clean_floor(item, policy, order, h, hz, radius, fitted)
             for radius, fitted, _ in radial]
    floors = [float(cell["clean_floor"]) for cell in cells]
    weights = [radius if cylindrical else 1.0 for radius, _, _ in radial]
    denominator = sum_up(weights)
    if norm in {"l1", "cyl_l1"}:
        clean_floor = up_div(sum_up([up_mul(weight, value)
                                     for weight, value in zip(weights, floors)]),
                             denominator)
    elif norm in {"l2", "cyl_l2"}:
        mean_square = up_div(sum_up([up_mul(weight, up_mul(value, value))
                                     for weight, value in zip(weights, floors)]),
                             denominator)
        clean_floor = up(math.sqrt(mean_square))
    elif norm in {"linfinity", "cyl_linfinity", "raw_error"}:
        clean_floor = max(floors)
    else:
        raise RuntimeError(f"unknown coefficient-floor norm {norm}")
    component_names = ("production_magnitude", "propagated_input",
                       "production_roundoff", "oracle_roundoff",
                       "subtraction_roundoff")
    components = {name: max(float(cell[name]) for cell in cells)
                  for name in component_names}
    return {"family_sha256": cells[0]["family_sha256"], "norm": norm,
            "coefficient_row_sha256": cells[0]["coefficient_row_sha256"],
            "fit_fixture_sha256": cells[0]["fit_fixture_sha256"],
            "mask_weight": "positive_rho" if cylindrical else "uniform",
            "radial_sample_count": len(cells),
            "fitted_count": sum(cell["branch"] == "fitted" for cell in cells),
            "raw_count": sum(cell["branch"] == "raw" for cell in cells),
            **components, "clean_floor": clean_floor}


def probe_clean_floor(item: dict[str, object], policy: dict[str, object], order: int,
                      resolution: int, row: dict[str, str],
                      domain: tuple[float, float, float, float]) -> dict[str, object]:
    h = (domain[1] - domain[0]) / resolution
    hz = (domain[3] - domain[2]) / resolution
    radius = abs(finite_value(row, "actual_rho"))
    layer = math.floor(radius / h)
    fitted = layer < order // 2 + 1
    record = cell_clean_floor(item, policy, order, h, hz, radius, fitted)
    return {**record, "norm": "raw_error", "mask_weight": "point",
            "layer_index": layer}


def evaluate_rate_samples(values: list[dict[str, float]], expected: float,
                          margin: float, lane: str) -> dict[str, object]:
    rates: list[float | None] = []
    rate_status = []
    prefix_rates = []
    nonmonotone_intervals = []
    for sample in values:
        if (not math.isfinite(sample["error"]) or sample["error"] <= 0.0 or
                not math.isfinite(sample["clean_floor"]) or sample["clean_floor"] <= 0.0):
            raise RuntimeError("rate sample/error floor must be finite and positive")
        sample["applied_floor"] = sample["clean_floor"]
        if lane != "clean":
            if not math.isfinite(sample["direct_delta"]) or sample["direct_delta"] < 0.0:
                raise RuntimeError("noisy rate sample requires a finite direct delta")
            sample["applied_floor"] = max(sample["clean_floor"],
                                           up_mul(8.0, sample["direct_delta"]))
        sample["floor_application"] = {
            "clean_floor": sample["clean_floor"],
            "direct_delta": sample["direct_delta"],
            "noisy_delta_floor": 0.0 if lane == "clean" else
            up_mul(8.0, sample["direct_delta"]),
            "applied_floor": sample["applied_floor"]}
    saturated_at = next((index for index, sample in enumerate(values)
                         if sample["error"] <= sample["applied_floor"]), len(values))
    for index in range(1, saturated_at):
        if values[index]["error"] > values[index - 1]["error"]:
            nonmonotone_intervals.append(index - 1)
    for index, (coarse, fine) in enumerate(zip(values, values[1:])):
        if index + 1 >= saturated_at:
            rates.append(None)
            rate_status.append("excluded_saturated")
        elif index in nonmonotone_intervals:
            rates.append(None)
            rate_status.append("excluded_pre_floor_nonmonotone")
        else:
            rate = math.log(coarse["error"] / fine["error"]) / \
                math.log(fine["resolution"] / coarse["resolution"])
            rates.append(rate)
            rate_status.append("included_rate")
            prefix_rates.append(rate)
    unsaturated_prefix = len(prefix_rates)
    passed = (not nonmonotone_intervals and unsaturated_prefix >= 2 and
              min(float(value) for value in prefix_rates[-2:]) >= expected - margin)
    if nonmonotone_intervals:
        reason = "pre_floor_nonmonotone"
    elif unsaturated_prefix < 2:
        reason = "saturated_insufficient" if saturated_at < len(values) else \
            "insufficient_ratios"
    elif not passed:
        reason = "rate_miss"
    else:
        reason = "pass"
    interval_reasons = []
    for index, status in enumerate(rate_status):
        interval_reasons.append({
            "coarse_resolution": values[index]["resolution"],
            "fine_resolution": values[index + 1]["resolution"],
            "status": status,
            "reason": {"included_rate": "finite_unsaturated_interval",
                       "excluded_saturated": "fine_endpoint_at_or_below_floor",
                       "excluded_pre_floor_nonmonotone":
                       "fine_error_exceeds_coarse_error"}[status]})
    return {"rates": rates, "unsaturated_prefix_ratios": unsaturated_prefix,
            "rate_status": rate_status, "saturation_absorbing": saturated_at < len(values),
            "saturated_at_resolution": None if saturated_at == len(values) else
            values[saturated_at]["resolution"],
            "interval_reasons": interval_reasons,
            "outcome_reason": reason, "passed": passed}


def output_forecast(resolutions_by_order: dict[int, tuple[int, ...]], phases: list[int],
                    ranks: int, rank_comparison: bool) -> dict[str, int]:
    norm_rows = 0
    probe_rows = 0
    case_count = 0
    for order, resolutions in resolutions_by_order.items():
        nghost = order // 2 + 1
        cases = len(resolutions) * len(phases)
        case_count += cases
        norm_rows += cases * 171 * (7 + 2 * nghost)
        probe_rows += cases * (171 * (6 + 2 * nghost) + 161)
    case_products = (norm_rows * 2048 + probe_rows * 1024 +
                     case_count * (5 + ranks) * 1024 * 1024)
    aggregate_products = norm_rows * 8192 + probe_rows * 4096 + 128 * 1024 * 1024
    comparison_products = 512 * 1024 * 1024 if rank_comparison else 0
    total = case_products + aggregate_products + comparison_products
    if min(case_count, norm_rows, probe_rows, total) <= 0:
        raise RuntimeError("invalid checked output forecast inventory")
    return {"case_count": case_count, "norm_rows": norm_rows,
            "probe_rows": probe_rows, "case_products": case_products,
            "aggregate_products": aggregate_products,
            "rank_comparison_products": comparison_products,
            "estimated_output_bytes_upper_bound": total}


def git_value(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()


def launcher_command(launcher: str, ranks: int) -> list[str]:
    words = shlex.split(launcher)
    executable = Path(words[0]).name
    if executable in {"mpirun", "mpiexec"}:
        return words + ["-np", str(ranks)]
    if executable == "srun":
        return words + ["--ntasks", str(ranks)]
    return words + [str(ranks)]


def execution_environment() -> dict[str, str]:
    return {name: os.environ[name] for name in CASE_ENVIRONMENT_KEYS
            if name in os.environ}


def canonical_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def validate_semantic_tables(powers: dict[str, tuple[int, int, int]],
                             weights: dict[str, str]) -> None:
    if powers != BLOCK_POWERS:
        raise RuntimeError("h/z/rho power table differs from the frozen family map")
    if weights != {"uniform": "one", "cylindrical": "positive_rho"}:
        raise RuntimeError("mask weight policy differs from the frozen contract")


def validate_roundoff_inventory(inventory: dict[str, object]) -> None:
    policy = inventory.get("roundoff_policy")
    series = inventory.get("series")
    if not isinstance(policy, dict) or not isinstance(series, list) or len(series) != 171:
        raise RuntimeError("coefficient floor inventory is incomplete")
    rationals = {order: {name: value.get("rational")
                         for name, value in dict(row).items()}
                 for order, row in dict(policy.get("coefficients", {})).items()}
    if rationals != EXPECTED_COEFFICIENT_RATIONALS:
        raise RuntimeError("coefficient table differs from the frozen finite contract")
    for order, row in dict(policy["coefficients"]).items():
        for name, value in dict(row).items():
            rational = Fraction(str(value["rational"]))
            nearest = float(rational)
            expected_hex = (math.nextafter(nearest, math.inf)
                            if Fraction.from_float(nearest) < rational else nearest).hex()
            if value.get("hex") != expected_hex:
                raise RuntimeError(f"coefficient {order}/{name} is not exact upward binary64")
    families = [[item.get("name"), item.get("roundoff_family")] for item in series]
    if canonical_digest(families) != FAMILY_MAPPING_SHA256:
        raise RuntimeError("171-row roundoff-family mapping differs from the frozen map")
    kinds = {str(block["kind"]).removeprefix("product_")
             for item in series for branch in ("active", "fitted", "raw")
             for block in dict(item["roundoff_family"])[branch]}
    if not kinds <= set(BLOCK_POWERS):
        raise RuntimeError("roundoff family uses an unclassified h/rho power")
    validate_semantic_tables(BLOCK_POWERS, MASK_WEIGHT_POLICY)


def expected_case_files(ranks: int) -> set[str]:
    return CASE_FIXED_FILES | {f"rank_binding_{rank:04d}.json"
                              for rank in range(ranks)}


def verify_no_evolution(case_id: str, stdout_text: str,
                        result: dict[str, object]) -> None:
    sentinels = (result.get("initial_cycle"), result.get("pgen_final_cycle"),
                 result.get("initial_time"), result.get("pgen_final_time"))
    if sentinels != (0, 0, 0, 0):
        raise RuntimeError(f"case {case_id} pgen cycle/time sentinel changed")
    states = re.findall(r"time=([^\s]+)\s+cycle=([+-]?\d+)", stdout_text)
    if any(float(time_value) != 0.0 or int(cycle) != 0
           for time_value, cycle in states):
        raise RuntimeError(f"case {case_id} executed a physical evolution step")


def self_test_no_evolution_parser() -> None:
    sentinels = {"initial_cycle": 0, "pgen_final_cycle": 0,
                 "initial_time": 0, "pgen_final_time": 0}
    observed_static_stdout = (
        "Cartoon derivative MMS passed: order=2 cells=1024 max_error=0 "
        "axis_error=0\n")
    verify_no_evolution("self-test-static", observed_static_stdout, sentinels)
    try:
        verify_no_evolution("self-test-nonzero",
                            observed_static_stdout + "time=0.125 cycle=1\n", sentinels)
    except RuntimeError:
        return
    raise RuntimeError("no-evolution parser accepted a nonzero stdout record")


def expect_runtime_error(action, label: str) -> None:
    try:
        action()
    except (RuntimeError, ValueError):
        return
    raise RuntimeError(f"CPU-audit self-test accepted {label}")


def self_test_cpu_audit_policy() -> None:
    """Exercise the audit plumbing without launching Athena or creating a campaign."""
    if rate_policy(6, "fitted_layer_0_negative", "clean", "l1") != (6.0, 0.25):
        raise RuntimeError("fitted clean margin is not p-0.25")
    if rate_policy(4, "raw_transition_positive", "clean", "raw_error",
                   "raw_transition") != (3.0, 0.25):
        raise RuntimeError("raw-probe policy is not p-1 with the clean margin")
    if rate_policy(4, "full_signed_plane", "clean", "cyl_l2") != (4.0, 0.25):
        raise RuntimeError("full-plane cylindrical L2 policy is not p")
    if rate_policy(4, "full_signed_plane", "clean", "cyl_linfinity") != (3.0, 0.25):
        raise RuntimeError("full-plane cylindrical Linfinity policy is not p-1")

    # Immutable examples are copied from convergence.json in CPU job 56586376.
    o2_miss = [
        {"resolution": 32, "error": 4.182343874808808e-4, "direct_delta": 0.0,
         "clean_floor": 1.0e-30},
        {"resolution": 64, "error": 1.6220836147908628e-4, "direct_delta": 0.0,
         "clean_floor": 1.0e-30},
        {"resolution": 128, "error": 5.573831257431492e-5, "direct_delta": 0.0,
         "clean_floor": 1.0e-30},
        {"resolution": 256, "error": 1.77800308574143e-5, "direct_delta": 0.0,
         "clean_floor": 1.0e-30},
    ]
    miss = evaluate_rate_samples(o2_miss, 2.0, 0.25, "clean")
    if miss["passed"] or miss["rate_status"] != ["included_rate"] * 3:
        raise RuntimeError("pre-floor order-2 miss was hidden or reclassified")
    o6_roundoff = [
        {"resolution": 32, "error": 6.0337018612239125e-9, "direct_delta": 0.0,
         "clean_floor": 1.0e-30},
        {"resolution": 64, "error": 4.7733002365378305e-11, "direct_delta": 0.0,
         "clean_floor": 1.0e-30},
        {"resolution": 128, "error": 3.7376593184852193e-13, "direct_delta": 0.0,
         "clean_floor": 1.0e-12},
        {"resolution": 256, "error": 5.558874451157167e-15, "direct_delta": 0.0,
         "clean_floor": 1.0e-12},
    ]
    saturated = evaluate_rate_samples(o6_roundoff, 6.0, 0.25, "clean")
    if (saturated["passed"] or saturated["rate_status"] !=
            ["included_rate", "excluded_saturated", "excluded_saturated"] or
            saturated["unsaturated_prefix_ratios"] != 1):
        raise RuntimeError("absorbing saturation invented or re-entered a rate")

    valid_norm = {field: "" for field in NORM_FIELDS}
    valid_norm.update({"operator": "scalar.first.0", "mask": "regular_negative",
                       "count": "16", "nonfinite": "0", "cyl_count": "0",
                       "cylindrical_applicable": "false",
                       "radius_applicable": "false", "mask_xor": "0"})
    for field in ("l1", "l2", "linfinity", "shared_l1", "shared_l2",
                  "shared_linfinity", "shared_delta_l1", "shared_delta_l2",
                  "shared_delta_linfinity", "independent_l1", "independent_l2",
                  "independent_linfinity", "independent_delta_l1",
                  "independent_delta_l2", "independent_delta_linfinity",
                  "rotation_linfinity"):
        valid_norm[field] = "0"
    validate_norm_row(valid_norm)
    legacy_norm = {key: value for key, value in valid_norm.items()
                   if key not in {"cylindrical_applicable", "radius_applicable"}}
    legacy_norm.update({"cyl_l1": "nan", "cyl_l2": "nan",
                        "cyl_linfinity": "nan", "target_abs_rho": "nan",
                        "actual_abs_rho": "nan"})
    normalized_norm, _ = normalize_legacy_norm_row(legacy_norm)
    if any(normalized_norm[field] != valid_norm[field] for field in
           ("l1", "l2", "linfinity", "shared_l1", "independent_l1")):
        raise RuntimeError("legacy norm normalization changed numerical evidence")
    for label, edit in (
        ("NaN numeric metadata", {"l1": "nan"}),
        ("infinite numeric metadata", {"l2": "inf"}),
        ("applicable blank cylindrical metadata",
         {"cyl_count": "1", "cylindrical_applicable": "true"}),
        ("inapplicable populated cylindrical metadata", {"cyl_l1": "0"}),
        ("radius applicability disagreement", {"radius_applicable": "true"}),
        ("missing schema field", {"mask_xor": None}),
    ):
        candidate = dict(valid_norm)
        for key, value in edit.items():
            if value is None:
                candidate.pop(key)
            else:
                candidate[key] = value
        expect_runtime_error(lambda row=candidate: validate_norm_row(row), label)

    valid_probe = {"operator": "scalar.first.0", "mask": "raw_transition_negative",
                   "side": "negative", "layer_index": "3",
                   "classification": "raw_transition",
                   "target_rho_applicable": "false", "target_rho": "",
                   "actual_rho": "-0.4375", "target_z": "0", "actual_z": "0.0625",
                   "global_cell_id": "7", "raw_error": "0"}
    validate_probe_row(valid_probe)
    validate_probe_geometry(valid_probe, 4, 32, QUALIFICATION_DOMAIN)
    legacy_probe = {key: value for key, value in valid_probe.items()
                    if key != "target_rho_applicable"}
    legacy_probe["target_rho"] = "nan"
    normalized_probe, _ = normalize_legacy_probe_row(legacy_probe)
    if normalized_probe["raw_error"] != valid_probe["raw_error"]:
        raise RuntimeError("legacy probe normalization changed raw error")
    for label, edit in (
        ("nonfinite probe error", {"raw_error": "-inf"}),
        ("inapplicable populated target", {"target_rho": "-0.1"}),
        ("probe applicability disagreement", {"target_rho_applicable": "true"}),
        ("probe classification disagreement", {"classification": "fitted"}),
    ):
        candidate = {**valid_probe, **edit}
        expect_runtime_error(lambda row=candidate: validate_probe_row(row), label)
    for label, edit in (
        ("fitted mask/layer mismatch", {"mask": "fitted_layer_1_negative",
                                        "classification": "fitted",
                                        "layer_index": "0", "actual_rho": "-0.0625"}),
        ("raw transition away from NGHOST", {"layer_index": "2",
                                             "actual_rho": "-0.3125"}),
        ("probe z mismatch", {"actual_z": "0"}),
    ):
        candidate = {**valid_probe, **edit}
        expect_runtime_error(
            lambda row=candidate: validate_probe_geometry(
                row, 4, 32, QUALIFICATION_DOMAIN), label)

    fixed_keys = {probe_series_identity({**valid_probe,
                                        "mask": "fixed_rho_negative_0.5",
                                        "classification": "fixed_radius",
                                        "layer_index": str(layer)})
                  for layer in (4, 8, 16, 32)}
    regular_keys = {probe_series_identity({**valid_probe,
                                          "mask": "regular_negative",
                                          "classification": "regular",
                                          "layer_index": str(layer)})
                    for layer in (8, 16, 32, 64)}
    fitted_keys = {probe_series_identity({**valid_probe,
                                         "mask": "fitted_layer_0_negative",
                                         "classification": "fitted",
                                         "layer_index": str(layer)})
                   for layer in (0, 1)}
    if len(fixed_keys) != 1 or len(regular_keys) != 1 or len(fitted_keys) != 2:
        raise RuntimeError("probe series identity uses a resolution-dependent target layer")

    inventory_operator = ["scalar.first.0"]
    inventory_rows = [{"operator": inventory_operator[0], "mask": mask}
                      for mask in expected_masks(2)]
    inventory_probes = [{"operator": inventory_operator[0], "mask": mask}
                        for mask in expected_masks(2) if mask != "full_signed_plane"]
    inventory_probes.append({"operator": inventory_operator[0],
                             "mask": "diagnostic_axis"})
    validate_case_inventory(2, inventory_operator, inventory_rows, inventory_probes)
    expect_runtime_error(
        lambda: validate_case_inventory(2, inventory_operator,
                                        inventory_rows[:-1], inventory_probes),
        "missing whole operator/mask series")

    with tempfile.TemporaryDirectory(prefix="cartoon-mms-audit-") as directory:
        for label, token in (("resume manifest", "NaN"),
                             ("resumed result", "NaN"),
                             ("rank binding", "Infinity"),
                             ("reference campaign", "-Infinity")):
            nonfinite = Path(directory) / (label.replace(" ", "_") + ".json")
            nonfinite.write_text('{"value": ' + token + '}\n', encoding="utf-8")
            expect_runtime_error(lambda path=nonfinite: load_json_strict(path),
                                 f"nonfinite {label} JSON")

    valid_result = {field: 0 for field in
                    ("spatial_order", "nghost", "nx1", "nx2", "nx3", "mpi_ranks",
                     "initial_cycle", "pgen_final_cycle", "owned_cells",
                     "operator_count", "diagnostic_axis_operator_count",
                     "diagnostic_axis_nonfinite", "nonfinite_count")}
    valid_result.update({field: 0.0 for field in
                         ("initial_time", "pgen_final_time", "noise_amplitude",
                          "maximum_error", "maximum_noise_delta", "noise_delta_bound",
                          "maximum_rotation_residual", "rotation_residual_bound",
                          "diagnostic_axis_linf", "diagnostic_axis_tolerance")})
    validate_result_numbers(valid_result)
    for field, bad in (("noise_delta_bound", "NaN"),
                       ("diagnostic_axis_linf", "1.0"),
                       ("maximum_error", math.inf), ("operator_count", 171.0)):
        candidate = {**valid_result, field: bad}
        expect_runtime_error(lambda value=candidate: validate_result_numbers(value),
                             f"typed finite result mutation {field}")

    forecast = output_forecast({order: DIAGNOSTIC_RESOLUTIONS for order in (2, 4, 6)},
                               list(range(8)), 4, False)
    if (forecast["case_count"] != 96 or
            forecast["estimated_output_bytes_upper_bound"] != 4_313_219_072 or
            forecast["estimated_output_bytes_upper_bound"] <=
            PRESERVED_JOB_56586376_BYTES or
            any(len(value) != 64 for value in
                (PRESERVED_JOB_56586376_CONVERGENCE_SHA256,
                 PRESERVED_JOB_56586376_EVIDENCE_SHA256,
                 PRESERVED_JOB_56586376_LOG_SHA256)) or
            PRESERVED_JOB_56586376_AUDIT_COUNTS != (63_880, 18_508, 4_512, 3_696)):
        raise RuntimeError("output forecast/audit anchor differs from frozen CPU evidence")

    root = Path(__file__).resolve().parents[3]
    load_search_manifest(root / "tst/inputs/z4c_cartoon_mms_search_manifest.json")
    inventory = load_json_strict(
        root / "tst/unit/z4c/z4c_cartoon_derivatives_series.json")
    if not isinstance(inventory, dict) or not isinstance(inventory.get("roundoff_policy"), dict):
        raise RuntimeError("coefficient-floor self-test lacks its generated policy")
    metadata = {item["name"]: item for item in inventory["series"]}
    policy = inventory["roundoff_policy"]
    validate_roundoff_inventory(inventory)
    domain = (-2.0, 2.0, -2.0, 2.0)

    # One compact table-driven mutation loop owns every frozen coefficient,
    # family row, h/rho exponent and mask-weight declaration.
    for order, row in EXPECTED_COEFFICIENT_RATIONALS.items():
        for coefficient in row:
            for field, replacement in (("rational", "0"), ("hex", "0x0.0p+0")):
                mutated = json.loads(json.dumps(inventory))
                mutated["roundoff_policy"]["coefficients"][order][coefficient][field] = \
                    replacement
                expect_runtime_error(
                    lambda value=mutated: validate_roundoff_inventory(value),
                    f"coefficient mutation {order}/{coefficient}/{field}")
    for index in range(171):
        mutated = json.loads(json.dumps(inventory))
        mutated["series"][index]["roundoff_family"]["source_branch"] += "_mutated"
        expect_runtime_error(lambda value=mutated: validate_roundoff_inventory(value),
                             f"family mapping mutation {index}")
    for kind, powers in BLOCK_POWERS.items():
        for component in range(3):
            changed = dict(BLOCK_POWERS)
            replacement = list(powers)
            replacement[component] += 1
            changed[kind] = tuple(replacement)
            expect_runtime_error(
                lambda value=changed: validate_semantic_tables(value, MASK_WEIGHT_POLICY),
                f"h/rho power mutation {kind}/{component}")
    for weight in MASK_WEIGHT_POLICY:
        changed = dict(MASK_WEIGHT_POLICY)
        changed[weight] += "_mutated"
        expect_runtime_error(
            lambda value=changed: validate_semantic_tables(BLOCK_POWERS, value),
            f"mask-weight mutation {weight}")
    validate_certified_domain(domain, {2: (32,), 4: (32,), 6: (32,)}, True)
    expect_runtime_error(
        lambda: validate_certified_domain((-2.9, 2.9, -2.0, 2.0),
                                          {6: (32,)}, False),
        "domain outside certified active-plus-ghost reach")

    advection = metadata["tensor.lower.0.1.advective"]
    advection_values = []
    for resolution, error in zip((256, 512, 1024),
                                 (7.2582313459523817e-7,
                                  2.4344003218823573e-7,
                                  6.8603908651154732e-8)):
        row = {field: "" for field in PROBE_FIELDS}
        row.update({"actual_rho": "-1", "classification": "regular"})
        advection_values.append({
            "resolution": resolution, "error": error, "direct_delta": 0.0,
            "clean_floor": probe_clean_floor(
                advection, policy, 2, resolution, row, domain)["clean_floor"]})
    advection_gate = evaluate_rate_samples(advection_values, 2.0, 0.15, "clean")
    if (advection_gate["passed"] or
            advection_gate["rate_status"] != ["included_rate", "included_rate"]):
        raise RuntimeError("order-2 regular advection miss was hidden by the floor")

    fitted_tensor = metadata["tensor.lower.0.2.second.0.2"]
    for order in (4, 6):
        for resolution, expected_branch in ((16, "fitted"), (32, "raw")):
            h = 4.0 / resolution
            layer = math.floor(0.5 / h)
            for side in (-1.0, 1.0):
                row = {field: "" for field in PROBE_FIELDS}
                row["actual_rho"] = str(side * (layer + 0.5) * h)
                row["classification"] = "fixed_radius"
                observed = probe_clean_floor(fitted_tensor, policy, order,
                                             resolution, row, domain)
                direct = cell_clean_floor(fitted_tensor, policy, order, h, h,
                                           abs(float(row["actual_rho"])),
                                           expected_branch == "fitted")
                if (observed["branch"] != expected_branch or
                        observed["clean_floor"] != direct["clean_floor"]):
                    raise RuntimeError("fixed probe floor selected semantic label, not geometry")
    fitted_values = []
    for resolution, error in zip((256, 512, 1024),
                                 (1.1391698318701828e-10,
                                  7.0795043411548894e-12,
                                  9.324645886704154e-12)):
        fitted_values.append({
            "resolution": resolution, "error": error, "direct_delta": 0.0,
            "clean_floor": aggregate_clean_floor(
                fitted_tensor, policy, 2, resolution,
                "fitted_layer_0_negative", "l1", domain)["clean_floor"]})
    fitted_gate = evaluate_rate_samples(fitted_values, 2.0, 0.25, "clean")
    if (fitted_gate["passed"] or
            fitted_gate["rate_status"] != ["excluded_saturated"] * 2):
        raise RuntimeError("order-2 fitted saturation re-entered after the floor")

    scalar_second = metadata["scalar.second.0.0"]
    scalar_values = []
    scalar_errors = (3.769623255979633e-7, 1.9004190833079248e-8,
                     3.6095301648934983e-10, 5.9150357934579329e-12)
    for resolution, error in zip((16, 32, 64, 128), scalar_errors):
        scalar_values.append({
            "resolution": resolution, "error": error, "direct_delta": 0.0,
            "clean_floor": aggregate_clean_floor(
                scalar_second, policy, 6, resolution,
                "fitted_layer_2_negative", "l1", domain)["clean_floor"]})
    scalar_gate = evaluate_rate_samples(scalar_values, 6.0, 0.25, "clean")
    if scalar_gate["passed"]:
        raise RuntimeError("order-6 coarse fitted miss became a false pass")
    shared_values = [dict(value) for value in scalar_values]
    shared_values[-1]["direct_delta"] = 2.3347311319430147e-11
    shared_gate = evaluate_rate_samples(shared_values, 6.0, 0.5, "shared")
    if (shared_values[-1]["applied_floor"] < 1.8677849055544118e-10 or
            shared_gate["rate_status"][-1] != "excluded_saturated"):
        raise RuntimeError("shared-noise floor does not absorb the frozen N128 anchor")

    high_floor = [{"resolution": 32, "error": 1.0, "direct_delta": 0.0,
                   "clean_floor": 2.0},
                  {"resolution": 64, "error": 0.5, "direct_delta": 0.0,
                   "clean_floor": 2.0},
                  {"resolution": 128, "error": 0.25, "direct_delta": 0.0,
                   "clean_floor": 2.0}]
    high_outcome = evaluate_rate_samples(high_floor, 2.0, 0.25, "clean")
    if high_outcome["passed"] or high_outcome["outcome_reason"] != \
       "saturated_insufficient":
        raise RuntimeError("a high floor produced a passing ratio")
    increasing = [{"resolution": 32, "error": 1.0, "direct_delta": 0.0,
                   "clean_floor": 1.0e-12},
                  {"resolution": 64, "error": 2.0, "direct_delta": 0.0,
                   "clean_floor": 1.0e-12},
                  {"resolution": 128, "error": 1.0, "direct_delta": 0.0,
                   "clean_floor": 1.0e-12}]
    nonmonotone = evaluate_rate_samples(increasing, 2.0, 0.25, "clean")
    if nonmonotone["passed"] or nonmonotone["outcome_reason"] != \
       "pre_floor_nonmonotone":
        raise RuntimeError("pre-floor increase was not retained as a hard failure")
    for errors in ((1.0, 0.25, 0.5, 0.125),
                   (1.0, 0.25, 0.0625, 0.125)):
        samples = [{"resolution": resolution, "error": error,
                    "direct_delta": 0.0, "clean_floor": 1.0e-12}
                   for resolution, error in zip(DIAGNOSTIC_RESOLUTIONS, errors)]
        outcome = evaluate_rate_samples(samples, 2.0, 0.25, "clean")
        if outcome["passed"] or outcome["outcome_reason"] != \
           "pre_floor_nonmonotone":
            raise RuntimeError("middle/trailing increase was not retained")


def verified_complete(case: Path, identity: dict[str, object]) -> bool:
    manifest_path = case / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return False
    try:
        manifest = load_json_strict(manifest_path)
    except (ValueError, RuntimeError):
        return False
    if not isinstance(manifest, dict):
        return False
    if (set(manifest) != {"schema", "state", "identity", "files"} or
            manifest.get("schema") != SCHEMA or manifest.get("state") != "complete" or
            manifest.get("identity") != identity or
            not isinstance(manifest.get("files"), dict)):
        return False
    expected = expected_case_files(int(identity["ranks"]))
    if set(manifest["files"]) != expected:
        return False
    actual = {entry.name for entry in case.iterdir()}
    if actual != expected | {"manifest.json"}:
        return False
    return all((case / name).is_file() and not (case / name).is_symlink() and
               sha256(case / name) == digest
               for name, digest in manifest["files"].items())


def run_case(args: argparse.Namespace, root: Path, source: dict[str, str],
             order: int, resolution: int, phase: int) -> dict[str, object]:
    nghost = order // 2 + 1
    domain_hash = hashlib.sha256(json.dumps(args.domain).encode()).hexdigest()[:10]
    key = f"o{order}-ng{nghost}-n{resolution}-p{phase}-r{args.ranks}-d{domain_hash}"
    case_environment = execution_environment()
    environment_sha256 = canonical_digest(case_environment)
    case_uuid = str(uuid.uuid5(
        uuid.NAMESPACE_URL,
        json.dumps([SCHEMA, source, key, case_environment], sort_keys=True)))
    case = args.output / f"{key}-{case_uuid}"
    basename = "cartoon_mms"
    rendered = render_input(args.input, order, resolution, phase, basename, args.domain)
    wrapper = args.rank_wrapper
    wrapped = [sys.executable, str(wrapper), "--evidence-dir", "."]
    if args.require_backend == "Cuda":
        wrapped.append("--require-cuda")
    command = launcher_command(args.launcher, args.ranks) + wrapped + [
        "--", str(args.athena), "-i", "input.athinput"]
    identity = {"case_id": key, "uuid": case_uuid, "source": source,
                "athena_sha256": sha256(args.athena), "input_sha256":
                hashlib.sha256(rendered.encode()).hexdigest(), "command": command,
                "ranks": args.ranks, "backend_required": args.require_backend,
                "build_manifest_sha256": sha256(args.build_manifest),
                "rank_wrapper_sha256": sha256(args.rank_wrapper),
                "domain": args.domain, "execution_environment": case_environment,
                "execution_environment_sha256": environment_sha256}
    started = time.time()
    resumed_result = None
    if case.exists():
        if not verified_complete(case, identity):
            raise RuntimeError(f"refusing incomplete or mismatched case directory: {case}")
        resumed_result = load_json_strict(case / "result.json")
        if not isinstance(resumed_result, dict):
            raise RuntimeError(f"case {key} resumed result is not an object")
    else:
        case.mkdir(parents=True)
        (case / "input.athinput").write_text(rendered, encoding="utf-8")
        write_atomic(case / "manifest.json", {"schema": SCHEMA, "state": "running",
                                               "identity": identity})
        environment = os.environ.copy()
        with (case / "stdout.txt").open("wb") as stdout, \
             (case / "stderr.txt").open("wb") as stderr:
            completed = subprocess.run(command, cwd=case, env=environment,
                                       stdout=stdout, stderr=stderr, check=False)
        if completed.returncode:
            raise RuntimeError(f"case {key} failed with exit {completed.returncode}: {case}")
    stdout_text = (case / "stdout.txt").read_text(encoding="utf-8", errors="replace")
    raw_result = case / f"{basename}.mms.json"
    raw_csv = case / f"{basename}.mms.csv"
    probes_csv = case / f"{basename}.mms.probes.csv"
    result = load_json_strict(raw_result)
    if not isinstance(result, dict):
        raise RuntimeError(f"case {key} result JSON is not an object")
    validate_result_numbers(result)
    if result.get("status") != "pass" or result.get("operator_count") != 171:
        raise RuntimeError(f"case {key} did not produce the complete passing 171-series set")
    verify_no_evolution(key, stdout_text, result)
    if result.get("backend") != args.require_backend:
        raise RuntimeError(f"case {key} backend {result.get('backend')} != {args.require_backend}")
    if result.get("owned_cells") != resolution * resolution or \
       result.get("ownership_sequence") != "[0,N*N) exactly once":
        raise RuntimeError(f"case {key} failed exact MPI ownership proof")
    bindings = [load_json_strict(path)
                for path in sorted(case.glob("rank_binding_*.json"))]
    for binding in bindings:
        validate_rank_binding(binding)
    if (len(bindings) != args.ranks or
            sorted(item["rank"] for item in bindings) != list(range(args.ranks))):
        raise RuntimeError(f"case {key} lacks one binding record per MPI rank")
    if args.require_backend == "Cuda":
        uuids = [item["selected_uuid"] for item in bindings]
        if (args.ranks != 4 or None in uuids or len(set(uuids)) != 4 or
                any(item.get("binding_verified") is not True or
                    item.get("cuda_visible_devices") !=
                    item.get("visible_device_token") or
                    not item.get("visible_device_token") for item in bindings) or
                any("A100" not in (item.get("gpu_name") or "") for item in bindings)):
            raise RuntimeError(
                f"case {key} requires four concrete, distinct CUDA UUID bindings")
    rows = list(csv.DictReader(raw_csv.open(encoding="utf-8")))
    for row in rows:
        validate_norm_row(row)
    operator_set = {row["operator"] for row in rows}
    operator_names = result.get("operator_names")
    if (not isinstance(operator_names, list) or len(operator_names) != 171 or
            len(set(operator_names)) != 171 or operator_set != set(operator_names) or
            any(int(row["nonfinite"]) for row in rows)):
        raise RuntimeError(f"case {key} has incomplete or nonfinite CSV series")
    noise_bound = json_number(result, "noise_delta_bound", 0.0)
    if any(finite_value(row, "shared_delta_linfinity") > noise_bound or
           finite_value(row, "independent_delta_linfinity") > noise_bound for row in rows):
        raise RuntimeError(f"case {key} exceeds frozen direct noise-delta bound")
    probe_rows = list(csv.DictReader(probes_csv.open(encoding="utf-8")))
    for row in probe_rows:
        validate_probe_row(row)
    validate_case_inventory(order, operator_names, rows, probe_rows)
    for row in probe_rows:
        validate_probe_geometry(row, order, resolution, args.domain)
    for row in rows:
        radial = mask_radial_indices(row["mask"], order, resolution, args.domain)
        expected_count = len(radial) * resolution
        expected_cyl_count = sum(1 for _, _, positive in radial if positive) * resolution
        if (integer_value(row, "count") != expected_count or
                integer_value(row, "cyl_count") != expected_cyl_count):
            raise RuntimeError(f"case {key} mask counts differ from exact geometry")
    if {row["operator"] for row in probe_rows} != operator_set or \
       any(not math.isfinite(float(row["raw_error"])) for row in probe_rows) or \
       any(not row["layer_index"] or not row["classification"] for row in probe_rows):
        raise RuntimeError(f"case {key} has incomplete raw probe/layer records")
    axis_rows = [row for row in probe_rows if row["mask"] == "diagnostic_axis"]
    axis_names = operator_names[:161]
    axis_errors = [finite_value(row, "raw_error") for row in axis_rows]
    if (len(axis_rows) != 161 or [row["operator"] for row in axis_rows] != axis_names or
            any(row["side"] != "axis" or row["classification"] != "diagnostic_axis" or
                row["layer_index"] != "0" for row in axis_rows) or
            any(not math.isfinite(error) for error in axis_errors) or
            result.get("diagnostic_axis_operator_count") != 161 or
            result.get("diagnostic_axis_nonfinite") != 0 or
            max(axis_errors) != json_number(result, "diagnostic_axis_linf", 0.0) or
            max(axis_errors) > json_number(result, "diagnostic_axis_tolerance", 0.0)):
        raise RuntimeError(f"case {key} lacks the exact finite 161-series true-axis probe")
    if resumed_result is not None:
        if (resumed_result.get("csv_sha256") != sha256(raw_csv) or
                resumed_result.get("probes_csv_sha256") != sha256(probes_csv) or
                resumed_result.get("operator_names") != operator_names or
                resumed_result.get("rank_bindings") != bindings or
                tuple(resumed_result.get("domain", ())) != tuple(args.domain) or
                resumed_result.get("execution_environment_sha256") !=
                environment_sha256):
            raise RuntimeError(f"case {key} resumed evidence differs from its result")
        resumed_result["case_manifest_sha256"] = sha256(case / "manifest.json")
        return resumed_result
    result.update({"case_id": key, "case_uuid": case_uuid, "phase": phase,
                   "resolution": resolution, "elapsed_seconds": time.time() - started,
                   "csv_sha256": sha256(raw_csv),
                   "probes_csv_sha256": sha256(probes_csv),
                   "operator_names": operator_names,
                   "rank_bindings": bindings,
                   "execution_environment": case_environment,
                   "execution_environment_sha256": environment_sha256,
                   "domain": args.domain,
                   "output_bytes": raw_csv.stat().st_size + probes_csv.stat().st_size})
    (case / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True,
                                                  allow_nan=False) + "\n",
                                      encoding="utf-8")
    files = {name: sha256(case / name) for name in
             ("input.athinput", "stdout.txt", "stderr.txt", "cartoon_mms.mms.json",
              "cartoon_mms.mms.csv", "result.json")}
    files["cartoon_mms.mms.probes.csv"] = sha256(probes_csv)
    for binding in sorted(case.glob("rank_binding_*.json")):
        files[binding.name] = sha256(binding)
    if set(files) != expected_case_files(args.ranks):
        raise RuntimeError(f"case {key} produced an unexpected file inventory")
    write_atomic(case / "manifest.json", {"schema": SCHEMA, "state": "complete",
                                           "identity": identity, "files": files})
    result["case_manifest_sha256"] = sha256(case / "manifest.json")
    return result


def normalize_legacy_norm_row(row: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    normalized = dict(row)
    actions = []
    if "cylindrical_applicable" not in normalized:
        applicable = integer_value(normalized, "cyl_count") > 0
        normalized["cylindrical_applicable"] = str(applicable).lower()
        actions.append("derived_cylindrical_applicability")
        if not applicable:
            for field in ("cyl_l1", "cyl_l2", "cyl_linfinity"):
                if normalized.get(field, "").lower() == "nan":
                    normalized[field] = ""
                    actions.append(f"blanked_inapplicable_{field}")
    if "radius_applicable" not in normalized:
        applicable = normalized.get("mask", "").startswith("fixed_rho_")
        normalized["radius_applicable"] = str(applicable).lower()
        actions.append("derived_radius_applicability")
        if not applicable:
            for field in ("target_abs_rho", "actual_abs_rho"):
                if normalized.get(field, "").lower() == "nan":
                    normalized[field] = ""
                    actions.append(f"blanked_inapplicable_{field}")
    validate_norm_row(normalized)
    return normalized, actions


def normalize_legacy_probe_row(row: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    normalized = dict(row)
    actions = []
    if "target_rho_applicable" not in normalized:
        applicable = normalized.get("classification") in {"fixed_radius", "regular",
                                                            "diagnostic_axis"}
        normalized["target_rho_applicable"] = str(applicable).lower()
        actions.append("derived_target_rho_applicability")
        if not applicable and normalized.get("target_rho", "").lower() == "nan":
            normalized["target_rho"] = ""
            actions.append("blanked_inapplicable_target_rho")
    validate_probe_row(normalized)
    return normalized, actions


def verify_replay_campaign(raw_root: Path) -> tuple[list[dict[str, object]],
                                                    dict[str, object]]:
    preserved_convergence = raw_root / "results/ranks2/convergence.json"
    preserved_evidence = raw_root / "evidence/cpu-ranks2.json"
    if (sha256(preserved_convergence) != PRESERVED_JOB_56586376_CONVERGENCE_SHA256 or
            sha256(preserved_evidence) != PRESERVED_JOB_56586376_EVIDENCE_SHA256):
        raise RuntimeError("replay root is not immutable job56586376 evidence")
    if not isinstance(load_json_strict(preserved_convergence), dict) or \
       not isinstance(load_json_strict(preserved_evidence), dict):
        raise RuntimeError("preserved convergence/wrapper evidence is malformed")
    manifests = sorted((raw_root / "results/ranks2").glob("o*/manifest.json"))
    if len(manifests) != EXPECTED_CASES:
        raise RuntimeError("replay requires exactly 96 complete case manifests")
    expected = {(order, resolution, phase) for order in (2, 4, 6)
                for resolution in DIAGNOSTIC_RESOLUTIONS for phase in range(8)}
    observed = set()
    case_bindings = []
    cases = []
    verified_case_bytes = 0
    for manifest_path in manifests:
        case = manifest_path.parent
        manifest = load_json_strict(manifest_path)
        if (not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA or
                manifest.get("state") != "complete" or
                set(manifest) != {"schema", "state", "identity", "files"} or
                not isinstance(manifest.get("files"), dict)):
            raise RuntimeError(f"replay case {case.name} has a malformed manifest")
        files = manifest["files"]
        actual = {entry.name for entry in case.iterdir()}
        if actual != set(files) | {"manifest.json"}:
            raise RuntimeError(f"replay case {case.name} file inventory differs")
        for name, digest in files.items():
            path = case / name
            if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or
                    not path.is_file() or path.is_symlink() or sha256(path) != digest):
                raise RuntimeError(f"replay case {case.name} failed hash verification: {name}")
            verified_case_bytes += path.stat().st_size
        verified_case_bytes += manifest_path.stat().st_size
        manifest_sha = sha256(manifest_path)
        stored_result = load_json_strict(case / "result.json")
        if not isinstance(stored_result, dict):
            raise RuntimeError(f"replay case {case.name} result is not an object")
        validate_result_numbers(stored_result)
        if set(files) != expected_case_files(json_integer(stored_result, "mpi_ranks", 1)):
            raise RuntimeError(f"replay case {case.name} file set differs from exact schema")
        result = dict(stored_result)
        order = json_integer(result, "spatial_order", 2)
        resolution = json_integer(result, "resolution", 1)
        phase = json_integer(result, "phase")
        observed.add((order, resolution, phase))
        case_id, case_uuid = result.get("case_id"), result.get("case_uuid")
        if (not isinstance(case_id, str) or not isinstance(case_uuid, str) or
                case.name != f"{case_id}-{case_uuid}"):
            raise RuntimeError("replay case directory differs from result id/uuid")
        identity = manifest["identity"]
        if (not isinstance(identity, dict) or identity.get("case_id") != case_id or
                identity.get("uuid") != case_uuid or
                identity.get("ranks") != result.get("mpi_ranks") or
                identity.get("domain") != result.get("domain") or
                identity.get("execution_environment_sha256") !=
                result.get("execution_environment_sha256")):
            raise RuntimeError(f"replay case {case_id} manifest/result identity differs")
        bindings = stored_result.get("rank_bindings")
        operators = stored_result.get("operator_names")
        if (not isinstance(bindings, list) or
                len(bindings) != json_integer(stored_result, "mpi_ranks", 1) or
                not isinstance(operators, list) or len(operators) != 171 or
                len(set(operators)) != 171):
            raise RuntimeError(f"replay case {case_id} result inventory is incomplete")
        for binding_record in bindings:
            validate_rank_binding(binding_record)
        if (stored_result.get("case_id") != case_id or
                stored_result.get("case_uuid") != case_uuid or
                json_integer(stored_result, "resolution", 1) != resolution or
                json_integer(stored_result, "phase") != phase or
                stored_result.get("csv_sha256") != files.get("cartoon_mms.mms.csv") or
                stored_result.get("probes_csv_sha256") !=
                files.get("cartoon_mms.mms.probes.csv")):
            raise RuntimeError(f"replay case {case_id} result identity differs")
        result["case_manifest_sha256"] = manifest_sha
        result["_replay_case_directory"] = str(case)
        cases.append(result)
        case_bindings.append({"case_id": case_id, "case_uuid": case_uuid,
                              "manifest_sha256": manifest_sha,
                              "identity": manifest["identity"], "files": files})
    if observed != expected:
        raise RuntimeError("replay cases differ from exact 2/4/6 x 32/64/128/256 x 8")
    binding = {"schema": "athenak_z4c_cartoon_mms_replay_binding_v1",
               "raw_root": str(raw_root),
               "preserved_convergence_sha256": sha256(preserved_convergence),
               "preserved_wrapper_evidence_sha256": sha256(preserved_evidence),
               "raw_identity_sha256": canonical_digest(
                   [manifest["identity"] for manifest in
                    (load_json_strict(path) for path in manifests)]),
               "case_count": len(cases), "verified_case_bytes": verified_case_bytes,
               "cases": case_bindings}
    return cases, binding


def convergence_gate(cases: list[dict[str, object]], case_root: Path, output: Path,
                     series_manifest: Path,
                     resolutions_by_order: dict[int, tuple[int, ...]],
                     allow_legacy_nullable: bool = False) -> list[str]:
    inventory = load_json_strict(series_manifest)
    if not isinstance(inventory, dict) or inventory.get("count") != 171:
        raise RuntimeError("series manifest does not enumerate exactly 171 operators")
    validate_roundoff_inventory(inventory)
    metadata = {item["name"]: item for item in inventory["series"]}
    if len(metadata) != 171:
        raise RuntimeError("series manifest has duplicate operators")
    policy = inventory.get("roundoff_policy")
    if (not isinstance(policy, dict) or
            policy.get("binary64_epsilon_hex") != sys.float_info.epsilon.hex() or
            policy.get("fit_fixture_count") != 108):
        raise RuntimeError("series manifest lacks the frozen binary64 floor policy")
    grouped: dict[tuple[object, ...], list[dict[str, float]]] = {}
    exact_records = []
    failures = []
    seen_norm_rows = set()
    seen_probe_rows = set()
    floor_cache: dict[tuple[object, ...], dict[str, object]] = {}
    floor_evidence: dict[str, dict[str, object]] = {}
    normalization_actions: set[str] = set()

    def add_sample(key: tuple[object, ...], sample: dict[str, float]) -> None:
        grouped.setdefault(key, []).append(sample)

    for result in cases:
        case = (Path(str(result["_replay_case_directory"]))
                if "_replay_case_directory" in result else
                case_root / f"{result['case_id']}-{result['case_uuid']}")
        order = int(result["spatial_order"])
        resolution = int(result["resolution"])
        phase = int(result["phase"])
        domain = tuple(float(value) for value in result["domain"])
        noise_bound = json_number(result, "noise_delta_bound", 0.0)
        raw_rows = list(csv.DictReader(
            (case / "cartoon_mms.mms.csv").open(encoding="utf-8")))
        rows = []
        for raw_row in raw_rows:
            if allow_legacy_nullable:
                row, actions = normalize_legacy_norm_row(raw_row)
                normalization_actions.update(actions)
            else:
                row = raw_row
                validate_norm_row(row)
            rows.append(row)
        for row in rows:
            identity = (order, resolution, phase, row["operator"], row["mask"])
            if identity in seen_norm_rows:
                failures.append(f"duplicate norm row {identity}")
                continue
            seen_norm_rows.add(identity)
            item = metadata.get(row["operator"])
            if item is None:
                failures.append(f"unknown operator {row['operator']}")
                continue
            classification = item["classification"]
            if classification != "truncating":
                bound = (2.0e-10 if classification == "exact_discrete" else
                         SATURATION_FACTOR * sys.float_info.epsilon *
                         resolution * resolution)
                delta_values = {lane: finite_value(row, f"{lane}_delta_linfinity")
                                for lane in item["noise_lanes"]}
                passed = finite_value(row, "linfinity") <= bound and \
                    all(value <= noise_bound for value in delta_values.values())
                exact_records.append({
                    "source": "norm", "order": order, "phase": phase,
                    "resolution": resolution, "operator": row["operator"],
                    "mask": row["mask"], "classification": classification,
                    "value": finite_value(row, "linfinity"), "bound": bound,
                    "direct_delta_linfinity": delta_values,
                    "direct_delta_bound": noise_bound, "passed": passed})
                if not passed:
                    failures.append(
                        f"exact norm gate failed {row['operator']} {row['mask']} "
                        f"N={resolution}")
                continue
            clean_norms = ["l1", "l2", "linfinity"]
            if boolean_value(row, "cylindrical_applicable"):
                clean_norms += ["cyl_l1", "cyl_l2", "cyl_linfinity"]
            for lane in item["convergence_lanes"]:
                norms = clean_norms if lane == "clean" else ["l1", "l2", "linfinity"]
                for norm in norms:
                    error_field = norm if lane == "clean" else f"{lane}_{norm}"
                    delta_field = None if lane == "clean" else f"{lane}_delta_{norm}"
                    floor_key = (row["operator"], order, resolution, row["mask"],
                                 norm, domain)
                    if floor_key not in floor_cache:
                        floor_cache[floor_key] = aggregate_clean_floor(
                            item, policy, order, resolution, row["mask"], norm, domain)
                    floor_record = floor_cache[floor_key]
                    floor_id = canonical_digest(floor_key)
                    floor_evidence.setdefault(floor_id, {
                        "floor_id": floor_id, "operator": row["operator"],
                        "order": order, "resolution": resolution,
                        "mask": row["mask"], "norm": norm,
                        "domain": list(domain), **floor_record})
                    add_sample(("norm", order, phase,
                                row["operator"] + "|" + row["mask"], lane, norm),
                               {"resolution": resolution,
                                "error": finite_value(row, error_field),
                                "direct_delta": 0.0 if delta_field is None else
                                finite_value(row, delta_field),
                                "clean_floor": floor_record["clean_floor"],
                                "floor_id": floor_id})

        raw_probes = list(csv.DictReader(
            (case / "cartoon_mms.mms.probes.csv").open(encoding="utf-8")))
        probes = []
        for raw_probe in raw_probes:
            if allow_legacy_nullable:
                row, actions = normalize_legacy_probe_row(raw_probe)
                normalization_actions.update(actions)
            else:
                row = raw_probe
                validate_probe_row(row)
            validate_probe_geometry(row, order, resolution, domain)
            probes.append(row)
        for row in probes:
            if row["classification"] == "diagnostic_axis":
                continue
            identity = (order, resolution, phase, row["operator"], row["mask"],
                        row["side"], row["layer_index"])
            if identity in seen_probe_rows:
                failures.append(f"duplicate raw probe row {identity}")
                continue
            seen_probe_rows.add(identity)
            item = metadata.get(row["operator"])
            if item is None:
                failures.append(f"unknown raw-probe operator {row['operator']}")
                continue
            classification = item["classification"]
            if classification != "truncating":
                bound = (2.0e-10 if classification == "exact_discrete" else
                         SATURATION_FACTOR * sys.float_info.epsilon *
                         resolution * resolution)
                value = finite_value(row, "raw_error")
                passed = value <= bound
                exact_records.append({
                    "source": "probe", "order": order, "phase": phase,
                    "resolution": resolution, "operator": row["operator"],
                    "mask": row["mask"], "side": row["side"],
                    "layer_index": integer_value(row, "layer_index"),
                    "classification": classification, "value": value,
                    "bound": bound, "passed": passed})
                if not passed:
                    failures.append(
                        f"exact probe gate failed {row['operator']} {row['mask']} "
                        f"N={resolution}")
                continue
            series = probe_series_identity(row)
            floor_record = probe_clean_floor(
                item, policy, order, resolution, row, domain)
            floor_id = canonical_digest((row["operator"], order, resolution,
                                         row["mask"], row["side"], domain))
            floor_evidence.setdefault(floor_id, {
                "floor_id": floor_id, "operator": row["operator"],
                "order": order, "resolution": resolution, "mask": row["mask"],
                "side": row["side"], "domain": list(domain), **floor_record})
            add_sample(("probe", order, phase, series, "clean", "raw_error"),
                       {"resolution": resolution,
                        "error": finite_value(row, "raw_error"),
                        "direct_delta": 0.0,
                        "clean_floor": floor_record["clean_floor"],
                        "floor_id": floor_id,
                        "probe_classification": row["classification"],
                        "layer_index": integer_value(row, "layer_index"),
                        "target_rho": None if not boolean_value(
                            row, "target_rho_applicable") else
                        finite_value(row, "target_rho"),
                        "actual_rho": finite_value(row, "actual_rho"),
                        "target_z": finite_value(row, "target_z"),
                        "actual_z": finite_value(row, "actual_z")})

    records = []
    for key, all_values in grouped.items():
        source, order, phase, series, lane, norm = key
        all_values.sort(key=lambda item: item["resolution"])
        required = tuple(resolutions_by_order[order])
        observed = tuple(item["resolution"] for item in all_values)
        if observed != required:
            reason = f"incomplete_resolution_set expected={required} observed={observed}"
            failures.append(f"{key}: {reason}")
            records.append({"source": source, "order": order, "phase": phase,
                            "series": series, "lane": lane, "norm": norm,
                            "expected": None, "margin": None,
                            "diagnostic_resolutions": list(required),
                            "samples": all_values, "rates": [],
                            "rate_status": [], "interval_reasons": [],
                            "outcome_reason": "incomplete_resolution_set",
                            "outcome_detail": reason, "passed": False})
            continue
        selected = all_values
        mask = series.split("|")[1]
        probe_classification = (selected[0].get("probe_classification")
                                if selected else None)
        expected, margin = rate_policy(order, mask, lane, norm,
                                       probe_classification)
        evaluation = evaluate_rate_samples(selected, expected, margin, lane)
        if not evaluation["passed"]:
            failures.append(
                f"{key}: rates={evaluation['rates']} "
                f"status={evaluation['rate_status']} "
                f"expected>={expected-margin}")
        record = {"source": source, "order": order, "phase": phase,
                  "series": series, "lane": lane, "norm": norm,
                  "expected": expected, "margin": margin,
                  "diagnostic_resolutions": list(required),
                  "samples": selected}
        record.update(evaluation)
        records.append(record)

    records.sort(key=lambda item: (item["order"], item["phase"], item["source"],
                                   item["series"], item["lane"], item["norm"]))
    exact_records.sort(key=lambda item: (item["order"], item["phase"],
                                         item["resolution"], item["source"],
                                         item["operator"], item["mask"]))
    table_rows = []
    for record in records:
        samples = record["samples"]
        for index, status in enumerate(record.get("rate_status", [])):
            rate = record["rates"][index]
            table_rows.append({
                "source": record["source"], "order": record["order"],
                "phase": record["phase"], "series": record["series"],
                "lane": record["lane"], "norm": record["norm"],
                "coarse_resolution": samples[index]["resolution"],
                "fine_resolution": samples[index + 1]["resolution"],
                "coarse_error": f"{samples[index]['error']:.17g}",
                "fine_error": f"{samples[index + 1]['error']:.17g}",
                "coarse_direct_delta": f"{samples[index]['direct_delta']:.17g}",
                "fine_direct_delta": f"{samples[index + 1]['direct_delta']:.17g}",
                "coarse_floor": f"{samples[index]['applied_floor']:.17g}",
                "fine_floor": f"{samples[index + 1]['applied_floor']:.17g}",
                "rate_status": status,
                "interval_reason": record["interval_reasons"][index]["reason"],
                "observed_rate": "" if rate is None else f"{rate:.17g}",
                "expected_rate": f"{record['expected']:.17g}",
                "passed": int(record["passed"]),
            })
    fields = ["source", "order", "phase", "series", "lane", "norm",
              "coarse_resolution", "fine_resolution", "coarse_error", "fine_error",
              "coarse_direct_delta", "fine_direct_delta", "coarse_floor", "fine_floor",
              "rate_status", "interval_reason", "observed_rate",
              "expected_rate", "passed"]
    csv_path = output / "convergence.csv"
    data_path = output / "convergence_rates.pgfplots.dat"
    write_csv_atomic(csv_path, fields, table_rows)
    plot_rows = [{"order": row["order"], "phase": row["phase"],
                  "lane_id": {"clean": 0, "shared": 1,
                              "independent": 2}[row["lane"]],
                  "fine_resolution": row["fine_resolution"],
                  "observed_rate": row["observed_rate"],
                  "expected_rate": row["expected_rate"], "passed": row["passed"]}
                 for row in table_rows if row["rate_status"] == "included_rate"]
    plot_fields = ["order", "phase", "lane_id", "fine_resolution",
                   "observed_rate", "expected_rate", "passed"]
    write_csv_atomic(data_path, plot_fields, plot_rows, delimiter=" ")
    plot_path = output / "convergence_plot.tex"
    plot_path.write_text(
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[xlabel={fine resolution},ylabel={observed order},"
        "xmode=log,log basis x=2,legend pos=south east]\n"
        "\\addplot+[only marks,mark size=.35pt] table[x=fine_resolution,"
        "y=observed_rate] {convergence_rates.pgfplots.dat};\n"
        "\\addlegendentry{diagnostic unsaturated series}\n"
        "\\end{axis}\n\\end{tikzpicture}\n", encoding="utf-8")
    write_atomic(output / "convergence.json", {
        "schema": SCHEMA, "series_manifest_sha256": sha256(series_manifest),
        "diagnostic_resolutions_by_order": {str(order): list(values)
                                             for order, values in
                                             resolutions_by_order.items()},
        "coefficient_floor_policy_sha256": canonical_digest(policy),
        "coefficient_floor_complexity": "O(171*nx1); active-z multiplicity analytic",
        "floor_decompositions": sorted(floor_evidence.values(),
                                       key=lambda item: item["floor_id"]),
        "legacy_normalization_actions": sorted(normalization_actions),
        "records": records, "exact_records": exact_records,
        "artifacts": {"convergence.csv": sha256(csv_path),
                      "convergence_rates.pgfplots.dat": sha256(data_path),
                      "convergence_plot.tex": sha256(plot_path)},
        "failures": failures})
    return failures


def compare_rank_campaigns(cases: list[dict[str, object]], output: Path,
                           reference_root: Path) -> dict[str, object]:
    reference_campaign_path = reference_root / "campaign.json"
    reference = load_json_strict(reference_campaign_path)
    campaign_fields = {"schema", "source", "build_manifest",
                       "build_manifest_sha256", "environment", "environment_sha256",
                       "ranks", "backend", "diagnostic_resolutions",
                       "reduction_tolerance_factor", "convergence_artifacts", "cases"}
    if (not isinstance(reference, dict) or set(reference) != campaign_fields or
            reference.get("schema") != SCHEMA or
            not isinstance(reference.get("cases"), list)):
        raise RuntimeError("reference campaign differs from the exact evidence schema")
    reference_case_manifests = [item.get("case_manifest_sha256")
                                for item in reference.get("cases", [])]
    if (not reference_case_manifests or None in reference_case_manifests or
            any(not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in reference_case_manifests)):
        raise RuntimeError("reference campaign does not bind every case manifest")
    reference_cases = {(item["spatial_order"], item["resolution"], item["phase"]): item
                       for item in reference["cases"]}
    comparisons = []
    failures = []
    for current in cases:
        key = (current["spatial_order"], current["resolution"], current["phase"])
        other = reference_cases.get(key)
        if other is None:
            failures.append(f"missing reference case {key}")
            continue
        current_dir = output / f"{current['case_id']}-{current['case_uuid']}"
        other_dir = reference_root / f"{other['case_id']}-{other['case_uuid']}"
        current_rows = list(csv.DictReader(
            (current_dir / "cartoon_mms.mms.csv").open(encoding="utf-8")))
        other_rows = list(csv.DictReader(
            (other_dir / "cartoon_mms.mms.csv").open(encoding="utf-8")))
        for row in current_rows + other_rows:
            validate_norm_row(row)
        indexed = {(row["operator"], row["mask"]): row for row in other_rows}
        current_index = {(row["operator"], row["mask"]): row for row in current_rows}
        if (len(current_index) != len(current_rows) or len(indexed) != len(other_rows) or
                set(current_index) != set(indexed) or
                current.get("operator_names") != other.get("operator_names")):
            failures.append(f"series/row inventory differs for {key}")
            continue
        for row in current_rows:
            row_key = (row["operator"], row["mask"])
            baseline = indexed.get(row_key)
            if baseline is None:
                failures.append(f"{key} missing row {row_key}")
                continue
            exact_fields = ("count", "nonfinite", "cyl_count",
                            "cylindrical_applicable", "linfinity",
                            "cyl_linfinity", "shared_linfinity",
                            "shared_delta_linfinity", "independent_linfinity",
                            "independent_delta_linfinity", "rotation_linfinity",
                            "target_abs_rho", "radius_applicable", "actual_abs_rho",
                            "mask_xor")
            exact = all(row[field] == baseline[field] for field in exact_fields)
            tolerance = (REDUCTION_TOLERANCE_FACTOR * sys.float_info.epsilon *
                         max(1.0, math.log2(int(current["resolution"]) ** 2)))
            numeric = True
            worst = 0.0
            numeric_fields = ["l1", "l2", "shared_l1", "shared_l2",
                              "shared_delta_l1", "shared_delta_l2",
                              "independent_l1", "independent_l2",
                              "independent_delta_l1", "independent_delta_l2"]
            if int(row["cyl_count"]) > 0:
                numeric_fields += ["cyl_l1", "cyl_l2"]
            for field in numeric_fields:
                left, right = float(row[field]), float(baseline[field])
                difference = abs(left - right)
                worst = max(worst, difference)
                numeric &= difference <= tolerance * max(1.0, abs(left), abs(right))
            passed = exact and numeric
            comparisons.append({"case": key, "row": row_key, "passed": passed,
                                "worst_absolute_difference": worst,
                                "relative_tolerance": tolerance})
            if not passed:
                failures.append(f"rank comparison failed {key} {row_key}")
        if current["ownership_fnv1a64"] != other["ownership_fnv1a64"]:
            failures.append(f"ownership hash differs for {key}")
        current_probes = list(csv.DictReader(
            (current_dir / "cartoon_mms.mms.probes.csv").open(encoding="utf-8")))
        other_probes = list(csv.DictReader(
            (other_dir / "cartoon_mms.mms.probes.csv").open(encoding="utf-8")))
        for row in current_probes + other_probes:
            validate_probe_row(row)
        validate_case_inventory(int(current["spatial_order"]),
                                list(current["operator_names"]),
                                current_rows, current_probes)
        validate_case_inventory(int(other["spatial_order"]),
                                list(other["operator_names"]),
                                other_rows, other_probes)
        probe_key = lambda row: (row["operator"], row["mask"], row["side"],
                                 row["layer_index"], row["classification"])
        current_probe_index = {probe_key(row): row for row in current_probes}
        other_probe_index = {probe_key(row): row for row in other_probes}
        if (len(current_probe_index) != len(current_probes) or
                len(other_probe_index) != len(other_probes) or
                set(current_probe_index) != set(other_probe_index)):
            failures.append(f"probe inventory differs for {key}")
        else:
            for probe_id, row in current_probe_index.items():
                baseline = other_probe_index[probe_id]
                if any(row[field] != baseline[field] for field in
                       ("target_rho_applicable", "target_rho", "actual_rho",
                        "target_z", "actual_z",
                        "global_cell_id", "raw_error")):
                    failures.append(f"probe value differs for {key} {probe_id}")
                    break
        for label, result in (("current", current), ("reference", other)):
            bindings = result.get("rank_bindings", [])
            for binding in bindings:
                validate_rank_binding(binding)
            expected_ranks = int(result.get("mpi_ranks", 0))
            if (len(bindings) != expected_ranks or
                    sorted(item.get("rank") for item in bindings) !=
                    list(range(expected_ranks))):
                failures.append(f"{label} rank binding inventory incomplete for {key}")
    write_atomic(output / "rank_comparison.json", {"schema": SCHEMA,
                                                    "reference": str(reference_root),
                                                    "comparisons": comparisons,
                                                    "failures": failures})
    if failures:
        raise RuntimeError("2-rank/4-rank comparison failed; see rank_comparison.json")
    return {"rank_comparison.json": sha256(output / "rank_comparison.json"),
            "reference_campaign_sha256": sha256(reference_campaign_path),
            "reference_case_manifest_sha256": reference_case_manifests}


def load_search_manifest(path: Path) -> dict[str, object]:
    value = load_json_strict(path)
    expected_pools = {
        "2": [32, 64, 128, 256, 512, 1024, 2048, 4096],
        "4": [32, 64, 128, 256],
        "6": [32, 48, 64, 80, 96, 112, 128, 160, 192, 256],
    }
    required = {"schema", "status", "diagnostic_resolutions", "resolution_pools",
                "staged_evaluation_pools", "existing_evidence",
                "pending_execution_stages", "qualification_domain",
                "qualification_windows",
                "minimum_consecutive_unsaturated_ratios", "thresholds", "floor_policy",
                "exact_series_treatment", "evidence_lifecycle", "series_binding",
                "lifecycle_transitions"}
    evidence = value.get("evidence_lifecycle", {}) if isinstance(value, dict) else {}
    if (not isinstance(value, dict) or set(value) != required or
            value.get("schema") != "athenak_z4c_cartoon_mms_prospective_search_v1" or
            value.get("status") != "prospective_unselected" or
            value.get("resolution_pools") != expected_pools or
            value.get("diagnostic_resolutions") != list(DIAGNOSTIC_RESOLUTIONS) or
            value.get("existing_evidence") != {
                "2": list(DIAGNOSTIC_RESOLUTIONS), "4": list(DIAGNOSTIC_RESOLUTIONS),
                "6": list(DIAGNOSTIC_RESOLUTIONS)} or
            value.get("qualification_domain") != list(QUALIFICATION_DOMAIN) or
            value.get("qualification_windows") is not None or
            value.get("minimum_consecutive_unsaturated_ratios") != 2 or
            value.get("series_binding") != {
                "count": 171,
                "class_counts": {"truncating": 137, "exact_identity": 12,
                                 "exact_plane_algebraic": 12, "exact_discrete": 10},
                "manifest_sha256":
                "eda57946cce1af9ae3acfb439c9d89e21b8980bbf48a9f375e8948e527ac2ba8"} or
            evidence != {"state": "checked_in_template", "source_commit": None,
                         "source_tree": None, "executable_sha256": None,
                         "result_sha256": None,
                         "input_sha256":
                         "10375d51677fc0544969271919b2d6a3b9c26d97e8899c2a6b429b01fe381bc0",
                         "oracle_header_sha256":
                         "57d8e9edd333e0450b94caf415df5ecdda917138de6514f77c9fa506269ebaf3",
                         "oracle_reference_sha256":
                         "4e3b61961097af105b915169c29f7f19e577218ec73061bb94204b7dbdfa43ad"} or
            value.get("lifecycle_transitions") != {
                "checked_in_template": "source/tree/executable/result are null",
                "prelaunch_bound": "fill source/tree/executable and retain result null",
                "postrun_finalized":
                "fill result hash after immutable case manifests close"}):
        raise RuntimeError("prospective search manifest differs from the frozen unselected schema")
    flattened = {order: [resolution for stage in value["staged_evaluation_pools"][order]
                         for resolution in stage] for order in expected_pools}
    if flattened != expected_pools:
        raise RuntimeError("prospective staged stopping differs from the fixed pools")
    if value["pending_execution_stages"] != {
            "2": [[512, 1024], [2048], [4096]], "4": [],
            "6": [[48, 80, 96], [112, 160, 192]]}:
        raise RuntimeError("prospective execution stages would rerun existing evidence")
    return value


def replay_campaign(raw_root: Path, analysis_output: Path, root: Path,
                    series_manifest: Path, prospective_search: bool) -> int:
    raw_root = raw_root.resolve()
    analysis_output = analysis_output.resolve()
    if not raw_root.is_dir() or analysis_output == raw_root or \
       raw_root in analysis_output.parents:
        raise RuntimeError("replay analysis output must be outside the read-only raw root")
    cases, binding = verify_replay_campaign(raw_root)
    domains = {tuple(float(value) for value in case.get("domain", ())) for case in cases}
    if len(domains) != 1:
        raise RuntimeError("replay cases do not share one explicit domain")
    resolutions = {order: DIAGNOSTIC_RESOLUTIONS for order in (2, 4, 6)}
    certification = validate_certified_domain(domains.pop(), resolutions, True)
    if analysis_output.exists() and any(analysis_output.iterdir()):
        raise RuntimeError("replay analysis output must be absent or empty")
    analysis_output.mkdir(parents=True, exist_ok=True)
    failures = convergence_gate(cases, raw_root, analysis_output, series_manifest,
                                resolutions, allow_legacy_nullable=True)
    artifacts = {name: sha256(analysis_output / name) for name in
                 ("convergence.json", "convergence.csv",
                  "convergence_rates.pgfplots.dat", "convergence_plot.tex")}
    if prospective_search:
        search = load_search_manifest(
            root / "tst/inputs/z4c_cartoon_mms_search_manifest.json")
        write_atomic(analysis_output / "prospective_search_manifest.json", {
            **search, "raw_evidence_binding_sha256": canonical_digest(binding)})
        artifacts["prospective_search_manifest.json"] = sha256(
            analysis_output / "prospective_search_manifest.json")
    write_atomic(analysis_output / "replay_manifest.json", {
        **binding, "analysis_policy_source_commit": git_value(root, "rev-parse", "HEAD"),
        "analysis_policy_source_tree": git_value(root, "rev-parse", "HEAD^{tree}"),
        "series_manifest_sha256": sha256(series_manifest),
        "domain_certification": certification, "artifacts": artifacts,
        "failures": failures, "passed": not failures})
    return 1 if failures else 0


def main() -> int:
    if sys.argv[1:] == ["--self-test-no-evolution-parser"]:
        self_test_no_evolution_parser()
        return 0
    if sys.argv[1:] == ["--self-test-cpu-audit-policy"]:
        self_test_cpu_audit_policy()
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--launcher")
    parser.add_argument("--ranks", type=int, choices=(2, 4))
    parser.add_argument("--orders", type=int, nargs="+", default=(2, 4, 6))
    parser.add_argument("--resolutions", type=int, nargs="+")
    parser.add_argument("--phases", type=int, nargs="+", default=tuple(range(8)))
    parser.add_argument("--require-backend", choices=("Serial", "Cuda"))
    parser.add_argument("--build-manifest", type=Path)
    parser.add_argument("--rank-wrapper", type=Path)
    parser.add_argument("--x1min", type=float, default=-2.0)
    parser.add_argument("--x1max", type=float, default=2.0)
    parser.add_argument("--x2min", type=float, default=-2.0)
    parser.add_argument("--x2max", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay-campaign", type=Path)
    parser.add_argument("--analysis-output", type=Path)
    parser.add_argument("--prospective-search", action="store_true")
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--compare-campaign", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    series_manifest = root / "tst/unit/z4c/z4c_cartoon_derivatives_series.json"
    if args.replay_campaign is not None:
        if args.analysis_output is None or any(value is not None for value in
                                               (args.athena, args.launcher, args.ranks,
                                                args.output)):
            raise RuntimeError("replay requires only RAW_ROOT and a separate analysis output")
        return replay_campaign(args.replay_campaign, args.analysis_output, root,
                               series_manifest, args.prospective_search)
    if args.analysis_output is not None:
        raise RuntimeError("--analysis-output is valid only with --replay-campaign")
    if args.athena is None or args.launcher is None or args.ranks is None or \
       args.output is None:
        raise RuntimeError("campaign requires athena, launcher, ranks, and output")
    if args.prospective_search:
        raise RuntimeError("prospective search is read-only and requires --replay-campaign")
    search = load_search_manifest(root / "tst/inputs/z4c_cartoon_mms_search_manifest.json")
    if args.resolutions is None:
        args.resolutions = list(DIAGNOSTIC_RESOLUTIONS)
    resolutions_by_order = {order: DIAGNOSTIC_RESOLUTIONS for order in (2, 4, 6)}
    if (set(args.orders) != {2, 4, 6} or len(args.orders) != 3 or
            set(args.resolutions) != set(DIAGNOSTIC_RESOLUTIONS) or
            len(args.resolutions) != 4 or set(args.phases) != set(range(8)) or
            len(args.phases) != 8):
        raise RuntimeError("diagnostics require exactly orders 2/4/6, resolutions "
                           "32/64/128/256, and phases 0..7")
    args.orders = [2, 4, 6]
    args.resolutions = sorted({resolution for values in resolutions_by_order.values()
                               for resolution in values})
    args.phases = list(range(8))
    if args.input is None:
        args.input = root / "tst/inputs/z4c_cartoon_derivatives.athinput"
    args.athena = args.athena.resolve()
    args.rank_wrapper = (args.rank_wrapper or
                         (root / "tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py")).resolve()
    args.build_manifest = (args.build_manifest or
                           (args.athena.parent / "mms_build_manifest.json")).resolve()
    build = load_json_strict(args.build_manifest)
    if not isinstance(build, dict):
        raise RuntimeError("build manifest is not an object")
    required_build_keys = {"schema", "source_commit", "source_tree", "kokkos_commit",
                           "source_clean", "backend", "executable_sha256",
                           "configure_cache_sha256", "compiler", "kokkos_runtime",
                           "configure", "build", "translation_units", "slowest_tus",
                           "configure_cache_contract"}
    if not required_build_keys.issubset(build) or not build["source_clean"]:
        raise RuntimeError("immutable build manifest is incomplete or not clean")
    if build["configure_cache_contract"].get("Athena_SINGLE_PRECISION") != "OFF":
        raise RuntimeError("Cartoon derivative MMS qualification is frozen to Real64")
    if build["executable_sha256"] != sha256(args.athena):
        raise RuntimeError("Athena executable does not match immutable build manifest")
    if args.require_backend is None:
        args.require_backend = build["backend"]
    elif args.require_backend != build["backend"]:
        raise RuntimeError("requested backend conflicts with immutable build manifest")
    if args.require_backend == "Cuda" and args.ranks != 4:
        raise RuntimeError("CUDA+MPI qualification requires exactly four ranks")
    args.domain = (args.x1min, args.x1max, args.x2min, args.x2max)
    certification = validate_certified_domain(args.domain, resolutions_by_order, True)
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    if args.output.exists() and not args.output.is_dir():
        raise RuntimeError("output is not a directory")
    source = {"commit": git_value(root, "rev-parse", "HEAD"),
              "tree": git_value(root, "rev-parse", "HEAD^{tree}"),
              "kokkos": git_value(root, "rev-parse", "HEAD:kokkos")}
    if git_value(root, "status", "--porcelain"):
        raise RuntimeError("campaign requires a clean source checkout")
    if (build["source_commit"] != source["commit"] or
            build["source_tree"] != source["tree"] or
            build["kokkos_commit"] != source["kokkos"]):
        raise RuntimeError("build manifest source identity does not match driver checkout")
    series_inventory = load_json_strict(series_manifest)
    if not isinstance(series_inventory, dict):
        raise RuntimeError("series manifest is not an object")
    expected_operators = [item["name"] for item in series_inventory.get("series", [])]
    if series_inventory.get("count") != 171 or len(expected_operators) != 171 or \
       len(set(expected_operators)) != 171:
        raise RuntimeError("frozen runtime series manifest is not exactly 171 unique entries")
    forecast = output_forecast(resolutions_by_order, args.phases, args.ranks,
                               args.compare_campaign is not None)
    disk_anchor = args.output
    while not disk_anchor.exists() and disk_anchor != disk_anchor.parent:
        disk_anchor = disk_anchor.parent
    free_bytes = shutil.disk_usage(disk_anchor).free
    preflight = {"schema": SCHEMA, "state": "preflight", **forecast,
                 "free_bytes_before_campaign": free_bytes,
                 "orders": args.orders, "resolutions": args.resolutions,
                 "resolutions_by_order": resolutions_by_order,
                 "phases": args.phases, "ranks": args.ranks,
                 "campaign_mode": "diagnostic_only",
                 "domain_certification": certification,
                 "search_manifest_sha256": sha256(
                     root / "tst/inputs/z4c_cartoon_mms_search_manifest.json"),
                 "series_manifest_sha256": sha256(series_manifest)}
    if free_bytes < 2 * forecast["estimated_output_bytes_upper_bound"]:
        raise RuntimeError("campaign output forecast exceeds half the available space")
    args.output.mkdir(parents=True, exist_ok=True)
    write_atomic(args.output / "preflight.json", preflight)
    cases = [run_case(args, root, source, order, resolution, phase)
             for order in args.orders for resolution in resolutions_by_order[order]
             for phase in args.phases]
    for result in cases:
        if result.get("operator_names") != expected_operators:
            raise RuntimeError(f"{result['case_id']} operator ordering differs from frozen 171-series manifest")
    failures = convergence_gate(cases, args.output, args.output, series_manifest,
                                resolutions_by_order)
    rank_evidence = None
    if args.compare_campaign and not failures:
        rank_evidence = compare_rank_campaigns(
            cases, args.output, args.compare_campaign.resolve())
    convergence_artifacts = {name: sha256(args.output / name) for name in
                             ("convergence.json", "convergence.csv",
                              "convergence_rates.pgfplots.dat",
                              "convergence_plot.tex", "preflight.json")}
    if rank_evidence is not None:
        convergence_artifacts.update(rank_evidence)
    campaign_environment = execution_environment()
    write_atomic(args.output / "campaign.json", {"schema": SCHEMA, "source": source,
                                                  "build_manifest": build,
                                                  "build_manifest_sha256":
                                                  sha256(args.build_manifest),
                                                  "environment": campaign_environment,
                                                  "environment_sha256":
                                                  canonical_digest(campaign_environment),
                                                  "ranks": args.ranks,
                                                  "backend": args.require_backend,
                                                  "diagnostic_resolutions":
                                                  list(DIAGNOSTIC_RESOLUTIONS),
                                                  "reduction_tolerance_factor":
                                                  REDUCTION_TOLERANCE_FACTOR,
                                                  "convergence_artifacts":
                                                  convergence_artifacts,
                                                  "cases": cases})
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
