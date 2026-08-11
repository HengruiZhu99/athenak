#!/usr/bin/env python3
"""Immutable CPU/CUDA MPI campaign driver for the input-selected Cartoon MMS.

This driver intentionally has no configure or build capability.  One already-built
Athena executable is checksum-bound and reused for the complete backend matrix.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
CAMPAIGN_FIELDS = {"schema", "source", "build_manifest", "build_manifest_sha256",
                   "environment", "environment_sha256", "ranks", "backend",
                   "campaign_mode", "accepted_window_sha256",
                   "window_execution_sha256", "diagnostic_resolutions",
                   "reduction_tolerance_factor", "convergence_artifacts", "cases"}
REDUCTION_TOLERANCE_FACTOR = 4096.0
SATURATION_FACTOR = 4096.0
DIAGNOSTIC_RESOLUTIONS = (32, 64, 128, 256)
QUALIFICATION_DOMAIN = (-2.0, 2.0, -2.0, 2.0)
CERTIFIED_COORDINATE_LIMIT = 3.0
EXPECTED_CASES = 3 * 4 * 8
PRESERVED_JOB_56586376_BYTES = 715_807_842
PRESERVED_JOB_56586376_CONVERGENCE_SHA256 = \
    "fdb4222c246b49d4df3c8ef40688dafacfd7983d5f090a2fe148d051538778a0"
PRESERVED_JOB_56586376_CONVERGENCE_BYTES = 372_674_932
PRESERVED_JOB_56586376_CONVERGENCE_NEGATIVE_INFINITY = 2_672
PRESERVED_JOB_56586376_EVIDENCE_SHA256 = \
    "347b210b251e6100413ffef5f691edb684b79a2b41fcb6c8757b8a6233fb1869"
PRESERVED_JOB_56586376_LOG_SHA256 = \
    "47522f4f70ad34d15c994e969f84c6e395dd434bcd5e238fff5f719ef2dd43b1"
PRESERVED_JOB_56586376_AUDIT_COUNTS = (63_880, 18_508, 4_512, 3_696)
PRESERVED_JOB_56586376_MANIFEST_SET_SHA256 = \
    "d2aa1ff2ff9b68302170e0271d3f6fca150d86acb183f9d89af62965427d2aa3"
PRESERVED_JOB_56586376_NORMALIZATION_COUNTS = {
    "derived_cylindrical_applicability": 213_408,
    "derived_radius_applicability": 213_408,
    "derived_target_rho_applicability": 212_448,
    "blanked_inapplicable_target_abs_rho": 180_576,
    "blanked_inapplicable_actual_abs_rho": 180_576,
    "blanked_inapplicable_target_rho": 131_328,
    "blanked_inapplicable_cyl_l1": 98_496,
    "blanked_inapplicable_cyl_l2": 98_496,
    "blanked_inapplicable_cyl_linfinity": 98_496,
}
PRESERVED_JOB_56587561_INVENTORY_SHA256 = \
    "b884d46e636e2beea8ca5eaa7f85d4745968c4a1465c371a70fc8765c4441c6d"
RAW_RESULT_FIELDS = {
    "case_id", "case_uuid", "phase", "resolution", "elapsed_seconds",
    "csv_sha256", "probes_csv_sha256", "operator_names", "rank_bindings",
    "execution_environment", "execution_environment_sha256", "domain",
    "output_bytes",
}
WINDOW_EXECUTION_FIELDS = {
    "schema", "state", "purpose", "attempt_id", "accepted_window_sha256",
    "accepted_review", "source", "build_manifest_sha256", "executable_sha256",
    "input_sha256", "oracle_header_sha256", "series_manifest_sha256", "backend",
    "ranks", "domain", "immutable_root_bindings", "execution_tuples", "attempts",
    "result_set_sha256", "stop_reason", "authorization_files",
}
FINAL_CONVERGENCE_ARTIFACTS = {
    "convergence.json", "convergence.csv", "convergence_rates.pgfplots.dat",
    "convergence_plot.tex", "preflight.json",
}
OPTIONAL_RANK_ARTIFACTS = {
    "rank_comparison.json", "reference_campaign_sha256",
    "reference_case_manifest_sha256",
}
CONVERGENCE_FIELDS = {
    "schema", "series_manifest_sha256", "diagnostic_resolutions_by_order",
    "coefficient_floor_policy_sha256", "coefficient_floor_complexity",
    "evidence_scope", "floor_decompositions", "legacy_normalization_actions",
    "records", "exact_records", "legacy_pre_coefficient_floor_partition",
    "coefficient_floor_partition", "artifacts", "failures", "passed",
}
PREFLIGHT_FIELDS = {
    "schema", "state", "case_count", "norm_rows", "probe_rows", "case_products",
    "aggregate_products", "rank_comparison_products",
    "estimated_output_bytes_upper_bound", "free_bytes_before_campaign", "orders",
    "resolutions", "resolutions_by_order", "phases", "ranks", "campaign_mode",
    "stage_id", "frozen_window_sha256", "run_tuples", "domain_certification",
    "search_manifest_sha256", "series_manifest_sha256",
}
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


def _verify_opaque_legacy_failed_report(
        path: Path, expected_sha256: str, expected_bytes: int,
        expected_negative_infinity: int) -> None:
    if (not path.is_file() or path.is_symlink() or
            path.stat().st_size != expected_bytes or sha256(path) != expected_sha256):
        raise RuntimeError("opaque legacy failed report provenance differs")
    with path.open("rb") as stream:
        negative_infinity = sum(line.count(b"-Infinity") for line in stream)
    if negative_infinity != expected_negative_infinity:
        raise RuntimeError("opaque legacy failed report nonfinite signature differs")


def verify_preserved_job56586376_failed_convergence(path: Path) -> None:
    # This sole immutable failed report is provenance only.  It is never parsed or
    # imported; all raw cases and every current/recomputed product remain strict JSON.
    _verify_opaque_legacy_failed_report(
        path, PRESERVED_JOB_56586376_CONVERGENCE_SHA256,
        PRESERVED_JOB_56586376_CONVERGENCE_BYTES,
        PRESERVED_JOB_56586376_CONVERGENCE_NEGATIVE_INFINITY)


def load_json_strict(path: Path) -> object:
    def reject_constant(token: str) -> None:
        raise ValueError(f"nonfinite JSON token {token}")
    def finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ValueError(f"overflowing JSON number {token}")
        return value
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key}")
            value[key] = item
        return value
    value = json.loads(path.read_text(encoding="utf-8"),
                       parse_constant=reject_constant, parse_float=finite_float,
                       object_pairs_hook=unique_object)
    def reject_nested_nonfinite(item: object) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("nested nonfinite JSON number")
        if isinstance(item, dict):
            for child in item.values():
                reject_nested_nonfinite(child)
        elif isinstance(item, list):
            for child in item:
                reject_nested_nonfinite(child)
    reject_nested_nonfinite(value)
    return value


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


def validate_norm_geometry(row: dict[str, str], order: int, resolution: int,
                           domain: tuple[float, float, float, float]) -> None:
    radial = mask_radial_indices(row["mask"], order, resolution, domain)
    if (integer_value(row, "count") != len(radial) * resolution or
            integer_value(row, "cyl_count") !=
            sum(positive for _, _, positive in radial) * resolution):
        raise RuntimeError("norm row count differs from exact mask geometry")


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
        if (not math.isfinite(sample["error"]) or sample["error"] < 0.0 or
                not math.isfinite(sample["clean_floor"]) or sample["clean_floor"] <= 0.0):
            raise RuntimeError(
                "rate sample error must be finite/nonnegative and floor positive")
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


def evaluate_legacy_rate_samples(values: list[dict[str, float]], expected: float,
                                 margin: float, lane: str) -> dict[str, object]:
    """Frozen pre-coefficient-floor evaluator used for job56586376 only."""
    if any(not math.isfinite(sample["error"]) or sample["error"] < 0.0 or
           not math.isfinite(sample["direct_delta"]) or
           sample["direct_delta"] < 0.0 for sample in values):
        raise RuntimeError("legacy rate errors/deltas must be finite and nonnegative")
    rates: list[float | None] = []
    statuses = []
    usable = []
    for coarse, fine in zip(values, values[1:]):
        floor = SATURATION_FACTOR * sys.float_info.epsilon * \
            max(1.0, coarse["error"])
        if lane != "clean":
            floor = max(floor, 8.0 * max(coarse["direct_delta"],
                                         fine["direct_delta"]))
        if fine["error"] <= floor:
            rates.append(None)
            statuses.append("legacy_saturated")
        elif fine["error"] > 0.0 and coarse["error"] >= fine["error"]:
            rate = math.log(coarse["error"] / fine["error"]) / \
                math.log(fine["resolution"] / coarse["resolution"])
            rates.append(rate)
            statuses.append("legacy_rate")
            usable.append(rate)
        else:
            rates.append(float("-inf"))
            statuses.append("legacy_nonmonotone")
            usable.append(float("-inf"))
    passed = len(usable) >= 2 and min(usable[-2:]) >= expected - margin
    if passed:
        outcome = "pass"
    elif len(usable) < 2:
        outcome = "legacy_inconclusive"
    else:
        outcome = "legacy_rate_miss"
    return {"rates": rates, "rate_status": statuses,
            "usable_prefix_ratios": len(usable), "saturation_absorbing": False,
            "outcome_reason": outcome, "passed": passed}


def output_forecast(resolutions_by_order: dict[int, tuple[int, ...]], phases: list[int],
                    ranks: int, rank_comparison: bool,
                    run_tuples: list[tuple[int, int, int]] | None = None) -> dict[str, int]:
    norm_rows = 0
    probe_rows = 0
    case_count = 0
    order_counts = ({order: len(resolutions) * len(phases)
                     for order, resolutions in resolutions_by_order.items()}
                    if run_tuples is None else
                    {order: sum(item[0] == order for item in run_tuples)
                     for order in resolutions_by_order})
    for order, cases in order_counts.items():
        nghost = order // 2 + 1
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


def require_canonical_digest(value: object, expected: str, label: str) -> None:
    if canonical_digest(value) != expected:
        raise RuntimeError(f"{label} canonical digest changed")


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


def bound_directory_inventory(directory: Path) -> list[dict[str, object]]:
    if not directory.is_dir() or directory.is_symlink():
        raise RuntimeError("evidence directory is missing or a symlink")
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in files):
        raise RuntimeError("evidence directory contains a file symlink")
    return [{"path": str(path.relative_to(directory)), "sha256": sha256(path),
             "bytes": path.stat().st_size} for path in files]


def require_exact_root_entries(directory: Path, expected: set[str], label: str) -> None:
    if {entry.name for entry in directory.iterdir()} != expected:
        raise RuntimeError(f"{label} contains an unclaimed entry")


def require_exact_regular_files(directory: Path, expected: set[str], label: str) -> None:
    if not directory.is_dir() or directory.is_symlink():
        raise RuntimeError(f"{label} is missing or a symlink")
    require_exact_root_entries(directory, expected, label)
    if any(not (directory / name).is_file() or (directory / name).is_symlink()
           for name in expected):
        raise RuntimeError(f"{label} contains a non-regular entry")


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
    root = Path(__file__).resolve().parents[3]
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
    positive_series = [
        {"resolution": resolution, "error": 4.0 ** (-index),
         "direct_delta": 0.0, "clean_floor": 1.0e-12}
        for index, resolution in enumerate((32, 64, 128, 256))]
    zero_first = evaluate_rate_samples(
        [{**sample, "error": 0.0} if index == 0 else sample
         for index, sample in enumerate(positive_series)], 2.0, 0.25, "clean")
    zero_middle = evaluate_rate_samples(
        [{**sample, "error": 0.0} if index == 1 else sample
         for index, sample in enumerate(positive_series)], 2.0, 0.25, "clean")
    zero_fine = evaluate_rate_samples(
        [{**sample, "error": 0.0} if index == 3 else sample
         for index, sample in enumerate(positive_series)], 2.0, 0.25, "clean")
    if (zero_first["rates"] != [None, None, None] or zero_first["passed"] or
            zero_first["outcome_reason"] != "saturated_insufficient" or
            zero_middle["rates"] != [None, None, None] or zero_middle["passed"] or
            zero_middle["outcome_reason"] != "saturated_insufficient" or
            zero_fine["rates"] != [2.0, 2.0, None] or not zero_fine["passed"] or
            zero_fine["rate_status"] != ["included_rate", "included_rate",
                                          "excluded_saturated"] or
            not zero_fine["saturation_absorbing"]):
        raise RuntimeError("coefficient-aware zero-error saturation semantics changed")
    for label, value in (("negative coefficient-aware error", -1.0),
                         ("nonfinite coefficient-aware error", float("inf"))):
        expect_runtime_error(
            lambda error=value: evaluate_rate_samples(
                [{**positive_series[0], "error": error}, *positive_series[1:]],
                2.0, 0.25, "clean"), label)
    legacy_low = [{"resolution": item["resolution"], "error": item["error"],
                   "direct_delta": 0.0, "clean_floor": 1.0e-30}
                  for item in o6_roundoff]
    legacy_high = [{**item, "clean_floor": 1.0e-8} for item in legacy_low]
    legacy_first = evaluate_legacy_rate_samples(legacy_low, 6.0, 0.25, "clean")
    legacy_second = evaluate_legacy_rate_samples(legacy_high, 6.0, 0.25, "clean")
    new_low = evaluate_rate_samples(legacy_low, 6.0, 0.25, "clean")
    new_high = evaluate_rate_samples(legacy_high, 6.0, 0.25, "clean")
    if (legacy_first != legacy_second or
            new_low["rate_status"] == new_high["rate_status"]):
        raise RuntimeError("coefficient floor altered the immutable legacy classifier")
    legacy_zero = [{"resolution": resolution, "error": 0.0,
                    "direct_delta": 0.0} for resolution in (32, 64, 128, 256)]
    zero_result = evaluate_legacy_rate_samples(
        legacy_zero, 2.0, 0.25, "clean")
    if (zero_result["rates"] != [None, None, None] or
            zero_result["rate_status"] != ["legacy_saturated"] * 3 or
            zero_result["outcome_reason"] != "legacy_inconclusive"):
        raise RuntimeError("legacy zero-error series did not saturate interval-locally")
    legacy_reentry = [
        {"resolution": 32, "error": 1.0, "direct_delta": 0.1},
        {"resolution": 64, "error": 0.5, "direct_delta": 0.0},
        {"resolution": 128, "error": 0.25, "direct_delta": 0.0},
        {"resolution": 256, "error": 0.125, "direct_delta": 0.0}]
    reentry = evaluate_legacy_rate_samples(legacy_reentry, 1.0, 0.0, "shared")
    if (reentry["rates"] != [None, 1.0, 1.0] or not reentry["passed"] or
            reentry["rate_status"] != ["legacy_saturated", "legacy_rate",
                                        "legacy_rate"] or
            reentry["saturation_absorbing"]):
        raise RuntimeError("legacy interval-local saturation became absorbing")
    legacy_nonmonotone = [
        {"resolution": 32, "error": 1.0, "direct_delta": 0.0},
        {"resolution": 64, "error": 2.0, "direct_delta": 0.0},
        {"resolution": 128, "error": 1.0, "direct_delta": 0.0},
        {"resolution": 256, "error": 0.5, "direct_delta": 0.0}]
    nonmonotone = evaluate_legacy_rate_samples(
        legacy_nonmonotone, 1.0, 0.0, "clean")
    if (nonmonotone["rates"] != [float("-inf"), 1.0, 1.0] or
            nonmonotone["rate_status"][0] != "legacy_nonmonotone" or
            not nonmonotone["passed"]):
        raise RuntimeError("legacy nonmonotone interval no longer records -inf")
    for label, mutation in (
        ("negative legacy error", {"error": -1.0}),
        ("nonfinite legacy delta", {"direct_delta": float("inf")})):
        expect_runtime_error(
            lambda edit=mutation: evaluate_legacy_rate_samples(
                [{**sample, **edit} if index == 0 else sample
                 for index, sample in enumerate(legacy_zero)], 2.0, 0.25, "clean"),
            label)

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
    radial_fixture = mask_radial_indices("regular_negative", 4, 32,
                                         QUALIFICATION_DOMAIN)
    geometry_row = {**valid_norm,
                    "count": str(len(radial_fixture) * 32), "cyl_count": "0"}
    validate_norm_geometry(geometry_row, 4, 32, QUALIFICATION_DOMAIN)
    expect_runtime_error(
        lambda: validate_norm_geometry({**geometry_row,
                                        "count": str(len(radial_fixture) * 32 + 1)},
                                       4, 32, QUALIFICATION_DOMAIN),
        "coherently rehashed geometry count")
    legacy_norm = {key: value for key, value in valid_norm.items()
                   if key not in {"cylindrical_applicable", "radius_applicable"}}
    legacy_norm.update({"cyl_l1": "nan", "cyl_l2": "nan",
                        "cyl_linfinity": "nan", "target_abs_rho": "nan",
                        "actual_abs_rho": "nan"})
    normalized_norm, _ = normalize_legacy_norm_row(legacy_norm)
    if any(normalized_norm[field] != valid_norm[field] for field in
           ("l1", "l2", "linfinity", "shared_l1", "independent_l1")):
        raise RuntimeError("legacy norm normalization changed numerical evidence")
    zero_cylindrical = {**legacy_norm, "cyl_l1": "0", "cyl_l2": "0",
                        "cyl_linfinity": "0"}
    normalized_zero, zero_actions = normalize_legacy_norm_row(zero_cylindrical)
    if (any(normalized_zero[field] != "" for field in
            ("cyl_l1", "cyl_l2", "cyl_linfinity")) or
            Counter(zero_actions)["blanked_inapplicable_cyl_l1"] != 1 or
            Counter(zero_actions)["blanked_inapplicable_cyl_l2"] != 1 or
            Counter(zero_actions)["blanked_inapplicable_cyl_linfinity"] != 1):
        raise RuntimeError("legacy exact-zero cylindrical norms did not normalize")
    applicable_zero = {**legacy_norm, "cyl_count": "1", "cyl_l1": "0",
                       "cyl_l2": "1", "cyl_linfinity": "1"}
    normalized_applicable, applicable_actions = normalize_legacy_norm_row(
        applicable_zero)
    if (any(normalized_applicable[field] != applicable_zero[field] for field in
            ("cyl_l1", "cyl_l2", "cyl_linfinity")) or
            any(action.startswith("blanked_inapplicable_cyl_")
                for action in applicable_actions)):
        raise RuntimeError("legacy applicable cylindrical zero was altered")
    for label, row in (
        ("legacy inapplicable nonexact zero", {**zero_cylindrical, "cyl_l1": "0.0"}),
        ("legacy inapplicable other token", {**zero_cylindrical, "cyl_l2": "1"})):
        expect_runtime_error(lambda value=row: normalize_legacy_norm_row(value), label)
    audited_counts = Counter(PRESERVED_JOB_56586376_NORMALIZATION_COUNTS)
    validate_preserved_job56586376_normalization_counts(audited_counts)
    for action in PRESERVED_JOB_56586376_NORMALIZATION_COUNTS:
        changed = audited_counts.copy()
        changed[action] += 1
        expect_runtime_error(
            lambda counts=changed:
            validate_preserved_job56586376_normalization_counts(counts),
            f"mutated replay normalization count {action}")
    expect_runtime_error(
        lambda: validate_preserved_job56586376_normalization_counts(
            audited_counts + Counter({"unexpected_action": 1})),
        "unexpected replay normalization action")
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
        overflow = Path(directory) / "overflow.json"
        overflow.write_text('{"nested": {"value": 1e400}}\n', encoding="utf-8")
        expect_runtime_error(lambda: load_json_strict(overflow),
                             "exponent-overflow nested JSON")
        opaque = Path(directory) / "opaque_legacy_failed_convergence.json"
        opaque_bytes = b'{"failed_rate": -Infinity}\n'
        opaque.write_bytes(opaque_bytes)
        opaque_sha256 = hashlib.sha256(opaque_bytes).hexdigest()
        _verify_opaque_legacy_failed_report(
            opaque, opaque_sha256, len(opaque_bytes), 1)
        expect_runtime_error(lambda: load_json_strict(opaque),
                             "opaque legacy report used as current JSON")
        opaque.write_bytes(opaque_bytes + b"tamper")
        expect_runtime_error(
            lambda: _verify_opaque_legacy_failed_report(
                opaque, opaque_sha256, len(opaque_bytes), 1),
            "tampered opaque legacy report")
        opaque.write_bytes(opaque_bytes)
        expect_runtime_error(
            lambda: _verify_opaque_legacy_failed_report(
                opaque, "0" * 64, len(opaque_bytes), 1),
            "wrong opaque legacy report hash")
        expect_runtime_error(
            lambda: _verify_opaque_legacy_failed_report(
                opaque, opaque_sha256, len(opaque_bytes), 0),
            "wrong opaque legacy nonfinite count")
        opaque.unlink()
        expect_runtime_error(
            lambda: _verify_opaque_legacy_failed_report(
                opaque, opaque_sha256, len(opaque_bytes), 1),
            "missing opaque legacy report")

    raw = {"status": "pass", "operator_count": 171}
    augmented = {**raw, **{field: None for field in RAW_RESULT_FIELDS}}
    validate_raw_result_projection(raw, augmented)
    expect_runtime_error(
        lambda: validate_raw_result_projection({**raw, "operator_count": 170},
                                               augmented),
        "coherently rehashed raw/result disagreement")
    expect_runtime_error(
        lambda: validate_raw_result_projection(raw, {**augmented, "extra": 1}),
        "augmented result extra field")
    legacy_result = {key: value for key, value in augmented.items()
                     if key != "domain"}
    legacy_identity = {"domain": list(QUALIFICATION_DOMAIN)}
    legacy_normalizations = [normalize_preserved_job56586376_result(
        raw, legacy_result, legacy_identity) for _ in range(EXPECTED_CASES)]
    if (len(legacy_normalizations) != EXPECTED_CASES or
            any(result.get("domain") != list(QUALIFICATION_DOMAIN) or
                action != "derived_domain_from_authenticated_manifest_identity"
                for result, action in legacy_normalizations)):
        raise RuntimeError("96-case legacy-domain fixture did not normalize exactly")
    expect_runtime_error(
        lambda: validate_raw_result_projection(raw, legacy_result),
        "legacy replay shape used on current projection path")
    for label, stored, identity in (
        ("existing legacy domain", augmented, legacy_identity),
        ("conflicting legacy domain",
         {**legacy_result, "domain": [-1.0, 1.0, -2.0, 2.0]}, legacy_identity),
        ("missing authenticated identity domain", legacy_result, {}),
        ("noncanonical authenticated identity domain", legacy_result,
         {"domain": [-1.0, 1.0, -2.0, 2.0]}),
        ("other missing legacy result key",
         {key: value for key, value in legacy_result.items() if key != "case_id"},
         legacy_identity),
        ("extra legacy result key", {**legacy_result, "extra": None}, legacy_identity),
        ("differing legacy raw value",
         {**legacy_result, "operator_count": 170}, legacy_identity)):
        expect_runtime_error(
            lambda result=stored, manifest=identity:
            normalize_preserved_job56586376_result(raw, result, manifest), label)
    manifest_fixture = [{"path": "a/manifest.json", "sha256": "0" * 64}]
    fixture_digest = canonical_digest(manifest_fixture)
    require_canonical_digest(manifest_fixture, fixture_digest, "fixture manifest")
    expect_runtime_error(
        lambda: require_canonical_digest(
            [{"path": "a/manifest.json", "sha256": "1" * 64}],
            fixture_digest, "mutated fixture manifest"),
        "coherently rehashed case manifest")

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
    semantic_result = {
        **valid_result, "status": "pass", "spatial_order": 2, "nghost": 2,
        "nx1": 32, "nx2": 32, "nx3": 1, "mpi_ranks": 2,
        "owned_cells": 1024, "operator_count": 171,
        "diagnostic_axis_operator_count": 161, "backend": "Serial",
        "phase": 0, "resolution": 32, "domain": list(QUALIFICATION_DOMAIN),
        "ownership_sequence": "[0,N*N) exactly once",
        "diagnostic_axis_tolerance": 1.0}
    validate_raw_case_invariants(semantic_result, 2, 32, 0, "Serial", 2,
                                 QUALIFICATION_DOMAIN)
    for label, field, value in (
        ("raw status", "status", "fail"), ("raw nghost", "nghost", 3),
        ("raw nx1", "nx1", 31), ("raw nx2", "nx2", 31),
        ("raw nx3", "nx3", 2), ("raw ranks", "mpi_ranks", 4),
        ("raw operator count", "operator_count", 170),
        ("raw nonfinite count", "nonfinite_count", 1),
        ("raw axis count", "diagnostic_axis_operator_count", 160),
        ("raw axis nonfinite", "diagnostic_axis_nonfinite", 1),
        ("raw ownership count", "owned_cells", 1023),
        ("raw ownership sequence", "ownership_sequence", "duplicated"),
        ("stored backend", "backend", "Cuda"),
        ("stored ranks", "mpi_ranks", 4),
        ("stored domain", "domain", [-1.0, 1.0, -2.0, 2.0])):
        expect_runtime_error(
            lambda key=field, replacement=value: validate_raw_case_invariants(
                {**semantic_result, key: replacement}, 2, 32, 0, "Serial", 2,
                QUALIFICATION_DOMAIN), label)
    summary_operators = [f"operator.{index}" for index in range(171)]
    summary_rows = [{"operator": name, "nonfinite": "0",
                     "shared_delta_linfinity": "0",
                     "independent_delta_linfinity": "0"}
                    for name in summary_operators]
    summary_probes = [
        {"operator": name, "mask": "diagnostic_axis", "side": "axis",
         "classification": "diagnostic_axis", "layer_index": "0",
         "raw_error": "0"} for name in summary_operators[:161]]
    validate_case_numerical_summary(semantic_result, summary_operators,
                                    summary_rows, summary_probes)
    expect_runtime_error(
        lambda: validate_case_numerical_summary(
            semantic_result, summary_operators,
            [{**summary_rows[0], "shared_delta_linfinity": "2"}] + summary_rows[1:],
            summary_probes), "noise-bound mutation")
    expect_runtime_error(
        lambda: validate_case_numerical_summary(
            {**semantic_result, "maximum_noise_delta": 2.0}, summary_operators,
            summary_rows, summary_probes), "noise-summary mutation")
    expect_runtime_error(
        lambda: validate_case_numerical_summary(
            semantic_result, summary_operators, summary_rows, summary_probes[:-1]),
        "axis-count mutation")
    expect_runtime_error(
        lambda: validate_case_numerical_summary(
            {**semantic_result, "diagnostic_axis_tolerance": 0.0},
            summary_operators, summary_rows,
            [{**summary_probes[0], "raw_error": "1"}] + summary_probes[1:]),
        "axis-tolerance mutation")
    with tempfile.TemporaryDirectory(prefix="cartoon-mms-resume-") as directory:
        raw_path = Path(directory) / "raw.json"
        result_path = Path(directory) / "result.json"
        raw_resume = {**valid_result, "status": "pass", "mpi_ranks": 1,
                      "operator_count": 171}
        environment = {"SLURM_NTASKS": "1"}
        binding = {"rank": 0, "local_rank": 0, "hostname": "fixture",
                   "cuda_visible_devices": None, "visible_device_token": None,
                   "selected_uuid": None, "gpu_name": None,
                   "binding_verified": True}
        resumed = {**raw_resume, "case_id": "fixture", "case_uuid": str(uuid.uuid4()),
                   "phase": 0, "resolution": 32, "elapsed_seconds": 0.0,
                   "csv_sha256": "1" * 64, "probes_csv_sha256": "2" * 64,
                   "operator_names": [f"operator.{index}" for index in range(171)],
                   "rank_bindings": [binding], "execution_environment": environment,
                   "execution_environment_sha256": canonical_digest(environment),
                   "domain": list(QUALIFICATION_DOMAIN), "output_bytes": 0}
        write_atomic(raw_path, raw_resume)
        write_atomic(result_path, resumed)
        load_augmented_result(raw_path, result_path)
        write_atomic(result_path, {**resumed, "status": "fail"})
        expect_runtime_error(lambda: load_augmented_result(raw_path, result_path),
                             "coherently rehashed resumed raw/result disagreement")
    with tempfile.TemporaryDirectory(prefix="cartoon-mms-case-integrity-") as directory:
        case = Path(directory) / "case"
        case.mkdir()
        identity = {"ranks": 1}
        for name in expected_case_files(1):
            (case / name).write_text(f"{name}\n", encoding="utf-8")
        files = {name: sha256(case / name) for name in expected_case_files(1)}
        write_atomic(case / "manifest.json", {
            "schema": SCHEMA, "state": "complete", "identity": identity,
            "files": files})
        if not verified_complete(case, identity):
            raise RuntimeError("case-integrity fixture is not initially complete")
        for name in ("cartoon_mms.mms.json", "result.json",
                     "cartoon_mms.mms.csv", "cartoon_mms.mms.probes.csv"):
            original = (case / name).read_bytes()
            (case / name).write_bytes(original + b"mutated")
            if verified_complete(case, identity):
                raise RuntimeError(f"case-integrity fixture accepted mutated {name}")
            (case / name).write_bytes(original)
        extra = case / "unexpected"
        extra.write_text("extra\n", encoding="utf-8")
        if verified_complete(case, identity):
            raise RuntimeError("case-integrity fixture accepted an extra entry")
        extra.unlink()
        manifest_path = case / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        mutated_manifest = load_json_strict(manifest_path)
        mutated_manifest["state"] = "running"
        write_atomic(manifest_path, mutated_manifest)
        if verified_complete(case, identity):
            raise RuntimeError("case-integrity fixture accepted a mutated manifest")
        manifest_path.write_bytes(manifest_bytes)
        manifest_copy = Path(directory) / "manifest-copy.json"
        manifest_copy.write_bytes(manifest_path.read_bytes())
        manifest_path.unlink()
        manifest_path.symlink_to(manifest_copy)
        if verified_complete(case, identity):
            raise RuntimeError("case-integrity fixture accepted a manifest symlink")
        reference = Path(directory) / "reference"
        reference.mkdir()
        reference_link = Path(directory) / "reference-link"
        reference_link.symlink_to(reference.name, target_is_directory=True)
        expect_runtime_error(
            lambda: verify_rank_reference_root(
                reference_link, 4, "accepted_frozen_window", "0" * 64,
                {"commit": "0" * 40, "tree": "1" * 40, "kokkos": "2" * 40},
                "Serial", "3" * 64), "rank-reference root symlink")
    with tempfile.TemporaryDirectory(prefix="cartoon-mms-launch-provenance-") as directory:
        case = Path(directory) / "case"
        case.mkdir()
        executable = Path(directory) / "athena"
        executable.write_text("fixture executable\n", encoding="utf-8")
        wrapper = root / "tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py"
        template = root / "tst/inputs/z4c_cartoon_derivatives.athinput"
        rendered = render_input(template, 2, 32, 0, "cartoon_mms",
                                QUALIFICATION_DOMAIN)
        input_path = case / "input.athinput"
        input_path.write_text(rendered, encoding="utf-8")
        bindings = [
            {"rank": rank, "local_rank": rank, "hostname": "fixture",
             "cuda_visible_devices": None, "visible_device_token": None,
             "selected_uuid": None, "gpu_name": None, "binding_verified": True}
            for rank in range(2)]
        for rank, binding in enumerate(bindings):
            write_atomic(case / f"rank_binding_{rank:04d}.json", binding)
        command = ["srun", "--ntasks", "2", sys.executable, str(wrapper),
                   "--evidence-dir", ".", "--", str(executable), "-i",
                   "input.athinput"]
        identity = {"input_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                    "rank_wrapper_sha256": sha256(wrapper), "command": command}
        stored = {"rank_bindings": bindings}
        validate_case_launch_provenance(
            case, identity, stored, sha256(executable), "Serial", 2,
            QUALIFICATION_DOMAIN, 2, 32, 0, template, wrapper)
        wrong_input = rendered + "# mutation\n"
        input_path.write_text(wrong_input, encoding="utf-8")
        expect_runtime_error(
            lambda: validate_case_launch_provenance(
                case, {**identity, "input_sha256":
                       hashlib.sha256(wrong_input.encode()).hexdigest()},
                stored, sha256(executable), "Serial", 2,
                QUALIFICATION_DOMAIN, 2, 32, 0, template, wrapper),
            "coherently rehashed wrong input")
        input_path.write_text(rendered, encoding="utf-8")
        for label, mutation in (
            ("wrapper digest mutation", {"rank_wrapper_sha256": "0" * 64}),
            ("command mutation", {"command": command[:-1] + ["wrong.input"]})):
            expect_runtime_error(
                lambda edit=mutation: validate_case_launch_provenance(
                    case, {**identity, **edit}, stored, sha256(executable),
                    "Serial", 2, QUALIFICATION_DOMAIN, 2, 32, 0, template,
                    wrapper), label)
        write_atomic(case / "rank_binding_0001.json",
                     {**bindings[1], "hostname": "mutated"})
        expect_runtime_error(
            lambda: validate_case_launch_provenance(
                case, identity, stored, sha256(executable), "Serial", 2,
                QUALIFICATION_DOMAIN, 2, 32, 0, template, wrapper),
            "archived rank-binding mutation")

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
    partition_names = ("norm_inconclusive", "norm_rate_miss",
                       "probe_inconclusive", "probe_rate_miss")
    frozen_partition = dict(zip(
        partition_names, PRESERVED_JOB_56586376_AUDIT_COUNTS))
    validate_preserved_job56586376_legacy_partition(frozen_partition)
    for name in partition_names:
        expect_runtime_error(
            lambda field=name: validate_preserved_job56586376_legacy_partition(
                {**frozen_partition, field: frozen_partition[field] + 1}),
            f"mutated frozen legacy partition {name}")

    root = Path(__file__).resolve().parents[3]
    search = load_search_manifest(root / "tst/inputs/z4c_cartoon_mms_search_manifest.json")
    fake_source = {"commit": "a" * 40, "tree": "b" * 40, "kokkos": "c" * 40}
    fake_roots = [{"name": "job56586376", "root": "/immutable/a",
                   "binding_sha256": "d" * 64},
                  {"name": "job56587561", "root": "/immutable/b",
                   "binding_sha256": "e" * 64}]
    materialized = materialize_search_stage(
        search, "o6_phase0_stage1", fake_source, "f" * 64, "1" * 64,
        "2" * 64, "3" * 64, "4" * 64, "Serial", 2, fake_roots)
    if (materialized["state"] != "prelaunch_bound" or
            materialized["materialization"]["stage_id"] != "o6_phase0_stage1"):
        raise RuntimeError("search stage did not materialize exact prelaunch identity")
    expect_runtime_error(
        lambda: materialize_search_stage(search, "o2_phase0_continuation",
                                         fake_source, "f" * 64, "1" * 64,
                                         "2" * 64, "3" * 64, "4" * 64,
                                         "Serial", 2, fake_roots),
        "blocked O2 OOM continuation")
    expect_runtime_error(
        lambda: materialize_search_stage(search, "o6_phase0_stage1",
                                         fake_source, "f" * 64, "1" * 64,
                                         "2" * 64, "3" * 64, "4" * 64,
                                         "Serial", 2, fake_roots[:1]),
        "incomplete immutable-root merge")
    lifecycle_attempts = [
        {"tuple": item, "status": "complete",
         "case_manifest_sha256": str(index + 5) * 64,
         "reason": "case_complete"}
        for index, item in enumerate([[6, 48, 0], [6, 80, 0], [6, 96, 0]])]
    transition_search_stage(materialized, "stage_finalized", lifecycle_attempts,
                            "stage_complete_pending_numerical_review")
    transition_search_stage(materialized, "analysis_finalized", lifecycle_attempts,
                            "numerical_failures_retained", "6" * 64)
    if materialized["materialization"]["analysis_sha256"] != "6" * 64:
        raise RuntimeError("search postrun lifecycle did not bind analysis")
    authorized = [[6, 48, 0], [6, 80, 0], [6, 96, 0]]
    for failed_index in range(3):
        partial = materialize_search_stage(
            search, "o6_phase0_stage1", fake_source, "f" * 64, "1" * 64,
            "2" * 64, "3" * 64, "4" * 64, "Serial", 2, fake_roots)
        partial_attempts = []
        for index, item in enumerate(authorized):
            status = ("complete" if index < failed_index else
                      "failed" if index == failed_index else
                      "not_attempted_after_stop")
            partial_attempts.append({
                "tuple": item, "status": status,
                "case_manifest_sha256": (str(index + 1) * 64
                                         if status == "complete" else None),
                "reason": status})
        transition_search_stage(partial, "stage_partial", partial_attempts,
                                "integrity_or_resource_failure")
        if stage_missing_tuples(partial) != [tuple(item)
                                             for item in authorized[failed_index:]]:
            raise RuntimeError("partial stage continuation would rerun completed work")
    chain_bindings = list(fake_roots)
    copied_cases = []
    partial_bindings = []
    for retry_index, failure_index in enumerate((1, 2)):
        retry = materialize_search_stage(
            search, "o6_phase0_stage1", fake_source, "f" * 64, "1" * 64,
            "2" * 64, "3" * 64, "4" * 64, "Serial", 2, chain_bindings)
        retry_attempts = []
        retry_cases = []
        for index, item in enumerate(authorized):
            status = ("complete" if index < failure_index else
                      "failed" if index == failure_index else
                      "not_attempted_after_stop")
            digest = str(index + 1) * 64 if status == "complete" else None
            retry_attempts.append({"tuple": item, "status": status,
                                   "case_manifest_sha256": digest,
                                   "reason": status})
            if status == "complete":
                retry_cases.append({"spatial_order": item[0],
                                    "resolution": item[1], "phase": item[2],
                                    "case_manifest_sha256": digest})
        transition_search_stage(retry, "stage_partial", retry_attempts,
                                "integrity_or_resource_failure")
        expected_missing = [tuple(item) for item in authorized[failure_index:]]
        if stage_missing_tuples(retry) != expected_missing:
            raise RuntimeError("repeated partial stage did not retain exact suffix")
        copied_cases = merge_case_ledgers(copied_cases, retry_cases)
        fake_binding = {"root": f"/immutable/partial-{retry_index}",
                        "stage_id": "o6_phase0_stage1", "state": "stage_partial",
                        "stage_campaign_sha256": str(retry_index + 7) * 64}
        partial_binding = stage_binding_record(fake_binding)
        partial_bindings.append(partial_binding)
        chain_bindings.append(partial_binding)
    complete_attempts = [
        {"tuple": item, "status": "complete",
         "case_manifest_sha256": str(index + 1) * 64,
         "reason": "case_complete"} for index, item in enumerate(authorized)]
    complete_cases = [
        {"spatial_order": item[0], "resolution": item[1], "phase": item[2],
         "case_manifest_sha256": str(index + 1) * 64}
        for index, item in enumerate(authorized)]
    single_retry = materialize_search_stage(
        search, "o6_phase0_stage1", fake_source, "f" * 64, "1" * 64,
        "2" * 64, "3" * 64, "4" * 64, "Serial", 2,
        list(fake_roots) + partial_bindings[:1])
    transition_search_stage(single_retry, "stage_finalized", complete_attempts,
                            "stage_complete_pending_offline_analysis")
    if (stage_missing_tuples(single_retry) or
            len(merge_case_ledgers(complete_cases[:1], complete_cases)) != 3):
        raise RuntimeError("partial1-success continuation is not composable")
    final_retry = materialize_search_stage(
        search, "o6_phase0_stage1", fake_source, "f" * 64, "1" * 64,
        "2" * 64, "3" * 64, "4" * 64, "Serial", 2, chain_bindings)
    transition_search_stage(final_retry, "stage_finalized", complete_attempts,
                            "stage_complete_pending_offline_analysis")
    if stage_missing_tuples(retry) != [(6, 96, 0)] or len(copied_cases) != 2 or \
       stage_missing_tuples(final_retry) or \
       len(merge_case_ledgers(copied_cases, complete_cases)) != 3 or \
       final_retry["materialization"]["immutable_root_bindings"] != chain_bindings:
        raise RuntimeError("partial1-partial2-final dependency chain is not composable")
    authorized_set = {tuple(item) for item in authorized}
    first_cases, second_cases = complete_cases[:1], complete_cases[:2]
    first_missing = {tuple(item) for item in authorized[1:]}
    second_missing = {tuple(authorized[2])}
    validate_stage_lineage([], authorized_set, first_cases, first_missing,
                           authorized_set)
    validate_stage_lineage(first_cases, first_missing, second_cases,
                           second_missing, authorized_set)
    validate_stage_lineage(first_cases, first_missing, first_cases,
                           first_missing, authorized_set)
    for label, later, missing_set in (
        ("partial completed regression", second_cases[1:],
         {tuple(authorized[0]), tuple(authorized[2])}),
        ("partial conflicting manifest",
         [{**first_cases[0], "case_manifest_sha256": "9" * 64}, second_cases[1]],
         second_missing),
        ("partial incorrect missing set", second_cases, first_missing)):
        expect_runtime_error(
            lambda cases=later, missing=missing_set: validate_stage_lineage(
                first_cases, first_missing, cases, missing, authorized_set), label)
    with tempfile.TemporaryDirectory(prefix="cartoon-mms-stage-header-") as directory:
        stage_fixture = materialize_search_stage(
            search, "o6_phase0_stage1", fake_source, "f" * 64, "1" * 64,
            sha256(root / "tst/inputs/z4c_cartoon_derivatives.athinput"),
            sha256(root / "src/pgen/unit_tests/z4c_cartoon_derivatives_oracle.hpp"),
            sha256(root / "tst/unit/z4c/z4c_cartoon_derivatives_series.json"),
            "Serial", 2, fake_roots)
        transition_search_stage(stage_fixture, "stage_finalized", lifecycle_attempts,
                                "stage_complete_pending_offline_analysis")
        stage_path = Path(directory) / "search_stage_manifest.json"
        write_atomic(stage_path, stage_fixture)
        campaign_fixture = stage_campaign_record(stage_fixture, stage_path, [], [])
        validate_stage_identity(stage_fixture, campaign_fixture, search, fake_roots)
        for label, field, value in (
            ("stage source", "source", {**fake_source, "commit": "0" * 40}),
            ("stage executable", "executable_sha256", "0" * 64),
            ("stage backend", "backend", "Cuda"),
            ("stage ranks", "ranks", 4),
            ("stage old-root binding", "immutable_root_bindings", fake_roots[:1])):
            expect_runtime_error(
                lambda key=field, replacement=value: validate_stage_identity(
                    stage_fixture, {**campaign_fixture, key: replacement},
                    search, fake_roots), label)
        input_mutation = json.loads(json.dumps(stage_fixture))
        input_mutation["materialization"]["input_sha256"] = "0" * 64
        expect_runtime_error(
            lambda: validate_stage_identity(
                input_mutation, {**campaign_fixture, "input_sha256": "0" * 64},
                search, fake_roots), "stage input")
        expected_entries = {"search_stage_manifest.json"}
        require_exact_root_entries(Path(directory), expected_entries,
                                   "stage fixture root")
        (Path(directory) / "unexpected").write_text("unclaimed", encoding="utf-8")
        expect_runtime_error(
            lambda: require_exact_root_entries(
                Path(directory), expected_entries, "stage fixture root"),
            "stage extra root entry")
    rank_campaign = {field: None for field in CAMPAIGN_FIELDS}
    rank_campaign.update({"schema": SCHEMA, "source": fake_source,
                          "build_manifest_sha256": "f" * 64,
                          "ranks": 2, "backend": "Serial",
                          "campaign_mode": "accepted_frozen_window",
                          "accepted_window_sha256": "7" * 64, "cases": []})
    validate_rank_campaign_header(rank_campaign, 4, "accepted_frozen_window",
                                  "7" * 64, fake_source, "Serial", "f" * 64)
    for label, mutation in (
        ("rank campaign purpose", {"campaign_mode": "diagnostic_only"}),
        ("rank campaign window", {"accepted_window_sha256": "8" * 64}),
        ("rank campaign schema", {"unexpected": None})):
        expect_runtime_error(
            lambda value={**rank_campaign, **mutation}: validate_rank_campaign_header(
                value, 4, "accepted_frozen_window", "7" * 64, fake_source,
                "Serial", "f" * 64), label)
    aggregate_artifacts = {name: str(index + 1) * 64 for index, name in
                           enumerate(sorted(FINAL_CONVERGENCE_ARTIFACTS))}
    internal_artifacts = {name: aggregate_artifacts[name] for name in
                          ("convergence.csv", "convergence_rates.pgfplots.dat",
                           "convergence_plot.tex")}
    aggregate_execution = {"series_manifest_sha256": "4" * 64,
                           "domain": list(QUALIFICATION_DOMAIN)}
    aggregate_resolutions = (32, 64, 128, 256)
    aggregate_required = [(2, resolution, 0)
                          for resolution in aggregate_resolutions]
    series_inventory = load_json_strict(
        root / "tst/unit/z4c/z4c_cartoon_derivatives_series.json")
    aggregate_policy = series_inventory["roundoff_policy"]
    aggregate_policy_sha = canonical_digest(aggregate_policy)
    aggregate_item = next(item for item in series_inventory["series"]
                          if item["name"] == "scalar.first.0")
    aggregate_record_keys = {
        ("norm", 2, 0, "scalar.first.0|full_signed_plane", "clean", norm)
        for norm in ("l1", "l2")}
    aggregate_exact_keys = {
        ("norm", 2, 0, resolution, "scalar.first.2", "full_signed_plane",
         None, None) for resolution in aggregate_resolutions}
    aggregate_records = []
    aggregate_floors = []
    for norm in ("l1", "l2"):
        samples = []
        for resolution in aggregate_resolutions:
            floor = aggregate_clean_floor(
                aggregate_item, aggregate_policy, 2, resolution,
                "full_signed_plane", norm, QUALIFICATION_DOMAIN)
            floor_id = canonical_digest((
                "scalar.first.0", 2, resolution, "full_signed_plane", norm,
                QUALIFICATION_DOMAIN))
            samples.append({
                "resolution": resolution, "error": (32.0 / resolution) ** 2,
                "direct_delta": 0.0, "clean_floor": floor["clean_floor"],
                "floor_id": floor_id, "evidence_source": "current_output"})
            aggregate_floors.append({
                "floor_id": floor_id, "operator": "scalar.first.0",
                "order": 2, "resolution": resolution,
                "mask": "full_signed_plane", "norm": norm,
                "domain": list(QUALIFICATION_DOMAIN), **floor})
        expected, margin = rate_policy(
            2, "full_signed_plane", "clean", norm, None)
        record = {
            "source": "norm", "order": 2, "phase": 0,
            "series": "scalar.first.0|full_signed_plane", "lane": "clean",
            "norm": norm, "expected": expected, "margin": margin,
            "diagnostic_resolutions": list(aggregate_resolutions),
            "samples": samples,
            "legacy_evaluation": evaluate_legacy_rate_samples(
                samples, expected, margin, "clean")}
        record.update(evaluate_rate_samples(samples, expected, margin, "clean"))
        aggregate_records.append(record)
    aggregate_exact_records = [{
        "source": "norm", "order": 2, "phase": 0,
        "resolution": resolution, "operator": "scalar.first.2",
        "mask": "full_signed_plane", "classification": "exact_identity",
        "value": 0.0,
        "bound": SATURATION_FACTOR * sys.float_info.epsilon * resolution ** 2,
        "direct_delta_linfinity": {"shared": 0.0},
        "direct_delta_bound": 0.0, "passed": True}
        for resolution in aggregate_resolutions]
    aggregate_partition = {
        "norm_inconclusive": 0, "norm_rate_miss": 0,
        "probe_inconclusive": 0, "probe_rate_miss": 0}
    aggregate_convergence = {field: None for field in CONVERGENCE_FIELDS}
    aggregate_convergence.update({
        "schema": SCHEMA, "series_manifest_sha256": "4" * 64,
        "diagnostic_resolutions_by_order": {"2": list(aggregate_resolutions)},
        "coefficient_floor_policy_sha256": aggregate_policy_sha,
        "coefficient_floor_complexity":
            "O(171*nx1); active-z multiplicity analytic",
        "legacy_normalization_actions": [],
        "evidence_scope": "fresh_single_source_final_qualification",
        "artifacts": internal_artifacts, "failures": [], "passed": True,
        "records": aggregate_records, "exact_records": aggregate_exact_records,
        "floor_decompositions": sorted(
            aggregate_floors, key=lambda item: item["floor_id"]),
        "legacy_pre_coefficient_floor_partition": aggregate_partition,
        "coefficient_floor_partition": aggregate_partition})
    aggregate_forecast = output_forecast(
        {2: aggregate_resolutions}, [0], 2, False, aggregate_required)
    aggregate_preflight = {field: 0 for field in PREFLIGHT_FIELDS}
    aggregate_preflight.update({
        "schema": SCHEMA, "state": "preflight", "orders": [2],
        "resolutions": list(aggregate_resolutions),
        "resolutions_by_order": {"2": list(aggregate_resolutions)},
        "phases": [0], "ranks": 2, "campaign_mode": "accepted_frozen_window",
        "stage_id": None, "frozen_window_sha256": "7" * 64,
        "run_tuples": [list(item) for item in aggregate_required],
        "domain_certification": validate_certified_domain(
            QUALIFICATION_DOMAIN, {2: aggregate_resolutions}, True),
        "series_manifest_sha256": "4" * 64,
        "search_manifest_sha256": sha256(
            root / "tst/inputs/z4c_cartoon_mms_search_manifest.json"),
        "free_bytes_before_campaign":
            2 * aggregate_forecast["estimated_output_bytes_upper_bound"],
        **aggregate_forecast})
    if reference_artifact_files(aggregate_artifacts) != FINAL_CONVERGENCE_ARTIFACTS:
        raise RuntimeError("current writer/reference artifact schema differs")
    validate_final_reference_aggregates(
        aggregate_convergence, aggregate_preflight, aggregate_artifacts,
        aggregate_execution, 2, "7" * 64, aggregate_required,
        aggregate_record_keys, aggregate_exact_keys, aggregate_policy_sha, False)
    for label, mutation in (
        ("failed reference convergence", {"failures": ["rate miss"],
                                           "passed": False}),
        ("diagnostic reference scope", {"evidence_scope":
                                         "single_source_diagnostic"}),
        ("altered convergence artifact hash",
         {"artifacts": {**internal_artifacts,
                         "convergence.csv": "0" * 64}})):
        expect_runtime_error(
            lambda edit=mutation: validate_final_reference_aggregates(
                {**aggregate_convergence, **edit}, aggregate_preflight,
                aggregate_artifacts, aggregate_execution, 2, "7" * 64,
                aggregate_required, aggregate_record_keys, aggregate_exact_keys,
                aggregate_policy_sha, False), label)
    for label, mutation in (
        ("empty convergence inventory", {"records": []}),
        ("truncated convergence inventory", {"records": aggregate_records[:1]}),
        ("duplicate convergence inventory",
         {"records": [aggregate_records[0], aggregate_records[0]]}),
        ("empty exact inventory", {"exact_records": []}),
        ("inconsistent convergence partition",
         {"coefficient_floor_partition":
          {**aggregate_partition, "norm_rate_miss": 1}}),
        ("wrong coefficient policy",
         {"coefficient_floor_policy_sha256": "0" * 64})):
        expect_runtime_error(
            lambda edit=mutation: validate_final_reference_aggregates(
                {**aggregate_convergence, **edit}, aggregate_preflight,
                aggregate_artifacts, aggregate_execution, 2, "7" * 64,
                aggregate_required, aggregate_record_keys, aggregate_exact_keys,
                aggregate_policy_sha, False), label)
    expect_runtime_error(
        lambda: validate_final_reference_aggregates(
            aggregate_convergence,
            {**aggregate_preflight, "run_tuples": []}, aggregate_artifacts,
            aggregate_execution, 2, "7" * 64, aggregate_required,
            aggregate_record_keys, aggregate_exact_keys, aggregate_policy_sha,
            False),
        "mutated final preflight")
    for label, mutation in (
        ("zeroed final forecast", {"estimated_output_bytes_upper_bound": 0}),
        ("wrong final search manifest", {"search_manifest_sha256": "0" * 64})):
        expect_runtime_error(
            lambda edit=mutation: validate_final_reference_aggregates(
                aggregate_convergence, {**aggregate_preflight, **edit},
                aggregate_artifacts, aggregate_execution, 2, "7" * 64,
                aggregate_required, aggregate_record_keys, aggregate_exact_keys,
                aggregate_policy_sha, False), label)
    expect_runtime_error(
        lambda: reference_artifact_files(
            {key: value for key, value in aggregate_artifacts.items()
             if key != "convergence.json"}), "missing convergence artifact")
    expect_runtime_error(
        lambda: reference_artifact_files(
            {**aggregate_artifacts, "unexpected.json": "0" * 64}),
        "unexpected convergence artifact")
    with tempfile.TemporaryDirectory(prefix="cartoon-mms-aggregate-products-") \
            as directory:
        archived = Path(directory) / "archived"
        recomputed = Path(directory) / "recomputed"
        archived.mkdir()
        recomputed.mkdir()
        for product_root in (archived, recomputed):
            write_atomic(product_root / "convergence.json", aggregate_convergence)
            for name in RECOMPUTED_CONVERGENCE_PRODUCTS - {"convergence.json"}:
                (product_root / name).write_text(
                    f"deterministic writer product {name}\n", encoding="utf-8")
        require_recomputed_reference_products(archived, recomputed)
        product_mutations = (
            ("sample error", lambda value: value["records"][0]["samples"][0].
             __setitem__("error", 2.0)),
            ("sample direct delta", lambda value: value["records"][0]["samples"][0].
             __setitem__("direct_delta", 1.0)),
            ("sample applied floor", lambda value: value["records"][0]["samples"][0].
             __setitem__("applied_floor", 1.0)),
            ("record rate", lambda value: value["records"][0]["rates"].
             __setitem__(0, 0.0)),
            ("record rate status", lambda value: value["records"][0]["rate_status"].
             __setitem__(0, "excluded_saturated")),
            ("exact bound", lambda value: value["exact_records"][0].
             __setitem__("bound", 0.0)),
            ("exact value", lambda value: value["exact_records"][0].
             __setitem__("value", 1.0)),
            ("exact direct delta", lambda value: value["exact_records"][0][
             "direct_delta_linfinity"].__setitem__("shared", 1.0)),
            ("floor decomposition", lambda value: value["floor_decompositions"][0].
             __setitem__("clean_floor", 0.0)),
            ("nondictionary floor", lambda value: value["floor_decompositions"].
             append("malformed")),
        )
        for label, mutate in product_mutations:
            changed = json.loads(json.dumps(aggregate_convergence))
            mutate(changed)
            write_atomic(archived / "convergence.json", changed)
            expect_runtime_error(
                lambda: require_recomputed_reference_products(
                    archived, recomputed), label)
        write_atomic(archived / "convergence.json", aggregate_convergence)
        require_recomputed_reference_products(archived, recomputed)
    prior_tuples = {(order, resolution, phase) for order in (2, 4, 6)
                    for resolution in DIAGNOSTIC_RESOLUTIONS for phase in range(8)}
    prior_tuples.update({(2, 512, 0), (2, 1024, 0)})
    missing = [(6, 48, phase) for phase in range(8)]
    with tempfile.TemporaryDirectory(prefix="cartoon-mms-window-") as directory:
        authorization = Path(directory) / "authorization"
        authorization.mkdir()
        selected = [[6, 48]]
        qualification_window = {"6": [48]}
        candidate = {
            "schema": "athenak_z4c_cartoon_mms_window_v1",
            "state": "candidate_resolved", "purpose": "characterization_completion",
            "qualification_window": qualification_window,
            "selected_extra_points": selected,
            "execution_tuples": [list(item) for item in missing],
            "source_policy_commit": fake_source["commit"],
            "source_policy_tree": fake_source["tree"],
            "search_template_sha256": sha256(
                root / "tst/inputs/z4c_cartoon_mms_search_manifest.json"),
            "series_manifest_sha256": "4" * 64,
            "immutable_root_bindings": fake_roots,
            "observed_tuples_sha256": "9" * 64,
            "resolutions_by_order": {"6": [32, 48, 64]},
            "domain_certification": {"domain": list(QUALIFICATION_DOMAIN)},
            "convergence_sha256": "a" * 64, "numerical_failures": [],
            "qualification_claim": None, "backend_intent": "Serial",
            "ranks_intent": [2, 4]}
        candidate_path = authorization / "candidate_manifest.json"
        write_atomic(candidate_path, candidate)
        review_artifact = {
            "schema": "athenak_z4c_cartoon_mms_window_review_v1",
            "disposition": "ACCEPT", "candidate_manifest_sha256": sha256(candidate_path),
            "source_policy_commit": fake_source["commit"],
            "source_policy_tree": fake_source["tree"],
            "purpose": "characterization_completion",
            "selected_extra_points": selected,
            "qualification_window": qualification_window,
            "backend": "Serial", "ranks": [2, 4]}
        review_path = authorization / "review_artifact.json"
        write_atomic(review_path, review_artifact)
        window = {
            "schema": "athenak_z4c_cartoon_mms_window_v1",
            "state": "accepted_frozen_window",
            "purpose": "characterization_completion",
            "search_template_sha256": candidate["search_template_sha256"],
            "candidate_manifest_path": candidate_path.name,
            "candidate_manifest_sha256": sha256(candidate_path),
            "source_commit": fake_source["commit"], "source_tree": fake_source["tree"],
            "kokkos_commit": fake_source["kokkos"], "executable_sha256": "1" * 64,
            "build_manifest_sha256": "f" * 64, "input_sha256": "2" * 64,
            "oracle_header_sha256": "3" * 64, "series_manifest_sha256": "4" * 64,
            "qualification_window": qualification_window,
            "selected_extra_points": selected,
            "execution_tuples": [list(item) for item in missing],
            "prior_complete_tuples_sha256": canonical_digest(sorted(prior_tuples)),
            "backend": "Serial", "ranks": [2, 4],
            "immutable_root_bindings_sha256": canonical_digest(fake_roots),
            "observed_tuples_sha256": "9" * 64,
            "convergence_sha256": "a" * 64,
            "review": {"artifact_path": review_path.name,
                       "artifact_sha256": sha256(review_path)}}
        write_atomic(authorization / "accepted_window.json", window)
        observed_missing = validate_frozen_window(
            window, window["search_template_sha256"], fake_source, "f" * 64,
            "1" * 64, "2" * 64, "3" * 64, "4" * 64, prior_tuples,
            authorization, "Serial", 2)
        if observed_missing != missing:
            raise RuntimeError("accepted frozen window did not produce exact missing tuples")
        for label, mutation in (
            ("unreviewed frozen window", {"state": "candidate_unreviewed"}),
            ("fake candidate hash", {"candidate_manifest_sha256": "7" * 64}),
            ("fake review hash", {"review": {"artifact_path": review_path.name,
                                              "artifact_sha256": "8" * 64}}),
            ("mismatched frozen window", {"execution_tuples":
                                           [list(item) for item in missing[:-1]]})):
            expect_runtime_error(
                lambda value={**window, **mutation}: validate_frozen_window(
                    value, window["search_template_sha256"], fake_source, "f" * 64,
                    "1" * 64, "2" * 64, "3" * 64, "4" * 64, prior_tuples,
                    authorization, "Serial", 2), label)
        fresh = sorted(
            {(order, resolution, phase) for order in (2, 4, 6)
             for resolution in DIAGNOSTIC_RESOLUTIONS for phase in range(8)} |
            {(6, 48, phase) for phase in range(8)})
        candidate.update({"purpose": "final_qualification",
                          "execution_tuples": [list(item) for item in fresh]})
        write_atomic(candidate_path, candidate)
        review_artifact.update({"purpose": "final_qualification",
                                "candidate_manifest_sha256": sha256(candidate_path)})
        write_atomic(review_path, review_artifact)
        final_window = {
            **window, "purpose": "final_qualification",
            "candidate_manifest_sha256": sha256(candidate_path),
            "execution_tuples": [list(item) for item in fresh],
            "prior_complete_tuples_sha256": canonical_digest([]),
            "review": {"artifact_path": review_path.name,
                       "artifact_sha256": sha256(review_path)}}
        write_atomic(authorization / "accepted_window.json", final_window)
        if validate_frozen_window(
                final_window, window["search_template_sha256"], fake_source, "f" * 64,
                "1" * 64, "2" * 64, "3" * 64, "4" * 64, set(), authorization,
                "Serial", 4) != fresh:
            raise RuntimeError("final frozen window did not require a fresh complete matrix")
        expect_runtime_error(
            lambda: validate_frozen_window(
                {**final_window, "purpose": "characterization_completion"},
                window["search_template_sha256"], fake_source, "f" * 64,
                "1" * 64, "2" * 64, "3" * 64, "4" * 64, set(), authorization,
                "Serial", 2), "cross-used final window")
        for label, entry, target in (
            ("candidate symlink", candidate_path, review_path),
            ("review symlink", review_path, candidate_path)):
            original = entry.read_bytes()
            entry.unlink()
            entry.symlink_to(target.name)
            expect_runtime_error(
                lambda: validate_frozen_window(
                    final_window, window["search_template_sha256"], fake_source,
                    "f" * 64, "1" * 64, "2" * 64, "3" * 64, "4" * 64,
                    set(), authorization, "Serial", 4), label)
            entry.unlink()
            entry.write_bytes(original)
        authorization_link = Path(directory) / "authorization-link"
        authorization_link.symlink_to(authorization.name, target_is_directory=True)
        expect_runtime_error(
            lambda: validate_frozen_window(
                final_window, window["search_template_sha256"], fake_source,
                "f" * 64, "1" * 64, "2" * 64, "3" * 64, "4" * 64, set(),
                authorization_link, "Serial", 4), "authorization-directory symlink")
        extra = authorization / "unexpected.json"
        extra.write_text("{}\n", encoding="utf-8")
        expect_runtime_error(
            lambda: validate_frozen_window(
                final_window, window["search_template_sha256"], fake_source,
                "f" * 64, "1" * 64, "2" * 64, "3" * 64, "4" * 64, set(),
                authorization, "Serial", 4), "authorization extra entry")
        extra.unlink()
        duplicate = Path(directory) / "duplicate.json"
        duplicate.write_text('{"schema":"x","schema":"y"}\n', encoding="utf-8")
        expect_runtime_error(lambda: load_json_strict(duplicate),
                             "duplicate-key authorization JSON")
    ledger_fixture = merge_case_ledgers(
        [{"spatial_order": 2, "resolution": 256, "phase": 0,
          "case_manifest_sha256": "1" * 64}],
        [{"spatial_order": 2, "resolution": 512, "phase": 0,
          "case_manifest_sha256": "2" * 64}])
    if [(item["spatial_order"], item["resolution"], item["phase"])
            for item in ledger_fixture] != [(2, 256, 0), (2, 512, 0)]:
        raise RuntimeError("characterization evidence was omitted from merged ledger")
    expect_runtime_error(
        lambda: merge_case_ledgers(
            ledger_fixture,
            [{**ledger_fixture[0], "case_manifest_sha256": "3" * 64}]),
        "conflicting immutable evidence roots")
    if len(merge_case_ledgers(ledger_fixture, [ledger_fixture[0]])) != 2:
        raise RuntimeError("byte-identical copied-prefix case was not deduplicated")
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
        try:
            validate_norm_geometry(row, order, resolution, args.domain)
        except RuntimeError as error:
            raise RuntimeError(f"case {key} mask counts differ from exact geometry") from error
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
        validate_augmented_result(result, resumed_result)
        if (resumed_result.get("case_id") != key or
                resumed_result.get("case_uuid") != case_uuid or
                resumed_result.get("phase") != phase or
                resumed_result.get("resolution") != resolution or
                resumed_result.get("csv_sha256") != sha256(raw_csv) or
                resumed_result.get("probes_csv_sha256") != sha256(probes_csv) or
                resumed_result.get("operator_names") != operator_names or
                resumed_result.get("rank_bindings") != bindings or
                tuple(resumed_result.get("domain", ())) != tuple(args.domain) or
                resumed_result.get("execution_environment_sha256") !=
                environment_sha256):
            raise RuntimeError(f"case {key} resumed evidence differs from its result")
        verified, _, _, manifest_sha = verify_complete_case_evidence(
            case, source, sha256(args.build_manifest), sha256(args.athena),
            args.require_backend, args.ranks, tuple(args.domain), order,
            resolution, phase, operator_names, args.input, args.rank_wrapper)
        if verified != resumed_result:
            raise RuntimeError(f"case {key} resumed campaign/result differs")
        resumed_result["case_manifest_sha256"] = manifest_sha
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
    validate_augmented_result(load_json_strict(raw_result), result)
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
    verified, _, _, manifest_sha = verify_complete_case_evidence(
        case, source, sha256(args.build_manifest), sha256(args.athena),
        args.require_backend, args.ranks, tuple(args.domain), order, resolution,
        phase, operator_names, args.input, args.rank_wrapper)
    if verified != result:
        raise RuntimeError(f"case {key} finalized campaign/result differs")
    result["case_manifest_sha256"] = manifest_sha
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
                token = normalized.get(field, "")
                if token.lower() == "nan" or token == "0":
                    normalized[field] = ""
                    actions.append(f"blanked_inapplicable_{field}")
                else:
                    raise RuntimeError(
                        f"legacy inapplicable {field} is neither NaN nor exact zero")
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


def validate_raw_result_projection(raw: object, augmented: object) -> None:
    if (not isinstance(raw, dict) or not isinstance(augmented, dict) or
            set(augmented) != set(raw) | RAW_RESULT_FIELDS or
            any(augmented[key] != value for key, value in raw.items())):
        raise RuntimeError("raw Athena JSON is not an exact augmented-result projection")


def normalize_preserved_job56586376_result(
        raw: object, stored: object,
        manifest_identity: object) -> tuple[dict[str, object], str]:
    if (not isinstance(raw, dict) or not isinstance(stored, dict) or
            not isinstance(manifest_identity, dict) or "domain" in raw or
            set(stored) != (set(raw) | RAW_RESULT_FIELDS) - {"domain"} or
            manifest_identity.get("domain") != list(QUALIFICATION_DOMAIN)):
        raise RuntimeError("legacy replay result is not the exact missing-domain shape")
    normalized = {**stored, "domain": list(QUALIFICATION_DOMAIN)}
    validate_raw_result_projection(raw, normalized)
    return normalized, "derived_domain_from_authenticated_manifest_identity"


def validate_preserved_job56586376_normalization_counts(
        counts: Counter[str]) -> None:
    if dict(counts) != PRESERVED_JOB_56586376_NORMALIZATION_COUNTS:
        raise RuntimeError("replay legacy normalization accounting differs")


def validate_preserved_job56586376_legacy_partition(partition: object) -> None:
    if not isinstance(partition, dict) or tuple(partition.get(name) for name in
            ("norm_inconclusive", "norm_rate_miss",
             "probe_inconclusive", "probe_rate_miss")) != \
            PRESERVED_JOB_56586376_AUDIT_COUNTS:
        raise RuntimeError("job56586376 legacy pre-coefficient-floor partition changed")


def validate_augmented_result(raw: object, augmented: object) -> None:
    validate_raw_result_projection(raw, augmented)
    if not isinstance(augmented, dict):
        raise RuntimeError("augmented result is not an object")
    validate_result_numbers(augmented)
    if (not isinstance(augmented.get("case_id"), str) or
            not isinstance(augmented.get("case_uuid"), str) or
            not isinstance(augmented.get("execution_environment"), dict) or
            any(not isinstance(key, str) or not isinstance(value, str)
                for key, value in augmented["execution_environment"].items()) or
            augmented.get("execution_environment_sha256") != canonical_digest(
                augmented["execution_environment"]) or
            any(not re.fullmatch(r"[0-9a-f]{64}", str(augmented.get(field, "")))
                for field in ("csv_sha256", "probes_csv_sha256"))):
        raise RuntimeError("augmented result identity/environment is malformed")
    try:
        uuid.UUID(augmented["case_uuid"])
    except (ValueError, TypeError, AttributeError) as error:
        raise RuntimeError("augmented result UUID is malformed") from error
    json_integer(augmented, "phase")
    json_integer(augmented, "resolution", 1)
    json_integer(augmented, "output_bytes")
    json_number(augmented, "elapsed_seconds", 0.0)
    names = augmented.get("operator_names")
    bindings = augmented.get("rank_bindings")
    domain = augmented.get("domain")
    if (not isinstance(names, list) or len(names) != 171 or
            len(set(names)) != 171 or any(not isinstance(name, str) for name in names) or
            not isinstance(bindings, list) or
            len(bindings) != json_integer(augmented, "mpi_ranks", 1) or
            not isinstance(domain, (list, tuple)) or len(domain) != 4 or
            any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(float(value)) for value in domain)):
        raise RuntimeError("augmented result inventory/domain is malformed")
    for binding in bindings:
        validate_rank_binding(binding)


def load_augmented_result(raw_path: Path, result_path: Path) -> dict[str, object]:
    raw = load_json_strict(raw_path)
    result = load_json_strict(result_path)
    validate_augmented_result(raw, result)
    return result


def validate_raw_case_invariants(result: dict[str, object], order: int,
                                 resolution: int, phase: int, backend: str,
                                 ranks: int,
                                 domain: tuple[float, float, float, float]) -> None:
    validate_result_numbers(result)
    if (result.get("status") != "pass" or
            json_integer(result, "spatial_order", 2) != order or
            json_integer(result, "nghost", 2) != order // 2 + 1 or
            json_integer(result, "nx1", 1) != resolution or
            json_integer(result, "nx2", 1) != resolution or
            json_integer(result, "nx3", 1) != 1 or
            json_integer(result, "mpi_ranks", 1) != ranks or
            json_integer(result, "operator_count", 1) != 171 or
            json_integer(result, "nonfinite_count") != 0 or
            json_integer(result, "diagnostic_axis_operator_count", 1) != 161 or
            json_integer(result, "diagnostic_axis_nonfinite") != 0 or
            json_integer(result, "owned_cells", 1) != resolution * resolution or
            result.get("ownership_sequence") != "[0,N*N) exactly once" or
            result.get("backend") != backend or
            json_integer(result, "phase") != phase or
            json_integer(result, "resolution", 1) != resolution or
            tuple(result.get("domain", ())) != tuple(domain)):
        raise RuntimeError("case raw physics/grid/ownership identity differs")


def validate_binding_inventory(result: dict[str, object], backend: str,
                               ranks: int) -> None:
    bindings = result.get("rank_bindings")
    if (not isinstance(bindings, list) or len(bindings) != ranks or
            sorted(item.get("rank") for item in bindings
                   if isinstance(item, dict)) != list(range(ranks))):
        raise RuntimeError("case lacks one binding record per MPI rank")
    for binding in bindings:
        validate_rank_binding(binding)
    if backend == "Cuda":
        uuids = [item["selected_uuid"] for item in bindings]
        if (ranks != 4 or None in uuids or len(set(uuids)) != 4 or
                any(item.get("binding_verified") is not True or
                    item.get("cuda_visible_devices") !=
                    item.get("visible_device_token") or
                    not item.get("visible_device_token") for item in bindings) or
                any("A100" not in (item.get("gpu_name") or "")
                    for item in bindings)):
            raise RuntimeError("case lacks four concrete distinct CUDA bindings")


def validate_case_numerical_summary(result: dict[str, object],
                                    expected_operators: list[str],
                                    rows: list[dict[str, str]],
                                    probes: list[dict[str, str]]) -> None:
    if (any(integer_value(row, "nonfinite") != 0 for row in rows) or
            {row["operator"] for row in rows} != set(expected_operators)):
        raise RuntimeError("case CSV contains incomplete or nonfinite series")
    noise_bound = json_number(result, "noise_delta_bound", 0.0)
    if (json_number(result, "maximum_noise_delta", 0.0) > noise_bound or
            any(finite_value(row, "shared_delta_linfinity") > noise_bound or
           finite_value(row, "independent_delta_linfinity") > noise_bound
                for row in rows)):
        raise RuntimeError("case exceeds the frozen direct-noise bound")
    axis = [row for row in probes if row["mask"] == "diagnostic_axis"]
    axis_errors = [finite_value(row, "raw_error") for row in axis]
    if (len(axis) != 161 or
            [row["operator"] for row in axis] != expected_operators[:161] or
            any(row["side"] != "axis" or
                row["classification"] != "diagnostic_axis" or
                row["layer_index"] != "0" for row in axis) or
            max(axis_errors) != json_number(result, "diagnostic_axis_linf", 0.0) or
            max(axis_errors) >
            json_number(result, "diagnostic_axis_tolerance", 0.0)):
        raise RuntimeError("case lacks the exact finite 161-series axis inventory")


def validate_case_launch_provenance(
        case: Path, identity: dict[str, object], stored: dict[str, object],
        executable_sha256: str, backend: str, ranks: int,
        domain: tuple[float, float, float, float], order: int, resolution: int,
        phase: int, input_template: Path, rank_wrapper: Path) -> None:
    expected_input = render_input(input_template, order, resolution, phase,
                                  "cartoon_mms", domain)
    input_path = case / "input.athinput"
    if (identity.get("input_sha256") !=
            hashlib.sha256(expected_input.encode()).hexdigest() or
            sha256(input_path) != identity.get("input_sha256") or
            input_path.read_text(encoding="utf-8") != expected_input or
            identity.get("rank_wrapper_sha256") != sha256(rank_wrapper)):
        raise RuntimeError("case input/wrapper provenance differs")
    command = identity.get("command")
    wrapper_tail = [str(rank_wrapper), "--evidence-dir", "."] + \
        (["--require-cuda"] if backend == "Cuda" else []) + ["--"]
    delimiter = (len(command) - 4 if isinstance(command, list) else -1)
    python_index = delimiter - len(wrapper_tail)
    launcher_prefix = command[:python_index] if isinstance(command, list) else []
    launcher_name = Path(launcher_prefix[0]).name if launcher_prefix else ""
    launcher_ranks_ok = (
        launcher_prefix[-2:] == ["-np", str(ranks)]
        if launcher_name in {"mpirun", "mpiexec"} else
        launcher_prefix[-2:] == ["--ntasks", str(ranks)]
        if launcher_name == "srun" else
        bool(launcher_prefix) and launcher_prefix[-1] == str(ranks))
    if (not isinstance(command, list) or
            any(not isinstance(token, str) for token in command) or
            python_index < 1 or not launcher_ranks_ok or
            not Path(command[python_index]).name.startswith("python") or
            command[delimiter - len(wrapper_tail) + 1:
                    delimiter + 1] != wrapper_tail or
            Path(command[delimiter + 1]).is_symlink() or
            not Path(command[delimiter + 1]).is_file() or
            sha256(Path(command[delimiter + 1])) != executable_sha256 or
            command[delimiter + 2:] != ["-i", "input.athinput"]):
        raise RuntimeError("case launcher/wrapper/executable command differs")
    archived_bindings = [load_json_strict(path)
                         for path in sorted(case.glob("rank_binding_*.json"))]
    if archived_bindings != stored.get("rank_bindings"):
        raise RuntimeError("archived rank-binding objects differ from result")


def verify_complete_case_evidence(
        case: Path, source: dict[str, str], build_manifest_sha256: str,
        executable_sha256: str, backend: str, ranks: int,
        domain: tuple[float, float, float, float], order: int, resolution: int,
        phase: int, expected_operators: list[str], input_template: Path,
        rank_wrapper: Path) -> \
        tuple[dict[str, object], list[dict[str, str]], list[dict[str, str]], str]:
    if not case.is_dir() or case.is_symlink():
        raise RuntimeError("case directory is missing or a symlink")
    manifest_path = case / "manifest.json"
    manifest = load_json_strict(manifest_path)
    identity = manifest.get("identity") if isinstance(manifest, dict) else None
    if not isinstance(identity, dict) or not verified_complete(case, identity):
        raise RuntimeError("case manifest/files are incomplete or changed")
    stored = load_augmented_result(case / "cartoon_mms.mms.json",
                                   case / "result.json")
    case_id, case_uuid = stored.get("case_id"), stored.get("case_uuid")
    if (case.name != f"{case_id}-{case_uuid}" or
            identity.get("case_id") != case_id or identity.get("uuid") != case_uuid or
            identity.get("source") != source or
            identity.get("athena_sha256") != executable_sha256 or
            identity.get("build_manifest_sha256") != build_manifest_sha256 or
            identity.get("backend_required") != backend or
            identity.get("ranks") != ranks or
            tuple(identity.get("domain", ())) != tuple(domain) or
            identity.get("execution_environment_sha256") !=
            stored.get("execution_environment_sha256")):
        raise RuntimeError("case manifest/result identity differs")
    validate_case_launch_provenance(
        case, identity, stored, executable_sha256, backend, ranks, domain,
        order, resolution, phase, input_template, rank_wrapper)
    validate_raw_case_invariants(stored, order, resolution, phase, backend,
                                 ranks, domain)
    verify_no_evolution(str(case_id),
                        (case / "stdout.txt").read_text(
                            encoding="utf-8", errors="replace"), stored)
    validate_binding_inventory(stored, backend, ranks)
    if stored.get("operator_names") != expected_operators:
        raise RuntimeError("case operator order differs from the frozen 171 series")
    csv_path = case / "cartoon_mms.mms.csv"
    probes_path = case / "cartoon_mms.mms.probes.csv"
    if (stored.get("csv_sha256") != sha256(csv_path) or
            stored.get("probes_csv_sha256") != sha256(probes_path)):
        raise RuntimeError("case result does not bind CSV/probe products")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    probes = list(csv.DictReader(probes_path.open(encoding="utf-8")))
    for row in rows:
        validate_norm_row(row)
        validate_norm_geometry(row, order, resolution, domain)
    for row in probes:
        validate_probe_row(row)
        validate_probe_geometry(row, order, resolution, domain)
    validate_case_inventory(order, expected_operators, rows, probes)
    validate_case_numerical_summary(stored, expected_operators, rows, probes)
    return stored, rows, probes, sha256(manifest_path)


def verify_replay_campaign(raw_root: Path) -> tuple[list[dict[str, object]],
                                                    dict[str, object]]:
    preserved_convergence = raw_root / "results/ranks2/convergence.json"
    preserved_evidence = raw_root / "evidence/cpu-ranks2.json"
    preserved_log = raw_root / "evidence/cpu-ranks2.log"
    build_record = raw_root / "evidence/cpu-build/build.json"
    build_manifest = raw_root / "build-mms-cpu-mpi/src/mms_build_manifest.json"
    verify_preserved_job56586376_failed_convergence(preserved_convergence)
    if (sha256(preserved_evidence) != PRESERVED_JOB_56586376_EVIDENCE_SHA256 or
            sha256(preserved_log) != PRESERVED_JOB_56586376_LOG_SHA256 or
            sha256(build_record) !=
            "69c236d67283272c8877685f5e5d533efa10a302b4ffd4eb66a7656f90d64f31" or
            sha256(build_manifest) !=
            "60356b252d40b3657b07c681ed6dcc72d94a5be341455a4dce42ba8978a0a06d"):
        raise RuntimeError("replay root is not immutable job56586376 evidence")
    if not isinstance(load_json_strict(preserved_evidence), dict):
        raise RuntimeError("preserved wrapper evidence is malformed")
    manifests = sorted((raw_root / "results/ranks2").glob("o*/manifest.json"))
    if len(manifests) != EXPECTED_CASES:
        raise RuntimeError("replay requires exactly 96 complete case manifests")
    if any(path.is_symlink() or path.parent.is_symlink() for path in manifests):
        raise RuntimeError("replay case/manifest paths may not be symlinks")
    manifest_set = [{"path": str(path.relative_to(raw_root)), "sha256": sha256(path)}
                    for path in manifests]
    require_canonical_digest(manifest_set,
                             PRESERVED_JOB_56586376_MANIFEST_SET_SHA256,
                             "job56586376 exact 96-case manifest set")
    inventory = load_json_strict(
        Path(__file__).resolve().parents[3] /
        "tst/unit/z4c/z4c_cartoon_derivatives_series.json")
    expected_operators = [item["name"] for item in inventory["series"]]
    expected = {(order, resolution, phase) for order in (2, 4, 6)
                for resolution in DIAGNOSTIC_RESOLUTIONS for phase in range(8)}
    observed = set()
    case_bindings = []
    cases = []
    legacy_result_normalization_actions = []
    legacy_csv_normalization_counts: Counter[str] = Counter()
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
        legacy_stored_result = load_json_strict(case / "result.json")
        raw_result = load_json_strict(case / "cartoon_mms.mms.json")
        stored_result, normalization_action = normalize_preserved_job56586376_result(
            raw_result, legacy_stored_result, manifest.get("identity"))
        legacy_result_normalization_actions.append(normalization_action)
        validate_result_numbers(stored_result)
        json_number(stored_result, "elapsed_seconds", 0.0)
        json_integer(stored_result, "output_bytes")
        if set(files) != expected_case_files(json_integer(stored_result, "mpi_ranks", 1)):
            raise RuntimeError(f"replay case {case.name} file set differs from exact schema")
        result = dict(stored_result)
        order = json_integer(result, "spatial_order", 2)
        resolution = json_integer(result, "resolution", 1)
        phase = json_integer(result, "phase")
        if order not in (2, 4, 6):
            raise RuntimeError(f"replay case {case.name} raw physics/grid state changed")
        validate_raw_case_invariants(result, order, resolution, phase, "Serial", 2,
                                     QUALIFICATION_DOMAIN)
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
                result.get("execution_environment_sha256") or
                identity.get("source") != {
                    "commit": "e29413e15fbe86d105fe27096113dffbcdd0eede",
                    "tree": "be855bfae9f5d12908a5bee1cbbbf0033c7894ef",
                    "kokkos": "6739bc623081648af9e752b616d9671527922cbf"} or
                identity.get("athena_sha256") !=
                "cd0c60d84198c157dd536c6b349ae13fd6f9fb559f40eb8dbc3c8102464e8c09"):
            raise RuntimeError(f"replay case {case_id} manifest/result identity differs")
        bindings = stored_result.get("rank_bindings")
        operators = stored_result.get("operator_names")
        if (not isinstance(bindings, list) or
                len(bindings) != json_integer(stored_result, "mpi_ranks", 1) or
                operators != expected_operators):
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
        rows = []
        for row in csv.DictReader(
                (case / "cartoon_mms.mms.csv").open(encoding="utf-8")):
            normalized, actions = normalize_legacy_norm_row(row)
            legacy_csv_normalization_counts.update(actions)
            rows.append(normalized)
        probes = []
        for row in csv.DictReader(
                (case / "cartoon_mms.mms.probes.csv").open(encoding="utf-8")):
            normalized, actions = normalize_legacy_probe_row(row)
            legacy_csv_normalization_counts.update(actions)
            validate_probe_geometry(normalized, order, resolution,
                                    tuple(float(value) for value in result["domain"]))
            probes.append(normalized)
        validate_case_inventory(order, expected_operators, rows, probes)
        domain = tuple(float(value) for value in result["domain"])
        noise_bound = json_number(result, "noise_delta_bound", 0.0)
        if any(finite_value(row, "shared_delta_linfinity") > noise_bound or
               finite_value(row, "independent_delta_linfinity") > noise_bound
               for row in rows):
            raise RuntimeError(f"replay case {case_id} noise bound changed")
        axis_errors = [finite_value(row, "raw_error") for row in probes
                       if row["classification"] == "diagnostic_axis"]
        if (len(axis_errors) != 161 or max(axis_errors) !=
                json_number(result, "diagnostic_axis_linf", 0.0) or
                max(axis_errors) > json_number(result, "diagnostic_axis_tolerance", 0.0)):
            raise RuntimeError(f"replay case {case_id} axis summary changed")
        for row in rows:
            try:
                validate_norm_geometry(row, order, resolution, domain)
            except RuntimeError as error:
                raise RuntimeError(f"replay case {case_id} geometry count changed") from error
        result["case_manifest_sha256"] = manifest_sha
        result["_replay_case_directory"] = str(case)
        cases.append(result)
        case_bindings.append({"case_id": case_id, "case_uuid": case_uuid,
                              "manifest_sha256": manifest_sha,
                              "identity": manifest["identity"], "files": files})
    if observed != expected:
        raise RuntimeError("replay cases differ from exact 2/4/6 x 32/64/128/256 x 8")
    if (len(legacy_result_normalization_actions) != EXPECTED_CASES or
            set(legacy_result_normalization_actions) != {
                "derived_domain_from_authenticated_manifest_identity"}):
        raise RuntimeError("replay legacy-domain normalization accounting differs")
    validate_preserved_job56586376_normalization_counts(
        legacy_csv_normalization_counts)
    binding = {"schema": "athenak_z4c_cartoon_mms_replay_binding_v1",
               "raw_root": str(raw_root),
               "preserved_convergence_sha256":
               PRESERVED_JOB_56586376_CONVERGENCE_SHA256,
               "preserved_convergence_bytes":
               PRESERVED_JOB_56586376_CONVERGENCE_BYTES,
               "preserved_convergence_negative_infinity":
               PRESERVED_JOB_56586376_CONVERGENCE_NEGATIVE_INFINITY,
               "preserved_convergence_trust": "opaque_failed_report_not_imported",
               "legacy_result_normalization_actions": sorted(
                   set(legacy_result_normalization_actions)),
               "legacy_result_normalization_count":
               len(legacy_result_normalization_actions),
               "legacy_csv_normalization_counts": dict(sorted(
                   legacy_csv_normalization_counts.items())),
               "preserved_wrapper_evidence_sha256": sha256(preserved_evidence),
               "preserved_log_sha256": sha256(preserved_log),
               "build_record_sha256": sha256(build_record),
               "build_manifest_sha256": sha256(build_manifest),
               "case_manifest_set_sha256": canonical_digest(manifest_set),
               "raw_identity_sha256": canonical_digest(
                   [manifest["identity"] for manifest in
                    (load_json_strict(path) for path in manifests)]),
               "case_count": len(cases), "verified_case_bytes": verified_case_bytes,
               "cases": case_bindings}
    return cases, binding


def verify_characterization_root(raw_root: Path,
                                 template: dict[str, object]) -> \
        tuple[dict[str, object], list[dict[str, object]]]:
    if not raw_root.is_dir() or raw_root.is_symlink():
        raise RuntimeError("job56587561 root is missing or a symlink")
    files = sorted(path for path in raw_root.rglob("*") if path.is_file())
    if len(files) != 35 or any(path.is_symlink() for path in files):
        raise RuntimeError("job56587561 exact 35-file inventory changed")
    inventory = [{"path": str(path.relative_to(raw_root)), "sha256": sha256(path),
                  "bytes": path.stat().st_size} for path in files]
    if sum(item["bytes"] for item in inventory) != 4_533_784:
        raise RuntimeError("job56587561 canonical inventory changed")
    require_canonical_digest(inventory, PRESERVED_JOB_56587561_INVENTORY_SHA256,
                             "job56587561 inventory")
    ledger = template["immutable_roots"]["job56587561"]
    provenance = raw_root / "provenance"
    if (sha256(provenance / "characterize_cpu_perlmutter.sh") !=
            ledger["provenance_script_sha256"] or
            sha256(provenance / "inputs.sha256") != ledger["inputs_manifest_sha256"] or
            sha256(provenance / "slurm-job.txt") != ledger["slurm_sha256"]):
        raise RuntimeError("job56587561 provenance files changed")
    input_binding = (provenance / "inputs.sha256").read_text(encoding="utf-8")
    if (ledger["executable_sha256"] not in input_binding or
            ledger["build_manifest_sha256"] not in input_binding):
        raise RuntimeError("job56587561 executable/build identity is unbound")
    claimed_paths = {"provenance/characterize_cpu_perlmutter.sh",
                     "provenance/inputs.sha256", "provenance/slurm-job.txt"}
    expected_operators = [item["name"] for item in load_json_strict(
        Path(__file__).resolve().parents[3] /
        "tst/unit/z4c/z4c_cartoon_derivatives_series.json")["series"]]
    cases = []
    for record in ledger["completed"]:
        order, resolution, phase = record["tuple"]
        case = raw_root / "cases" / f"o{order}-n{resolution}"
        expected_hashes = {
            "input.athinput": record["input_sha256"],
            "cartoon_mms.mms.json": record["raw_json_sha256"],
            "cartoon_mms.mms.csv": record["csv_sha256"],
            "cartoon_mms.mms.probes.csv": record["probes_sha256"],
            "stdout.txt": record["stdout_sha256"],
            "stderr.txt": record["stderr_sha256"], "time.txt": record["time_sha256"]}
        claimed_paths.update(f"cases/o{order}-n{resolution}/{name}"
                             for name in expected_hashes)
        if (not case.is_dir() or case.is_symlink() or
                any(not (case / name).is_file() or (case / name).is_symlink() or
                    sha256(case / name) != digest
                    for name, digest in expected_hashes.items())):
            raise RuntimeError(f"job56587561 O{order}/N{resolution} files changed")
        result = load_json_strict(case / "cartoon_mms.mms.json")
        if not isinstance(result, dict):
            raise RuntimeError("job56587561 raw result is not an object")
        validate_result_numbers(result)
        domain = QUALIFICATION_DOMAIN
        if (result.get("status") != "pass" or result.get("backend") != "Serial" or
                json_integer(result, "spatial_order", 2) != order or
                json_integer(result, "mpi_ranks", 1) != 2 or
                json_integer(result, "nx1", 1) != resolution or
                json_integer(result, "nx2", 1) != resolution or
                json_integer(result, "nx3", 1) != 1 or
                json_integer(result, "owned_cells", 1) != resolution * resolution or
                result.get("operator_names") != expected_operators):
            raise RuntimeError("job56587561 raw result identity changed")
        rows = []
        for row in csv.DictReader(
                (case / "cartoon_mms.mms.csv").open(encoding="utf-8")):
            normalized, _ = normalize_legacy_norm_row(row)
            validate_norm_geometry(normalized, order, resolution, domain)
            rows.append(normalized)
        probes = []
        for row in csv.DictReader(
                (case / "cartoon_mms.mms.probes.csv").open(encoding="utf-8")):
            normalized, _ = normalize_legacy_probe_row(row)
            validate_probe_geometry(normalized, order, resolution, domain)
            probes.append(normalized)
        validate_case_inventory(order, expected_operators, rows, probes)
        axis_errors = [finite_value(row, "raw_error") for row in probes
                       if row["classification"] == "diagnostic_axis"]
        noise_bound = json_number(result, "noise_delta_bound", 0.0)
        if (len(axis_errors) != 161 or max(axis_errors) !=
                json_number(result, "diagnostic_axis_linf", 0.0) or
                any(finite_value(row, "shared_delta_linfinity") > noise_bound or
                    finite_value(row, "independent_delta_linfinity") > noise_bound
                    for row in rows)):
            raise RuntimeError("job56587561 numerical inventory changed")
        result.update({"resolution": resolution, "phase": phase,
                       "domain": list(domain),
                       "_replay_case_directory": str(case),
                       "_evidence_source": "job56587561"})
        cases.append(result)
    for record in ledger["bounded_characterization"]:
        order, resolution, phase = record["tuple"]
        case = raw_root / "cases" / f"o{order}-n{resolution}"
        expected_hashes = {
            "input.athinput": record["input_sha256"],
            "cartoon_mms.mms.json": record["raw_json_sha256"],
            "cartoon_mms.mms.csv": record["csv_sha256"],
            "cartoon_mms.mms.probes.csv": record["probes_sha256"],
            "stdout.txt": record["stdout_sha256"],
            "stderr.txt": record["stderr_sha256"], "time.txt": record["time_sha256"]}
        claimed_paths.update(f"cases/o{order}-n{resolution}/{name}"
                             for name in expected_hashes)
        if any(not (case / name).is_file() or sha256(case / name) != digest
               for name, digest in expected_hashes.items()):
            raise RuntimeError("job56587561 bounded characterization changed")
        raw = load_json_strict(case / "cartoon_mms.mms.json")
        if (not isinstance(raw, dict) or raw.get("status") != "pass" or
                json_integer(raw, "spatial_order", 2) != order or
                json_integer(raw, "mpi_ranks", 1) != 2 or
                json_integer(raw, "nx3", 1) != 1 or phase != 0):
            raise RuntimeError("job56587561 bounded characterization identity changed")
    for record in ledger["attempts"]:
        order, resolution, phase = record["tuple"]
        case = raw_root / "cases" / f"o{order}-n{resolution}"
        expected_hashes = {"input.athinput": record["input_sha256"],
                           "stdout.txt": record["stdout_sha256"],
                           "stderr.txt": record["stderr_sha256"],
                           "time.txt": record["time_sha256"]}
        claimed_paths.update(f"cases/o{order}-n{resolution}/{name}"
                             for name in expected_hashes)
        if (record.get("status") != "out_of_memory" or phase != 0 or
                any(not (case / name).is_file() or sha256(case / name) != digest
                    for name, digest in expected_hashes.items()) or
                any((case / name).exists() for name in
                    ("cartoon_mms.mms.json", "cartoon_mms.mms.csv",
                     "cartoon_mms.mms.probes.csv"))):
            raise RuntimeError("job56587561 OOM attempt identity changed")
    if claimed_paths != {item["path"] for item in inventory}:
        raise RuntimeError("job56587561 ledger does not own the exact root inventory")
    binding = {"root": str(raw_root), "file_count": len(files),
               "total_bytes": sum(item["bytes"] for item in inventory),
               "canonical_inventory_sha256": canonical_digest(inventory),
               "source_commit": ledger["source_commit"],
               "source_tree": ledger["source_tree"],
               "kokkos_commit": ledger["kokkos_commit"],
               "executable_sha256": ledger["executable_sha256"],
               "build_manifest_sha256": ledger["build_manifest_sha256"],
               "completed_tuples": [record["tuple"] for record in ledger["completed"]]}
    return binding, cases


def validate_stage_identity(stage: object, campaign: object,
                            template: dict[str, object],
                            expected_root_bindings: list[dict[str, object]]) -> \
        tuple[dict[str, object], dict[str, str]]:
    campaign_fields = {"schema", "state", "stage_manifest_sha256", "source",
                       "build_manifest_sha256", "executable_sha256", "input_sha256",
                       "oracle_header_sha256", "series_manifest_sha256", "backend",
                       "ranks", "domain", "immutable_root_bindings", "cases",
                       "failed_case_inventories"}
    if (not isinstance(stage, dict) or stage.get("state") not in
            {"stage_partial", "stage_finalized"} or set(stage) != set(template) or
            any(stage[key] != template[key] for key in template
                if key not in {"state", "materialization"}) or
            not isinstance(campaign, dict) or set(campaign) != campaign_fields or
            campaign.get("state") != stage.get("state") or
            not isinstance(campaign.get("cases"), list)):
        raise RuntimeError("completed stage lifecycle/campaign is malformed")
    material = stage.get("materialization")
    root = Path(__file__).resolve().parents[3]
    if (not isinstance(material, dict) or
            set(material) != set(template["materialization"]) or
            material.get("immutable_root_bindings") != expected_root_bindings or
            material.get("input_sha256") != sha256(
                root / "tst/inputs/z4c_cartoon_derivatives.athinput") or
            material.get("oracle_header_sha256") != sha256(
                root / "src/pgen/unit_tests/z4c_cartoon_derivatives_oracle.hpp") or
            material.get("series_manifest_sha256") != sha256(
                root / "tst/unit/z4c/z4c_cartoon_derivatives_series.json") or
            material.get("domain") != list(QUALIFICATION_DOMAIN)):
        raise RuntimeError("stage materialization identity differs")
    material_source = {"commit": material.get("source_commit"),
                       "tree": material.get("source_tree"),
                       "kokkos": material.get("kokkos_commit")}
    if (campaign.get("source") != material_source or
            campaign.get("build_manifest_sha256") !=
            material.get("build_manifest_sha256") or
            campaign.get("executable_sha256") != material.get("executable_sha256") or
            campaign.get("input_sha256") != material.get("input_sha256") or
            campaign.get("oracle_header_sha256") !=
            material.get("oracle_header_sha256") or
            campaign.get("series_manifest_sha256") !=
            material.get("series_manifest_sha256") or
            campaign.get("backend") != material.get("backend") or
            campaign.get("ranks") != material.get("ranks") or
            campaign.get("domain") != material.get("domain") or
            campaign.get("immutable_root_bindings") != expected_root_bindings):
        raise RuntimeError("stage campaign/materialization identity differs")
    return material, material_source


def verify_stage_campaign_root(raw_root: Path,
                               template: dict[str, object],
                               expected_root_bindings: list[dict[str, object]]) -> \
        tuple[list[dict[str, object]], dict[str, object],
              list[tuple[int, int, int]]]:
    if not raw_root.is_dir() or raw_root.is_symlink():
        raise RuntimeError("completed stage root is missing or a symlink")
    stage_path = raw_root / "search_stage_manifest.json"
    campaign_path = raw_root / "stage_campaign.json"
    if any(not path.is_file() or path.is_symlink()
           for path in (stage_path, campaign_path)):
        raise RuntimeError("completed stage lifecycle files are missing or symlinks")
    stage = load_json_strict(stage_path)
    campaign = load_json_strict(campaign_path)
    material, material_source = validate_stage_identity(
        stage, campaign, template, expected_root_bindings)
    if campaign.get("stage_manifest_sha256") != sha256(stage_path):
        raise RuntimeError("stage campaign does not bind its materialization")
    stage_id = material.get("stage_id")
    expected_tuples = [tuple(item) for item in
                       template["stages"].get(stage_id, {}).get("tuples", [])]
    cases = []
    observed = []
    expected_operators = [item["name"] for item in load_json_strict(
        Path(__file__).resolve().parents[3] /
        "tst/unit/z4c/z4c_cartoon_derivatives_series.json")["series"]]
    for augmented in campaign["cases"]:
        if not isinstance(augmented, dict):
            raise RuntimeError("completed stage case is not an object")
        case = raw_root / f"{augmented.get('case_id')}-{augmented.get('case_uuid')}"
        order, resolution, phase = (json_integer(augmented, "spatial_order", 2),
                                    json_integer(augmented, "resolution", 1),
                                    json_integer(augmented, "phase"))
        stored, _, _, manifest_sha = verify_complete_case_evidence(
            case, material_source, material["build_manifest_sha256"],
            material["executable_sha256"], material["backend"], material["ranks"],
            tuple(material["domain"]), order, resolution, phase,
            expected_operators, Path(__file__).resolve().parents[3] /
            "tst/inputs/z4c_cartoon_derivatives.athinput",
            Path(__file__).resolve().parents[3] /
            "tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py")
        manifest = load_json_strict(case / "manifest.json")
        identity = manifest["identity"]
        if augmented != {**stored, "case_manifest_sha256": manifest_sha}:
            raise RuntimeError("completed stage campaign/result differs")
        expected_input = render_input(
            Path(__file__).resolve().parents[3] /
            "tst/inputs/z4c_cartoon_derivatives.athinput", order, resolution,
            phase, "cartoon_mms", tuple(material["domain"]))
        if (identity.get("source") != material_source or
                identity.get("athena_sha256") != material.get("executable_sha256") or
                identity.get("build_manifest_sha256") !=
                material.get("build_manifest_sha256") or
                identity.get("input_sha256") !=
                hashlib.sha256(expected_input.encode()).hexdigest() or
                identity.get("backend_required") != material.get("backend") or
                identity.get("ranks") != material.get("ranks") or
                list(identity.get("domain", ())) != material.get("domain")):
            raise RuntimeError("stage case identity differs from materialization")
        result = dict(augmented)
        result["_replay_case_directory"] = str(case)
        result["_evidence_source"] = f"stage:{stage_id}"
        cases.append(result)
        observed.append((order, resolution, phase))
    attempts = material.get("attempts")
    if (not isinstance(attempts, list) or
            [item.get("tuple") for item in attempts] !=
            [list(item) for item in expected_tuples] or
            any(set(item) != {"tuple", "status", "case_manifest_sha256", "reason"} or
                item.get("status") not in
                {"complete", "failed", "not_attempted_after_stop"}
                for item in attempts)):
        raise RuntimeError("stage attempt ledger differs from authorization")
    completed_tuples = [tuple(item["tuple"]) for item in attempts
                        if item.get("status") == "complete"]
    missing_tuples = stage_missing_tuples(stage)
    failed_indices = [index for index, item in enumerate(attempts)
                      if item.get("status") == "failed"]
    if (len(failed_indices) > 1 or
            (failed_indices and any(item.get("status") != "not_attempted_after_stop"
                                    for item in attempts[failed_indices[0] + 1:]))):
        raise RuntimeError("stage attempt ledger continued after failure")
    if observed != completed_tuples or (stage["state"] == "stage_finalized" and
                                        missing_tuples):
        raise RuntimeError("completed stage tuples differ from attempt ledger")
    if (material.get("result_set_sha256") != canonical_digest(
            [case["case_manifest_sha256"] for case in cases]) or
            [item.get("case_manifest_sha256") for item in attempts
             if item.get("status") == "complete"] !=
            [case["case_manifest_sha256"] for case in cases] or
            (stage["state"] == "stage_finalized") !=
            (material.get("stop_reason") ==
             "stage_complete_pending_offline_analysis")):
        raise RuntimeError("completed stage result-set/stop reason changed")
    binding = {"root": str(raw_root), "stage_id": stage_id,
               "state": stage["state"],
               "stage_manifest_sha256": sha256(stage_path),
               "stage_campaign_sha256": sha256(campaign_path),
               "case_manifest_sha256": [case["case_manifest_sha256"] for case in cases],
               "completed_tuples": [list(item) for item in completed_tuples],
               "missing_tuples": [list(item) for item in missing_tuples]}
    expected_root_entries = {"preflight.json", "search_stage_manifest.json",
                             "stage_campaign.json"}
    for case in cases:
        directory = raw_root / f"{case['case_id']}-{case['case_uuid']}"
        expected_root_entries.add(directory.name)
    failed_inventories = campaign.get("failed_case_inventories")
    if not isinstance(failed_inventories, list):
        raise RuntimeError("stage failed-case inventory is malformed")
    for item in failed_inventories:
        if (not isinstance(item, dict) or set(item) != {"directory", "files"} or
                not isinstance(item["directory"], str) or
                "/" in item["directory"] or not isinstance(item["files"], list)):
            raise RuntimeError("stage failed-case inventory schema differs")
        directory = raw_root / item["directory"]
        if bound_directory_inventory(directory) != item["files"]:
            raise RuntimeError("stage failed-case evidence changed")
        expected_root_entries.add(item["directory"])
    require_exact_root_entries(raw_root, expected_root_entries, "stage root")
    return cases, binding, missing_tuples


def verify_window_campaign_root(
        raw_root: Path, expected_root_bindings: list[dict[str, object]],
        completed_tuples: set[tuple[int, int, int]], template_sha256: str) -> \
        tuple[list[dict[str, object]], dict[str, object]]:
    execution_path = raw_root / "frozen_window_execution.json"
    campaign_path = raw_root / "window_campaign.json"
    if (not raw_root.is_dir() or raw_root.is_symlink() or
            any(not path.is_file() or path.is_symlink()
                for path in (execution_path, campaign_path))):
        raise RuntimeError("completion root/lifecycle files are missing or symlinks")
    execution = load_json_strict(execution_path)
    campaign = load_json_strict(campaign_path)
    campaign_fields = {"schema", "state", "purpose", "accepted_window_sha256",
                       "execution_manifest_sha256", "source", "build_manifest_sha256",
                       "executable_sha256", "input_sha256", "oracle_header_sha256",
                       "series_manifest_sha256", "backend", "ranks", "domain",
                       "immutable_root_bindings", "cases", "qualification_claim"}
    if (not isinstance(execution, dict) or set(execution) != WINDOW_EXECUTION_FIELDS or
            execution.get("schema") !=
            "athenak_z4c_cartoon_mms_window_execution_v1" or
            execution.get("state") != "execution_finalized" or
            execution.get("purpose") != "characterization_completion" or
            execution.get("immutable_root_bindings") != expected_root_bindings or
            not isinstance(campaign, dict) or set(campaign) != campaign_fields or
            campaign.get("state") != "execution_finalized" or
            campaign.get("purpose") != "characterization_completion" or
            campaign.get("execution_manifest_sha256") != sha256(execution_path) or
            campaign.get("qualification_claim") is not None):
        raise RuntimeError("characterization-completion lifecycle schema differs")
    for field in ("source", "build_manifest_sha256", "executable_sha256",
                  "input_sha256", "oracle_header_sha256", "series_manifest_sha256",
                  "backend", "ranks", "domain", "immutable_root_bindings"):
        if campaign.get(field) != execution.get(field):
            raise RuntimeError("completion campaign/execution identity differs")
    authorization = raw_root / "authorization"
    files = execution.get("authorization_files")
    if (not isinstance(files, dict) or set(files) !=
            {"accepted_window.json", "candidate_manifest.json", "review_artifact.json"}):
        raise RuntimeError("completion authorization artifacts changed")
    require_exact_regular_files(authorization, set(files),
                                "completion authorization directory")
    if any(sha256(authorization / name) != digest for name, digest in files.items()):
        raise RuntimeError("completion authorization artifacts changed")
    accepted_path = authorization / "accepted_window.json"
    accepted = load_json_strict(accepted_path)
    if (not isinstance(accepted, dict) or
            accepted.get("immutable_root_bindings_sha256") !=
            canonical_digest(expected_root_bindings) or
            execution.get("accepted_review") != accepted.get("review")):
        raise RuntimeError("completion accepted window uses different immutable roots")
    required = validate_frozen_window(
        accepted, template_sha256, execution["source"],
        execution["build_manifest_sha256"], execution["executable_sha256"],
        execution["input_sha256"], execution["oracle_header_sha256"],
        execution["series_manifest_sha256"], completed_tuples, authorization,
        execution["backend"], execution["ranks"])
    if (sha256(accepted_path) != execution["accepted_window_sha256"] or
            campaign.get("accepted_window_sha256") !=
            execution["accepted_window_sha256"] or
            [tuple(item) for item in execution["execution_tuples"]] != required):
        raise RuntimeError("completion accepted-window binding differs")
    attempts = execution.get("attempts")
    if (not isinstance(attempts, list) or
            [tuple(item.get("tuple", ())) for item in attempts] != required or
            any(set(item) != {"tuple", "status", "case_manifest_sha256", "reason"} or
                item.get("status") != "complete" or
                not re.fullmatch(r"[0-9a-f]{64}",
                                 str(item.get("case_manifest_sha256", "")))
                for item in attempts) or
            execution.get("stop_reason") !=
            "accepted_window_execution_complete_pending_merged_analysis"):
        raise RuntimeError("completion attempt ledger differs")
    expected_operators = [item["name"] for item in load_json_strict(
        Path(__file__).resolve().parents[3] /
        "tst/unit/z4c/z4c_cartoon_derivatives_series.json")["series"]]
    cases = []
    observed = []
    for augmented in campaign["cases"]:
        if not isinstance(augmented, dict):
            raise RuntimeError("completion campaign case is not an object")
        case = raw_root / f"{augmented.get('case_id')}-{augmented.get('case_uuid')}"
        order, resolution, phase = (json_integer(augmented, "spatial_order", 2),
                                    json_integer(augmented, "resolution", 1),
                                    json_integer(augmented, "phase"))
        stored, _, _, manifest_sha = verify_complete_case_evidence(
            case, execution["source"], execution["build_manifest_sha256"],
            execution["executable_sha256"], execution["backend"],
            execution["ranks"], tuple(execution["domain"]), order, resolution,
            phase, expected_operators, Path(__file__).resolve().parents[3] /
            "tst/inputs/z4c_cartoon_derivatives.athinput",
            Path(__file__).resolve().parents[3] /
            "tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py")
        manifest = load_json_strict(case / "manifest.json")
        identity = manifest["identity"]
        if augmented != {**stored, "case_manifest_sha256": manifest_sha}:
            raise RuntimeError("completion campaign/result differs")
        expected_input = render_input(
            Path(__file__).resolve().parents[3] /
            "tst/inputs/z4c_cartoon_derivatives.athinput", order, resolution,
            phase, "cartoon_mms", tuple(execution["domain"]))
        if (identity.get("source") != execution["source"] or
                identity.get("athena_sha256") != execution["executable_sha256"] or
                identity.get("build_manifest_sha256") !=
                execution["build_manifest_sha256"] or
                identity.get("input_sha256") !=
                hashlib.sha256(expected_input.encode()).hexdigest() or
                identity.get("backend_required") != execution["backend"] or
                identity.get("ranks") != execution["ranks"] or
                list(identity.get("domain", ())) != execution["domain"]):
            raise RuntimeError("completion case identity differs")
        result = dict(augmented)
        result["_replay_case_directory"] = str(case)
        result["_evidence_source"] = "characterization_completion"
        cases.append(result)
        observed.append((order, resolution, phase))
    if (observed != required or
            execution.get("result_set_sha256") != canonical_digest(
                [case["case_manifest_sha256"] for case in cases]) or
            [item["case_manifest_sha256"] for item in attempts] !=
            [case["case_manifest_sha256"] for case in cases]):
        raise RuntimeError("completion result set differs from accepted window")
    expected_entries = (
        {"preflight.json", "frozen_window_execution.json",
         "window_campaign.json", "authorization"} |
        {f"{case['case_id']}-{case['case_uuid']}" for case in cases})
    require_exact_root_entries(raw_root, expected_entries, "completion root")
    binding = {"root": str(raw_root), "purpose": "characterization_completion",
               "execution_sha256": sha256(execution_path),
               "campaign_sha256": sha256(campaign_path),
               "accepted_window_sha256": sha256(accepted_path),
               "case_manifest_sha256": [case["case_manifest_sha256"] for case in cases]}
    return cases, binding


def convergence_gate(cases: list[dict[str, object]], case_root: Path, output: Path,
                     series_manifest: Path,
                     resolutions_by_order: dict[int, tuple[int, ...]],
                     allow_legacy_nullable: bool = False,
                     evidence_scope: str = "single_source_diagnostic") -> list[str]:
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
        evidence_source = str(result.get(
            "_evidence_source", "job56586376" if "_replay_case_directory" in result
            else "current_output"))
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
                                "floor_id": floor_id,
                                "evidence_source": evidence_source})

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
                        "evidence_source": evidence_source,
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
                            "legacy_evaluation": None,
                            "outcome_reason": "incomplete_resolution_set",
                            "outcome_detail": reason, "passed": False})
            continue
        selected = all_values
        mask = series.split("|")[1]
        probe_classification = (selected[0].get("probe_classification")
                                if selected else None)
        expected, margin = rate_policy(order, mask, lane, norm,
                                       probe_classification)
        legacy_evaluation = evaluate_legacy_rate_samples(
            selected, expected, margin, lane)
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
                  "samples": selected, "legacy_evaluation": legacy_evaluation}
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
    inconclusive_reasons = {"saturated_insufficient", "insufficient_ratios"}
    coefficient_partition = {
        "norm_inconclusive": sum(record["source"] == "norm" and
                                 not record["passed"] and
                                 record.get("outcome_reason") in inconclusive_reasons
                                 for record in records),
        "norm_rate_miss": sum(record["source"] == "norm" and
                              not record["passed"] and
                              record.get("outcome_reason") not in inconclusive_reasons
                              for record in records),
        "probe_inconclusive": sum(record["source"] == "probe" and
                                  not record["passed"] and
                                  record.get("outcome_reason") in inconclusive_reasons
                                  for record in records),
        "probe_rate_miss": sum(record["source"] == "probe" and
                               not record["passed"] and
                               record.get("outcome_reason") not in inconclusive_reasons
                               for record in records),
    }
    legacy_partition = {
        "norm_inconclusive": sum(
            record["source"] == "norm" and
            record.get("legacy_evaluation", {}).get("outcome_reason") ==
            "legacy_inconclusive" for record in records
            if record.get("legacy_evaluation") is not None),
        "norm_rate_miss": sum(
            record["source"] == "norm" and
            record.get("legacy_evaluation", {}).get("outcome_reason") ==
            "legacy_rate_miss" for record in records
            if record.get("legacy_evaluation") is not None),
        "probe_inconclusive": sum(
            record["source"] == "probe" and
            record.get("legacy_evaluation", {}).get("outcome_reason") ==
            "legacy_inconclusive" for record in records
            if record.get("legacy_evaluation") is not None),
        "probe_rate_miss": sum(
            record["source"] == "probe" and
            record.get("legacy_evaluation", {}).get("outcome_reason") ==
            "legacy_rate_miss" for record in records
            if record.get("legacy_evaluation") is not None),
    }
    write_atomic(output / "convergence.json", {
        "schema": SCHEMA, "series_manifest_sha256": sha256(series_manifest),
        "diagnostic_resolutions_by_order": {str(order): list(values)
                                             for order, values in
                                             resolutions_by_order.items()},
        "coefficient_floor_policy_sha256": canonical_digest(policy),
        "coefficient_floor_complexity": "O(171*nx1); active-z multiplicity analytic",
        "evidence_scope": evidence_scope,
        "floor_decompositions": sorted(floor_evidence.values(),
                                       key=lambda item: item["floor_id"]),
        "legacy_normalization_actions": sorted(normalization_actions),
        "records": records, "exact_records": exact_records,
        "legacy_pre_coefficient_floor_partition": legacy_partition,
        "coefficient_floor_partition": coefficient_partition,
        "artifacts": {"convergence.csv": sha256(csv_path),
                      "convergence_rates.pgfplots.dat": sha256(data_path),
                      "convergence_plot.tex": sha256(plot_path)},
        "failures": failures, "passed": not failures})
    return failures


def validate_rank_campaign_header(reference: object, current_ranks: int,
                                  campaign_mode: str,
                                  accepted_window_sha256: str | None,
                                  source: dict[str, str], backend: str,
                                  build_manifest_sha256: str) -> None:
    if (not isinstance(reference, dict) or set(reference) != CAMPAIGN_FIELDS or
            reference.get("schema") != SCHEMA or
            not isinstance(reference.get("cases"), list) or
            reference.get("campaign_mode") != campaign_mode or
            reference.get("accepted_window_sha256") != accepted_window_sha256 or
            reference.get("source") != source or reference.get("backend") != backend or
            reference.get("build_manifest_sha256") != build_manifest_sha256 or
            {reference.get("ranks"), current_ranks} != {2, 4}):
        raise RuntimeError("reference campaign differs from the exact evidence schema")


def validate_final_reference_aggregates(
        convergence: object, preflight: object, artifacts: dict[str, object],
        execution: dict[str, object], reference_ranks: int,
        accepted_window_sha256: str,
        required: list[tuple[int, int, int]],
        expected_record_keys: set[tuple[object, ...]],
        expected_exact_keys: set[tuple[object, ...]],
        expected_policy_sha256: str, rank_comparison: bool) -> None:
    internal_artifacts = {name: artifacts[name] for name in
                          ("convergence.csv", "convergence_rates.pgfplots.dat",
                           "convergence_plot.tex")}
    resolutions_by_order = {
        str(order): sorted({resolution for item_order, resolution, _ in required
                            if item_order == order})
        for order in sorted({item[0] for item in required})}
    records = convergence.get("records") if isinstance(convergence, dict) else None
    exact_records = (convergence.get("exact_records")
                     if isinstance(convergence, dict) else None)
    valid_records = (records if isinstance(records, list) and
                     all(isinstance(item, dict) for item in records) else [])
    valid_exact = (exact_records if isinstance(exact_records, list) and
                   all(isinstance(item, dict) for item in exact_records) else [])
    record_keys = [
        (item.get("source"), item.get("order"), item.get("phase"),
         item.get("series"), item.get("lane"), item.get("norm"))
        for item in valid_records]
    exact_keys = [
        (item.get("source"), item.get("order"), item.get("phase"),
         item.get("resolution"), item.get("operator"), item.get("mask"),
         item.get("side"), item.get("layer_index"))
        for item in valid_exact]
    floor_decompositions = (convergence.get("floor_decompositions")
                            if isinstance(convergence, dict) else None)
    floor_ids = ([] if not isinstance(floor_decompositions, list) else
                 [item.get("floor_id") for item in floor_decompositions
                  if isinstance(item, dict)])
    referenced_floors = {
        sample.get("floor_id") for record in valid_records
        for sample in record.get("samples", [])
        if isinstance(sample, dict)}
    legacy_outcomes = [
        (item, item.get("legacy_evaluation")
         if isinstance(item.get("legacy_evaluation"), dict) else {})
        for item in valid_records]
    legacy_partition = {
        "norm_inconclusive": sum(
            item.get("source") == "norm" and
            legacy.get("outcome_reason") ==
            "legacy_inconclusive" for item, legacy in legacy_outcomes),
        "norm_rate_miss": sum(
            item.get("source") == "norm" and
            legacy.get("outcome_reason") ==
            "legacy_rate_miss" for item, legacy in legacy_outcomes),
        "probe_inconclusive": sum(
            item.get("source") == "probe" and
            legacy.get("outcome_reason") ==
            "legacy_inconclusive" for item, legacy in legacy_outcomes),
        "probe_rate_miss": sum(
            item.get("source") == "probe" and
            legacy.get("outcome_reason") ==
            "legacy_rate_miss" for item, legacy in legacy_outcomes)}
    zero_partition = {name: 0 for name in legacy_partition}
    if (not isinstance(convergence, dict) or set(convergence) != CONVERGENCE_FIELDS or
            convergence.get("schema") != SCHEMA or
            convergence.get("series_manifest_sha256") !=
            execution["series_manifest_sha256"] or
            convergence.get("diagnostic_resolutions_by_order") !=
            resolutions_by_order or
            convergence.get("coefficient_floor_policy_sha256") !=
            expected_policy_sha256 or
            convergence.get("coefficient_floor_complexity") !=
            "O(171*nx1); active-z multiplicity analytic" or
            convergence.get("legacy_normalization_actions") != [] or
            convergence.get("evidence_scope") !=
            "fresh_single_source_final_qualification" or
            convergence.get("artifacts") != internal_artifacts or
            convergence.get("failures") != [] or convergence.get("passed") is not True or
            not expected_record_keys or not expected_exact_keys or
            len(record_keys) != len(expected_record_keys) or
            set(record_keys) != expected_record_keys or
            len(exact_keys) != len(expected_exact_keys) or
            set(exact_keys) != expected_exact_keys or
            any(item.get("passed") is not True or
                item.get("outcome_reason") != "pass" or
                not isinstance(item.get("samples"), list) or
                len(item["samples"]) != len(
                    resolutions_by_order[str(item.get("order"))]) or
                {sample.get("resolution") for sample in item["samples"]
                 if isinstance(sample, dict)} !=
                set(resolutions_by_order[str(item.get("order"))]) or
                item.get("diagnostic_resolutions") !=
                resolutions_by_order[str(item.get("order"))]
                for item in valid_records) or
            any(item.get("passed") is not True for item in valid_exact) or
            any(not isinstance(value, str) or
                re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in floor_ids) or
            len(floor_ids) != len(set(floor_ids)) or
            set(floor_ids) != referenced_floors or
            convergence.get("legacy_pre_coefficient_floor_partition") !=
            legacy_partition or
            convergence.get("coefficient_floor_partition") != zero_partition):
        raise RuntimeError("rank-reference convergence result is not a passing fresh final")
    expected_orders = sorted({item[0] for item in required})
    expected_resolutions = sorted({item[1] for item in required})
    expected_phases = sorted({item[2] for item in required})
    expected_certification = validate_certified_domain(
        tuple(execution["domain"]),
        {int(order): tuple(values) for order, values in resolutions_by_order.items()},
        True)
    forecast = output_forecast(
        {int(order): tuple(values) for order, values in resolutions_by_order.items()},
        expected_phases, reference_ranks, rank_comparison, required)
    if (not isinstance(preflight, dict) or set(preflight) != PREFLIGHT_FIELDS or
            preflight.get("schema") != SCHEMA or preflight.get("state") != "preflight" or
            preflight.get("orders") != expected_orders or
            preflight.get("resolutions") != expected_resolutions or
            preflight.get("resolutions_by_order") != resolutions_by_order or
            preflight.get("phases") != expected_phases or
            preflight.get("ranks") != reference_ranks or
            preflight.get("campaign_mode") != "accepted_frozen_window" or
            preflight.get("stage_id") is not None or
            preflight.get("frozen_window_sha256") != accepted_window_sha256 or
            preflight.get("run_tuples") != [list(item) for item in required] or
            preflight.get("domain_certification") != expected_certification or
            preflight.get("series_manifest_sha256") !=
            execution["series_manifest_sha256"] or
            preflight.get("search_manifest_sha256") != sha256(
                Path(__file__).resolve().parents[3] /
                "tst/inputs/z4c_cartoon_mms_search_manifest.json") or
            any(preflight.get(name) != value for name, value in forecast.items()) or
            isinstance(preflight.get("free_bytes_before_campaign"), bool) or
            not isinstance(preflight.get("free_bytes_before_campaign"), int) or
            preflight["free_bytes_before_campaign"] <
            2 * forecast["estimated_output_bytes_upper_bound"]):
        raise RuntimeError("rank-reference preflight identity differs")


RECOMPUTED_CONVERGENCE_PRODUCTS = {
    "convergence.json", "convergence.csv",
    "convergence_rates.pgfplots.dat", "convergence_plot.tex",
}


def require_recomputed_reference_products(archived: Path, recomputed: Path) -> None:
    require_exact_regular_files(
        recomputed, RECOMPUTED_CONVERGENCE_PRODUCTS,
        "recomputed rank-reference convergence products")
    for name in sorted(RECOMPUTED_CONVERGENCE_PRODUCTS):
        archived_path = archived / name
        recomputed_path = recomputed / name
        if (not archived_path.is_file() or archived_path.is_symlink() or
                archived_path.read_bytes() != recomputed_path.read_bytes()):
            raise RuntimeError(
                f"rank-reference {name} differs from verified-case recomputation")


def reference_artifact_files(artifacts: object) -> set[str]:
    allowed = {frozenset(FINAL_CONVERGENCE_ARTIFACTS),
               frozenset(FINAL_CONVERGENCE_ARTIFACTS | OPTIONAL_RANK_ARTIFACTS)}
    if not isinstance(artifacts, dict) or frozenset(artifacts) not in allowed:
        raise RuntimeError("rank-reference convergence inventory is malformed")
    files = set(FINAL_CONVERGENCE_ARTIFACTS)
    if OPTIONAL_RANK_ARTIFACTS <= set(artifacts):
        if (not re.fullmatch(r"[0-9a-f]{64}", str(
                artifacts["reference_campaign_sha256"])) or
                not isinstance(artifacts["reference_case_manifest_sha256"], list) or
                any(not re.fullmatch(r"[0-9a-f]{64}", str(value))
                    for value in artifacts["reference_case_manifest_sha256"])):
            raise RuntimeError("rank-reference comparison binding is malformed")
        files.add("rank_comparison.json")
    return files


def verify_rank_reference_root(
        reference_root: Path, current_ranks: int, campaign_mode: str,
        accepted_window_sha256: str | None, source: dict[str, str], backend: str,
        build_manifest_sha256: str) -> tuple[dict[str, object],
                                               list[dict[str, object]]]:
    if not reference_root.is_dir() or reference_root.is_symlink():
        raise RuntimeError("rank-reference root is missing or a symlink")
    campaign_path = reference_root / "campaign.json"
    execution_path = reference_root / "frozen_window_execution.json"
    if any(not path.is_file() or path.is_symlink()
           for path in (campaign_path, execution_path)):
        raise RuntimeError("rank-reference campaign/execution is missing or a symlink")
    reference = load_json_strict(campaign_path)
    validate_rank_campaign_header(reference, current_ranks, campaign_mode,
                                  accepted_window_sha256, source, backend,
                                  build_manifest_sha256)
    execution = load_json_strict(execution_path)
    if (not isinstance(execution, dict) or set(execution) != WINDOW_EXECUTION_FIELDS or
            execution.get("schema") !=
            "athenak_z4c_cartoon_mms_window_execution_v1" or
            execution.get("state") != "execution_finalized" or
            execution.get("purpose") != "final_qualification" or
            execution.get("source") != source or
            execution.get("build_manifest_sha256") != build_manifest_sha256 or
            execution.get("backend") != backend or
            execution.get("ranks") != reference.get("ranks") or
            execution.get("immutable_root_bindings") != [] or
            execution.get("accepted_window_sha256") != accepted_window_sha256 or
            execution.get("stop_reason") !=
            "accepted_window_execution_complete_pending_merged_analysis" or
            reference.get("window_execution_sha256") != sha256(execution_path)):
        raise RuntimeError("rank-reference execution identity differs")
    authorization = reference_root / "authorization"
    authorization_files = execution.get("authorization_files")
    if not isinstance(authorization_files, dict) or set(authorization_files) != {
            "accepted_window.json", "candidate_manifest.json", "review_artifact.json"}:
        raise RuntimeError("rank-reference authorization inventory differs")
    require_exact_regular_files(authorization, set(authorization_files),
                                "rank-reference authorization directory")
    if any(sha256(authorization / name) != digest
           for name, digest in authorization_files.items()):
        raise RuntimeError("rank-reference authorization artifact changed")
    accepted_path = authorization / "accepted_window.json"
    accepted = load_json_strict(accepted_path)
    required = validate_frozen_window(
        accepted, sha256(Path(__file__).resolve().parents[3] /
                         "tst/inputs/z4c_cartoon_mms_search_manifest.json"),
        source, build_manifest_sha256, execution["executable_sha256"],
        execution["input_sha256"], execution["oracle_header_sha256"],
        execution["series_manifest_sha256"], set(), authorization, backend,
        reference["ranks"])
    if (sha256(accepted_path) != accepted_window_sha256 or
            execution.get("accepted_review") != accepted.get("review") or
            [tuple(item) for item in execution.get("execution_tuples", [])] != required):
        raise RuntimeError("rank-reference accepted-window binding differs")
    series_inventory = load_json_strict(
        Path(__file__).resolve().parents[3] /
        "tst/unit/z4c/z4c_cartoon_derivatives_series.json")
    expected_operators = [item["name"] for item in series_inventory["series"]]
    metadata = {item["name"]: item for item in series_inventory["series"]}
    expected_record_keys: set[tuple[object, ...]] = set()
    expected_exact_keys: set[tuple[object, ...]] = set()
    verified_cases = []
    for advertised in reference["cases"]:
        if not isinstance(advertised, dict):
            raise RuntimeError("rank-reference campaign case is malformed")
        order = json_integer(advertised, "spatial_order", 2)
        resolution = json_integer(advertised, "resolution", 1)
        phase = json_integer(advertised, "phase")
        case = reference_root / \
            f"{advertised.get('case_id')}-{advertised.get('case_uuid')}"
        stored, rows, probes, manifest_sha = verify_complete_case_evidence(
            case, source, build_manifest_sha256, execution["executable_sha256"],
            backend, reference["ranks"], tuple(execution["domain"]), order,
            resolution, phase, expected_operators,
            Path(__file__).resolve().parents[3] /
            "tst/inputs/z4c_cartoon_derivatives.athinput",
            Path(__file__).resolve().parents[3] /
            "tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py")
        if advertised != {**stored, "case_manifest_sha256": manifest_sha}:
            raise RuntimeError("rank-reference campaign/result differs")
        for row in rows:
            item = metadata[row["operator"]]
            if item["classification"] != "truncating":
                expected_exact_keys.add((
                    "norm", order, phase, resolution, row["operator"], row["mask"],
                    None, None))
                continue
            clean_norms = ["l1", "l2", "linfinity"]
            if boolean_value(row, "cylindrical_applicable"):
                clean_norms += ["cyl_l1", "cyl_l2", "cyl_linfinity"]
            for lane in item["convergence_lanes"]:
                norms = clean_norms if lane == "clean" else ["l1", "l2", "linfinity"]
                for norm in norms:
                    expected_record_keys.add((
                        "norm", order, phase, row["operator"] + "|" + row["mask"],
                        lane, norm))
        for row in probes:
            if row["classification"] == "diagnostic_axis":
                continue
            item = metadata[row["operator"]]
            if item["classification"] != "truncating":
                expected_exact_keys.add((
                    "probe", order, phase, resolution, row["operator"], row["mask"],
                    row["side"], integer_value(row, "layer_index")))
            else:
                expected_record_keys.add((
                    "probe", order, phase, probe_series_identity(row), "clean",
                    "raw_error"))
        verified_cases.append(advertised)
    attempts = execution.get("attempts")
    manifests = [case["case_manifest_sha256"] for case in verified_cases]
    tuples = [(case["spatial_order"], case["resolution"], case["phase"])
              for case in verified_cases]
    if (not isinstance(attempts, list) or
            [tuple(item.get("tuple", ())) for item in attempts] != tuples or
            any(not isinstance(item, dict) or item.get("status") != "complete"
                for item in attempts) or
            [item.get("case_manifest_sha256") for item in attempts] != manifests or
            execution.get("result_set_sha256") != canonical_digest(manifests)):
        raise RuntimeError("rank-reference attempt/result-set ledger differs")
    artifacts = reference.get("convergence_artifacts")
    artifact_files = reference_artifact_files(artifacts)
    for name in artifact_files:
        path = reference_root / name
        if (Path(name).name != name or not path.is_file() or path.is_symlink() or
                not isinstance(artifacts[name], str) or
                sha256(path) != artifacts[name]):
            raise RuntimeError("rank-reference convergence artifact changed")
    convergence = load_json_strict(reference_root / "convergence.json")
    preflight = load_json_strict(reference_root / "preflight.json")
    validate_final_reference_aggregates(
        convergence, preflight, artifacts, execution, reference["ranks"],
        str(accepted_window_sha256), required, expected_record_keys,
        expected_exact_keys, canonical_digest(series_inventory["roundoff_policy"]),
        OPTIONAL_RANK_ARTIFACTS <= set(artifacts))
    resolutions_by_order = {
        order: tuple(sorted({resolution for item_order, resolution, _ in required
                             if item_order == order}))
        for order in sorted({item[0] for item in required})}
    with tempfile.TemporaryDirectory(prefix="cartoon-mms-rank-reference-") as directory:
        recomputed = Path(directory)
        failures = convergence_gate(
            verified_cases, reference_root, recomputed,
            Path(__file__).resolve().parents[3] /
            "tst/unit/z4c/z4c_cartoon_derivatives_series.json",
            resolutions_by_order,
            evidence_scope="fresh_single_source_final_qualification")
        if failures:
            raise RuntimeError(
                "rank-reference verified cases fail fresh aggregate recomputation")
        require_recomputed_reference_products(reference_root, recomputed)
    expected_entries = ({"campaign.json", "frozen_window_execution.json",
                         "authorization"} | artifact_files |
                        {f"{case['case_id']}-{case['case_uuid']}"
                         for case in verified_cases})
    require_exact_root_entries(reference_root, expected_entries,
                               "rank-reference root")
    return reference, verified_cases


def compare_rank_campaigns(cases: list[dict[str, object]], output: Path,
                           reference_root: Path, current_ranks: int,
                           campaign_mode: str,
                           accepted_window_sha256: str | None,
                           source: dict[str, str], backend: str,
                           build_manifest_sha256: str) -> dict[str, object]:
    reference_root = reference_root.absolute()
    reference_campaign_path = reference_root / "campaign.json"
    reference, verified_reference_cases = verify_rank_reference_root(
        reference_root, current_ranks, campaign_mode, accepted_window_sha256,
        source, backend, build_manifest_sha256)
    reference_case_manifests = [item.get("case_manifest_sha256")
                                for item in verified_reference_cases]
    if (not reference_case_manifests or None in reference_case_manifests or
            any(not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in reference_case_manifests)):
        raise RuntimeError("reference campaign does not bind every case manifest")
    reference_cases = {(item["spatial_order"], item["resolution"], item["phase"]): item
                       for item in verified_reference_cases}
    current_keys = {(item["spatial_order"], item["resolution"], item["phase"])
                    for item in cases}
    if (len(reference_cases) != len(reference["cases"]) or
            len(current_keys) != len(cases) or set(reference_cases) != current_keys):
        raise RuntimeError("rank campaigns do not contain the same exact tuple set")
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
    required = {"schema", "state", "qualification_domain", "qualification_window",
                "policy", "immutable_roots", "stages", "materialization"}
    if (not isinstance(value, dict) or set(value) != required or
            value.get("schema") != "athenak_z4c_cartoon_mms_search_v2" or
            value.get("state") != "checked_in_template" or
            value.get("qualification_domain") != list(QUALIFICATION_DOMAIN) or
            value.get("qualification_window") is not None):
        raise RuntimeError("search manifest differs from the exact template schema")
    policy = value.get("policy")
    if (not isinstance(policy, dict) or policy.get("diagnostic_resolutions") !=
            list(DIAGNOSTIC_RESOLUTIONS) or policy.get("resolution_pools") != {
                "2": [32, 64, 128, 256, 512, 1024, 2048, 4096],
                "4": [32, 64, 128, 256],
                "6": [32, 48, 64, 80, 96, 112, 128, 160, 192, 256]} or
            policy.get("minimum_consecutive_unsaturated_ratios") != 2 or
            policy.get("series_count") != 171):
        raise RuntimeError("search policy/pools differ from the frozen contract")
    roots, stages = value.get("immutable_roots"), value.get("stages")
    if not isinstance(roots, dict) or set(roots) != {"job56586376", "job56587561"} or \
       not isinstance(stages, dict):
        raise RuntimeError("search evidence ledger is incomplete")
    diagnostic = roots["job56586376"]
    if (diagnostic.get("case_manifest_set_sha256") !=
            "d2aa1ff2ff9b68302170e0271d3f6fca150d86acb183f9d89af62965427d2aa3" or
            diagnostic.get("case_count") != 96 or diagnostic.get("phases") !=
            list(range(8))):
        raise RuntimeError("job56586376 immutable case binding changed")
    characterization = roots["job56587561"]
    complete = {tuple(item["tuple"]) for item in characterization.get("completed", [])}
    bounded = {tuple(item["tuple"]) for item in
               characterization.get("bounded_characterization", [])}
    attempts = characterization.get("attempts", [])
    if (characterization.get("canonical_inventory_sha256") !=
            "b884d46e636e2beea8ca5eaa7f85d4745968c4a1465c371a70fc8765c4441c6d" or
            characterization.get("build_manifest_sha256") !=
            "60356b252d40b3657b07c681ed6dcc72d94a5be341455a4dce42ba8978a0a06d" or
            bounded != {(4, 16, 0), (6, 16, 0)} or
            complete != {(2, 512, 0), (2, 1024, 0)} or len(attempts) != 1 or
            attempts[0].get("tuple") != [2, 2048, 0] or
            attempts[0].get("status") != "out_of_memory"):
        raise RuntimeError("job56587561 phase/attempt ledger changed")
    seen = complete | {(order, resolution, phase) for order in (2, 4, 6)
                       for resolution in DIAGNOSTIC_RESOLUTIONS for phase in range(8)}
    stage_tuples = []
    for stage_id, stage in stages.items():
        tuples = [tuple(item) for item in stage.get("tuples", [])]
        if len(tuples) != len(set(tuples)):
            raise RuntimeError(f"stage {stage_id} contains duplicate tuples")
        if stage.get("rerun_completed") is not False or seen.intersection(tuples):
            raise RuntimeError(f"stage {stage_id} would rerun completed evidence")
        stage_tuples.extend(tuples)
    if len(stage_tuples) != len(set(stage_tuples)):
        raise RuntimeError("search stages overlap")
    if (stages.get("o6_phase0_stage1", {}).get("tuples") !=
            [[6, 48, 0], [6, 80, 0], [6, 96, 0]] or
            stages.get("o6_phase0_stage1", {}).get("state") !=
            "authorized_missing_phase0" or
            stages.get("o2_phase0_continuation", {}).get("state") !=
            "blocked_pending_resource_decision" or
            stages.get("qualification_extra_phases", {}).get("tuples") != []):
        raise RuntimeError("search phase/stage authorization changed")
    material = value.get("materialization")
    if not isinstance(material, dict) or any(item not in (None, [])
                                             for item in material.values()):
        raise RuntimeError("checked-in search template is already materialized")
    return value


def materialize_search_stage(template: dict[str, object], stage_id: str,
                             source: dict[str, str], build_manifest_sha256: str,
                             executable_sha256: str, input_sha256: str,
                             oracle_sha256: str, series_sha256: str,
                             backend: str, ranks: int,
                             root_bindings: list[dict[str, object]]) -> dict[str, object]:
    stage = template["stages"].get(stage_id)
    if not isinstance(stage, dict) or stage.get("state") != "authorized_missing_phase0":
        raise RuntimeError(f"stage {stage_id} is not authorized for execution")
    value = json.loads(json.dumps(template))
    value["state"] = "prelaunch_bound"
    material = value["materialization"]
    material.update({"stage_id": stage_id, "source_commit": source["commit"],
                     "source_tree": source["tree"], "kokkos_commit": source["kokkos"],
                     "executable_sha256": executable_sha256,
                     "build_manifest_sha256": build_manifest_sha256,
                     "input_sha256": input_sha256,
                     "oracle_header_sha256": oracle_sha256,
                     "series_manifest_sha256": series_sha256, "backend": backend,
                     "ranks": ranks, "domain": list(QUALIFICATION_DOMAIN),
                     "attempt_id": str(uuid.uuid4()),
                     "immutable_root_bindings": root_bindings})
    names = {item.get("name") for item in root_bindings}
    partial_pattern = re.compile(rf"partial:{re.escape(stage_id)}:[0-9a-f]{{16}}")
    if (len(root_bindings) < 2 or
            [item.get("name") for item in root_bindings[:2]] !=
            ["job56586376", "job56587561"] or len(names) != len(root_bindings) or
            any(not isinstance(item, dict) or
                not partial_pattern.fullmatch(str(item.get("name", "")))
                for item in root_bindings[2:])):
        raise RuntimeError("stage materialization requires exact prior/partial roots")
    return value


def transition_search_stage(value: dict[str, object], state: str,
                            attempts: list[dict[str, object]], stop_reason: str,
                            analysis_sha256: str | None = None) -> None:
    allowed = {"prelaunch_bound": {"stage_partial", "stage_finalized"},
               "stage_partial": {"stage_finalized"},
               "stage_finalized": {"analysis_finalized"}}
    if state not in allowed.get(str(value.get("state")), set()):
        raise RuntimeError("invalid search-manifest lifecycle transition")
    material = value["materialization"]
    expected_tuples = value["stages"][material["stage_id"]]["tuples"]
    for attempt in attempts:
        if (set(attempt) != {"tuple", "status", "case_manifest_sha256", "reason"} or
                attempt["status"] not in
                {"complete", "failed", "not_attempted_after_stop"}):
            raise RuntimeError("stage attempt differs from exact lifecycle schema")
    if ([item["tuple"] for item in attempts] != expected_tuples or
            sum(item["status"] == "failed" for item in attempts) > 1 or
            any(item["case_manifest_sha256"] is not None
                for item in attempts if item["status"] != "complete") or
            any(not re.fullmatch(r"[0-9a-f]{64}", str(item["case_manifest_sha256"]))
                for item in attempts if item["status"] == "complete")):
        raise RuntimeError("stage attempt ledger does not cover exact authorized tuples")
    failed = next((index for index, item in enumerate(attempts)
                   if item["status"] == "failed"), None)
    if failed is not None and any(item["status"] != "not_attempted_after_stop"
                                  for item in attempts[failed + 1:]):
        raise RuntimeError("stage attempted work after its first failure")
    value["state"] = state
    material["attempts"] = attempts
    completed = [item["case_manifest_sha256"] for item in attempts
                 if item["status"] == "complete"]
    material["case_manifest_sha256"] = completed
    material["result_set_sha256"] = canonical_digest(completed)
    material["stop_reason"] = stop_reason
    material["analysis_sha256"] = analysis_sha256


def stage_campaign_record(stage: dict[str, object], stage_path: Path,
                          cases: list[dict[str, object]],
                          failed_case_inventories: list[dict[str, object]]) -> \
        dict[str, object]:
    material = stage["materialization"]
    return {"schema": SCHEMA, "state": stage["state"],
            "stage_manifest_sha256": sha256(stage_path),
            "source": {"commit": material["source_commit"],
                       "tree": material["source_tree"],
                       "kokkos": material["kokkos_commit"]},
            "build_manifest_sha256": material["build_manifest_sha256"],
            "executable_sha256": material["executable_sha256"],
            "input_sha256": material["input_sha256"],
            "oracle_header_sha256": material["oracle_header_sha256"],
            "series_manifest_sha256": material["series_manifest_sha256"],
            "backend": material["backend"], "ranks": material["ranks"],
            "domain": material["domain"],
            "immutable_root_bindings": material["immutable_root_bindings"],
            "cases": cases, "failed_case_inventories": failed_case_inventories}


def stage_missing_tuples(stage: dict[str, object]) -> list[tuple[int, int, int]]:
    material = stage.get("materialization", {})
    attempts = material.get("attempts", []) if isinstance(material, dict) else []
    return [tuple(item["tuple"]) for item in attempts
            if item.get("status") in {"failed", "not_attempted_after_stop"}]


def validate_stage_lineage(
        prior_cases: list[dict[str, object]],
        prior_missing: set[tuple[int, int, int]],
        current_cases: list[dict[str, object]],
        current_missing: set[tuple[int, int, int]],
        authorized: set[tuple[int, int, int]]) -> None:
    def completed(cases: list[dict[str, object]]) -> \
            dict[tuple[int, int, int], str]:
        result = {}
        for case in cases:
            item = (int(case["spatial_order"]), int(case["resolution"]),
                    int(case["phase"]))
            digest = case.get("case_manifest_sha256")
            if (item in result or item not in authorized or
                    not isinstance(digest, str) or
                    not re.fullmatch(r"[0-9a-f]{64}", digest)):
                raise RuntimeError("partial lineage case ledger is malformed")
            result[item] = digest
        return result
    previous = completed(prior_cases)
    current = completed(current_cases)
    if (set(previous) | prior_missing != authorized or
            set(previous) & prior_missing or
            set(current) | current_missing != authorized or
            set(current) & current_missing or
            not set(previous).issubset(current) or
            any(current[item] != digest for item, digest in previous.items()) or
            not current_missing.issubset(prior_missing) or
            not (set(current) - set(previous)).issubset(prior_missing)):
        raise RuntimeError("partial lineage regressed or conflicts with prior evidence")


def verify_stage_roots(specifications: list[str] | None) -> \
        tuple[list[dict[str, object]], list[dict[str, object]],
              list[dict[str, object]]]:
    roots = {}
    for specification in specifications or []:
        if "=" not in specification:
            raise RuntimeError("immutable root must be NAME=PATH")
        name, raw_path = specification.split("=", 1)
        if name in roots:
            raise RuntimeError("duplicate immutable root name")
        roots[name] = Path(raw_path).resolve()
    if set(roots) != {"job56586376", "job56587561"}:
        raise RuntimeError("stage requires exact job56586376/job56587561 roots")
    diagnostic_cases, diagnostic = verify_replay_campaign(roots["job56586376"])
    template = load_search_manifest(
        Path(__file__).resolve().parents[3] /
        "tst/inputs/z4c_cartoon_mms_search_manifest.json")
    characterization, characterization_cases = verify_characterization_root(
        roots["job56587561"], template)
    return ([{"name": "job56586376", "binding_sha256": canonical_digest(diagnostic),
              "root": str(roots["job56586376"])},
            {"name": "job56587561",
              "binding_sha256": canonical_digest(characterization),
              "root": str(roots["job56587561"])}], diagnostic_cases,
            characterization_cases)


def validate_frozen_window(value: object, template_sha256: str,
                           source: dict[str, str], build_manifest_sha256: str,
                           executable_sha256: str, input_sha256: str,
                           oracle_sha256: str, series_sha256: str,
                           completed_tuples: set[tuple[int, int, int]],
                           authorization_root: Path, backend: str, ranks: int) -> \
        list[tuple[int, int, int]]:
    fields = {"schema", "state", "purpose", "search_template_sha256",
              "candidate_manifest_path", "candidate_manifest_sha256",
              "source_commit", "source_tree", "kokkos_commit", "executable_sha256",
              "build_manifest_sha256", "input_sha256", "oracle_header_sha256",
              "series_manifest_sha256", "qualification_window", "selected_extra_points",
              "execution_tuples", "prior_complete_tuples_sha256", "backend", "ranks",
              "immutable_root_bindings_sha256", "observed_tuples_sha256",
              "convergence_sha256", "review"}
    if (not isinstance(value, dict) or set(value) != fields or
            value.get("schema") != "athenak_z4c_cartoon_mms_window_v1" or
            value.get("state") != "accepted_frozen_window" or
            value.get("purpose") not in
            {"characterization_completion", "final_qualification"} or
            value.get("search_template_sha256") != template_sha256 or
            value.get("source_commit") != source["commit"] or
            value.get("source_tree") != source["tree"] or
            value.get("kokkos_commit") != source["kokkos"] or
            value.get("build_manifest_sha256") != build_manifest_sha256 or
            value.get("executable_sha256") != executable_sha256 or
            value.get("input_sha256") != input_sha256 or
            value.get("oracle_header_sha256") != oracle_sha256 or
            value.get("series_manifest_sha256") != series_sha256 or
            value.get("backend") != backend or
            value.get("ranks") != ([4] if backend == "Cuda" else [2, 4]) or
            ranks not in value.get("ranks", []) or
            not isinstance(value.get("qualification_window"), dict)):
        raise RuntimeError("accepted frozen-window identity/schema differs")
    review = value.get("review")
    if (not isinstance(review, dict) or
            set(review) != {"artifact_path", "artifact_sha256"} or
            not re.fullmatch(r"[0-9a-f]{64}", str(review.get("artifact_sha256", ""))) or
            not re.fullmatch(r"[0-9a-f]{64}", str(value.get("candidate_manifest_sha256", "")))):
        raise RuntimeError("frozen window lacks independent accepted review binding")
    if (value.get("candidate_manifest_path") != "candidate_manifest.json" or
            review.get("artifact_path") != "review_artifact.json"):
        raise RuntimeError("authorization artifact filenames are not canonical")
    if value.get("prior_complete_tuples_sha256") != canonical_digest(
            sorted(completed_tuples)):
        raise RuntimeError("frozen window prior-complete ledger differs")
    authorization_root = authorization_root.absolute()
    require_exact_regular_files(
        authorization_root,
        {"accepted_window.json", "candidate_manifest.json", "review_artifact.json"},
        "authorization directory")
    if load_json_strict(authorization_root / "accepted_window.json") != value:
        raise RuntimeError("accepted window differs from its canonical artifact")
    raw_selected = value.get("selected_extra_points")
    raw_execution = value.get("execution_tuples")
    if not isinstance(raw_selected, list) or not isinstance(raw_execution, list):
        raise RuntimeError("frozen window point/tuple inventory is malformed")
    if any(not isinstance(item, list) or len(item) != 2 or
           any(isinstance(entry, bool) or not isinstance(entry, int) for entry in item)
           for item in raw_selected) or \
       any(not isinstance(item, list) or len(item) != 3 or
           any(isinstance(entry, bool) or not isinstance(entry, int) for entry in item)
           for item in raw_execution):
        raise RuntimeError("frozen window point/tuple inventory is malformed")
    selected = [tuple(item) for item in raw_selected]
    execution = [tuple(item) for item in raw_execution]
    if (not selected or selected != sorted(set(selected)) or
            any(item[0] not in (2, 6) for item in selected) or
            execution != sorted(set(execution))):
        raise RuntimeError("frozen window point/tuple inventory is malformed")
    pools = {int(order): set(pool) for order, pool in
             load_search_manifest(Path(__file__).resolve().parents[3] /
                                  "tst/inputs/z4c_cartoon_mms_search_manifest.json")
             ["policy"]["resolution_pools"].items()}
    if any(resolution not in pools[order] or resolution in DIAGNOSTIC_RESOLUTIONS
           for order, resolution in selected):
        raise RuntimeError("frozen window selects a point outside prospective extras")
    expected_window = {str(order): sorted(resolution for item_order, resolution in selected
                                          if item_order == order)
                       for order in sorted({item[0] for item in selected})}
    if value["qualification_window"] != expected_window:
        raise RuntimeError("frozen qualification window differs from selected points")
    def bound_path(token: object) -> Path:
        if not isinstance(token, str):
            raise RuntimeError("authorization artifact path is not a string")
        relative = Path(token)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("authorization artifact path escapes its root")
        unresolved = authorization_root / relative
        if unresolved.is_symlink() or not unresolved.is_file():
            raise RuntimeError("authorization artifact is missing or a symlink")
        path = unresolved.resolve()
        if authorization_root.resolve() not in path.parents:
            raise RuntimeError("authorization artifact is missing, escaped, or a symlink")
        return path
    candidate_path = bound_path(value.get("candidate_manifest_path"))
    review_path = bound_path(review.get("artifact_path"))
    if (sha256(candidate_path) != value["candidate_manifest_sha256"] or
            sha256(review_path) != review["artifact_sha256"]):
        raise RuntimeError("candidate or review artifact digest changed")
    candidate = load_json_strict(candidate_path)
    review_artifact = load_json_strict(review_path)
    candidate_fields = {
        "schema", "state", "purpose", "qualification_window",
        "selected_extra_points", "execution_tuples", "source_policy_commit",
        "source_policy_tree", "search_template_sha256", "series_manifest_sha256",
        "immutable_root_bindings", "observed_tuples_sha256",
        "resolutions_by_order", "domain_certification", "convergence_sha256",
        "numerical_failures", "qualification_claim", "backend_intent",
        "ranks_intent"}
    if (not isinstance(candidate, dict) or set(candidate) != candidate_fields or
            candidate.get("schema") != "athenak_z4c_cartoon_mms_window_v1" or
            candidate.get("state") != "candidate_resolved" or
            candidate.get("purpose") != value["purpose"] or
            candidate.get("qualification_window") != value["qualification_window"] or
            candidate.get("selected_extra_points") != raw_selected or
            candidate.get("execution_tuples") != raw_execution or
            candidate.get("source_policy_commit") != source["commit"] or
            candidate.get("source_policy_tree") != source["tree"] or
            candidate.get("search_template_sha256") != template_sha256 or
            candidate.get("series_manifest_sha256") != series_sha256 or
            candidate.get("backend_intent") != backend or
            candidate.get("ranks_intent") != value["ranks"] or
            canonical_digest(candidate.get("immutable_root_bindings")) !=
            value.get("immutable_root_bindings_sha256") or
            candidate.get("observed_tuples_sha256") !=
            value.get("observed_tuples_sha256") or
            candidate.get("convergence_sha256") != value.get("convergence_sha256") or
            candidate.get("qualification_claim") is not None):
        raise RuntimeError("resolved candidate does not project into accepted window")
    review_fields = {"schema", "disposition", "candidate_manifest_sha256",
                     "source_policy_commit", "source_policy_tree", "purpose",
                     "selected_extra_points", "qualification_window", "backend",
                     "ranks"}
    if (not isinstance(review_artifact, dict) or set(review_artifact) != review_fields or
            review_artifact.get("schema") !=
            "athenak_z4c_cartoon_mms_window_review_v1" or
            review_artifact.get("disposition") != "ACCEPT" or
            review_artifact.get("candidate_manifest_sha256") != sha256(candidate_path) or
            review_artifact.get("source_policy_commit") != source["commit"] or
            review_artifact.get("source_policy_tree") != source["tree"] or
            review_artifact.get("purpose") != value["purpose"] or
            review_artifact.get("selected_extra_points") != raw_selected or
            review_artifact.get("qualification_window") != value["qualification_window"] or
            review_artifact.get("backend") != backend or
            review_artifact.get("ranks") != value["ranks"]):
        raise RuntimeError("independent review artifact did not accept this candidate")
    if value["purpose"] == "characterization_completion":
        required = sorted((order, resolution, phase)
                          for order, resolution in selected for phase in range(8)
                          if (order, resolution, phase) not in completed_tuples)
    else:
        if completed_tuples:
            raise RuntimeError("final qualification cannot subtract old-source tuples")
        required = sorted(
            {(order, resolution, phase) for order in (2, 4, 6)
             for resolution in DIAGNOSTIC_RESOLUTIONS for phase in range(8)} |
            {(order, resolution, phase) for order, resolution in selected
             for phase in range(8)})
    if execution != required or set(execution).intersection(completed_tuples):
        raise RuntimeError("frozen window does not execute exactly missing phases")
    return execution


def replay_campaign(raw_root: Path, analysis_output: Path, root: Path,
                    series_manifest: Path) -> int:
    raw_root = raw_root.resolve()
    analysis_output = analysis_output.resolve()
    if not raw_root.is_dir() or raw_root.is_symlink() or \
       (raw_root / "results/ranks2").is_symlink() or analysis_output == raw_root or \
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
    convergence = load_json_strict(analysis_output / "convergence.json")
    partition = convergence.get("legacy_pre_coefficient_floor_partition", {})
    validate_preserved_job56586376_legacy_partition(partition)
    artifacts = {name: sha256(analysis_output / name) for name in
                 ("convergence.json", "convergence.csv",
                  "convergence_rates.pgfplots.dat", "convergence_plot.tex")}
    write_atomic(analysis_output / "replay_manifest.json", {
        **binding, "analysis_policy_source_commit": git_value(root, "rev-parse", "HEAD"),
        "analysis_policy_source_tree": git_value(root, "rev-parse", "HEAD^{tree}"),
        "series_manifest_sha256": sha256(series_manifest),
        "domain_certification": certification, "artifacts": artifacts,
        "failures": failures, "passed": not failures})
    return 1 if failures else 0


def merge_case_ledgers(*groups: list[dict[str, object]]) -> list[dict[str, object]]:
    merged = []
    seen: dict[tuple[int, int, int], str | None] = {}
    for group in groups:
        for case in group:
            identity = (int(case["spatial_order"]), int(case["resolution"]),
                        int(case["phase"]))
            if identity in seen:
                manifest_sha = case.get("case_manifest_sha256")
                if (seen[identity] is None or
                        not isinstance(manifest_sha, str) or
                        seen[identity] != manifest_sha):
                    raise RuntimeError("immutable evidence roots conflict")
                continue
            manifest_sha = case.get("case_manifest_sha256")
            seen[identity] = (manifest_sha if isinstance(manifest_sha, str) and
                              re.fullmatch(r"[0-9a-f]{64}", manifest_sha) else None)
            merged.append(case)
    return merged


def stage_binding_record(binding: dict[str, object]) -> dict[str, object]:
    state = binding.get("state")
    prefix = "partial" if state == "stage_partial" else "stage"
    digest = str(binding.get("stage_campaign_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RuntimeError("stage binding lacks its campaign digest")
    return {"name": f"{prefix}:{binding['stage_id']}:{digest[:16]}",
            "root": binding["root"],
            "binding_sha256": canonical_digest(binding)}


def analyze_search(immutable_specs: list[str] | None, stage_specs: list[str] | None,
                   analysis_output: Path, root: Path, series_manifest: Path) -> int:
    bindings, diagnostic_cases, characterization_cases = verify_stage_roots(
        immutable_specs)
    template = load_search_manifest(
        root / "tst/inputs/z4c_cartoon_mms_search_manifest.json")
    stage_cases = []
    stage_lineages: dict[str, tuple[list[dict[str, object]],
                                   set[tuple[int, int, int]]]] = {}
    completed_tuples = {
        (int(case["spatial_order"]), int(case["resolution"]), int(case["phase"]))
        for case in diagnostic_cases + characterization_cases}
    for raw_path in stage_specs or []:
        stage_root = Path(raw_path).absolute()
        if (stage_root / "window_campaign.json").is_file():
            cases, binding = verify_window_campaign_root(
                stage_root, bindings, completed_tuples,
                sha256(root / "tst/inputs/z4c_cartoon_mms_search_manifest.json"))
            missing = []
            name = "window:characterization_completion"
        else:
            cases, binding, missing = verify_stage_campaign_root(
                stage_root, template, bindings)
            name = (f"partial:{binding['stage_id']}" if missing else
                    f"stage:{binding['stage_id']}")
            stage_id = str(binding["stage_id"])
            authorized = {tuple(item) for item in
                          template["stages"][stage_id]["tuples"]}
            prior_cases, prior_missing = stage_lineages.get(
                stage_id, ([], set(authorized)))
            validate_stage_lineage(prior_cases, prior_missing, cases,
                                   set(missing), authorized)
            stage_lineages[stage_id] = (cases, set(missing))
        stage_cases = merge_case_ledgers(stage_cases, cases)
        completed_tuples.update(
            (int(case["spatial_order"]), int(case["resolution"]),
             int(case["phase"])) for case in cases)
        bindings.append(
            ({"name": name, "root": binding["root"],
              "binding_sha256": canonical_digest(binding)}
             if name.startswith("window:") else stage_binding_record(binding)))
    cases = merge_case_ledgers(diagnostic_cases, characterization_cases, stage_cases)
    requested = {(int(case["spatial_order"]), int(case["phase"])) for case in cases}
    tuples = [(int(case["spatial_order"]), int(case["resolution"]),
               int(case["phase"])) for case in cases]
    resolutions = {order: tuple(sorted({resolution for item_order, resolution, _ in tuples
                                        if item_order == order}))
                   for order, _ in requested}
    domains = {tuple(float(value) for value in case["domain"]) for case in cases}
    if len(domains) != 1:
        raise RuntimeError("mixed-source search domains differ")
    certification = validate_certified_domain(domains.pop(), resolutions, True)
    analysis_output = analysis_output.resolve()
    raw_roots = [Path(item["root"]).resolve() for item in bindings]
    if any(analysis_output == raw or raw in analysis_output.parents for raw in raw_roots):
        raise RuntimeError("search analysis output must be outside every immutable root")
    if analysis_output.exists() and any(analysis_output.iterdir()):
        raise RuntimeError("search analysis output must be absent or empty")
    analysis_output.mkdir(parents=True, exist_ok=True)
    failures = convergence_gate(
        cases, analysis_output, analysis_output, series_manifest, resolutions,
        allow_legacy_nullable=True, evidence_scope="mixed_source_characterization_only")
    write_atomic(analysis_output / "candidate_window_manifest.json", {
        "schema": "athenak_z4c_cartoon_mms_window_v1",
        "state": "candidate_unreviewed", "purpose": None,
        "qualification_window": None,
        "selected_extra_points": None, "execution_tuples": None,
        "source_policy_commit": git_value(root, "rev-parse", "HEAD"),
        "source_policy_tree": git_value(root, "rev-parse", "HEAD^{tree}"),
        "search_template_sha256": sha256(
            root / "tst/inputs/z4c_cartoon_mms_search_manifest.json"),
        "series_manifest_sha256": sha256(series_manifest),
        "immutable_root_bindings": bindings,
        "observed_tuples_sha256": canonical_digest(sorted(tuples)),
        "resolutions_by_order": {str(order): list(values)
                                 for order, values in resolutions.items()},
        "domain_certification": certification,
        "convergence_sha256": sha256(analysis_output / "convergence.json"),
        "numerical_failures": failures,
        "qualification_claim": None, "backend_intent": None,
        "ranks_intent": None})
    return 0


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
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--characterization-stage")
    parser.add_argument("--immutable-root", action="append")
    parser.add_argument("--analyze-search", action="store_true")
    parser.add_argument("--stage-root", action="append")
    parser.add_argument("--frozen-window-manifest", type=Path)
    parser.add_argument("--compare-campaign", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    series_manifest = root / "tst/unit/z4c/z4c_cartoon_derivatives_series.json"
    if args.analyze_search:
        if (args.analysis_output is None or args.replay_campaign is not None or
                any(value is not None for value in
                    (args.athena, args.launcher, args.ranks, args.output,
                     args.characterization_stage, args.frozen_window_manifest))):
            raise RuntimeError("search analysis is read-only and requires roots/output only")
        return analyze_search(args.immutable_root, args.stage_root,
                              args.analysis_output, root, series_manifest)
    if args.replay_campaign is not None:
        if args.analysis_output is None or any(value is not None for value in
                                               (args.athena, args.launcher, args.ranks,
                                                args.output,
                                                args.characterization_stage,
                                                args.immutable_root,
                                                args.stage_root,
                                                args.frozen_window_manifest)):
            raise RuntimeError("replay requires only RAW_ROOT and a separate analysis output")
        return replay_campaign(args.replay_campaign, args.analysis_output, root,
                               series_manifest)
    if args.analysis_output is not None:
        raise RuntimeError("--analysis-output is valid only with --replay-campaign")
    if args.athena is None or args.launcher is None or args.ranks is None or \
       args.output is None:
        raise RuntimeError("campaign requires athena, launcher, ranks, and output")
    search = load_search_manifest(root / "tst/inputs/z4c_cartoon_mms_search_manifest.json")
    frozen_window = None
    frozen_window_purpose = None
    frozen_window_path = None
    if args.frozen_window_manifest is not None:
        if (args.diagnostic_only or args.characterization_stage is not None or
                args.orders != (2, 4, 6) or
                args.resolutions is not None or args.phases != tuple(range(8))):
            raise RuntimeError("frozen-window selection comes only from the accepted manifest")
        frozen_window_path = args.frozen_window_manifest.absolute()
        if (frozen_window_path.is_symlink() or not frozen_window_path.is_file() or
                frozen_window_path.parent.is_symlink()):
            raise RuntimeError("accepted-window path/directory is missing or a symlink")
        frozen_window = load_json_strict(frozen_window_path)
        if not isinstance(frozen_window, dict) or not isinstance(
                frozen_window.get("execution_tuples"), list):
            raise RuntimeError("frozen-window execution tuple inventory is missing")
        frozen_window_purpose = frozen_window.get("purpose")
        if frozen_window_purpose not in {
                "characterization_completion", "final_qualification"}:
            raise RuntimeError("frozen-window purpose is missing or unsupported")
        if (args.compare_campaign is not None and
                frozen_window_purpose != "final_qualification"):
            raise RuntimeError("rank comparison is allowed only for final qualification")
        run_tuples = []
        for item in frozen_window["execution_tuples"]:
            if (not isinstance(item, list) or len(item) != 3 or
                    any(isinstance(value, bool) or not isinstance(value, int)
                        for value in item)):
                raise RuntimeError("frozen-window execution tuple is malformed")
            run_tuples.append(tuple(item))
        if not run_tuples or run_tuples != sorted(set(run_tuples)):
            raise RuntimeError("frozen-window execution tuples are empty, duplicated, or unordered")
        args.orders = sorted({item[0] for item in run_tuples})
        args.resolutions = sorted({item[1] for item in run_tuples})
        args.phases = sorted({item[2] for item in run_tuples})
        resolutions_by_order = {
            order: tuple(sorted({item[1] for item in run_tuples if item[0] == order}))
            for order in args.orders}
        campaign_mode = "accepted_frozen_window"
    elif args.characterization_stage is not None:
        if (args.diagnostic_only or args.compare_campaign is not None or
                args.orders != (2, 4, 6) or
                args.resolutions is not None or args.phases != tuple(range(8))):
            raise RuntimeError("characterization selection comes only from the manifest")
        stage = search["stages"].get(args.characterization_stage)
        if not isinstance(stage, dict) or stage.get("state") != \
           "authorized_missing_phase0":
            raise RuntimeError("requested characterization stage is not authorized")
        run_tuples = [tuple(item) for item in stage["tuples"]]
        args.orders = sorted({item[0] for item in run_tuples})
        args.resolutions = sorted({item[1] for item in run_tuples})
        args.phases = sorted({item[2] for item in run_tuples})
        resolutions_by_order = {order: tuple(item[1] for item in run_tuples
                                              if item[0] == order)
                                for order in args.orders}
        campaign_mode = "characterization_stage"
    else:
        if args.stage_root:
            raise RuntimeError("--stage-root requires --analyze-search or --frozen-window-manifest")
        if args.resolutions is None:
            args.resolutions = list(DIAGNOSTIC_RESOLUTIONS)
        if (set(args.orders) != {2, 4, 6} or len(args.orders) != 3 or
                set(args.resolutions) != set(DIAGNOSTIC_RESOLUTIONS) or
                len(args.resolutions) != 4 or set(args.phases) != set(range(8)) or
                len(args.phases) != 8):
            raise RuntimeError("diagnostic-only mode owns exactly the old 96-case matrix")
        args.orders = [2, 4, 6]
        args.resolutions = list(DIAGNOSTIC_RESOLUTIONS)
        args.phases = list(range(8))
        resolutions_by_order = {order: DIAGNOSTIC_RESOLUTIONS for order in args.orders}
        run_tuples = [(order, resolution, phase) for order in args.orders
                      for resolution in DIAGNOSTIC_RESOLUTIONS for phase in args.phases]
        campaign_mode = "diagnostic_only"
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
    frozen_window_sha256 = None
    continuation_cases = []
    if frozen_window is not None:
        immutable_root_bindings = []
        completed_tuples = set()
        if frozen_window_purpose == "characterization_completion":
            immutable_root_bindings, diagnostic_cases, characterization_cases = \
                verify_stage_roots(args.immutable_root)
            completed_tuples = {
                (int(case["spatial_order"]), int(case["resolution"]),
                 int(case["phase"]))
                for case in diagnostic_cases + characterization_cases}
            lineage_cases = []
            lineage_cases_by_stage: dict[str, list[dict[str, object]]] = {}
            lineage_missing: dict[str, set[tuple[int, int, int]]] = {}
            for stage_root in args.stage_root or []:
                stage_cases, binding, missing = verify_stage_campaign_root(
                    Path(stage_root).absolute(), search, immutable_root_bindings)
                stage_id = str(binding["stage_id"])
                authorized = {tuple(item) for item in
                              search["stages"][stage_id]["tuples"]}
                validate_stage_lineage(
                    lineage_cases_by_stage.get(stage_id, []),
                    lineage_missing.get(stage_id, set(authorized)), stage_cases,
                    set(missing), authorized)
                lineage_cases = merge_case_ledgers(lineage_cases, stage_cases)
                lineage_cases_by_stage[stage_id] = stage_cases
                lineage_missing[stage_id] = set(missing)
                completed_tuples.update(
                    (int(case["spatial_order"]), int(case["resolution"]),
                     int(case["phase"])) for case in stage_cases)
                immutable_root_bindings.append(stage_binding_record(binding))
        elif args.immutable_root or args.stage_root:
            raise RuntimeError("final qualification cannot consume old-source roots")
        if (frozen_window_purpose == "characterization_completion" and
                frozen_window.get("immutable_root_bindings_sha256") !=
                canonical_digest(immutable_root_bindings)):
            raise RuntimeError("accepted characterization window binds different roots")
        validated_tuples = validate_frozen_window(
            frozen_window,
            sha256(root / "tst/inputs/z4c_cartoon_mms_search_manifest.json"),
            source, sha256(args.build_manifest), sha256(args.athena),
            sha256(args.input),
            sha256(root / "src/pgen/unit_tests/z4c_cartoon_derivatives_oracle.hpp"),
            sha256(series_manifest), completed_tuples,
            frozen_window_path.parent,
            args.require_backend, args.ranks)
        if validated_tuples != run_tuples:
            raise RuntimeError("frozen-window tuple selection changed after identity validation")
        frozen_window_sha256 = sha256(frozen_window_path)
    elif args.characterization_stage is not None:
        immutable_root_bindings, _, _ = verify_stage_roots(
            args.immutable_root)
        if args.stage_root:
            stage_id = args.characterization_stage
            authorized = {tuple(item) for item in search["stages"][stage_id]["tuples"]}
            prior_lineage_cases = []
            prior_missing = set(authorized)
            for stage_root in args.stage_root:
                cases_from_root, binding, missing = verify_stage_campaign_root(
                    Path(stage_root).absolute(), search, immutable_root_bindings)
                if (not missing or
                        binding["stage_id"] != args.characterization_stage or
                        binding["state"] != "stage_partial"):
                    raise RuntimeError(
                        "stage continuation root has no matching missing tuples")
                validate_stage_lineage(prior_lineage_cases, prior_missing,
                                       cases_from_root, set(missing), authorized)
                continuation_cases = merge_case_ledgers(
                    continuation_cases, cases_from_root)
                prior_lineage_cases = cases_from_root
                prior_missing = set(missing)
                immutable_root_bindings.append(stage_binding_record(binding))
            run_tuples = missing
            args.orders = sorted({item[0] for item in run_tuples})
            args.resolutions = sorted({item[1] for item in run_tuples})
            args.phases = sorted({item[2] for item in run_tuples})
            resolutions_by_order = {
                order: tuple(sorted({item[1] for item in run_tuples
                                     if item[0] == order}))
                for order in args.orders}
            certification = validate_certified_domain(
                args.domain, resolutions_by_order, True)
    else:
        if args.immutable_root or args.stage_root:
            raise RuntimeError("immutable roots require characterization or frozen-window mode")
        immutable_root_bindings = []
    series_inventory = load_json_strict(series_manifest)
    if not isinstance(series_inventory, dict):
        raise RuntimeError("series manifest is not an object")
    expected_operators = [item["name"] for item in series_inventory.get("series", [])]
    if series_inventory.get("count") != 171 or len(expected_operators) != 171 or \
       len(set(expected_operators)) != 171:
        raise RuntimeError("frozen runtime series manifest is not exactly 171 unique entries")
    forecast = output_forecast(resolutions_by_order, args.phases, args.ranks,
                               args.compare_campaign is not None, run_tuples)
    disk_anchor = args.output
    while not disk_anchor.exists() and disk_anchor != disk_anchor.parent:
        disk_anchor = disk_anchor.parent
    free_bytes = shutil.disk_usage(disk_anchor).free
    preflight = {"schema": SCHEMA, "state": "preflight", **forecast,
                 "free_bytes_before_campaign": free_bytes,
                 "orders": args.orders, "resolutions": args.resolutions,
                 "resolutions_by_order": resolutions_by_order,
                 "phases": args.phases, "ranks": args.ranks,
                 "campaign_mode": campaign_mode,
                 "stage_id": args.characterization_stage,
                 "frozen_window_sha256": frozen_window_sha256,
                 "run_tuples": [list(item) for item in run_tuples],
                 "domain_certification": certification,
                 "search_manifest_sha256": sha256(
                     root / "tst/inputs/z4c_cartoon_mms_search_manifest.json"),
                 "series_manifest_sha256": sha256(series_manifest)}
    if free_bytes < 2 * forecast["estimated_output_bytes_upper_bound"]:
        raise RuntimeError("campaign output forecast exceeds half the available space")
    args.output.mkdir(parents=True, exist_ok=True)
    authorization_files = None
    if frozen_window is not None:
        authorization = args.output / "authorization"
        authorization.mkdir(exist_ok=True)
        source_files = {
            "accepted_window.json": frozen_window_path,
            "candidate_manifest.json":
            frozen_window_path.parent /
            frozen_window["candidate_manifest_path"],
            "review_artifact.json":
            frozen_window_path.parent /
            frozen_window["review"]["artifact_path"]}
        for name, source_path in source_files.items():
            destination = authorization / name
            if destination.exists():
                if sha256(destination) != sha256(source_path):
                    raise RuntimeError("resumed authorization artifact changed")
            else:
                shutil.copy2(source_path, destination)
        require_exact_regular_files(authorization, set(source_files),
                                    "campaign authorization directory")
        authorization_files = {name: sha256(authorization / name)
                               for name in source_files}
    resumed_stage_cases = []
    for prior in continuation_cases:
        source_directory = Path(str(prior["_replay_case_directory"]))
        destination = args.output / source_directory.name
        if destination.exists():
            if bound_directory_inventory(destination) != \
               bound_directory_inventory(source_directory):
                raise RuntimeError("continued stage copied-case evidence changed")
        else:
            shutil.copytree(source_directory, destination)
        resumed_stage_cases.append({key: value for key, value in prior.items()
                                    if not key.startswith("_")})
    stage_manifest_path = args.output / "search_stage_manifest.json"
    stage_manifest = None
    window_execution_path = args.output / "frozen_window_execution.json"
    window_execution = None
    if args.characterization_stage is not None:
        stage_manifest = materialize_search_stage(
            search, args.characterization_stage, source, sha256(args.build_manifest),
            sha256(args.athena), sha256(args.input),
            sha256(root / "src/pgen/unit_tests/z4c_cartoon_derivatives_oracle.hpp"),
            sha256(series_manifest), args.require_backend, args.ranks,
            immutable_root_bindings)
        if stage_manifest_path.exists():
            existing = load_json_strict(stage_manifest_path)
            if not isinstance(existing, dict) or existing.get("state") != \
               "prelaunch_bound":
                raise RuntimeError("refusing to rerun a finalized characterization stage")
            stage_manifest["materialization"]["attempt_id"] = \
                existing["materialization"]["attempt_id"]
            if stage_manifest != existing:
                raise RuntimeError("resumed characterization manifest identity changed")
        else:
            write_atomic(stage_manifest_path, stage_manifest)
    elif frozen_window is not None:
        window_execution = {
            "schema": "athenak_z4c_cartoon_mms_window_execution_v1",
            "state": "prelaunch_bound", "purpose": frozen_window_purpose,
            "attempt_id": str(uuid.uuid4()),
            "accepted_window_sha256": frozen_window_sha256,
            "accepted_review": frozen_window["review"],
            "source": source, "build_manifest_sha256": sha256(args.build_manifest),
            "executable_sha256": sha256(args.athena),
            "input_sha256": sha256(args.input),
            "oracle_header_sha256": sha256(
                root / "src/pgen/unit_tests/z4c_cartoon_derivatives_oracle.hpp"),
            "series_manifest_sha256": sha256(series_manifest),
            "backend": args.require_backend, "ranks": args.ranks,
            "domain": list(args.domain),
            "immutable_root_bindings": immutable_root_bindings,
            "execution_tuples": [list(item) for item in run_tuples],
            "attempts": [], "result_set_sha256": None, "stop_reason": None,
            "authorization_files": authorization_files}
        if window_execution_path.exists():
            existing = load_json_strict(window_execution_path)
            if (not isinstance(existing, dict) or
                    existing.get("state") != "prelaunch_bound"):
                raise RuntimeError("refusing to rerun a finalized frozen-window execution")
            window_execution["attempt_id"] = existing.get("attempt_id")
            if window_execution != existing:
                raise RuntimeError("resumed frozen-window execution identity changed")
        else:
            write_atomic(window_execution_path, window_execution)
    write_atomic(args.output / "preflight.json", preflight)
    cases = list(resumed_stage_cases)
    prior_case_hashes = {
        (int(case["spatial_order"]), int(case["resolution"]), int(case["phase"])):
        case["case_manifest_sha256"] for case in resumed_stage_cases}
    attempts = [{"tuple": list(item), "status": "complete",
                 "case_manifest_sha256": prior_case_hashes[tuple(item)],
                 "reason": "reused_verified_partial_stage"}
                for item in search["stages"].get(
                    args.characterization_stage, {}).get("tuples", [])
                if tuple(item) in prior_case_hashes]
    for tuple_index, (order, resolution, phase) in enumerate(run_tuples):
        try:
            result = run_case(args, root, source, order, resolution, phase)
            cases.append(result)
            attempts.append({"tuple": [order, resolution, phase], "status": "complete",
                             "case_manifest_sha256":
                             result["case_manifest_sha256"], "reason": "case_complete"})
        except RuntimeError as error:
            attempts.append({"tuple": [order, resolution, phase], "status": "failed",
                             "case_manifest_sha256": None,
                             "reason": f"integrity_or_resource_failure: {error}"})
            if stage_manifest is None and window_execution is None:
                raise
            if stage_manifest is not None:
                attempts.extend({"tuple": list(item),
                                 "status": "not_attempted_after_stop",
                                 "case_manifest_sha256": None,
                                 "reason": "stopped_after_prior_failure"}
                                for item in run_tuples[tuple_index + 1:])
                transition_search_stage(stage_manifest, "stage_partial", attempts,
                                        "integrity_or_resource_failure")
                write_atomic(stage_manifest_path, stage_manifest)
                complete_directories = {
                    f"{case['case_id']}-{case['case_uuid']}" for case in cases}
                failed_case_inventories = [
                    {"directory": entry.name,
                     "files": bound_directory_inventory(entry)}
                    for entry in args.output.iterdir()
                    if entry.is_dir() and entry.name not in complete_directories]
                write_atomic(args.output / "stage_campaign.json",
                             stage_campaign_record(
                                 stage_manifest, stage_manifest_path, cases,
                                 failed_case_inventories))
            else:
                window_execution.update({
                    "state": "execution_failed", "attempts": attempts,
                    "result_set_sha256": canonical_digest(
                        [item["case_manifest_sha256"] for item in attempts
                         if item["status"] == "complete"]),
                    "stop_reason": "integrity_or_resource_failure"})
                write_atomic(window_execution_path, window_execution)
            return 1
    for result in cases:
        if result.get("operator_names") != expected_operators:
            raise RuntimeError(f"{result['case_id']} operator ordering differs from frozen 171-series manifest")
    if stage_manifest is not None:
        transition_search_stage(stage_manifest, "stage_finalized", attempts,
                                "stage_complete_pending_offline_analysis")
        write_atomic(stage_manifest_path, stage_manifest)
        write_atomic(args.output / "stage_campaign.json",
                     stage_campaign_record(stage_manifest, stage_manifest_path,
                                           cases, []))
        return 0
    if window_execution is not None:
        window_execution.update({
            "state": "execution_finalized", "attempts": attempts,
            "result_set_sha256": canonical_digest(
                [item["case_manifest_sha256"] for item in attempts]),
            "stop_reason": "accepted_window_execution_complete_pending_merged_analysis"})
        write_atomic(window_execution_path, window_execution)
        if frozen_window_purpose == "characterization_completion":
            write_atomic(args.output / "window_campaign.json", {
                "schema": SCHEMA, "state": "execution_finalized",
                "purpose": frozen_window_purpose,
                "accepted_window_sha256": frozen_window_sha256,
                "execution_manifest_sha256": sha256(window_execution_path),
                "source": source,
                "build_manifest_sha256": sha256(args.build_manifest),
                "executable_sha256": sha256(args.athena),
                "input_sha256": sha256(args.input),
                "oracle_header_sha256": sha256(
                    root / "src/pgen/unit_tests/z4c_cartoon_derivatives_oracle.hpp"),
                "series_manifest_sha256": sha256(series_manifest),
                "backend": args.require_backend, "ranks": args.ranks,
                "domain": list(args.domain),
                "immutable_root_bindings": immutable_root_bindings,
                "cases": cases, "qualification_claim": None})
            return 0
    failures = convergence_gate(cases, args.output, args.output, series_manifest,
                                resolutions_by_order,
                                evidence_scope=(
                                    "fresh_single_source_final_qualification"
                                    if frozen_window_purpose == "final_qualification"
                                    else "single_source_diagnostic"))
    rank_evidence = None
    if args.compare_campaign:
        rank_evidence = compare_rank_campaigns(
            cases, args.output, args.compare_campaign.absolute(), args.ranks,
            campaign_mode, frozen_window_sha256, source, args.require_backend,
            sha256(args.build_manifest))
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
                                                  "campaign_mode": campaign_mode,
                                                  "accepted_window_sha256":
                                                  frozen_window_sha256,
                                                  "window_execution_sha256":
                                                  (sha256(window_execution_path)
                                                   if window_execution is not None
                                                   else None),
                                                  "diagnostic_resolutions":
                                                  list(args.resolutions),
                                                  "reduction_tolerance_factor":
                                                  REDUCTION_TOLERANCE_FACTOR,
                                                  "convergence_artifacts":
                                                  convergence_artifacts,
                                                  "cases": cases})
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
