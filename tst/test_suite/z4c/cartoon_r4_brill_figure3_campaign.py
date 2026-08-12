#!/usr/bin/env python3
"""Run the prospectively frozen off-center Brill Figure-3 analogue campaign.

This is deliberately a campaign driver, not a scheduler framework.  It binds
one input-selected AthenaK executable to the accepted IrisK 48x32 handoff,
renders exactly three 2-D Cartoon inputs, launches them in the declared order,
and emits machine-readable central-curvature and completeness evidence.  Its
bounded comparison subcommand samples both a selected Athena curve and each
frozen published vector centerline on 1025 inclusive, uniformly spaced points
over their explicit common proper-time interval.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE = SCRIPT_DIR / "cartoon_r4_brill_figure3.athinput"
DESIGN_PATH = SCRIPT_DIR / "cartoon_r4_brill_figure3_campaign.json"
STATE_SCHEMA = "athenak_cartoon_r4_brill_figure3_campaign_v1"
ANALYSIS_SCHEMA = "athenak_cartoon_r4_brill_figure3_analysis_v1"
PUBLISHED_COMPARISON_SCHEMA = "athenak_cartoon_r4_figure3_published_comparison_v1"
PUBLISHED_CURVES_SHA256 = (
    "947f92f72af32caf631d5efe9d720c7c56800b2e4578dcb6bcc26107cabc8adf")
PUBLISHED_METADATA_SHA256 = (
    "b623b340ded776acb4165c8d69fc79a41d41a92dc5f459b131cab8a94927859d")
PUBLISHED_SERIES = ("bamps", "prague", "sphGR")
PUBLISHED_CURVE_HEADER = (
    "series", "index", "form_x_pt", "form_y_pt", "page_x_pt", "page_y_pt",
    "tau", "log10_abs_I", "abs_I", "original_pdf_vertex")
ATHENA_CURVE_HEADER = (
    "resolution", "cycle", "coordinate_time", "proper_time",
    "abs_kretschmann_I", "log10_abs_kretschmann_I", "axis_lapse",
    "normalized_H", "normalized_M", "meshblocks", "max_refinement_level",
    "max_meshblocks_per_rank")
PUBLISHED_COMPARISON_SAMPLES = 1025
PAYLOAD_SENTINEL = "__IRISK_BRILL_FIGURE3_PAYLOAD__"
COEFFICIENT_SENTINEL = "__IRISK_BRILL_FIGURE3_COEFFICIENTS__"


def load_r1() -> Any:
    path = SCRIPT_DIR / "cartoon_r1_campaign.py"
    spec = importlib.util.spec_from_file_location("cartoon_r1_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load established Cartoon campaign utilities")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r1 = load_r1()


def require(condition: bool, message: str) -> None:
    r1.require(condition, message)


def sha256(path: Path) -> str:
    return r1.sha256(path)


def strict_load(path: Path) -> dict[str, Any]:
    return r1.strict_load(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    r1.atomic_json(path, value)


def git(root: Path, *arguments: str) -> str:
    return r1.git(root, *arguments)


def require_file_hash(path: Path, expected: str, label: str) -> str:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
            f"{label} expected hash is malformed")
    actual = sha256(path)
    require(actual == expected, f"{label} hash mismatch")
    return actual


def strict_csv_rows(path: Path, header: tuple[str, ...], label: str
                    ) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and
                tuple(reader.fieldnames) == header and
                len(set(reader.fieldnames)) == len(reader.fieldnames),
                f"{label} CSV inventory is not exact")
        rows = list(reader)
    require(rows and all(None not in row and set(row) == set(header) for row in rows),
            f"{label} CSV contains a malformed record")
    return rows


def finite_float(token: str, label: str) -> float:
    try:
        value = float(token)
    except ValueError as error:
        raise RuntimeError(f"{label} is malformed") from error
    require(math.isfinite(value), f"{label} is nonfinite")
    return value


def finite_integer(token: str, label: str) -> int:
    value = finite_float(token, label)
    require(value.is_integer(), f"{label} is not integral")
    return int(value)


def read_published_curves(path: Path, expected_hash: str,
                          metadata_path: Path, metadata_hash: str
                          ) -> tuple[dict[str, dict[str, list[float]]],
                                     dict[str, Any]]:
    require_file_hash(path, expected_hash, "published vector-curves CSV")
    require_file_hash(metadata_path, metadata_hash, "published vector metadata")
    metadata = strict_load(metadata_path)
    require(set(metadata) == {"axis_transform", "independent_render_check",
                              "inventory", "schema", "scientific_binding",
                              "source", "tools", "uncertainty"} and
            metadata["schema"] ==
            "axisymmetric-cartoon.figure3-published-vector-curves.v1",
            "published vector metadata schema/inventory changed")
    binding = metadata["scientific_binding"]
    require(binding.get("A") == -0.047 and binding.get("rho0") == 5.0 and
            binding.get("family") == "Brill" and
            binding.get("series") == list(PUBLISHED_SERIES) and
            binding.get("recovered_quantity") ==
            "rendered PDF polyline centerlines",
            "published vector scientific binding changed")
    require(metadata["source"].get("paper") == "arXiv:2607.10843v1" and
            metadata["source"].get("figure") == 3 and
            metadata["source"].get("contains_embedded_original_numeric_data")
            is False,
            "published vector source binding changed")
    uncertainty = metadata["uncertainty"]
    for key in ("coordinate_only_log10_abs_I_half_unit",
                "coordinate_only_tau_half_unit"):
        require(isinstance(uncertainty.get(key), (int, float)) and
                math.isfinite(float(uncertainty[key])) and
                float(uncertainty[key]) > 0.0,
                f"published vector uncertainty {key} is invalid")
    rows = strict_csv_rows(path, PUBLISHED_CURVE_HEADER, "published vector-curves")
    curves = {name: {"tau": [], "log10_abs_I": []} for name in PUBLISHED_SERIES}
    expected_index = {name: 0 for name in PUBLISHED_SERIES}
    for number, row in enumerate(rows, 2):
        series = row["series"]
        require(series in curves, f"unexpected published series at row {number}")
        require(row["original_pdf_vertex"] in {"true", "false"},
                f"malformed published vertex flag at row {number}")
        index = finite_integer(row["index"], f"published index at row {number}")
        require(index == expected_index[series],
                f"noncontiguous published index for {series}")
        expected_index[series] += 1
        for key in ("form_x_pt", "form_y_pt", "page_x_pt", "page_y_pt"):
            finite_float(row[key], f"published {key} at row {number}")
        tau = finite_float(row["tau"], f"published tau at row {number}")
        log_value = finite_float(row["log10_abs_I"],
                                 f"published log10 curvature at row {number}")
        absolute = finite_float(row["abs_I"],
                                f"published curvature at row {number}")
        require(absolute > 0.0 and
                abs(math.log10(absolute) - log_value) <= 5.0e-14,
                f"published linear/log curvature mismatch at row {number}")
        curve = curves[series]
        require(not curve["tau"] or tau > curve["tau"][-1],
                f"published {series} proper time is not strictly increasing")
        curve["tau"].append(tau)
        curve["log10_abs_I"].append(log_value)
    require(all(len(curves[name]["tau"]) >= 2 for name in PUBLISHED_SERIES),
            "published vector series inventory is incomplete")
    return curves, metadata


def read_athena_curve(path: Path, expected_hash: str,
                      resolution: str) -> dict[str, list[float]]:
    require(resolution in {"n128", "n192", "n256"},
            "requested Athena resolution is invalid")
    require_file_hash(path, expected_hash, "Athena central curve CSV")
    rows = strict_csv_rows(path, ATHENA_CURVE_HEADER, "Athena central curve")
    curves: dict[str, dict[str, list[float]]] = {}
    for number, row in enumerate(rows, 2):
        name = row["resolution"]
        require(name in {"n128", "n192", "n256"},
                f"unexpected Athena resolution at row {number}")
        curve = curves.setdefault(name, {"tau": [], "log10_abs_I": []})
        for key in ("cycle", "meshblocks", "max_refinement_level",
                    "max_meshblocks_per_rank"):
            value = finite_integer(row[key], f"Athena {key} at row {number}")
            require(value >= 0, f"negative Athena {key} at row {number}")
        for key in ("coordinate_time", "axis_lapse"):
            finite_float(row[key], f"Athena {key} at row {number}")
        for key in ("normalized_H", "normalized_M"):
            if row[key]:
                finite_float(row[key], f"Athena {key} at row {number}")
        tau = finite_float(row["proper_time"],
                           f"Athena proper time at row {number}")
        absolute = finite_float(row["abs_kretschmann_I"],
                                f"Athena curvature at row {number}")
        log_value = finite_float(row["log10_abs_kretschmann_I"],
                                 f"Athena log10 curvature at row {number}")
        require(absolute > 0.0 and
                abs(math.log10(absolute) - log_value) <= 5.0e-13,
                f"Athena linear/log curvature mismatch at row {number}")
        require(not curve["tau"] or tau > curve["tau"][-1],
                f"Athena {name} proper time is not strictly increasing")
        curve["tau"].append(tau)
        curve["log10_abs_I"].append(log_value)
    require(set(curves) == {"n128", "n192", "n256"} and
            all(len(curves[name]["tau"]) >= 2 for name in curves),
            "Athena resolution inventory is not exact")
    return curves[resolution]


def design() -> dict[str, Any]:
    result = strict_load(DESIGN_PATH)
    require(set(result) == {"schema", "qualification_claim", "paper_reference",
                            "initial_data", "execution", "resolutions",
                            "fixed_policy", "acceptance"},
            "Figure-3 design inventory changed")
    require(result["schema"] == "athenak_cartoon_r4_brill_figure3_design_v1" and
            result["qualification_claim"] == "current_gauge_analogue_only",
            "Figure-3 design schema/claim changed")
    require(result["execution"] == {
                "order": ["n128", "n192", "n256"], "mpi_ranks": 4,
                "backend": "Cuda", "one_executable": True,
                "cmake_option": "Athena_ENABLE_IRISK_INTERPOLATOR:BOOL=ON",
                "required_import_coordinate_map":
                    "CartoonIrisInterpolationCoordinates:(x1,0,x2)"},
            "prospective execution contract changed")
    require(result["initial_data"].get("provenance_artifacts") == {
                "source": {"basename": "axisymmetric_wave_2607.md",
                           "location": "provenance", "size": 8577},
                "executable": {"basename": "iris_brill_wave_export_study",
                               "location": "provenance", "size": 1896008},
                "input": {"basename": "input.txt", "location": "root",
                          "size": 1289}},
            "Brill provenance relocation contract changed")
    require(result["initial_data"].get("import_mode") ==
            "direct_global_coefficients" and
            result["initial_data"].get("initial_lapse") ==
            "precollapsed_psi_minus_2" and
            result["initial_data"].get("direct_coefficient_stream_sha256") ==
            "ff0993c390513c15d6aa65857a0a3c710f2e2c3faf5717d9d63245203ccf2d6b",
            "direct Brill import/lapse/coefficient contract changed")
    expected = {"n128": (128, 128), "n192": (192, 192),
                "n256": (256, 256)}
    require(set(result["resolutions"]) == set(expected),
            "resolution pool changed")
    for name, shape in expected.items():
        record = result["resolutions"][name]
        require((record["nx1"], record["nx2"]) == shape and
                record["meshblock_nx1"] == 32 and
                record["meshblock_nx2"] == 32,
                f"resolution contract changed for {name}")
    return result


def validate_template() -> None:
    spec = design()
    blocks = r1.parse_athinput(TEMPLATE)
    require(blocks["problem"].get("pgen_name") == "z4c_irisk_xcts" and
            blocks["problem"].get("irisk_adm_import_mode") ==
            "direct_global_coefficients" and
            blocks["problem"].get("irisk_adm_spectral_file") == PAYLOAD_SENTINEL and
            blocks["problem"].get("brill_global_coefficients_file") ==
            COEFFICIENT_SENTINEL and
            blocks["problem"].get("brill_direct_initial_lapse") ==
            "precollapsed_psi_minus_2",
            "template does not select direct coefficients and pre-collapsed lapse")
    require(blocks["z4c"].get("symmetry") == "cartoon_so2" and
            blocks["z4c"].get("coordinate_map") ==
            "signed_rho_z_suppressed_y_v1" and
            blocks["mesh"].get("nx3") == "1" and
            blocks["meshblock"].get("nx3") == "1",
            "template is not collapsed signed-rho Cartoon")
    require(float(blocks["mesh"]["x1min"]) ==
            -float(blocks["mesh"]["x1max"]) and
            int(blocks["mesh"]["nx1"]) % int(blocks["meshblock"]["nx1"]) == 0 and
            (int(blocks["mesh"]["nx1"]) //
             int(blocks["meshblock"]["nx1"])) % 2 == 0,
            "template violates the internal-axis MeshBlock topology")
    policy = spec["fixed_policy"]
    require(blocks["mesh_refinement"].get("refinement") == "adaptive" and
            blocks["z4c_amr"].get("method") == policy["amr_method"] and
            float(blocks["z4c_amr"]["dchi_max"]) == policy["dchi_max"] and
            int(blocks["z4c_amr"]["max_ref_lev"]) ==
            policy["max_refinement_level"], "AMR policy changed")
    require(blocks["z4c"].get("telegraph_lapse") == "true" and
            blocks["z4c"].get("telegraph_max_K") == "true" and
            blocks["z4c"].get("lapse_oplog") == "0.0" and
            blocks["z4c"].get("lapse_harmonic") == "0.0",
            "current-gauge contract changed")
    outputs = [value for key, value in blocks.items() if key.startswith("output")]
    require([value.get("file_type") for value in outputs].count("hst") == 1 and
            [value.get("file_type") for value in outputs].count("rst") == 1,
            "history/restart inventory changed")
    require({value.get("variable") for value in outputs} >=
            {"z4c", "con", "adm", "weyl", "z4c_diag"},
            "state evidence inventory changed")


def validate_source_contract(source: Path) -> None:
    cmake = (source / "CMakeLists.txt").read_text(encoding="utf-8")
    dispatch = (source / "src/pgen/pgen.cpp").read_text(encoding="utf-8")
    importer = (source / "src/pgen/z4c_irisk_xcts.cpp").read_text(
        encoding="utf-8")
    symmetry = (source / "src/z4c/z4c_symmetry.cpp").read_text(
        encoding="utf-8")
    require("option(Athena_ENABLE_IRISK_INTERPOLATOR" in cmake,
            "IrisK interpolator feature option is absent")
    require('pgen_fun_name.compare("z4c_irisk_xcts")' in dispatch and
            "Z4cIrisXcts(pin, is_restart)" in dispatch,
            "input-selected IrisK dispatch is absent")
    require(importer.count("IrisAthenakSpectralOpen") == 1 and
            importer.count("IrisAthenakSpectralInterpolateCartesian") == 1,
            "IrisK importer boundary changed")
    require('input.problem_generator == "z4c_irisk_xcts"' in symmetry,
            "Cartoon admission for the IrisK importer is absent")
    # Mesh x2 is physical z and mesh x3 is the suppressed y direction in the
    # signed-rho map.  A Cartesian interpolator must therefore see (x1,0,x2),
    # not the ordinary Cartesian mesh tuple (x1,x2,x3).  The 01a base does not
    # yet meet this consuming production contract, so preparation must stop.
    require("cartoon_so2" in importer and
            "signed_rho_z_suppressed_y_v1" in importer and
            "CartoonIrisInterpolationCoordinates" in importer,
            "IrisK importer lacks reviewed signed-rho (x1,0,x2) coordinate mapping")
    require('import_mode == "direct_global_coefficients"' in importer and
            "ReadBrillGlobalCoefficients(resolved_filename)" in importer and
            'initial_lapse == "precollapsed_psi_minus_2"' in importer,
            "source lacks direct Brill/pre-collapsed import contract")


def load_cache(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw or raw.startswith(("#", "//")):
            continue
        require("=" in raw and ":" in raw.split("=", 1)[0],
                f"malformed CMake cache record at {path}:{number}")
        typed, value = raw.split("=", 1)
        key, _ = typed.split(":", 1)
        require(key not in values, f"duplicate CMake cache key {key}")
        values[key] = value
    return values


def validate_cache(path: Path, source: Path) -> dict[str, str]:
    require(path.is_file(), f"missing CMakeCache.txt: {path}")
    values = load_cache(path)
    require(values.get("Athena_ENABLE_IRISK_INTERPOLATOR") == "ON",
            "Athena_ENABLE_IRISK_INTERPOLATOR is not ON")
    require(values.get("Kokkos_ENABLE_CUDA") == "ON",
            "future Figure-3 executable is not CUDA-enabled")
    require(Path(values.get("CMAKE_HOME_DIRECTORY", "")).resolve() == source,
            "CMake cache is not bound to the requested source checkout")
    return values


def require_hash(value: Any, label: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value),
            f"malformed SHA-256 for {label}")
    return value


def resolved_provenance_artifact(label: str, record: dict[str, Any],
                                 contract: dict[str, Any],
                                 artifact_root: Path) -> Path:
    """Resolve one authenticated sidecar path without weakening its binding."""
    embedded_value = record.get("path")
    require(isinstance(embedded_value, str) and embedded_value,
            f"malformed external Brill provenance path: {label}")
    embedded = Path(embedded_value)
    require(embedded.is_absolute() and ".." not in embedded.parts,
            f"external Brill provenance path traversal: {label}")
    require(embedded.name == contract["basename"],
            f"external Brill provenance basename mismatch: {label}")
    expected_size = contract["size"]
    expected_hash = record["sha256"]

    # Preserve the producer's exact path whenever the original immutable file
    # is still available.  A stale producer path does not authorize a search
    # outside the explicit artifact root below.
    if (os.path.lexists(embedded) and not embedded.is_symlink() and
            embedded.is_file() and embedded.stat().st_size == expected_size and
            sha256(embedded) == expected_hash):
        return embedded.resolve(strict=True)

    require(os.path.lexists(artifact_root) and not artifact_root.is_symlink() and
            artifact_root.is_dir(),
            "immutable Brill artifact root is missing, invalid, or a symlink")
    root = artifact_root.resolve(strict=True)
    location = contract["location"]
    if location == "root":
        candidates = [artifact_root / contract["basename"]]
    else:
        require(location == "provenance",
                f"unknown Brill provenance location: {label}")
        provenance = artifact_root / "provenance"
        require(os.path.lexists(provenance) and not provenance.is_symlink() and
                provenance.is_dir(),
                "immutable Brill provenance directory is missing or a symlink")
        candidates = []
        for directory, directories, files in os.walk(provenance,
                                                     followlinks=False):
            parent = Path(directory)
            candidates.extend(parent / name for name in directories + files
                              if name == contract["basename"])
    require(candidates, f"missing relocated Brill provenance artifact: {label}")
    require(len(candidates) == 1,
            f"ambiguous relocated Brill provenance artifact: {label}")
    candidate = candidates[0]
    require(os.path.lexists(candidate),
            f"missing relocated Brill provenance artifact: {label}")
    require(not candidate.is_symlink() and candidate.is_file(),
            "relocated Brill provenance artifact is not a regular nonsymlink "
            f"file or resolves outside artifact root: {label}")
    resolved = candidate.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"relocated Brill provenance artifact is outside artifact root: {label}") from error
    cursor = artifact_root
    for part in candidate.relative_to(artifact_root).parts:
        cursor = cursor / part
        require(not cursor.is_symlink(),
                f"relocated Brill provenance artifact uses a symlink/outside path: {label}")
    require(relative.parts and candidate.stat().st_size == expected_size,
            f"relocated Brill provenance artifact size mismatch: {label}")
    require(sha256(candidate) == expected_hash,
            f"relocated Brill provenance artifact hash mismatch: {label}")
    return resolved


def validate_handoff(payload: Path, sidecar: Path,
                     expected: dict[str, Any],
                     artifact_root: Path) -> dict[str, Any]:
    require(payload.is_file(), f"missing Brill payload: {payload}")
    require(sidecar.is_file(), f"missing Brill sidecar: {sidecar}")
    require(sidecar.name == payload.name + ".manifest.json",
            "Brill sidecar is not adjacent-name bound to the payload")
    require(sha256(payload) == expected["payload_sha256"],
            "accepted Brill payload hash mismatch")
    require(sha256(sidecar) == expected["sidecar_sha256"],
            "accepted Brill sidecar hash mismatch")
    manifest = strict_load(sidecar)
    require(set(manifest) == {"schema_version", "artifact_format", "family",
                              "branch", "amplitude", "source_faithful",
                              "solver", "residuals", "export", "artifacts"},
            "Brill sidecar inventory is not exact")
    require(manifest["schema_version"] == 1 and
            manifest["artifact_format"] ==
            "IRIS_ATHENAK_SPECTRAL_ADM_PROVENANCE_V1" and
            manifest["family"] == expected["family"] and
            manifest["branch"] == expected["branch"] and
            manifest["amplitude"] == expected["amplitude"] and
            manifest["source_faithful"] is True,
            "Brill sidecar family/branch/amplitude/source contract mismatch")
    solver = manifest["solver"]
    require(isinstance(solver, dict) and set(solver) ==
            {"converged", "radial_points", "angular_points", "radial_scale",
             "adm_mass", "reciprocal_condition", "forward_error",
             "backward_error"} and solver.get("converged") is True and
            solver.get("radial_points") == expected["radial_points"] and
            solver.get("angular_points") == expected["angular_points"] and
            math.isfinite(solver.get("adm_mass", math.nan)) and
            solver["adm_mass"] > 0.0,
            "Brill solver record is not the accepted 48x32 solution")
    residuals = manifest["residuals"]
    require(isinstance(residuals, dict) and set(residuals) ==
            {"nonlinear", "hamiltonian", "momentum_max", "momentum_r",
             "momentum_theta", "maximal_lapse", "minimum_lapse",
             "oversampled_hamiltonian", "oversampled_momentum_r",
             "oversampled_momentum_theta", "oversampled_maximal_lapse"},
            "Brill residual inventory is not exact")
    export = manifest["export"]
    require(isinstance(export, dict) and set(export) ==
            {"order", "block_count", "nodes_per_block", "variable_count",
             "reconciled_traces"} and export["reconciled_traces"] is True and
            export["variable_count"] == 17,
            "Brill export inventory is not exact")
    artifacts = manifest["artifacts"]
    require(isinstance(artifacts, dict) and set(artifacts) ==
            {"source", "executable", "input", "coefficients", "adm_payload"},
            "Brill artifact inventory is not exact")
    for label in ("source", "executable", "input", "coefficients", "adm_payload"):
        require(isinstance(artifacts[label], dict),
                f"malformed Brill artifact {label}")
        require_hash(artifacts[label].get("sha256"), label)
    require(set(artifacts["source"]) == {"identifier", "path", "sha256"} and
            set(artifacts["executable"]) == {"path", "sha256"} and
            set(artifacts["input"]) == {"path", "sha256"} and
            set(artifacts["coefficients"]) ==
            {"description", "schema", "path", "sha256"} and
            set(artifacts["adm_payload"]) == {"path", "sha256"},
            "Brill nested artifact schema is not exact")
    require(artifacts["adm_payload"]["sha256"] == expected["payload_sha256"] and
            Path(artifacts["adm_payload"]["path"]).name == payload.name and
            artifacts["coefficients"]["sha256"] ==
            expected["coefficient_sha256"] and
            expected["irisk_commit"] in artifacts["source"].get("identifier", ""),
            "Brill payload/coefficient/source binding mismatch")
    relocation = expected.get("provenance_artifacts")
    require(isinstance(relocation, dict) and set(relocation) ==
            {"source", "executable", "input"},
            "Brill provenance relocation inventory is not exact")
    for label in ("source", "executable", "input"):
        contract = relocation[label]
        require(isinstance(contract, dict) and set(contract) ==
                {"basename", "location", "size"} and
                contract["location"] in {"root", "provenance"} and
                isinstance(contract["size"], int) and contract["size"] > 0,
                f"malformed Brill provenance relocation contract: {label}")
        resolved_provenance_artifact(label, artifacts[label], contract,
                                     artifact_root)
    return manifest


def validate_paper_figure(path: Path, expected: str) -> None:
    require(path.is_file() and sha256(path) == expected,
            "official Figure-3 vector PDF hash mismatch")


def render_input(name: str, payload: Path, coefficients: Path,
                 spec: dict[str, Any]) -> str:
    require(not any(character.isspace() for path in (payload, coefficients)
                    for character in str(path)),
            "direct Brill input path contains whitespace unsupported by Athena input")
    blocks = r1.parse_athinput(TEMPLATE)
    resolution = spec["resolutions"][name]
    text = r1.render_athinput(blocks, {
        "job/basename": f"cartoon_r4_brill_figure3_{name}",
        "mesh/nx1": str(resolution["nx1"]),
        "mesh/nx2": str(resolution["nx2"]),
        "meshblock/nx1": str(resolution["meshblock_nx1"]),
        "meshblock/nx2": str(resolution["meshblock_nx2"]),
        "problem/irisk_adm_spectral_file": str(payload),
        "problem/brill_global_coefficients_file": str(coefficients),
        "problem/constraint_summary_file":
            f"cartoon_r4_brill_figure3_{name}-constraints.dat",
    })
    rendered = r1.parse_athinput_text(text, f"rendered-{name}")
    require(rendered["mesh"]["nx3"] == "1" and
            rendered["meshblock"]["nx3"] == "1" and
            int(rendered["mesh"]["nx1"]) //
            int(rendered["meshblock"]["nx1"]) % 2 == 0,
            f"rendered {name} topology is invalid")
    return text


def contract_payload(state: dict[str, Any]) -> dict[str, Any]:
    result = {key: state[key] for key in
              ("schema", "source", "executable", "cmake_cache", "design",
               "initial_data", "direct_coefficients", "paper_figure",
               "backend", "ranks", "inputs", "execution_order")}
    # Runtime status, commands, and hashes are evidence rather than prospective
    # contract fields.  Bind only the exact case inventory here so state updates
    # cannot either invalidate or expand the declared pool.
    result["case_inventory"] = list(state["cases"])
    return result


def contract_digest(state: dict[str, Any]) -> str:
    encoded = json.dumps(contract_payload(state), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_state(state: dict[str, Any]) -> None:
    require(state.get("schema") == STATE_SCHEMA and
            state.get("backend") == "Cuda" and state.get("ranks") == 4,
            "unknown or altered Figure-3 campaign state")
    require(state.get("contract_sha256") == contract_digest(state),
            "Figure-3 campaign contract changed after preparation")
    source = Path(state["source"]["path"])
    require(git(source, "rev-parse", "HEAD") == state["source"]["commit"] and
            git(source, "rev-parse", "HEAD^{tree}") == state["source"]["tree"] and
            git(source, "rev-parse", "HEAD:kokkos") ==
            state["source"]["kokkos_commit"], "bound source identity changed")
    executable = Path(state["executable"]["path"])
    require(executable.is_file() and sha256(executable) ==
            state["executable"]["sha256"], "bound executable changed")
    cache = Path(state["cmake_cache"]["path"])
    require(cache.is_file() and sha256(cache) == state["cmake_cache"]["sha256"],
            "bound CMake cache changed")
    for group in ("initial_data", "paper_figure"):
        for key in ("archive_path",):
            path = Path(state[group][key])
            require(path.is_file() and sha256(path) == state[group]["sha256"],
                    f"bound {group} changed")
    sidecar = Path(state["initial_data"]["sidecar_archive_path"])
    require(sidecar.is_file() and sha256(sidecar) ==
            state["initial_data"]["sidecar_sha256"], "bound sidecar changed")
    coefficients = Path(state["direct_coefficients"]["archive_path"])
    require(coefficients.is_file() and not coefficients.is_symlink() and
            sha256(coefficients) == state["direct_coefficients"]["sha256"],
            "bound direct Brill coefficient stream changed")
    for record in state["inputs"].values():
        path = Path(record["path"])
        require(path.is_file() and sha256(path) == record["sha256"],
                f"bound rendered input changed: {path}")


def prepare(args: argparse.Namespace) -> None:
    validate_template()
    spec = design()
    source = args.source.resolve()
    executable = args.executable.resolve()
    cache = args.cmake_cache.resolve()
    payload = args.payload.resolve()
    sidecar = args.sidecar.resolve()
    coefficients = args.coefficients.resolve()
    artifact_root = args.artifact_root
    figure = args.paper_figure.resolve()
    output = args.output.resolve()
    validate_source_contract(source)
    require(git(source, "status", "--porcelain") == "",
            "campaign preparation requires a clean source checkout")
    require(executable.is_file(), f"missing AthenaK executable: {executable}")
    validate_cache(cache, source)
    manifest = validate_handoff(payload, sidecar, spec["initial_data"],
                                artifact_root)
    require_file_hash(coefficients,
                      spec["initial_data"]["direct_coefficient_stream_sha256"],
                      "direct Brill coefficient stream")
    validate_paper_figure(figure, spec["paper_reference"]["figure_sha256"])
    require(not output.exists() or not any(output.iterdir()),
            "campaign output root already contains evidence")
    provenance = output / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    archived_payload = provenance / payload.name
    archived_sidecar = provenance / sidecar.name
    archived_coefficients = provenance / coefficients.name
    archived_figure = provenance / "arxiv2607.10843v1_figure3.pdf"
    shutil.copyfile(payload, archived_payload)
    shutil.copyfile(sidecar, archived_sidecar)
    shutil.copyfile(coefficients, archived_coefficients)
    shutil.copyfile(figure, archived_figure)
    input_dir = output / "inputs"
    input_dir.mkdir()
    inputs: dict[str, dict[str, str]] = {}
    for name in spec["execution"]["order"]:
        path = input_dir / f"cartoon_r4_brill_figure3_{name}.athinput"
        path.write_text(render_input(name, archived_payload.resolve(),
                                     archived_coefficients.resolve(), spec),
                        encoding="utf-8")
        inputs[name] = {"path": str(path.resolve()), "sha256": sha256(path),
                        "template_sha256": sha256(TEMPLATE)}
    cases = {name: {"status": "pending"} for name in spec["execution"]["order"]}
    state: dict[str, Any] = {
        "schema": STATE_SCHEMA,
        "source": {"path": str(source), "commit": git(source, "rev-parse", "HEAD"),
                   "tree": git(source, "rev-parse", "HEAD^{tree}"),
                   "kokkos_commit": git(source, "rev-parse", "HEAD:kokkos")},
        "executable": {"path": str(executable), "sha256": sha256(executable)},
        "cmake_cache": {"path": str(cache), "sha256": sha256(cache)},
        "design": {"path": str(DESIGN_PATH), "sha256": sha256(DESIGN_PATH)},
        "initial_data": {
            "source_path": str(payload), "archive_path": str(archived_payload.resolve()),
            "sha256": sha256(archived_payload),
            "sidecar_source_path": str(sidecar),
            "sidecar_archive_path": str(archived_sidecar.resolve()),
            "sidecar_sha256": sha256(archived_sidecar),
            "artifact_root": str(artifact_root.resolve(strict=True)),
            "manifest": manifest,
        },
        "direct_coefficients": {
            "source_path": str(coefficients),
            "archive_path": str(archived_coefficients.resolve()),
            "sha256": sha256(archived_coefficients),
            "producer_coefficient_sha256":
                spec["initial_data"]["coefficient_sha256"],
            "schema": "IRIS_BRILL_GLOBAL_COEFFICIENTS_V1",
        },
        "paper_figure": {"source_path": str(figure),
                         "archive_path": str(archived_figure.resolve()),
                         "sha256": sha256(archived_figure),
                         "machine_readable_curve_available": False},
        "backend": "Cuda", "ranks": 4, "inputs": inputs,
        "execution_order": list(spec["execution"]["order"]), "cases": cases,
    }
    state["contract_sha256"] = contract_digest(state)
    atomic_json(output / "campaign_state.json", state)


def process_record(environment: dict[str, str] | None = None) -> dict[str, Any]:
    selected = os.environ if environment is None else environment
    keys = ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_NODELIST",
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "KOKKOS_NUM_THREADS",
            "MPICH_GPU_SUPPORT_ENABLED", "MPICH_GPU_IPC_ENABLED",
            "MPICH_OFI_NIC_POLICY")
    return {"utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
            "cwd": str(Path.cwd()),
            "environment": {key: selected.get(key) for key in keys}}


def run_case(args: argparse.Namespace) -> None:
    state_path = args.state.resolve()
    state = strict_load(state_path)
    verify_state(state)
    require(args.case in state["cases"], f"unknown case {args.case}")
    expected_next = next((name for name in state["execution_order"]
                          if state["cases"][name]["status"] != "complete"), None)
    require(args.case == expected_next,
            f"cases must run in prospective order; next is {expected_next}")
    case = state["cases"][args.case]
    require(case["status"] == "pending", f"case {args.case} is not pending")
    root = state_path.parent
    run_dir = root / "runs" / args.case
    require(not run_dir.exists() or not any(run_dir.iterdir()),
            f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    wrapper = Path(state["source"]["path"]) / (
        "tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py")
    require(wrapper.is_file(), "established rank/GPU evidence wrapper is missing")
    athena = [state["executable"]["path"], "-i",
              state["inputs"][args.case]["path"], "-d", str(run_dir)]
    command = ["srun", "--nodes=1", "--ntasks=4", "--ntasks-per-node=4",
               "--cpus-per-task=8", "--gpus-per-task=1",
               "--gpu-bind=map_gpu:0,1,2,3", "--cpu-bind=cores", "--exact",
               "--kill-on-bad-exit=1", sys.executable, str(wrapper),
               "--evidence-dir", str(run_dir / "bindings"), "--require-cuda",
               "--", *athena]
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "8", "KOKKOS_NUM_THREADS": "8",
                        "MPICH_GPU_SUPPORT_ENABLED": "1",
                        "MPICH_GPU_IPC_ENABLED": "0",
                        "MPICH_OFI_NIC_POLICY": "GPU"})
    case["command"] = command
    started = time.perf_counter()
    atomic_json(run_dir / "process_start.json",
                {"schema": "athenak_process_evidence_v1",
                 **process_record(environment),
                 "command": command})
    log = run_dir / "run.log"
    with log.open("wb") as stream:
        result = subprocess.run(command, cwd=root, env=environment, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    atomic_json(run_dir / "process_end.json",
                {"schema": "athenak_process_evidence_v1",
                 **process_record(environment), "exit_code": result.returncode,
                 "elapsed_seconds": time.perf_counter() - started})
    case.update({"exit_code": result.returncode, "log_path": str(log),
                 "log_sha256": sha256(log),
                 "status": "complete" if result.returncode == 0 else "failed"})
    atomic_json(state_path, state)
    require(result.returncode == 0, f"case {args.case} failed; evidence retained")


def linear_value(x: list[float], y: list[float], point: float) -> float:
    require(len(x) == len(y) and len(x) >= 2 and x[0] <= point <= x[-1],
            "interpolation point is outside the curve")
    low, high = 0, len(x) - 1
    while high - low > 1:
        middle = (low + high) // 2
        if x[middle] <= point:
            low = middle
        else:
            high = middle
    width = x[high] - x[low]
    require(width > 0.0, "curve proper-time coordinate is not strictly increasing")
    fraction = (point - x[low]) / width
    return y[low] + fraction * (y[high] - y[low])


def peak_on_interval(curve: dict[str, list[float]], start: float,
                     end: float) -> tuple[float, float]:
    points = [start, end]
    points.extend(point for point in curve["tau"] if start < point < end)
    values = [(linear_value(curve["tau"], curve["log10_abs_I"], point), point)
              for point in points]
    # Prefer the earlier time for a flat peak so the result is deterministic.
    return max(values, key=lambda item: (item[0], -item[1]))


def write_published_comparison_grid(
        path: Path,
        records: list[tuple[str, int, float, float, float, float, float, bool]]
        ) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("published_series", "grid_index", "proper_time",
                         "published_centerline_log10_abs_I",
                         "athena_current_gauge_log10_abs_I",
                         "athena_minus_published_log10_abs_I",
                         "published_coordinate_uncertainty_log10_abs_I",
                         "outside_coordinate_uncertainty"))
        for record in records:
            writer.writerow((*record[:-1], "true" if record[-1] else "false"))


def compare_published_curves(published_csv: Path, published_metadata: Path,
                             athena_csv: Path, athena_hash: str,
                             resolution: str, output: Path,
                             published_csv_hash: str = PUBLISHED_CURVES_SHA256,
                             published_metadata_hash: str =
                             PUBLISHED_METADATA_SHA256) -> dict[str, Any]:
    published, metadata = read_published_curves(
        published_csv, published_csv_hash, published_metadata,
        published_metadata_hash)
    athena = read_athena_curve(athena_csv, athena_hash, resolution)
    log_uncertainty = float(
        metadata["uncertainty"]["coordinate_only_log10_abs_I_half_unit"])
    tau_uncertainty = float(
        metadata["uncertainty"]["coordinate_only_tau_half_unit"])
    metrics: dict[str, Any] = {}
    grid_records: list[
        tuple[str, int, float, float, float, float, float, bool]] = []
    for name in PUBLISHED_SERIES:
        reference = published[name]
        start = max(reference["tau"][0], athena["tau"][0])
        end = min(reference["tau"][-1], athena["tau"][-1])
        require(start < end, f"{name} and Athena curves have no explicit overlap")
        grid = [start + (end - start) * index /
                (PUBLISHED_COMPARISON_SAMPLES - 1)
                for index in range(PUBLISHED_COMPARISON_SAMPLES)]
        reference_values = [linear_value(reference["tau"],
                                         reference["log10_abs_I"], point)
                            for point in grid]
        athena_values = [linear_value(athena["tau"], athena["log10_abs_I"], point)
                         for point in grid]
        residuals = [trial - paper for paper, trial in
                     zip(reference_values, athena_values)]
        rms = math.sqrt(sum(value * value for value in residuals) /
                        PUBLISHED_COMPARISON_SAMPLES)
        maximum = max(abs(value) for value in residuals)
        paper_peak, paper_peak_time = peak_on_interval(reference, start, end)
        athena_peak, athena_peak_time = peak_on_interval(athena, start, end)
        peak_ratio = 10.0 ** (athena_peak - paper_peak)
        require(math.isfinite(peak_ratio), f"{name} peak curvature ratio is nonfinite")
        metrics[name] = {
            "overlap_start_proper_time": start,
            "overlap_end_proper_time": end,
            "interpolation_grid": {
                "kind": "uniform_in_proper_time_inclusive",
                "samples": PUBLISHED_COMPARISON_SAMPLES,
            },
            "published_time_coverage_fraction":
                (end - start) / (reference["tau"][-1] - reference["tau"][0]),
            "athena_time_coverage_fraction":
                (end - start) / (athena["tau"][-1] - athena["tau"][0]),
            "log10_curvature_rms_error": rms,
            "log10_curvature_max_abs_error": maximum,
            "published_peak": {"proper_time": paper_peak_time,
                               "log10_abs_I": paper_peak,
                               "abs_I": 10.0 ** paper_peak},
            "athena_peak": {"proper_time": athena_peak_time,
                            "log10_abs_I": athena_peak,
                            "abs_I": 10.0 ** athena_peak},
            "peak_log10_value_offset": athena_peak - paper_peak,
            "peak_abs_I_ratio": peak_ratio,
            "peak_proper_time_offset": athena_peak_time - paper_peak_time,
            "uncertainty_aware_flags": {
                "basis": "published_vector_coordinate_serialization_only",
                "rms_exceeds_log10_coordinate_half_unit": rms > log_uncertainty,
                "max_exceeds_log10_coordinate_half_unit": maximum > log_uncertainty,
                "peak_value_offset_exceeds_log10_coordinate_half_unit":
                    abs(athena_peak - paper_peak) > log_uncertainty,
                "peak_time_offset_exceeds_tau_coordinate_half_unit":
                    abs(athena_peak_time - paper_peak_time) > tau_uncertainty,
                "grid_fraction_outside_log10_coordinate_half_unit":
                    sum(abs(value) > log_uncertainty for value in residuals) /
                    PUBLISHED_COMPARISON_SAMPLES,
            },
        }
        grid_records.extend((name, index, point, paper, trial, residual,
                             log_uncertainty, abs(residual) > log_uncertainty)
                            for index, (point, paper, trial, residual) in enumerate(
                                zip(grid, reference_values, athena_values, residuals)))
    grid_path = output.with_name(output.stem + "_grid.csv")
    write_published_comparison_grid(grid_path, grid_records)
    result = {
        "schema": PUBLISHED_COMPARISON_SCHEMA,
        "status": "quantitative_comparison_only_not_qualification",
        "qualification_claim": "current_gauge_analogue_only",
        "inputs": {
            "published_vector_curves": {"path": str(published_csv.resolve()),
                                        "sha256": published_csv_hash},
            "published_vector_metadata": {"path": str(published_metadata.resolve()),
                                          "sha256": published_metadata_hash},
            "athena_current_gauge_curve": {"path": str(athena_csv.resolve()),
                                           "sha256": athena_hash,
                                           "resolution": resolution},
        },
        "interpretation": {
            "published_data_kind": "rendered PDF polyline centerlines",
            "published_data_are_raw_simulation_samples": False,
            "coordinate_uncertainty_only": True,
            "current_gauge_caveat":
                ("AthenaK uses the frozen campaign's current gauge and is an "
                 "analogue, not an exact gauge-matched reproduction."),
            "use_limit": metadata["uncertainty"]["use_limit"],
        },
        "metrics": metrics,
        "plot_data": {"path": str(grid_path.resolve()), "sha256": sha256(grid_path)},
    }
    atomic_json(output, result)
    return result


def compare_published(args: argparse.Namespace) -> None:
    compare_published_curves(
        args.published_csv.resolve(), args.published_metadata.resolve(),
        args.athena_csv.resolve(), args.athena_sha256, args.resolution,
        args.output.resolve())


def curve_error(reference: dict[str, list[float]],
                trial: dict[str, list[float]], start: float, end: float,
                samples: int) -> float:
    require(start < end, "central curves have no common proper-time interval")
    points = [start + (end - start) * index / (samples - 1)
              for index in range(samples)]
    ref = [linear_value(reference["axisTau"], reference["axisKret"], point)
           for point in points]
    other = [linear_value(trial["axisTau"], trial["axisKret"], point)
             for point in points]
    scale = max(max(abs(value) for value in ref), sys.float_info.min)
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(ref, other)) /
                     samples) / scale


def evidence_hashes(run_dir: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(run_dir).as_posix()
        require(relative not in records, f"duplicate evidence path {relative}")
        records[relative] = sha256(path)
    return records


def collect_case(state: dict[str, Any], name: str, reader: Any) -> dict[str, Any]:
    root = Path(state["cases"][name]["log_path"]).parent
    histories = sorted(root.glob("*.hst"))
    require(len(histories) == 1, f"expected one history for {name}")
    history = r1.read_history(histories[0])
    require(tuple(history) == r1.HISTORY_KEYS,
            f"unexpected history inventory for {name}")
    count = len(history["axisTau"])
    require(count >= 2 and all(len(values) == count for values in history.values()),
            f"incomplete history columns for {name}")
    require(all(history["axisTau"][index] < history["axisTau"][index + 1]
                for index in range(count - 1)),
            f"proper time is not strictly increasing for {name}")
    require(all(value >= 0.0 for value in history["axisKret"]),
            f"central absolute Kretschmann is negative for {name}")
    groups = r1.final_binary_groups(root, reader)
    blocks = r1.parse_athinput(Path(state["inputs"][name]["path"]))
    root_blocks_x1 = int(blocks["mesh"]["nx1"]) // int(
        blocks["meshblock"]["nx1"])
    tree = r1.tree_summary(groups["adm"], root_blocks_x1)
    bindings = r1.binding_summary(root, "Cuda", 4)
    restart, carrier = r1.latest_restart(root)
    rendered_input = Path(state["inputs"][name]["path"]).read_text(
        encoding="utf-8")
    require(Path(state["initial_data"]["archive_path"]).name in rendered_input and
            Path(state["direct_coefficients"]["archive_path"]).name in
            rendered_input and
            "irisk_adm_import_mode = direct_global_coefficients" in rendered_input and
            "brill_direct_initial_lapse = precollapsed_psi_minus_2" in rendered_input,
            f"rendered {name} input lost direct Brill provenance or lapse policy")
    normalized_h: list[float | None] = []
    normalized_m: list[float | None] = []
    for volume, max_k, h2, m2 in zip(history["Volume"], history["max_abs_K"],
                                     history["H-norm2"], history["M-norm2"]):
        require(volume > 0.0 and h2 >= 0.0 and m2 >= 0.0,
                f"invalid constraint integral for {name}")
        if max_k == 0.0:
            normalized_h.append(None)
            normalized_m.append(None)
        else:
            normalized_h.append(math.sqrt(h2 / volume) / (max_k * max_k))
            normalized_m.append(math.sqrt(m2 / volume) / (max_k * max_k))
    return {"history": history, "normalized_h": normalized_h,
            "normalized_m": normalized_m, "tree": tree,
            "rank_bindings": bindings, "restart": {"path": str(restart),
                "sha256": sha256(restart), "central": carrier},
            "evidence_files": evidence_hashes(root)}


def write_curve_csv(path: Path, cases: dict[str, Any], order: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("resolution", "cycle", "coordinate_time", "proper_time",
                         "abs_kretschmann_I", "log10_abs_kretschmann_I",
                         "axis_lapse", "normalized_H", "normalized_M",
                         "meshblocks", "max_refinement_level",
                         "max_meshblocks_per_rank"))
        for name in order:
            case = cases[name]
            history = case["history"]
            for index, value in enumerate(history["axisKret"]):
                log_value: str | float = "" if value == 0.0 else math.log10(value)
                writer.writerow((name, history["cycle"][index],
                                 history["time"][index], history["axisTau"][index],
                                 value, log_value, history["axisLapse"][index],
                                 "" if case["normalized_h"][index] is None else
                                 case["normalized_h"][index],
                                 "" if case["normalized_m"][index] is None else
                                 case["normalized_m"][index],
                                 history["nmb_total"][index],
                                 history["maxRefLev"][index],
                                 history["maxNmbRank"][index]))


def analyze(args: argparse.Namespace) -> None:
    state_path = args.state.resolve()
    state = strict_load(state_path)
    verify_state(state)
    require(all(state["cases"][name]["status"] == "complete"
                for name in state["execution_order"]),
            "all three prospective cases must complete before analysis")
    reader = r1.import_binary_reader(Path(state["source"]["path"]))
    cases = {name: collect_case(state, name, reader)
             for name in state["execution_order"]}
    spec = design()
    acceptance = spec["acceptance"]
    common_start = max(cases[name]["history"]["axisTau"][0]
                       for name in state["execution_order"])
    common_end = min(cases[name]["history"]["axisTau"][-1]
                     for name in state["execution_order"])
    samples = max(acceptance["minimum_common_samples"], 256)
    low_medium = curve_error(cases["n192"]["history"],
                             cases["n128"]["history"], common_start, common_end,
                             samples)
    medium_high = curve_error(cases["n256"]["history"],
                              cases["n192"]["history"], common_start, common_end,
                              samples)
    constraint_values = [value for name in state["execution_order"]
                         for key in ("normalized_h", "normalized_m")
                         for value in cases[name][key] if value is not None]
    require(constraint_values, "all normalized constraint records are undefined")
    constraint_max = max(constraint_values)
    failures: list[str] = []
    if common_end < acceptance["minimum_common_proper_time"]:
        failures.append("insufficient_common_proper_time")
    if medium_high > acceptance["fine_medium_normalized_l2_max"]:
        failures.append("fine_medium_curve_difference")
    if medium_high > acceptance["required_error_contraction"] * low_medium:
        failures.append("central_curve_error_did_not_contract")
    if constraint_max > acceptance["normalized_constraint_linf_max"]:
        failures.append("normalized_constraint_limit")
    curve_path = state_path.parent / "plot_data" / "figure3_current_gauge.csv"
    write_curve_csv(curve_path, cases, state["execution_order"])
    comparison = {
        "claim": "current_gauge_analogue_only",
        "paper_raw_curve_machine_readable": False,
        "published_rendered_vector_centerline_available": True,
        "paper_figure_sha256": state["paper_figure"]["sha256"],
        "warning": ("No authors' raw curve samples are available. A separately "
                    "authenticated rendered-vector centerline freeze may be compared "
                    "with the compare-published subcommand, subject to its gauge and "
                    "coordinate-uncertainty caveats."),
        "common_proper_time_start": common_start,
        "common_proper_time_end": common_end, "sample_count": samples,
        "low_medium_normalized_l2": low_medium,
        "medium_high_normalized_l2": medium_high,
        "normalized_constraint_linf": constraint_max,
    }
    report = {"schema": ANALYSIS_SCHEMA,
              "qualification_claim": "current_gauge_analogue_only",
              "verdict": "passed" if not failures else "failed",
              "failures": failures, "provenance": contract_payload(state),
              "comparison": comparison, "cases": cases,
              "plot_data": {"path": str(curve_path),
                            "sha256": sha256(curve_path)}}
    output = state_path.parent / "figure3_analysis.json"
    atomic_json(output, report)
    state["analysis"] = {"path": str(output), "sha256": sha256(output),
                         "verdict": report["verdict"],
                         "plot_data_sha256": sha256(curve_path)}
    atomic_json(state_path, state)
    require(not failures, "Figure-3 analogue gates failed: " + ", ".join(failures))


def synthetic_manifest(payload: Path, source: Path, executable: Path,
                       input_path: Path, payload_hash: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_format": "IRIS_ATHENAK_SPECTRAL_ADM_PROVENANCE_V1",
        "family": "brill_gaussian", "branch": "unique_time_symmetric",
        "amplitude": -0.047, "source_faithful": True,
        "solver": {"converged": True, "radial_points": 48,
                   "angular_points": 32, "radial_scale": 3.0,
                   "adm_mass": 2.6, "reciprocal_condition": 1.0e-10,
                   "forward_error": 1.0e-10, "backward_error": 1.0e-16},
        "residuals": {"nonlinear": None, "hamiltonian": 1.0e-12,
                      "momentum_max": None, "momentum_r": None,
                      "momentum_theta": None, "maximal_lapse": None,
                      "minimum_lapse": None, "oversampled_hamiltonian": None,
                      "oversampled_momentum_r": None,
                      "oversampled_momentum_theta": None,
                      "oversampled_maximal_lapse": None},
        "export": {"order": 4, "block_count": 25, "nodes_per_block": 64,
                   "variable_count": 17, "reconciled_traces": True},
        "artifacts": {
            "source": {"identifier":
                       "irisk:2a069fd0497ef4352d4ecd28c6879ac47b84a5a1",
                       "path": str(source), "sha256": sha256(source)},
            "executable": {"path": str(executable),
                           "sha256": sha256(executable)},
            "input": {"path": str(input_path), "sha256": sha256(input_path)},
            "coefficients": {
                "description":
                    "psi_minus_one_rational_sine_even_radial_major_angular_fast",
                "schema": "IRIS_BRILL_WAVE_STATE_V1", "path": None,
                "sha256": "4" * 64},
            "adm_payload": {"path": str(payload), "sha256": payload_hash},
        },
    }


def expect_failure(action: Any, token: str) -> None:
    try:
        action()
    except (RuntimeError, ValueError, json.JSONDecodeError) as error:
        require(token in str(error), f"wrong failure for {token}: {error}")
    else:
        raise RuntimeError(f"synthetic {token} case was accepted")


def self_test() -> None:
    validate_template()
    spec = design()
    rendered = r1.parse_athinput_text(
        render_input("n192", Path("/immutable/brill.adm_spectral"),
                     Path("/immutable/brill.coefficients"), spec),
        "synthetic-render")
    require(rendered["mesh"]["nx1"] == "192" and
            rendered["mesh"]["nx2"] == "192",
            "runtime resolution selection failed")
    invalid = r1.render_athinput(rendered, {"mesh/nx3": "2"})
    expect_failure(lambda: require(
        r1.parse_athinput_text(invalid, "invalid-topology")["mesh"]["nx3"] == "1",
        "topology"), "topology")
    with tempfile.TemporaryDirectory(prefix="cartoon-r4-brill-") as directory:
        root = Path(directory)
        handoff = root / "handoff"
        original = root / "producer"
        artifact_root = root / "immutable"
        provenance = artifact_root / "provenance"
        handoff.mkdir()
        original.mkdir()
        provenance.mkdir(parents=True)
        payload = handoff / "brill.adm_spectral"
        sidecar = handoff / "brill.adm_spectral.manifest.json"
        source = original / "source.md"
        executable = original / "iris"
        input_path = original / "input.txt"
        for path, content in ((payload, b"payload"), (source, b"source"),
                              (executable, b"exe"), (input_path, b"input")):
            path.write_bytes(content)
        relocated = {"source": provenance / source.name,
                     "executable": provenance / executable.name,
                     "input": artifact_root / input_path.name}
        for label, producer in (("source", source),
                                ("executable", executable),
                                ("input", input_path)):
            shutil.copyfile(producer, relocated[label])
        expected = dict(spec["initial_data"])
        expected.update({"payload_sha256": sha256(payload),
                         "sidecar_sha256": "", "coefficient_sha256": "4" * 64})
        expected["provenance_artifacts"] = {
            "source": {"basename": source.name, "location": "provenance",
                       "size": source.stat().st_size},
            "executable": {"basename": executable.name,
                           "location": "provenance",
                           "size": executable.stat().st_size},
            "input": {"basename": input_path.name, "location": "root",
                      "size": input_path.stat().st_size},
        }
        manifest = synthetic_manifest(payload, source, executable, input_path,
                                      expected["payload_sha256"])
        def authenticate(value: dict[str, Any]) -> dict[str, Any]:
            sidecar.write_text(json.dumps(value, sort_keys=True) + "\n",
                               encoding="utf-8")
            result = dict(expected)
            result["sidecar_sha256"] = sha256(sidecar)
            return result

        expected = authenticate(manifest)
        immutable_sidecar = sidecar.read_bytes()
        immutable_sidecar_hash = sha256(sidecar)
        validate_handoff(payload, sidecar, expected, artifact_root)
        source.write_bytes(b"sourcf")
        validate_handoff(payload, sidecar, expected, artifact_root)
        source.write_bytes(b"source")
        source.unlink()
        executable.unlink()
        input_path.unlink()
        validate_handoff(payload, sidecar, expected, artifact_root)
        require(sidecar.read_bytes() == immutable_sidecar and
                sha256(sidecar) == immutable_sidecar_hash,
                "relocation rewrote the authenticated sidecar")
        expect_failure(lambda: validate_handoff(
            payload, sidecar, expected, root / "absent-artifact-root"),
                       "artifact root is missing")
        missing = root / "missing.adm_spectral"
        expect_failure(lambda: validate_handoff(
            missing, sidecar, expected, artifact_root),
                       "missing Brill payload")
        payload.write_bytes(b"changed")
        expect_failure(lambda: validate_handoff(
            payload, sidecar, expected, artifact_root),
                       "payload hash mismatch")
        payload.write_bytes(b"payload")

        held_source = root / "held-source"
        relocated["source"].rename(held_source)
        expect_failure(lambda: validate_handoff(
            payload, sidecar, expected, artifact_root),
                       "missing relocated")
        held_source.rename(relocated["source"])
        duplicate = provenance / "nested" / source.name
        duplicate.parent.mkdir()
        shutil.copyfile(relocated["source"], duplicate)
        expect_failure(lambda: validate_handoff(
            payload, sidecar, expected, artifact_root),
                       "ambiguous relocated")
        duplicate.unlink()
        duplicate.parent.rmdir()
        good_source = relocated["source"].read_bytes()
        relocated["source"].write_bytes(b"sourcf")
        expect_failure(lambda: validate_handoff(
            payload, sidecar, expected, artifact_root), "hash mismatch")
        relocated["source"].write_bytes(good_source + b"x")
        expect_failure(lambda: validate_handoff(
            payload, sidecar, expected, artifact_root), "size mismatch")
        relocated["source"].write_bytes(good_source)
        outside = root / source.name
        relocated["source"].rename(outside)
        relocated["source"].symlink_to(outside)
        expect_failure(lambda: validate_handoff(
            payload, sidecar, expected, artifact_root), "outside artifact root")
        relocated["source"].unlink()
        outside.rename(relocated["source"])

        traversal = json.loads(json.dumps(manifest))
        traversal["artifacts"]["source"]["path"] = "/abs/../source.md"
        traversal_expected = authenticate(traversal)
        expect_failure(lambda: validate_handoff(
            payload, sidecar, traversal_expected, artifact_root), "traversal")
        wrong_name = json.loads(json.dumps(manifest))
        wrong_name["artifacts"]["source"]["path"] = "/abs/wrong.md"
        wrong_expected = authenticate(wrong_name)
        expect_failure(lambda: validate_handoff(
            payload, sidecar, wrong_expected, artifact_root),
                       "basename mismatch")
        expected = authenticate(manifest)
        sidecar.write_text('{"amplitude": NaN}\n', encoding="utf-8")
        bad = dict(expected)
        bad["sidecar_sha256"] = sha256(sidecar)
        expect_failure(lambda: validate_handoff(
            payload, sidecar, bad, artifact_root),
                       "nonfinite JSON")
        sidecar.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
        bad["sidecar_sha256"] = sha256(sidecar)
        expect_failure(lambda: validate_handoff(
            payload, sidecar, bad, artifact_root),
                       "duplicate JSON key")
        sidecar.write_text('{broken\n', encoding="utf-8")
        bad["sidecar_sha256"] = sha256(sidecar)
        expect_failure(lambda: validate_handoff(
            payload, sidecar, bad, artifact_root),
                       "invalid strict JSON")
        cache = root / "CMakeCache.txt"
        cache.write_text("A:BOOL=ON\nA:BOOL=OFF\n", encoding="utf-8")
        expect_failure(lambda: load_cache(cache), "duplicate CMake cache key")
    synthetic_curve = {"axisTau": [0.0, 1.0, 2.0],
                       "axisKret": [1.0, 2.0, 1.0]}
    require(curve_error(synthetic_curve, synthetic_curve, 0.0, 2.0, 5) == 0.0,
            "curve comparison oracle failed")
    with tempfile.TemporaryDirectory(prefix="cartoon-r4-published-compare-") as directory:
        root = Path(directory)
        published_csv = root / "published.csv"
        metadata_path = root / "metadata.json"
        athena_csv = root / "athena.csv"
        output = root / "comparison.json"
        metadata = {
            "axis_transform": {}, "independent_render_check": {}, "inventory": {},
            "schema": "axisymmetric-cartoon.figure3-published-vector-curves.v1",
            "scientific_binding": {
                "A": -0.047, "rho0": 5.0, "family": "Brill",
                "series": list(PUBLISHED_SERIES),
                "recovered_quantity": "rendered PDF polyline centerlines",
            },
            "source": {"paper": "arXiv:2607.10843v1", "figure": 3,
                       "contains_embedded_original_numeric_data": False},
            "tools": {},
            "uncertainty": {
                "coordinate_only_log10_abs_I_half_unit": 1.0e-6,
                "coordinate_only_tau_half_unit": 2.0e-6,
                "use_limit": "synthetic rendered-centerline fixture",
            },
        }
        atomic_json(metadata_path, metadata)

        def write_published() -> None:
            with published_csv.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(PUBLISHED_CURVE_HEADER)
                for name in PUBLISHED_SERIES:
                    for index, (tau, log_value) in enumerate(
                            ((0.0, 0.0), (1.0, 1.0), (2.0, 0.0))):
                        writer.writerow((name, index, tau, log_value, tau, log_value,
                                         tau, log_value, 10.0 ** log_value, "true"))

        def write_athena(times: tuple[float, ...], offset: float = 0.0,
                         nonfinite: bool = False) -> None:
            with athena_csv.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(ATHENA_CURVE_HEADER)
                for resolution in ("n128", "n192", "n256"):
                    for index, (tau, base) in enumerate(
                            zip(times, (0.0, 1.0, 0.0))):
                        bad = nonfinite and resolution == "n256" and index == 1
                        log_value: float | str = "nan" if bad else base + offset
                        absolute = "nan" if bad else 10.0 ** float(log_value)
                        writer.writerow((resolution, index, tau, tau, absolute,
                                         log_value, 1.0, "", "", 4, 0, 4))

        write_published()
        write_athena((0.0, 1.0, 2.0))
        published_hash = sha256(published_csv)
        metadata_hash = sha256(metadata_path)
        exact = compare_published_curves(
            published_csv, metadata_path, athena_csv, sha256(athena_csv), "n256",
            output, published_hash, metadata_hash)
        require(all(record["log10_curvature_rms_error"] == 0.0 and
                    record["log10_curvature_max_abs_error"] == 0.0 and
                    record["peak_log10_value_offset"] == 0.0 and
                    record["peak_proper_time_offset"] == 0.0
                    for record in exact["metrics"].values()),
                "exact published-curve comparison fixture failed")
        exact_hashes = (sha256(output), sha256(output.with_name(
            output.stem + "_grid.csv")))
        compare_published_curves(
            published_csv, metadata_path, athena_csv, sha256(athena_csv), "n256",
            output, published_hash, metadata_hash)
        require(exact_hashes == (sha256(output), sha256(output.with_name(
                    output.stem + "_grid.csv"))),
                "published-curve comparison is not byte deterministic")
        write_athena((0.0, 1.0, 2.0), offset=0.25)
        offset = compare_published_curves(
            published_csv, metadata_path, athena_csv, sha256(athena_csv), "n256",
            output, published_hash, metadata_hash)
        require(all(abs(record["log10_curvature_rms_error"] - 0.25) < 1.0e-15 and
                    abs(record["log10_curvature_max_abs_error"] - 0.25) < 1.0e-15 and
                    abs(record["peak_log10_value_offset"] - 0.25) < 1.0e-15 and
                    record["uncertainty_aware_flags"][
                        "max_exceeds_log10_coordinate_half_unit"] is True
                    for record in offset["metrics"].values()),
                "offset published-curve comparison fixture failed")
        write_athena((3.0, 4.0, 5.0))
        expect_failure(lambda: compare_published_curves(
            published_csv, metadata_path, athena_csv, sha256(athena_csv), "n256",
            output, published_hash, metadata_hash), "no explicit overlap")
        write_athena((0.0, 1.0, 2.0), nonfinite=True)
        expect_failure(lambda: compare_published_curves(
            published_csv, metadata_path, athena_csv, sha256(athena_csv), "n256",
            output, published_hash, metadata_hash), "nonfinite")
        write_athena((0.0, 1.0, 2.0))
        authenticated_hash = sha256(athena_csv)
        with athena_csv.open("a", encoding="utf-8") as stream:
            stream.write("mutation\n")
        expect_failure(lambda: compare_published_curves(
            published_csv, metadata_path, athena_csv, authenticated_hash, "n256",
            output, published_hash, metadata_hash), "hash mismatch")
    synthetic_state = {
        "schema": STATE_SCHEMA, "source": {}, "executable": {},
        "cmake_cache": {}, "design": {}, "initial_data": {},
        "direct_coefficients": {},
        "paper_figure": {}, "backend": "Cuda", "ranks": 4,
        "inputs": {}, "execution_order": ["n128"],
        "cases": {"n128": {"status": "pending"}},
    }
    before = contract_digest(synthetic_state)
    synthetic_state["cases"]["n128"]["status"] = "complete"
    require(before == contract_digest(synthetic_state),
            "runtime evidence incorrectly changes the prospective contract")
    synthetic_state["direct_coefficients"] = {"sha256": "a" * 64}
    require(before != contract_digest(synthetic_state),
            "prospective contract did not bind direct coefficient provenance")
    synthetic_state["direct_coefficients"] = {}
    synthetic_state["cases"]["extra"] = {"status": "pending"}
    require(before != contract_digest(synthetic_state),
            "prospective contract did not bind the exact case inventory")
    print("Cartoon R4 Brill Figure-3 campaign tooling self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    make = subparsers.add_parser("prepare")
    make.add_argument("--source", required=True, type=Path)
    make.add_argument("--executable", required=True, type=Path)
    make.add_argument("--cmake-cache", required=True, type=Path)
    make.add_argument("--payload", required=True, type=Path)
    make.add_argument("--sidecar", required=True, type=Path)
    make.add_argument("--coefficients", required=True, type=Path)
    make.add_argument("--artifact-root", required=True, type=Path)
    make.add_argument("--paper-figure", required=True, type=Path)
    make.add_argument("--output", required=True, type=Path)
    run = subparsers.add_parser("run-case")
    run.add_argument("--state", required=True, type=Path)
    run.add_argument("--case", required=True)
    inspect = subparsers.add_parser("analyze")
    inspect.add_argument("--state", required=True, type=Path)
    compare = subparsers.add_parser("compare-published")
    compare.add_argument("--published-csv", required=True, type=Path)
    compare.add_argument("--published-metadata", required=True, type=Path)
    compare.add_argument("--athena-csv", required=True, type=Path)
    compare.add_argument("--athena-sha256", required=True)
    compare.add_argument("--resolution", required=True,
                         choices=("n128", "n192", "n256"))
    compare.add_argument("--output", required=True, type=Path)
    subparsers.add_parser("self-test")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    elif args.action == "run-case":
        run_case(args)
    elif args.action == "analyze":
        analyze(args)
    elif args.action == "compare-published":
        compare_published(args)
    else:
        self_test()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
