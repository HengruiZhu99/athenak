#!/usr/bin/env python3
"""Fail-closed source, generator, target-shape, and coverage gates for Cartoon MMS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


LEGACY_PROVIDER_SHA = "50a88d96cb9640c393982124d439b7a693c103d4735b5c9ce7daac84a3114055"
FINITE_DIFF_SHA = "203074210997f5d2bf6ce1960b4a9574d96c9c4e2df8a34f89b0b5f023f4cd42"
BASE = "949ccd7828adf18a122c352996aa1a6393d762e7"
BASE_ORACLE_SHA = "03699d48a07aad3b928fba75a45b6ae0d1b1ed78a16b57e16fc0e93fc163d84c"
CATEGORIES = {"lightweight_structural", "runtime_mms", "retained_oracle",
              "static_source"}
CHECKSUM_PATHS = {
    "tst/unit/z4c/generate_z4c_cartoon_derivative_oracle.py",
    "src/pgen/unit_tests/z4c_cartoon_derivatives_oracle.hpp",
    "tst/unit/z4c/z4c_cartoon_derivatives_reference.json",
    "tst/unit/z4c/z4c_cartoon_derivatives_series.json",
    "tst/unit/z4c/z4c_cartoon_mms_coverage.json",
    "tst/inputs/z4c_cartoon_derivatives.athinput",
    "tst/inputs/z4c_cartoon_mms_search_manifest.json",
    "tst/test_suite/unit_tests/test_z4c_cartoon_derivatives.py",
    "tst/unit/z4c/z4c_cartoon_mms_static_test.py",
    "reports/axisymmetric_cartoon_z4c_arxiv2607/sections/cartoon_derivative_mms.tex",
}
REQUIRED_COVERAGE = {
    "LEG-PARITY-METADATA": ("lightweight_structural", "athena.z4c_cartoon_mms_structure", "1400-1418"),
    "LEG-PI-ROTATION": ("runtime_mms", "test_z4c_cartoon_derivatives.py", "1420-1490"),
    "LEG-FIELD-JET-ORACLE": ("retained_oracle", "athena_cartoon_derivatives_unit_test", "101-181"),
    "LEG-DISS-ORACLE": ("retained_oracle", "athena_cartoon_derivatives_unit_test", "196-214"),
    "LEG-NOISE-DELTA": ("runtime_mms", "test_z4c_cartoon_derivatives.py", "336-360"),
    "LEG-SAMPLE-RESULT-SPACE": ("runtime_mms", "test_z4c_cartoon_derivatives.py", "364-575"),
    "LEG-FULL-CARTOON-API": ("runtime_mms", "test_z4c_cartoon_derivatives.py", "579-919"),
    "LEG-CARTESIAN-DELEGATION": ("static_source", "athena.z4c_cartoon_mms_static", "579-919"),
    "LEG-VARIANCE-INSTANTIATION": ("lightweight_structural", "athena.z4c_cartoon_mms_structure", "579-919"),
    "LEG-MINIMAL-FIT-REACH-NG2/3/4": ("lightweight_structural", "athena.z4c_cartoon_mms_structure", "922-1018"),
    "LEG-BLOCK-EDGE-REACH-NG2/3/4": ("lightweight_structural", "athena.z4c_cartoon_mms_structure", "1021-1124"),
    "LEG-FIXED-RAW-POS/NEG-NG2/3/4": ("runtime_mms", "test_z4c_cartoon_derivatives.py", "1127-1176"),
    "LEG-REGULAR-CONV-NG2/3/4": ("runtime_mms", "test_z4c_cartoon_derivatives.py", "1191-1220"),
    "LEG-FITTED-RAW-CONV-NG2/3/4": ("runtime_mms", "test_z4c_cartoon_derivatives.py", "1222-1284"),
    "LEG-SHARED-PARITY-NOISE-NG2/3/4": ("runtime_mms", "test_z4c_cartoon_derivatives.py", "1286-1354"),
    "LEG-INDEPENDENT-TENSOR-NOISE-NG2/3/4": ("runtime_mms", "test_z4c_cartoon_derivatives.py", "1356-1379"),
    "LEG-DIAGNOSTIC-AXIS-NG2/3/4": ("runtime_mms", "test_z4c_cartoon_derivatives.py", "1381-1397"),
    "LEG-DISS-EXACTLY-ONCE-NG2/3/4": ("lightweight_structural", "athena.z4c_cartoon_mms_static", "1381-1397"),
    "LEG-ORDER-AGGREGATION-NG2/3/4": ("retained_oracle", "athena_cartoon_derivatives_unit_test", "1179-1505"),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def lambda_bodies(source: str) -> list[str]:
    bodies = []
    cursor = 0
    while True:
        start = source.find("KOKKOS_LAMBDA", cursor)
        if start < 0:
            return bodies
        brace = source.find("{", start)
        require(brace >= 0, "KOKKOS_LAMBDA lacks a body")
        depth = 1
        end = brace + 1
        while end < len(source) and depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        require(depth == 0, "unterminated KOKKOS_LAMBDA body")
        bodies.append(source[brace:end])
        cursor = end


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_dir.resolve()
    provider = root / "src/z4c/cartoon_derivatives.hpp"
    finite_diff = root / "src/utils/finite_diff.hpp"
    legacy_provider = subprocess.check_output(
        ["git", "show", f"{BASE}:src/z4c/cartoon_derivatives.hpp"], cwd=root)
    require(digest(legacy_provider) == LEGACY_PROVIDER_SHA,
            "frozen signed-plane provider reference changed")
    provider_text = provider.read_text()
    for forbidden in ("FitRadialSamples", "RadialFit", "OddCoefficientFit",
                      "QuadraticCoefficientFit", "QuadraticDifferenceFit",
                      "NearAxisCell", "TargetLayer", "RegularCoefficientDerivative",
                      "EvenCoefficientDerivative", "OddCoefficientDerivative",
                      "QuadraticCoefficientDerivative",
                      "QuadraticDifferenceCoefficientDerivative"):
        require(forbidden not in provider_text,
                f"production provider retains layer-dependent closure helper {forbidden}")
    require("return ActiveFirst(RhoDirection(), field) / rho_;" in provider_text and
            "const Real radial_derivative = ActiveFirst(RhoDirection(), component, field);"
            in provider_text and
            "const Real radial_derivative = ActiveFirst(RhoDirection(), a, b, field);"
            in provider_text,
            "half-plane provider lacks the single all-bulk SO(2) path")
    require(digest(finite_diff.read_bytes()) == FINITE_DIFF_SHA,
            "generated finite_diff.hpp hash changed")
    base_oracle = subprocess.check_output(
        ["git", "show", f"{BASE}:tst/unit/z4c/cartoon_derivatives_test.cpp"], cwd=root)
    require(digest(base_oracle) == BASE_ORACLE_SHA, "frozen standalone source changed")
    generated = subprocess.run(
        [sys.executable,
         str(root / "tst/unit/z4c/generate_z4c_cartoon_derivative_oracle.py"),
         "--root", str(root), "--check"], check=False)
    require(generated.returncode == 0,
            "pinned SymPy generator --check failed or its dependency is unavailable")

    pgen = (root / "src/pgen/unit_tests/z4c_cartoon_derivatives.cpp").read_text()
    kernels = (root / "src/pgen/unit_tests/z4c_cartoon_derivatives_kernels.cpp").read_text()
    require("template <" not in pgen and "if constexpr" not in pgen and
            "KOKKOS_LAMBDA" not in pgen, "pgen entry is not non-templated host-only")
    require(kernels.count("RunMmsOrder<2>") == 1 and
            kernels.count("RunMmsOrder<3>") == 1 and
            kernels.count("RunMmsOrder<4>") == 1,
            "host dispatcher does not own exactly three order calls")
    require("DispatchCartoonZ4cKernel" in kernels and
            "switch (config.stencil_width)" not in kernels and
            "switch (fd_stencil)" not in kernels,
            "MMS bypasses the compiled production Cartoon host dispatcher")
    device_kernels = kernels[kernels.index("template <z4c::TensorVariance"):
                             kernels.index("std::vector<std::string> ResultNames()")]
    require("KOKKOS_INLINE_FUNCTION\nint TensorFirstIndex" in kernels and
            "KOKKOS_INLINE_FUNCTION\nint TensorSecondIndex" in kernels and
            "return component < 3 ? 0 : (component < 5 ? 1 : 2);" in kernels and
            "return component == 0 ? 0 : ((component == 1 || component == 3) ? 1 : 2);"
            in kernels and
            "kTensorFirst[" not in device_kernels and
            "kTensorSecond[" not in device_kernels,
            "symmetric tensor mapping is not scalar and device-callable")
    for body in lambda_bodies(kernels):
        require("Z4cSymmetryMode" not in body and "spatial_order" not in body and
                "z4c_symmetry" not in body,
                "device lambda captures or branches on runtime symmetry/order")
    require("constexpr int kResults = 171;" in kernels and
            "constexpr int kVariables = 10;" in kernels,
            "runtime operator inventory is not exactly 171")
    require("std::array<double, kResults - kVariables> errors" in kernels and
            "diagnostic_axis_operator_count" in kernels and
            "diagnostic_axis_nonfinite" in kernels and
            ",diagnostic_axis,axis,0,diagnostic_axis," in kernels,
            "true-axis probe is not bound to the explicit per-operator inventory")
    require("TensorVariance::all_lower" in kernels and
            "TensorVariance::all_upper" in kernels, "both tensor variances are required")
    require("Real observed[kResults]" not in kernels and
            "Real expected[kResults]" not in kernels and
            "AnalyticOracle oracle" not in kernels,
            "production MMS kernel has an O(171) thread-local oracle/result array")
    series = json.loads((root / "tst/unit/z4c/z4c_cartoon_derivatives_series.json").read_text())
    require(series.get("count") == 171 and len(series.get("series", [])) == 171 and
            [item["index"] for item in series["series"]] == list(range(171)) and
            len({item["name"] for item in series["series"]}) == 171,
            "generated runtime series inventory is not exact and ordered")
    require(series.get("contract_errata") == {
                "scalar.second.0.0":
                "active radial Dxx; scalar.second.2.2 owns 2 EvenP"},
            "scalar Dxx contract erratum is absent or changed")
    classifications = [item["classification"] for item in series["series"]]
    require(classifications.count("truncating") == 137 and
            classifications.count("exact_identity") == 12 and
            classifications.count("exact_plane_algebraic") == 12 and
            classifications.count("exact_discrete") == 10,
            "exact/truncating series classes differ from the frozen inventory")
    policy = series.get("roundoff_policy", {})
    coefficient_rationals = {
        order: {name: value["rational"] for name, value in row.items()}
        for order, row in policy.get("coefficients", {}).items()}
    field_rationals = {name: value["rational"]
                       for name, value in policy.get("field_maxima", {}).items()}
    require(policy.get("binary64_epsilon_hex") == float.fromhex("0x1p-52").hex() and
            policy.get("fit_fixture_count") == 108 and
            policy.get("fit_construction_condition_checked") is True and
            coefficient_rationals == {
                "2": {"C1": "1", "C2": "4", "CM": "1", "CU": "4",
                      "CKO": "16", "CE": "1", "CO": "4/3", "CQ": "20/9"},
                "3": {"C1": "3/2", "C2": "16/3", "CM": "9/4", "CU": "19/6",
                      "CKO": "64", "CE": "3/2", "CO": "28/15", "CQ": "226/75"},
                "4": {"C1": "11/6", "C2": "272/45", "CM": "121/36",
                      "CU": "3", "CKO": "256", "CE": "5/2", "CO": "76/35",
                      "CQ": "12598/3675"}} and
            field_rationals == {
                "scalar": "26/5", "vector.0": "339/50", "vector.1": "133/50",
                "vector.2": "81/20", "tensor.0.0": "284/25",
                "tensor.0.1": "93/50", "tensor.0.2": "567/100",
                "tensor.1.1": "249/100", "tensor.1.2": "159/100",
                "tensor.2.2": "191/100"},
            "coefficient-aware floor tables differ from the frozen finite contract")
    search = json.loads(
        (root / "tst/inputs/z4c_cartoon_mms_search_manifest.json").read_text())
    require(search.get("schema") == "athenak_z4c_cartoon_mms_search_v2" and
            search.get("state") == "checked_in_template" and
            search.get("policy", {}).get("resolution_pools") == {
                "2": [32, 64, 128, 256, 512, 1024, 2048, 4096],
                "4": [32, 64, 128, 256],
                "6": [32, 48, 64, 80, 96, 112, 128, 160, 192, 256]} and
            search.get("qualification_window") is None and
            search.get("qualification_domain") == [-2.0, 2.0, -2.0, 2.0] and
            search.get("policy", {}).get("class_counts") == {
                "truncating": 137, "exact_identity": 12,
                "exact_plane_algebraic": 12, "exact_discrete": 10} and
            search.get("stages", {}).get("o6_phase0_stage1", {}).get("tuples") ==
            [[6, 48, 0], [6, 80, 0], [6, 96, 0]] and
            search.get("immutable_roots", {}).get("job56586376", {}).get(
                "case_manifest_set_sha256") ==
            "d2aa1ff2ff9b68302170e0271d3f6fca150d86acb183f9d89af62965427d2aa3",
            "prospective search pools/lifecycle differ from the frozen unselected plan")
    split_counts = policy.get("split_function_operation_counts", {})
    require(len(split_counts) == 24 and
            all(count <= (128 if name.startswith("field.") else 256)
                for name, count in split_counts.items()) and
            all(set(item.get("roundoff_family", {})) ==
                {"source_branch", "fitted_t", "raw_class", "active", "fitted", "raw"}
                for item in series["series"]),
            "split-operation caps or 171-row roundoff-family mapping are incomplete")
    by_name = {item["name"]: item["roundoff_family"] for item in series["series"]}
    require([block["kind"] for block in
             by_name["tensor.lower.0.2.second.0.2"]["fitted"]] ==
            ["quad_value", "rho2_quad_derivative"] and
            [block["kind"] for block in
             by_name["tensor.lower.0.1.advective"]["active"]] == ["up", "up"] and
            by_name["tensor.lower.0.1.advective"]["raw"][0]["components"] ==
            ["vector.2", "tensor.1.2"],
            "quadratic/advection family, rho powers, or component permutation changed")
    independent = [item for item in series["series"]
                   if "independent" in item["noise_lanes"]]
    require(len(independent) == 60 and all(item["name"].startswith((
                "tensor.lower.0.0.", "tensor.lower.0.2.", "tensor.lower.2.2.",
                "tensor.upper.0.0.", "tensor.upper.0.2.",
                "tensor.upper.2.2.")) for item in independent),
            "independent tensor-noise affected-series inventory changed")
    require("(x,z,Y)" in (root / "tst/unit/z4c/generate_z4c_cartoon_derivative_oracle.py").read_text()
            or "(X,Z,Y)" in kernels, "coordinate permutation is not documented")

    cmake = (root / "CMakeLists.txt").read_text()
    require(re.search(r"athena_cartoon_derivatives_unit_test\s+EXCLUDE_FROM_ALL", cmake),
            "heavy standalone is not EXCLUDE_FROM_ALL")
    require("NAME athena.z4c_cartoon_mms_structure" in cmake,
            "ordinary lightweight structural CTest is missing")
    require("NAME athena_cartoon_derivatives_unit_test" not in cmake,
            "heavy standalone was registered as routine CTest")
    driver = (root / "tst/test_suite/unit_tests/test_z4c_cartoon_derivatives.py").read_text()
    wrapper = (root / "tst/test_suite/unit_tests/cartoon_mms_rank_wrapper.py").read_text()
    require("local_rank % len(records)" not in wrapper and
            "exactly one visible device per rank" in wrapper and
            '"binding_verified"' in wrapper,
            "CUDA rank wrapper does not fail closed on concrete one-device binding")
    require("execution_environment_sha256" in driver and
            "actual != expected |" in driver and
            "axis_names = operator_names[:161]" in driver and
            "len(axis_rows) != 161" in driver,
            "case resume does not bind environment and exact file inventory")
    require("def verify_complete_case_evidence" in driver and
            driver.count("verify_complete_case_evidence(") >= 6 and
            "def verify_rank_reference_root" in driver and
            "def stage_binding_record" in driver and
            "def validate_stage_lineage" in driver and
            "def validate_case_launch_provenance" in driver and
            "def validate_final_reference_aggregates" in driver and
            "expected_record_keys" in driver and
            "expected_exact_keys" in driver and
            "require_recomputed_reference_products(reference_root, recomputed)" in
            driver and
            "verify_preserved_job56586376_failed_convergence" in driver and
            "normalize_preserved_job56586376_result" in driver and
            "PRESERVED_JOB_56586376_NORMALIZATION_COUNTS" in driver and
            '"legacy_csv_normalization_counts"' in driver and
            "load_json_strict(preserved_convergence)" not in driver and
            'preflight.get("search_manifest_sha256")' in driver and
            '"fresh_single_source_final_qualification"' in driver and
            "require_exact_regular_files(authorization" in driver,
            "shared case verifier, partial lineage, or immutable rank/auth guard is missing")
    require("O(171*nx1)" in driver and "clean_floor" in driver and
            "saturated_at_resolution" in driver and
            'sample["error"] < 0.0' in driver and
            "def evaluate_legacy_rate_samples" in driver and
            'usable.append(float("-inf"))' in driver and
            '"saturation_absorbing": False' in driver and
            "validate_finite_text_product(csv_path)" in driver and
            "validate_finite_text_product(data_path)" in driver and
            "historical_interval_local_pre_coefficient_floor_partition" in driver and
            "prospective_absorbing_pre_coefficient_floor_partition" in driver and
            "evaluate_prospective_absorbing_legacy_rate_samples" in driver and
            "coefficient_floor_partition" in driver,
            "driver does not use the finite O(nx1) absorbing coefficient floor")
    for self_test in ("--self-test-no-evolution-parser",
                      "--self-test-cpu-audit-policy"):
        completed = subprocess.run(
            [sys.executable, "-B",
             str(root / "tst/test_suite/unit_tests/test_z4c_cartoon_derivatives.py"),
             self_test], check=False)
        require(completed.returncode == 0,
                f"campaign driver self-test failed: {self_test}")

    coverage_path = root / "tst/unit/z4c/z4c_cartoon_mms_coverage.json"
    coverage = json.loads(coverage_path.read_text())
    retained_false = {"LEG-FIELD-JET-ORACLE", "LEG-DISS-ORACLE",
                      "LEG-ORDER-AGGREGATION-NG2/3/4"}
    required_coverage = {identifier: (*properties, identifier not in retained_false)
                         for identifier, properties in REQUIRED_COVERAGE.items()}
    actual = {entry["id"]: (entry["category"], entry["owner"],
                             entry["source_range"], entry["retained_duplicate"])
              for entry in coverage["checks"]}
    require(len(actual) == len(coverage["checks"]), "duplicate primary legacy ID")
    require(actual == required_coverage,
            "legacy coverage IDs/categories/owners/ranges/retention differ from frozen inventory")
    require(len(actual) == 19, "legacy coverage must contain exactly 19 primary IDs")
    require(all(category in CATEGORIES for category, _, _, _ in actual.values()),
            "unknown primary coverage category")
    registered = {
        "athena.z4c_cartoon_mms_structure":
            "NAME athena.z4c_cartoon_mms_structure" in cmake,
        "athena.z4c_cartoon_mms_generated_reference":
            "NAME athena.z4c_cartoon_mms_generated_reference" in cmake,
        "athena.z4c_cartoon_mms_static":
            "NAME athena.z4c_cartoon_mms_static" in cmake,
        "athena_cartoon_derivatives_unit_test": bool(re.search(
            r"athena_cartoon_derivatives_unit_test\s+EXCLUDE_FROM_ALL", cmake)),
        "test_z4c_cartoon_derivatives.py":
            (root / "tst/test_suite/unit_tests/test_z4c_cartoon_derivatives.py").is_file()
            and "if __name__ == \"__main__\":" in driver,
    }
    require(set(coverage["owners"]) == set(registered),
            "coverage owner inventory differs from frozen targets")
    require(all(registered.values()),
            "coverage owner is not registered by CMake or the campaign driver")
    require(coverage["frozen_source_sha256"] == BASE_ORACLE_SHA,
            "coverage source hash is not frozen")
    require(coverage.get("schema") == "athenak_z4c_cartoon_mms_legacy_coverage_v1" and
            coverage.get("frozen_base") == BASE and
            coverage.get("frozen_source") ==
            "tst/unit/z4c/cartoon_derivatives_test.cpp",
            "coverage schema/base/source identity differs from frozen inventory")

    checksum_manifest = json.loads(
        (root / "tst/unit/z4c/z4c_cartoon_mms_checksums.json").read_text())
    files = checksum_manifest.get("files", {})
    require(checksum_manifest.get("schema") ==
            "athenak_z4c_cartoon_mms_checksums_v1" and
            set(files) == CHECKSUM_PATHS,
            "MMS checksum manifest does not bind the exact frozen source set")
    require(all(digest((root / relative).read_bytes()) == expected
                for relative, expected in files.items()),
            "MMS checksum-bound source artifact changed")

    require(not re.search(r"\bcmake\b", driver, re.IGNORECASE),
            "campaign driver contains a build-system invocation")
    require(not re.search(r"openmp", driver, re.IGNORECASE),
            "campaign driver contains forbidden OpenMP lane")
    print("Cartoon MMS static and coverage gates passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
