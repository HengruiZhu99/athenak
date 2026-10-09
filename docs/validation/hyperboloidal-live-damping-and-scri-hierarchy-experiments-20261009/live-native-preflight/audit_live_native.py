"""Independent binary64 audit of private live C0 damping-profile native preflights.

The existing restart/snapshot helpers are pinned byte-for-byte. New build,
source, link, initial-state and endpoint checks are specific to this live C0 damping-profile build.
No accepted helper, frozen gate or production source is modified.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FAMILY = ROOT / "build-layer-research/continuum/preferred/native-overlay/spatial-norm-family"
BUILD = HERE / "native-build"
HELPER = ROOT / "build-layer-research/time-projection-controls/audit_composed_native.py"
BASE_EXE = "dd1d189210abd4e094da339dd73e3014357924b343c9f08f658eaf8cd4ae172d"
PRIVATE_EXE = "580b6043f906ea37ae4870db2cd63d68f344c634c28319d88dafe53308d56005"
BUILD_RECEIPT = "5ade5d3c80b68913301957a7093929a6793af5da8a9fc19c4215f140d7e49a72"
LIVE_INDEX = "fcfd4a740fc25e999d0417598015608c009bf61de032aff5266f1f843e2d6b59"
PINNED = {
    HELPER: "088b456a99a9ff66ad905e7cc12c38da922656393e47c2835996ce7d8ddf6e69",
    HELPER.parent / "rst-reader-gate/restart_reader.py":
        "74c62c97afd113819438d8be42be84cc605902983bd974022ebb7e0747f3cb13",
    HELPER.parent / "rst-reader-gate/abi.json":
        "4e7599223e62ea4aa020b0efc691168a7ec2fcd6d83c38f6f45c4c6945d04cec",
    ROOT / "vis/python/bin_convert.py":
        "35e3ec2ebbe2d5a24af9bf1e4796cf1f20def58ce1d76c50f0166288b6d8f37c",
    ROOT / "tst/hyperboloidal/run_layer_validation.py":
        "ca81d60cc9a00b83f9325b79abce9fa3eaaade293e2c5f73c3a3d4bdbfac5d96",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_helper():
    for path, digest in PINNED.items():
        assert sha(path) == digest, path
    spec = importlib.util.spec_from_file_location("C1_pinned_snapshot_helper", HELPER)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def verify_index(path, digest):
    assert sha(path) == digest
    index = json.loads(path.read_text())
    for name, value in index["files"].items():
        assert sha(path.parent / name) == value
    for name, value in index["large_outputs_outside_snapshot"].items():
        source = path.parent.parent / name
        assert sha(source) == value["sha256"] and source.stat().st_size == value["bytes"]
    return index


def verify_build():
    base_path = FAMILY / "native-build-receipt.json"
    base = json.loads(base_path.read_text())
    assert len(base["source_sha256"]) == 369 and len(base["overlay_sha256"]) == 4
    assert all(sha(ROOT / key) == value for key, value in base["source_sha256"].items())
    assert all(sha(FAMILY / key) == value for key, value in base["overlay_sha256"].items())
    assert (HERE / "native.athinput").read_bytes() == (FAMILY / "native.athinput").read_bytes()
    private_path = BUILD / "build-receipt.json"
    assert sha(private_path) == BUILD_RECEIPT
    private = json.loads(private_path.read_text())
    assert private["base_build_receipt_sha256"] == sha(base_path)
    assert private["base_executable_sha256"] == BASE_EXE
    assert private["executable_sha256"] == PRIVATE_EXE
    assert private["link_exit_status"] == 0
    assert sha(ROOT / private["executable"]) == PRIVATE_EXE
    assert sha(BUILD / "build-source.py") == private["script_sha256"]
    assert sha(BUILD / "build.log") == private["build_log_sha256"]
    assert private["compiled_implementation"] == "27c19d20696ea6dd4704032c51dfd026218f64f2"
    assert len(private["private_source_sha256"]) == 2
    assert len(private["all_compiled_repository_dependencies_sha256"]) == 268
    for key in ("private_source_sha256", "all_compiled_repository_dependencies_sha256"):
        assert all(sha(ROOT / name) == digest for name, digest in private[key].items())
    gate_path = ROOT / private["live_gate_index"]
    assert private["live_gate_index_sha256"] == LIVE_INDEX
    gate = verify_index(gate_path, LIVE_INDEX)
    gate_receipt = json.loads((gate_path.parent / "receipt.json").read_text())
    assert len(gate_receipt["source_after"]) == 384
    assert gate_receipt["source_after"] == gate_receipt["source_before"]
    assert all(sha(ROOT / name) == digest for name, digest in gate_receipt["source_after"].items())
    assert len(gate_receipt["commands"]) == 17
    assert all(row["returncode"] == 0 for row in gate_receipt["commands"])
    assert gate_receipt["passed_finite_Omega_local_numerical_gate"] and not gate_receipt["global_native_or_scri_stability_accepted"]
    old_inputs = private["reused_base_link_inputs_sha256"]
    assert len(old_inputs) == 186
    assert sum(name.endswith(".o") for name in old_inputs) == 182
    assert sum(name.endswith(".a") for name in old_inputs) == 4
    assert all(sha(ROOT / name) == record["sha256"] and
               (ROOT / name).stat().st_size == record["bytes"]
               for name, record in old_inputs.items())
    base_build = ROOT / "build-layer-spatial-norm-native"
    compile_commands = json.loads((base_build / "compile_commands.json").read_text())
    base_commands = {Path(row["file"]): row for row in compile_commands}
    base_link = shlex.split((base_build / "src/CMakeFiles/athena.dir/link.txt").read_text())
    expected_link = base_link.copy()
    expected_link[expected_link.index("-o") + 1] = str(ROOT / private["executable"])
    assert len(private["compile_results"]) == 6
    compiled = []
    for row in private["compile_results"]:
        assert row["exit_status"] == 0
        command = row["command"]
        output, dependency = Path(command[command.index("-o") + 1]), Path(command[command.index("-MF") + 1])
        assert sha(output) == row["private_object_sha256"]
        assert sha(dependency) == row["dependency_file_sha256"]
        original = base_commands[Path(row["base_source"])]
        original_command = shlex.split(original["command"])
        original_object = (Path(original["directory"]) /
                           original_command[original_command.index("-o") + 1]).resolve()
        assert sha(original_object) == row["base_object_sha256"]
        expected = original_command.copy()
        expected[expected.index("-o") + 1] = str(output)
        expected[1:1] = ["-I" + str(BUILD / "include")]
        expected.extend(["-MD", "-MF", str(dependency)])
        assert command == expected
        matches = [i for i, token in enumerate(expected_link)
                   if token.endswith(".o") and
                   (Path(private["link_cwd"]) / token).resolve() == original_object]
        assert len(matches) == 1
        expected_link[matches[0]] = str(output)
        compiled.append({"source": row["base_source"], "object": sha(output),
                         "dependency": sha(dependency)})
    assert private["link_command"] == expected_link
    cart = BUILD / "include/z4c/hyperboloidal/cartesian_patch.hpp"
    restored=cart.read_text().replace('#include "live_damping_profile.hpp"\n','')
    pairs={
     'ConformalRHS(u,omega,damping/u.alpha.value,Real(0))':'ConformalRHS(u,omega,damping/u.alpha.value,ResearchLiveKappa2Profile(u,omega,p.radius,damping))',
     'ConformalRHS(background,omega0,damping,Real(0))':'ConformalRHS(background,omega0,damping,ResearchLiveKappa2Profile(background,omega0,p.radius,damping))',
     'ConformalRHS(u,CartesianOmega(u,p),\n                                        damping/u.alpha.value,Real(0))':'ConformalRHS(u,CartesianOmega(u,p),\n                                        damping/u.alpha.value,ResearchLiveKappa2Profile(u,CartesianOmega(u,p),p.radius,damping))',
     'ConformalRHS(background,CartesianOmega(background,p),\n                                        damping,Real(0))':'ConformalRHS(background,CartesianOmega(background,p),\n                                        damping,ResearchLiveKappa2Profile(background,CartesianOmega(background,p),p.radius,damping))'}
    for old,new in pairs.items():
        assert restored.count(new)==1
        restored=restored.replace(new,old)
    assert restored==(ROOT/"src/z4c/hyperboloidal/cartesian_patch.hpp").read_text()
    assert sha(BUILD/"include/live_damping_profile.hpp")=="69bbbc137486eb3398f94ed50a8b04583c375da049372b82b19330d8432fc153"
    assert sha(cart)=="a3acc7dce1c4fa98d2c7766801fb35ecb12088d23475e8f36a905ea7add4b395"
    return {"status": "PASS", "private_build_receipt_sha256": BUILD_RECEIPT,
            "baseline_sources": 369, "baseline_overlays": 4,
            "private_sources": 2, "private_objects": 6, "repository_dependencies": 268,
            "all182_original_objects_and4_libraries_unchanged": True,
            "actual_compile_and_link_replacements_verified": True,
            "compiled_objects": compiled, "native_override_byteidentical": True,
            "frozen_live_gate_index": LIVE_INDEX,
            "same_live_helper_in_evolution_reference_and_geometric_pole_diagnostic": True,
            "no_C1_or_covector_repair_and_no_new_double_pole": True,
            "baseline_executable_sha256": BASE_EXE, "private_executable_sha256": PRIVATE_EXE}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--baseline-case")
    parser.add_argument("--zero-reference", type=Path, default=FAMILY / "native-reference")
    parser.add_argument("--launch", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    helper = load_helper()
    build = verify_build()
    run, baseline = args.run.resolve(), args.baseline.resolve()
    private_result = json.loads((run / "results.json").read_text())
    base_result = json.loads((baseline / "results.json").read_text())
    assert private_result["sha256"] == PRIVATE_EXE and base_result["sha256"] == BASE_EXE
    pcase = helper.choose_case(private_result, args.case)
    bcase = helper.choose_case(base_result, args.baseline_case)
    assert pcase["requested_time"] == bcase["requested_time"]
    private = helper.run_data(run, pcase, PRIVATE_EXE)
    base = helper.run_data(baseline, bcase, BASE_EXE)
    assert len(private["snapshots"]) == 3
    pset, bset = private["settings"], base["settings"]
    differences = {key: [bset.get(key), pset.get(key)] for key in sorted(set(pset) | set(bset))
                   if bset.get(key) != pset.get(key)}
    allowed = {f"output{i}/dt" for i in range(1, 6)}
    assert set(differences) <= allowed
    assert pset["mesh/nghost"] == bset["mesh/nghost"] == "3"
    assert pset["z4c/hyperboloidal_kappa1"] == bset["z4c/hyperboloidal_kappa1"] == "10"
    assert private["checkpoint"]["mesh_size"] == base["checkpoint"]["mesh_size"]
    assert private["checkpoint"]["mb_indcs"] == base["checkpoint"]["mb_indcs"]
    assert np.array_equal(private["mask"], base["mask"])
    assert all(np.array_equal(x, y) for x, y in zip(private["coordinates"], base["coordinates"]))
    mask = private["mask"]
    assert np.array_equal(private["initial"][:, mask], base["initial"][:, mask])
    zero_path = args.zero_reference.resolve()
    zero_result = json.loads((zero_path / "results.json").read_text())
    zero_case = helper.choose_case(zero_result, None)
    zero = helper.run_data(zero_path, zero_case, BASE_EXE)
    geometry_indices = [i for i in range(25) if i not in (18, 19, 20, 21)]
    assert np.array_equal(mask, zero["mask"])
    assert np.array_equal(private["initial"][geometry_indices][:, mask],
                          zero["initial"][geometry_indices][:, mask])
    fields = {}
    for index, name in enumerate(helper.rst.VARIABLES):
        delta = private["final"][index][mask] - base["final"][index][mask]
        fields[name] = {"initial_max_difference": 0.,
                        "final_max_difference": float(abs(delta).max()),
                        "final_rms_difference": float(np.sqrt(np.mean(delta * delta)))}
    comparisons = {}
    for name, index in (("H", 2), ("M", 3), ("Z", 4), ("Theta", 5)):
        p, b = float(private["history"][-1, index]), float(base["history"][-1, index])
        comparisons[name] = {"private": p, "baseline": b, "ratio": p / b if b else None}
    launch_files = []
    if args.launch:
        launch = json.loads(args.launch.read_text())
        assert LIVE_INDEX in args.launch.read_text(), "Launch must explicitly pin the live gate"
        launch_files.append(args.launch.resolve())
    all_files = [p for directory in (run, baseline, zero_path)
                 for p in directory.rglob("*") if p.is_file()]
    out = {
        "status": "PASS", "scope": "Fixed finite-Omega live C0 damping-profile native preflight; no stability, scri closure or BH acceptance",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "seconds": time.monotonic() - started, "build_verification": build,
        "private_case": pcase["name"], "baseline_case": bcase["name"],
        "input_parameter_differences": differences,
        "initial_active25_binary64_bitwise_equal": True,
        "initial_geometry_bitwise_equal_zero_reference": True,
        "coordinates_masks_and_grid_bitwise_equal": True,
        "exact_final_comparison_time": pcase["requested_time"],
        "original_HST_diagnostics": comparisons,
        "final_equal_time_binary64_fields": fields,
        "private_snapshots": private["snapshots"], "baseline_snapshots": base["snapshots"],
        "zero_reference_snapshots": zero["snapshots"],
        "all_run_files": helper.records(all_files), "launch_files": helper.records(launch_files),
        "audit_inputs": {str(p.relative_to(ROOT)): digest for p, digest in PINNED.items()},
        "auditor_sha256": sha(Path(__file__)),
        "precision": "State is binary64 RST; all25 active BIN fields exactly match binary32 cast. Historical physical_metric_eigen keys mean Penrose gtilde/chi values; positivity is equivalent to physical SPD for Omega>0. Headerdt is at write time.",
    }
    assert all(sha(path) == digest for path, digest in PINNED.items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print("PASS", pcase["name"], "three private full-precision snapshots; initial bitwise equality")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
