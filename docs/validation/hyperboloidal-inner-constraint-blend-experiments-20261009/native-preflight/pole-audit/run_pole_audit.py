"""Actual snapshot geometry/pole audit; no native evolution or changed helpers."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess

import numpy as np


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
READER = ROOT / "build-layer-research/time-projection-controls/rst-reader-gate/restart_reader.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(READER) == "74c62c97afd113819438d8be42be84cc605902983bd974022ebb7e0747f3cb13"
    build_path = BASE / "native-build/build-receipt.json"
    assert sha(build_path) == "855afa6f0b950254dc59ca19ff11c984502257a7b1b2dab9917dadcc45c3fa69"
    build = json.loads(build_path.read_text())
    command = build["compile_results"][0]["command"].copy()
    command = command[:command.index("-o")] + ["-o", str(HERE / "check_snapshot"),
                                             str(HERE / "check_snapshot.cpp")]
    command += [str(ROOT / name) for name in build["reused_base_link_inputs_sha256"]
                if name.endswith(".a")]
    (HERE / "compile-command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (HERE / "compile.log").open("w") as log:
        compile_result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    compile_result.check_returncode()
    spec = importlib.util.spec_from_file_location("independent_pole_restart", READER)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    rows = []
    for run in ("reference", "short"):
        result = json.loads((BASE / run / "results.json").read_text())
        case, = result["cases"]
        casepath = BASE / run / case["name"]
        history = np.atleast_2d(np.loadtxt(casepath / "hyp.z4c.user.hst"))
        for rstpath in sorted((casepath / "rst").glob("*.rst")):
            checkpoint = reader.read_rst(rstpath)
            raw = HERE / (run + "-" + rstpath.stem + ".raw")
            ng = checkpoint["mb_indcs"]["ng"]
            steps = [checkpoint["mesh_size"][f"dx{a}"] for a in (1, 2, 3)]
            first = [checkpoint["mesh_size"][f"x{a}min"] + (.5-ng)*steps[a-1]
                     for a in (1, 2, 3)]
            with raw.open("wb") as stream:
                stream.write(struct.pack("<3i", *checkpoint["u"].shape[1:][::-1]))
                stream.write(struct.pack("<3d", *first))
                stream.write(struct.pack("<3d", *steps))
                stream.write(checkpoint["u"].astype("<f8").tobytes())
            args = [str(HERE / "check_snapshot"), str(raw)]
            process = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
            process.check_returncode()
            value = json.loads(process.stdout)
            nearest = int(np.argmin(abs(history[:, 0] - checkpoint["time"])))
            assert abs(history[nearest, 0] - checkpoint["time"]) < 1e-14
            comparison = {}
            for name, column in (("H", 2), ("M", 3), ("Z", 4), ("Theta", 5),
                                 ("actual_pole_deviation", 12)):
                expected = float(history[nearest, column])
                error = abs(value[name] - expected)
                assert error < 1e-11 * (1 + abs(expected))
                comparison[name] = {"recomputed": value[name], "native_HST": expected,
                                    "absolute_difference": error}
            assert value["synthetic_missing_double_pole_control"] > .01
            rows.append({"run": run, "time": checkpoint["time"],
                         "restart": str(rstpath.relative_to(ROOT)), "restart_sha256": sha(rstpath),
                         "raw_file": str(raw.relative_to(ROOT)), "raw_sha256": sha(raw),
                         "command": args, "computed": value,
                         "native_HST_comparisons": comparison,
                         "returncode": process.returncode, "stderr": process.stderr})
    assert len(rows) == 6
    receipt = {"status": "PASS", "scope": "Actual private native shell geometric pole numerator simple+double/Omega; not gauge poles or scri limit",
               "rows": rows, "compile_command": command, "compile_returncode": compile_result.returncode,
               "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in
                                 (Path(__file__), HERE / "check_snapshot.cpp", READER)},
               "private_build_receipt_sha256": sha(build_path),
               "utility_executable_sha256": sha(HERE / "check_snapshot"),
               "compile_log_sha256": sha(HERE / "compile.log"),
               "actual_inputs_unchanged": all(sha(ROOT / name) == digest for name, digest in
                                               build["all_compiled_repository_dependencies_sha256"].items())}
    assert receipt["actual_inputs_unchanged"]
    (HERE / "snapshot-results.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print("PASS six binary64 snapshots; exact H/M/Z/Theta/geometric-pole HST reproduction")
    print("manual/helper max", max(row["computed"]["synthetic_manual_vs_helper_max"] for row in rows),
          "missing-double control", rows[0]["computed"]["synthetic_missing_double_pole_control"])


if __name__ == "__main__":
    main()
