"""Read-only attribution of composed native states to two constraint functionals."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import time

import numpy as np


HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent
ROOT = CONTROL.parents[1]
BUILD = CONTROL / "composed-derivative-build"
READER = CONTROL / "rst-reader-gate/restart_reader.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def metadata(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
            "bytes": path.stat().st_size}


def main():
    started = time.monotonic()
    assert sha(READER) == "74c62c97afd113819438d8be42be84cc605902983bd974022ebb7e0747f3cb13"
    receipt_path = BUILD / "build-receipt.json"
    assert sha(receipt_path) == "f2c884662b04b68b4efef0cb1dedb2e11b26f242683427642d233afbf433c886"
    build = json.loads(receipt_path.read_text())
    for group in ("private_source_sha256", "all_compiled_repository_dependencies_sha256"):
        for name, digest in build[group].items():
            assert sha(ROOT / name) == digest, name
    for name, entry in build["reused_base_link_inputs_sha256"].items():
        assert sha(ROOT / name) == entry["sha256"], name
    assert sha(Path(build["executable"])) == build["executable_sha256"]
    original_header = BUILD / "include/z4c/hyperboloidal/cartesian_patch.hpp"
    overlay_header = HERE / "composed-diagnose-include/z4c/hyperboloidal/cartesian_patch.hpp"
    overlay_header.parent.mkdir(parents=True, exist_ok=True)
    text = original_header.read_text()
    target = "      auto u = LoadMeshJet<3>(q,idx,0,k,j,i);"
    assert text.count(target) == 1
    overlay_header.write_text(text.replace(target, target.replace("LoadMeshJet", "LoadComposedMeshJet")))
    compiles = []
    for mode in ("original", "composed"):
        command = build["compile_results"][0]["command"].copy()
        command = command[:command.index("-o")]
        if mode == "composed":
            command.insert(1, "-I" + str(HERE / "composed-diagnose-include"))
        command += ['-DDIAGNOSTIC_LOADER="' + mode + '"', "-o",
                    str(HERE / ("check_" + mode)), str(HERE / "check_snapshot.cpp")]
        command += [str(ROOT / name) for name in build["reused_base_link_inputs_sha256"]
                    if name.endswith(".a")]
        write(HERE / (mode + "-compile-command.json"), command)
        before = time.monotonic()
        with (HERE / (mode + "-compile.log")).open("w") as stream:
            process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        compiles.append({"mode": mode, "command": command, "returncode": process.returncode,
                         "seconds": time.monotonic() - before})
        process.check_returncode()
    spec = importlib.util.spec_from_file_location("read_only_composed_rst", READER)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    selections = [("N24-t0", "composed-N24-t2", 0.),
                  ("N24-t.02", "composed-short", .02),
                  ("N24-before-t.2", "composed-N24-t2", .175),
                  ("N24-near-t.2", "composed-N24-t2", .2),
                  ("N24-t2", "composed-N24-t2", 2.),
                  ("N36-t0", "composed-N36-t0.2", 0.),
                  ("N36-t.2", "composed-N36-t0.2", .2),
                  ("reference-t0", "composed-reference", 0.),
                  ("reference-t.05", "composed-reference", .05)]
    rows = []
    for label, run, requested in selections:
        result_path = CONTROL / run / "results.json"
        result = json.loads(result_path.read_text())
        case, = result["cases"]
        assert result["sha256"] == build["executable_sha256"] and case["exit_status"] == 0
        casepath = CONTROL / run / case["name"]
        paths = sorted((casepath / "rst").glob("*.rst"))
        candidates = [(path, reader.read_rst(path)) for path in paths]
        rstpath, checkpoint = min(candidates, key=lambda pair: abs(pair[1]["time"] - requested))
        assert checkpoint["mb_indcs"]["ng"] == 4
        assert checkpoint["u"].shape[0] == 25
        n = checkpoint["mb_indcs"]["nx1"]
        assert checkpoint["u"].shape == (25, n+8, n+8, n+8)
        assert n in (24, 36) and np.isfinite(checkpoint["u"]).all()
        for key, expected in {"problem/mass": "0", "z4c/hyperboloidal_curvature_radius": ".5",
                              "z4c/hyperboloidal_layer_r0": ".05",
                              "z4c/hyperboloidal_layer_r1": ".95",
                              "z4c/hyperboloidal_kappa1": "10"}.items():
            assert checkpoint["parameters"][key] == expected
        raw = HERE / (label + ".raw")
        steps = [checkpoint["mesh_size"][f"dx{a}"] for a in (1, 2, 3)]
        first = [checkpoint["mesh_size"][f"x{a}min"] + (.5-4)*steps[a-1] for a in (1, 2, 3)]
        with raw.open("wb") as stream:
            stream.write(struct.pack("<3i", n+8, n+8, n+8))
            stream.write(struct.pack("<3d", *first))
            stream.write(struct.pack("<3d", *steps))
            stream.write(checkpoint["u"].astype("<f8").tobytes())
        values, commands = {}, []
        for mode in ("original", "composed"):
            command = [str(HERE / ("check_" + mode)), str(raw), str(HERE / (label+"-"+mode+".cells"))]
            before = time.monotonic()
            process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            (HERE / (label+"-"+mode+".log")).write_text(process.stdout + process.stderr)
            process.check_returncode()
            values[mode] = json.loads(process.stdout)
            commands.append({"command": command, "returncode": process.returncode,
                             "seconds": time.monotonic() - before})
        assert values["original"]["direct"] == values["composed"]["direct"]
        assert sha(HERE / (label+"-original.cells")) == sha(HERE / (label+"-composed.cells"))
        for mode in values:
            value = values[mode]
            assert value["diagnose_loader"] == mode
            assert value["diagnose"]["H"] == value["direct"]["H_"+mode]
            assert value["diagnose"]["M"] == value["direct"]["M"]
            assert value["diagnose"]["Z"] == value["direct"]["Z"]
        hstpath = casepath / "hyp.z4c.user.hst"
        history = np.atleast_2d(np.loadtxt(hstpath))
        nearest = int(np.argmin(abs(history[:, 0]-checkpoint["time"])))
        assert abs(history[nearest, 0]-checkpoint["time"]) < 1e-14
        reproduction = {}
        for name, column in (("H", 2), ("M", 3), ("Z", 4), ("Theta", 5)):
            expected = float(history[nearest, column])
            observed = values["original"]["diagnose"][name]
            error = abs(observed-expected)
            assert error < 1e-11*(1+abs(expected)), (label, name, error)
            reproduction[name] = {"original_recomputed": observed, "native_HST": expected,
                                  "absolute_difference": error}
        cells = np.fromfile(HERE / (label+"-original.cells"), dtype="<f8").reshape(-1, 12)
        assert len(cells) == values["original"]["active_cells"]
        radius = np.linalg.norm(cells[:, :3], axis=1)
        regions = []
        for left, right in ((0., .05), (.05, .9), (.9, .95), (.95, 1.)):
            selected = (radius >= left) & (radius < right)
            region = {"r_min": left, "r_max": right, "cells": int(selected.sum())}
            for name, col in (("H_original", 3), ("H_composed", 4), ("H_difference", 9),
                              ("M", 6), ("Z", 7)):
                square = cells[:, col]**2
                region[name] = {"rms": float(np.sqrt(square[selected].mean())) if selected.any() else None,
                                "squared_norm_fraction": float(square[selected].sum()/square.sum())
                                if square.sum() else 0.}
            regions.append(region)
        rows.append({"label": label, "requested_time": requested, "exact_restart_time": checkpoint["time"],
                     "time_offset_from_request": checkpoint["time"]-requested,
                     "cycle": checkpoint["cycle"], "dt": checkpoint["dt"],
                     "restart": metadata(rstpath), "raw": metadata(raw),
                     "cell_data": metadata(HERE / (label+"-original.cells")),
                     "native_results": metadata(result_path), "native_input": metadata(casepath / "layer.athinput"),
                     "native_HST": metadata(hstpath), "native_executable_sha256": result["sha256"],
                     "original_HST_reproduction": reproduction, "original": values["original"],
                     "composed": values["composed"], "regions": regions, "commands": commands})
        print(label, checkpoint["time"], "H original/composed", values["original"]["diagnose"]["H"],
              values["composed"]["diagnose"]["H"], "M", values["original"]["diagnose"]["M"], flush=True)
    for group in ("private_source_sha256", "all_compiled_repository_dependencies_sha256"):
        assert all(sha(ROOT / name) == digest for name, digest in build[group].items())
    named = {row["label"]: row for row in rows}
    left, right, fine = (named[name] for name in
                         ("N24-before-t.2", "N24-near-t.2", "N36-t.2"))
    t0, t1 = left["exact_restart_time"], right["exact_restart_time"]
    assert t0 < .2 < t1 and fine["exact_restart_time"] == .2
    fraction = (.2-t0)/(t1-t0)
    comparison = {"target_time": .2, "N24_bracketing_times": [t0, t1],
                  "linear_interpolation_fraction": fraction,
                  "scope": "Linear interpolation of diagnostic RMS values, never interpolation of state fields; two-resolution descriptive ratios, not a spatial convergence-order claim",
                  "dt_at_nearest_checkpoint_N24_N36": [right["dt"], fine["dt"]],
                  "nearest_checkpoint": {}, "interpolated_diagnostic": {}}
    for name in ("H", "M", "Z"):
        coarse = right["composed"]["diagnose"][name]
        reference = fine["composed"]["diagnose"][name]
        interpolated = ((1-fraction)*left["composed"]["diagnose"][name]
                        + fraction*coarse)
        for kind, value in (("nearest_checkpoint", coarse),
                            ("interpolated_diagnostic", interpolated)):
            ratio = value/reference
            comparison[kind][name] = {"N24": value, "N36": reference,
                                     "N24_over_N36": ratio,
                                     "log_ratio_over_log_1.5": float(np.log(ratio)/np.log(1.5))}
    receipt = {"status": "PASS", "scope": "Read-only constraint-functional attribution of archived composed-RHS states; no evolution/stability/convergence claim",
               "seconds": time.monotonic()-started, "head_at_analysis": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "compiled_implementation": build["compiled_implementation"], "compiles": compiles, "rows": rows,
               "two_resolution_t.2_comparison": comparison,
               "private_build": metadata(receipt_path), "original_private_inputs_unchanged": True,
               "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in
                                 (Path(__file__), HERE / "check_snapshot.cpp", READER, READER.with_name("abi.json"), original_header, overlay_header)},
               "executables": {mode: metadata(HERE / ("check_"+mode)) for mode in ("original", "composed")},
               "cell_columns": ["x", "y", "z", "H_original", "H_composed", "predicted_delta_H", "M_conformal_norm", "Z_conformal_norm", "Theta_physical", "delta_H", "delta_M_norm", "delta_Z_norm"],
               "limitations": ["N24-near-t.2 uses the nearest actual checkpoint, not interpolated state fields.",
                               "Only the diagonal second-derivative functional changes; first/mixed derivatives and reconstruction/ghost plans remain the original private ones.",
                               "A composed flat linear constraint closure does not establish variable-coefficient product-rule closure."]}
    write(HERE / "receipt.json", receipt)
    print("PASS nine stored states, two Diagnose loaders, exact unchanged M/Z/Theta, independently predicted delta H")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
