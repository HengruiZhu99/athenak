"""Replay only constructed synthetic inputs in the original failure runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np
import scipy

from pade13_einsum import expm_pade13


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def peak(a):
    return float(np.max(np.abs(a))) if a.size else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    warnings.filterwarnings("error")
    np.seterr(all="raise")
    start = time.monotonic()
    failure_receipt = HERE.parent / "isolated-scipy-001/receipt.json"
    original = json.loads(failure_receipt.read_text())
    assert sys.executable == original["runtime"]["executable"]
    assert np.__version__ == original["runtime"]["numpy"]
    assert scipy.__version__ == original["runtime"]["scipy"]
    assert np.__file__ == original["runtime"]["numpy_file"]
    assert scipy.__file__ == original["runtime"]["scipy_file"]
    library_pins = []
    for pin in original["library_pins"]:
        assert sha(pin["path"]) == pin["sha256"]
        library_pins.append(pin)
    pins = {"pade13_einsum.py": sha(HERE / "pade13_einsum.py"),
            "check_failure_runtime.py": sha(HERE / "check_failure_runtime.py"),
            "isolated_failure_receipt": sha(failure_receipt)}
    records, arrays = [], {}
    for folder in ("synthetic-001/results", "large-norm-001/results"):
        source = HERE / folder
        receipt = json.loads((source / "receipt.json").read_text())
        assert receipt["status"] == "PASS"
        assert receipt["source_sha256"]["pade13_einsum.py"] == pins["pade13_einsum.py"]
        data_path = source / "synthetic-arrays.npz"
        assert sha(data_path) == receipt["output_sha256"]["synthetic-arrays.npz"]
        pins[folder + "/receipt.json"] = sha(source / "receipt.json")
        pins[folder + "/synthetic-arrays.npz"] = sha(data_path)
        # These two data paths are exclusively the constructed synthetic gates.
        data = np.load(data_path, allow_pickle=False)
        for case in receipt["cases"]:
            name = case["name"]
            argument = data[name + "_argument"]
            value, info = expm_pade13(argument, return_info=True)
            prior = data[name + "_exponential"]
            key = name + ("_oracle" if name + "_oracle" in data.files else "_closed")
            reference = data[key]
            scaled = peak(value - reference) / max(1.0, peak(reference))
            prior_scaled = peak(value - prior) / max(1.0, peak(prior))
            assert scaled <= 2e-11
            assert prior_scaled <= 2e-11
            assert info["rational_solve_residual"] <= 2e-14
            records.append({"name": name, "shape": list(argument.shape),
                            "forward_error": scaled, "prior_runtime_error": prior_scaled,
                            "prior_runtime_bitwise_equal": bool(np.array_equal(value, prior)),
                            "info": info})
            arrays[name] = value
    np.savez(out / "synthetic-exponentials.npz", **arrays)
    result = {"status": "PASS", "scope": "synthetic-only same-runtime check",
              "cases": records, "case_count": len(records), "source_sha256": pins,
              "python": sys.version, "executable": sys.executable,
              "executable_sha256": sha(Path(sys.executable).resolve()),
              "numpy": np.__version__, "scipy": scipy.__version__,
              "numpy_file": np.__file__, "scipy_file": scipy.__file__,
              "platform": platform.platform(), "library_pins": library_pins,
              "numpy_error_mode": np.geterr(), "warnings_as_errors": True,
              "original_underflow_mode": original["runtime"]["errstate"]["under"],
              "new_underflow_mode_stricter": "raise", "scipy_expm_called": False,
              "scientific_array_loaded": False, "seconds": time.monotonic() - start,
              "environment": {name: os.environ.get(name) for name in
                              ("PYTHONPATH", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
              "output_sha256": sha(out / "synthetic-exponentials.npz")}
    (out / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "cases": len(records),
                      "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
