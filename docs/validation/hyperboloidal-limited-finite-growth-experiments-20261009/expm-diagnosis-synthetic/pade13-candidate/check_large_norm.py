"""Additional large-norm synthetic inputs; no saved operator is an input."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import mpmath as mp
import numpy as np

from check_synthetic import error, mp_oracle, oracle_difference
from pade13_einsum import expm_pade13, mm


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    warnings.filterwarnings("error")
    np.seterr(all="raise")
    start = time.monotonic()
    here = Path(__file__).resolve().parent
    names = ("pade13_einsum.py", "check_synthetic.py", "check_large_norm.py",
             "PLAN-STIFF-SUPPLEMENT.md")
    pins = {name: sha(here / name) for name in names}
    cases = []
    for amplitude in (1024.0, 16384.0, 131072.0):
        a = np.zeros((3, 3))
        a[0, 1], a[1, 2] = amplitude, -0.5 * amplitude
        closed = np.eye(3) + a + 0.5 * mm(a, a)
        cases.append(("nilpotent3_{}".format(int(amplitude)), a, closed))
    s = np.array([[1.0, 1024.0, -512.0], [0.0, 1.0, 32.0], [0.0, 0.0, 1.0]])
    d = np.diag([-1.0, 0.5, 1.0])
    a = np.linalg.solve(s.T, mm(s, d).T).T
    closed = np.linalg.solve(s.T, mm(s, np.diag(np.exp(np.diag(d)))).T).T
    cases.append(("large_nonnormal_similarity", a, closed))
    records, arrays, oracles = [], {}, {}
    for name, a, closed in cases:
        value, info = expm_pade13(a, return_info=True)
        first, second = mp_oracle(a, 100), mp_oracle(a, 130)
        difference = oracle_difference(first, second)
        assert difference <= mp.mpf("1e-85")
        reference = np.array(second, dtype=float)
        row = {"name": name, "shape": list(a.shape), "info": info,
               "closed_form_error": error(value, closed),
               "mpmath_forward_error": error(value, reference),
               "oracle_100_130_difference": mp.nstr(difference, 20)}
        assert row["closed_form_error"] <= 2e-11
        assert row["mpmath_forward_error"] <= 2e-11
        assert info["rational_solve_residual"] <= 2e-14
        records.append(row)
        arrays[name + "_argument"], arrays[name + "_exponential"] = a, value
        arrays[name + "_oracle"], arrays[name + "_closed"] = reference, closed
        oracles[name] = {"digits100": first, "digits130": second}
    np.savez(out / "synthetic-arrays.npz", **arrays)
    (out / "high-precision-oracles.json").write_text(json.dumps(oracles, indent=2) + "\n")
    result = {"status": "PASS", "scope": "four large-norm synthetic inputs only",
              "python": sys.version, "executable": sys.executable,
              "numpy": np.__version__, "mpmath": mp.__version__,
              "numpy_error_mode": np.geterr(), "source_sha256": pins,
              "thresholds": {"forward": 2e-11, "solve": 2e-14, "oracle": "1e-85"},
              "cases": records, "seconds": time.monotonic() - start,
              "output_sha256": {name: sha(out / name) for name in
                                ("synthetic-arrays.npz", "high-precision-oracles.json")}}
    assert pins == {name: sha(here / name) for name in names}
    (out / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "cases": len(records),
                      "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
