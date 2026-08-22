#!/usr/bin/env python3
"""Run the native-cell Ref-GH planted-exponent calibration matrix."""

import argparse
import json
import re
import subprocess
from pathlib import Path


RESULT = re.compile(
    r"estimator initialized: e_G=(?P<e_g>[-+0-9.eE]+) "
    r"e_alpha=(?P<e_alpha>[-+0-9.eE]+) shell-valid=(?P<valid>[01])"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    resolutions = (32, 48, 64)
    mismatches = (-0.25, -0.10, 0.0, 0.10, 0.25)
    records = []
    for resolution in resolutions:
        for mismatch in mismatches:
            command = [
                str(args.executable), "-v", "-i", str(args.input),
                f"meshblock/nx1={resolution}",
                f"meshblock/nx2={resolution}",
                f"meshblock/nx3={resolution}",
                f"ref_gh/controller_delta_q={mismatch:.17g}",
                f"ref_gh/controller_delta_p={mismatch:.17g}",
            ]
            completed = subprocess.run(
                command, check=False, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            match = RESULT.search(completed.stdout)
            if completed.returncode != 0 or match is None:
                raise RuntimeError(
                    f"calibration failed for nx={resolution}, delta={mismatch}:\n"
                    f"{completed.stdout}"
                )
            measured_e_g = float(match.group("e_g"))
            measured_e_alpha = float(match.group("e_alpha"))
            expected_e_g = 2.0*mismatch
            expected_e_alpha = -mismatch
            records.append({
                "nx": resolution,
                "delta_q": mismatch,
                "delta_p": mismatch,
                "expected_e_G": expected_e_g,
                "measured_e_G": measured_e_g,
                "absolute_e_G_error": abs(measured_e_g - expected_e_g),
                "expected_e_alpha": expected_e_alpha,
                "measured_e_alpha": measured_e_alpha,
                "absolute_e_alpha_error": abs(
                    measured_e_alpha - expected_e_alpha
                ),
                "shell_valid": bool(int(match.group("valid"))),
                "command": command,
            })

    maximum_g_error = max(record["absolute_e_G_error"] for record in records)
    maximum_alpha_error = max(
        record["absolute_e_alpha_error"] for record in records
    )
    result = {
        "schema": "ref-gh-fixed-shell-estimator-calibration-v1",
        "fixed_shell_M": [0.15, 0.40],
        "weight": "(M/r)^3 on native Cartesian cells",
        "resolutions": list(resolutions),
        "mismatches": list(mismatches),
        "maximum_absolute_e_G_error": maximum_g_error,
        "maximum_absolute_e_alpha_error": maximum_alpha_error,
        "passed": (
            all(record["shell_valid"] for record in records)
            and maximum_g_error <= 5.0e-12
            and maximum_alpha_error <= 5.0e-12
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "maximum_absolute_e_G_error",
        "maximum_absolute_e_alpha_error",
        "passed",
    )}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
