#!/usr/bin/env python3
"""Bounded native-VC direct-Brill initialization qualification."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import shutil
import subprocess
from pathlib import Path


COEFFICIENT_SHA256 = (
    "ff0993c390513c15d6aa65857a0a3c710f2e2c3faf5717d9d63245203ccf2d6b"
)
EXPECTED_ADM_MASS = 2.660301967997158


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_numbers(text: str) -> list[float]:
    values = [float(value) for value in re.findall(
        r"(?<![A-Za-z_])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
        text,
    )]
    require(values and all(math.isfinite(value) for value in values),
            "direct Brill evidence contains a nonfinite or empty numeric inventory")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--coefficients", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()

    athena = args.athena.resolve()
    input_path = args.input.resolve()
    coefficients = args.coefficients.resolve()
    work_dir = args.work_dir.resolve()
    for path, label in ((athena, "Athena executable"),
                        (input_path, "input"),
                        (coefficients, "coefficient artifact")):
        require(path.is_file(), f"{label} is missing: {path}")
    require(sha256(coefficients) == COEFFICIENT_SHA256,
            "direct Brill coefficient authority hash mismatch")
    input_text = input_path.read_text(encoding="utf-8")
    require(re.search(r"(?m)^grid_centering\s*=\s*vertex\s*$", input_text)
            is not None,
            "direct Brill test input does not select native vertex centering")

    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    command = [
        str(athena), "-i", str(input_path), "-d", str(work_dir),
        f"problem/brill_global_coefficients_file={coefficients}",
    ]
    result = subprocess.run(command, cwd=work_dir, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=False)
    evidence = result.stdout + "\n" + result.stderr
    require(result.returncode == 0,
            f"native-VC direct Brill initialization failed ({result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    require("mode=direct_global_coefficients" in result.stdout,
            "run did not use direct coefficient import")
    require("lapse=psi^-2" in result.stdout,
            "run did not initialize the pre-collapsed lapse")
    require("Terminating on cycle limit" in result.stdout,
            "bounded initialization did not terminate at nlim=0")
    require(not list(work_dir.rglob("z4c_state_failure.json")),
            "initialization emitted a fail-closed Z4c state artifact")

    import_match = re.search(
        r"adm_mass=([^ ]+) min_psi4=([^ ]+) lapse=psi\^-2 "
        r"min_lapse=([^ ]+) max_lapse=([^\s]+)", result.stdout)
    require(import_match is not None,
            "direct Brill import diagnostics are incomplete")
    adm_mass, min_psi4, min_lapse, max_lapse = map(
        float, import_match.groups())
    require(all(math.isfinite(value) for value in
                (adm_mass, min_psi4, min_lapse, max_lapse)),
            "direct Brill import diagnostic is nonfinite")
    require(abs(adm_mass - EXPECTED_ADM_MASS) <= 1.0e-14,
            "direct Brill ADM mass changed")
    require(min_psi4 > 0.0 and 0.0 < min_lapse <= max_lapse <= 1.0,
            "direct Brill conformal factor or lapse is outside its admissible range")

    constraint_path = work_dir / "z4c_vc_brill_direct_fixed.constraints.dat"
    require(constraint_path.is_file(),
            "direct Brill initialization omitted the constraint summary")
    rows = [line.split() for line in constraint_path.read_text(
        encoding="utf-8").splitlines() if line and not line.startswith("#")]
    require(len(rows) == 4,
            f"expected four constraint regions, found {len(rows)}")
    require({(row[4], row[5]) for row in rows} == {
        ("box", "coordinate"), ("box", "proper"),
        ("r<=1", "coordinate"), ("r<=1", "proper")},
            "constraint region/weighting inventory changed")
    for row in rows:
        require(len(row) == 14, "constraint summary row schema changed")
        finite_numbers(" ".join(row[:4] + row[6:]))
        require(float(row[6]) > 0.0 and int(row[7]) > 0,
                "constraint summary has empty support")
        require(all(float(value) >= 0.0 for value in row[8:]),
                "constraint norm is negative")

    print("native-VC direct Brill initialization passed: "
          f"adm_mass={adm_mass:.16g} min_lapse={min_lapse:.16g} "
          f"max_lapse={max_lapse:.16g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
