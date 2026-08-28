#!/usr/bin/env python3
"""Fail-closed source contract for shock-avoiding Bona-Masso slicing."""

from __future__ import annotations

import argparse
import math
import pathlib
import subprocess
import tempfile


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=pathlib.Path, required=True)
    parser.add_argument("--athena", type=pathlib.Path)
    args = parser.parse_args()
    source = args.source_dir
    header = (source / "src/z4c/z4c.hpp").read_text(encoding="utf-8")
    setup = (source / "src/z4c/z4c.cpp").read_text(encoding="utf-8")
    rhs = (source / "src/z4c/z4c_calcrhs.cpp").read_text(encoding="utf-8")
    newdt = (source / "src/z4c/z4c_newdt.cpp").read_text(encoding="utf-8")
    policy_input = (source / "tst/inputs/z4c_shock_avoiding_gauge.athinput").read_text(
        encoding="utf-8")

    for token in (
        "bool lapse_shock_avoiding;",
        "Real lapse_shock_avoiding_kappa;",
    ):
        require(header.count(token) == 1, f"missing or duplicate option: {token}")
    for token in (
        'pin->DoesParameterExist("z4c", "lapse_shock_avoiding")',
        'pin->GetBoolean("z4c", "lapse_shock_avoiding")',
        'pin->DoesParameterExist("z4c", "lapse_shock_avoiding_kappa")',
        'pin->GetReal("z4c", "lapse_shock_avoiding_kappa")',
        "opt.lapse_shock_avoiding_kappa <= 0.0",
        "opt.telegraph_lapse || opt.slow_start_lapse || opt.lapse_oplog != 0.0",
        "opt.lapse_harmonic != 0.0",
    ):
        require(setup.count(token) == 1, f"missing or duplicate validation: {token}")
    for token in (
        "if (opt.lapse_shock_avoiding)",
        "(alpha * alpha + opt.lapse_shock_avoiding_kappa) *",
        "(z4c.vKhat(m, k, j, i) + 2.0 * z4c.vTheta(m, k, j, i));",
        "opt.lapse_advect * Lalpha - f * alpha * z4c.vKhat(m, k, j, i);",
    ):
        require(rhs.count(token) == 1, f"missing or duplicate RHS policy: {token}")

    shock_block = rhs.split("if (opt.lapse_shock_avoiding)", 1)[1].split("} else", 1)[0]
    for forbidden in ("abs", "max", "floor", "clip", "telegraph"):
        require(forbidden not in shock_block.lower(),
                f"shock-avoiding driver contains forbidden {forbidden}")
    require(newdt.count("(!opt.lapse_shock_avoiding && !(alpha > 0.0))") == 2,
            "shock-avoiding timestep path still rejects negative lapse")
    require("Kokkos::fabs(alpha) * Kokkos::sqrt(physical_inverse)" in newdt,
            "physical light speed does not use the negative-lapse magnitude")
    admissibility = setup.split(
        "void Z4c::CheckStateAdmissibility(Driver *driver", 1)[1].split(
            "\nvoid Z4c::", 1)[0]
    require(admissibility.count(
        "const bool require_positive_lapse = !opt.lapse_shock_avoiding;") == 1,
        "admissibility policy is not copied out of the host Z4c object")
    require(admissibility.count(
        "EvaluateZ4cState(values, nz4c, require_positive_lapse)") == 1,
        "device admissibility scan does not use the capture-safe lapse policy")
    require(admissibility.count(
        "EvaluateZ4cState(values.data(), nz4c, require_positive_lapse)") == 1,
        "host admissibility report disagrees with the device lapse policy")
    device_scan = admissibility.split("KOKKOS_LAMBDA", 1)[1].split("});", 1)[0]
    require("opt." not in device_scan,
            "admissibility device lambda captures the host Z4c options object")
    require(policy_input.count("lapse_shock_avoiding = false") == 1,
            "regression input must exercise the default-off legacy path")
    require(policy_input.count("lapse_shock_avoiding_kappa = 1.0") == 1,
            "regression input must expose the prospective kappa override")

    if args.athena is not None:
        input_path = source / "tst/inputs/z4c_shock_avoiding_gauge.athinput"
        with tempfile.TemporaryDirectory(prefix="athena-shock-gauge-") as temporary:
            root = pathlib.Path(temporary)
            common = [
                str(args.athena), "-i", str(input_path),
                "z4c/telegraph_lapse=false", "z4c/lapse_oplog=0",
                "z4c/lapse_harmonic=0", "z4c/lapse_shock_avoiding=true",
                "z4c/shift_Gamma=0", "z4c/shift_alpha2Gamma=0",
                "z4c/shift_H=0", "z4c/shift_advect=0", "z4c/shift_eta=0",
            ]
            for label, kappa in (("kappa1", "1.0"),
                                 ("kappa2over3", "0.6666666666666666")):
                case = root / label
                case.mkdir()
                command = common + ["-d", str(case), f"job/basename=shock_{label}",
                                    f"z4c/lapse_shock_avoiding_kappa={kappa}"]
                run = subprocess.run(command, check=False, text=True,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
                require(run.returncode == 0 and "cycle=1" in run.stdout,
                        f"{label} production smoke failed: {run.stdout}")
                histories = list(case.glob("*.hst"))
                require(len(histories) == 1, f"{label} history inventory changed")
                rows = [line.split() for line in histories[0].read_text().splitlines()
                        if line and not line.startswith("#")]
                require(rows and all(math.isfinite(float(value)) for value in rows[-1]),
                        f"{label} final history is not finite")

            invalid = root / "invalid"
            invalid.mkdir()
            run = subprocess.run(
                common + ["-d", str(invalid),
                          "z4c/lapse_shock_avoiding_kappa=0"],
                check=False, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT)
            require(run.returncode != 0 and
                    "lapse_shock_avoiding_kappa must be positive" in run.stdout,
                    "nonpositive shock-avoiding kappa did not fail closed")
    print("Shock-avoiding Bona-Masso source contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
