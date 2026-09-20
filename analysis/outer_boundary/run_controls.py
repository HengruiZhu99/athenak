#!/usr/bin/env python3
"""Prepare/run isolated, reproducible boundary controls. Never submits cluster jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def parameter(text, section, key, value):
    def replace(match):
        body = match[2]
        pattern = rf"(?m)^\s*{re.escape(key)}\s*=.*$"
        if re.search(pattern, body):
            body = re.sub(pattern, f"{key} = {value}", body)
        else:
            body += f"\n{key} = {value}\n"
        return match[1] + body
    text, count = re.subn(rf"(<{section}>\n)(.*?)(?=\n<|\Z)",
                         replace, text, flags=re.S)
    assert count == 1, f"Missing or duplicate section: {section}"
    return text


CASES = {
    "outer-linear": {},
    "outer-quadratic": {"z4c/extrap_order": 3},
    "outer-zero": {"problem/vacuum_gauge_pulse_amplitude": 0},
    "outer-eight-blocks": {f"meshblock/nx{i}": 8 for i in (1, 2, 3)},
    "outer-radiation-linear": {
        "z4c/characteristic_bc_source": "damped_constraint_radiation"},
    "outer-radiation-cubic": {
        "z4c/characteristic_bc_source": "damped_constraint_radiation",
        "z4c/extrap_order": 4},
    "outer-sponge-wide": {"problem/outer_sponge_enabled": "true",
        "problem/outer_sponge_geometry": "face", "problem/outer_sponge_width": 256,
        "problem/outer_sponge_rate": .02},
    "outer-sponge-narrow": {"problem/outer_sponge_enabled": "true",
        "problem/outer_sponge_geometry": "face", "problem/outer_sponge_width": 128,
        "problem/outer_sponge_rate": .02},
    "centered-linear": {},
    "centered-radiation": {
        "z4c/characteristic_bc_source": "damped_constraint_radiation"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--target", type=float)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    exe = args.exe.resolve()
    digest = hashlib.sha256(exe.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    for case in args.cases:
        run = args.output / case
        run.mkdir()
        geometry = "outer" if case.startswith("outer") else "centered"
        text = (Path(__file__).parent / "inputs" / f"{geometry}_patch.athinput").read_text()
        target = args.target if args.target is not None else (5000 if geometry == "outer" else 1000)
        changes = {"time/tlim": target, **CASES[case]}
        for path, value in changes.items():
            section, key = path.split("/")
            text = parameter(text, section, key, value)
        (run / "input.athinput").write_text(text)
        result = {"executable": str(exe), "binary_sha256": digest,
                  "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "OMP_NUM_THREADS": args.threads, "case": case,
                  "note": "Finite completion alone is not a stability pass. Radiation is a rejected experimental control; sponge is mitigation."}
        (run / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        if not args.prepare_only:
            started = time.monotonic()
            with (run / "run.log").open("w") as log:
                proc = subprocess.run([str(exe), "-i", "input.athinput"], cwd=run,
                    stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, OMP_NUM_THREADS=str(args.threads)))
            result.update(exit_code=proc.returncode, elapsed_seconds=time.monotonic()-started)
            (run / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
            print(case, "exit", proc.returncode, flush=True)


if __name__ == "__main__":
    main()
