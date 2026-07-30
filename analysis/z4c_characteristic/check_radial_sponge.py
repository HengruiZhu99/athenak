#!/usr/bin/env python3
"""Validate a radial-sponge pulse against a matched undamped control."""

import argparse
import math
import pathlib
import re

import numpy as np


DAMPED_FIELDS = (
    "Theta-max",
    "beta-res",
    "Gam-res",
    "res-ramp",
    "res-outer",
)


def load_history(root):
    matches = [
        path for path in root.glob("*.user.hst")
        if not path.name.endswith(".z4c.user.hst")
    ]
    if len(matches) != 1:
        raise SystemExit(
            "{}: expected one pgen user history".format(root))
    lines = matches[0].read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise SystemExit("{}: incomplete history".format(matches[0]))
    labels = {
        match.group(2): int(match.group(1)) - 1
        for match in re.finditer(r"\[(\d+)\]=(\S+)", lines[1])
    }
    required = (
        "time",
        "bad-metric",
        "res-inner",
    ) + DAMPED_FIELDS
    missing = [label for label in required if label not in labels]
    if missing:
        raise SystemExit(
            "{}: missing columns {}".format(
                matches[0], ", ".join(missing)))
    rows = np.asarray([
        [float(value) for value in line.split()]
        for line in lines[2:]
        if line.strip() and not line.lstrip().startswith("#")
    ])
    if rows.ndim != 2 or len(rows) < 2 or not np.all(np.isfinite(rows)):
        raise SystemExit("{}: missing or nonfinite rows".format(matches[0]))
    return {label: rows[:, index] for label, index in labels.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("control", type=pathlib.Path)
    parser.add_argument("sponge", type=pathlib.Path)
    parser.add_argument("--damping-time", type=float, default=16.0)
    parser.add_argument("--relative-tolerance", type=float, default=0.02)
    parser.add_argument("--interior-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--interior-absolute-tolerance", type=float, default=5.0e-15)
    args = parser.parse_args()

    if not math.isfinite(args.damping_time) or args.damping_time <= 0.0:
        raise SystemExit("damping time must be finite and positive")
    control = load_history(args.control)
    sponge = load_history(args.sponge)
    if (
        control["time"].shape != sponge["time"].shape
        or not np.allclose(
            control["time"], sponge["time"], rtol=0.0, atol=1.0e-12)
    ):
        raise SystemExit("control and sponge time samples do not match")
    if np.any(control["bad-metric"] != 0.0) or np.any(
            sponge["bad-metric"] != 0.0):
        raise SystemExit("a run reported an invalid metric")

    times = control["time"]
    selected = times > 0.0
    expected = np.exp(-times[selected] / args.damping_time)
    failures = []
    for label in DAMPED_FIELDS:
        denominator = control[label][selected]
        if np.any(denominator <= 0.0):
            failures.append("{} has a nonpositive control value".format(label))
            continue
        ratio = sponge[label][selected] / denominator
        relative_error = np.max(np.abs(ratio / expected - 1.0))
        print(
            "{} final_control={:.8e} final_sponge={:.8e} "
            "final_ratio={:.8e} expected={:.8e} "
            "maximum_relative_error={:.8e}".format(
                label, control[label][-1], sponge[label][-1], ratio[-1],
                expected[-1], relative_error))
        if relative_error >= args.relative_tolerance:
            failures.append(
                "{} attenuation error {:.6g} exceeds {:.6g}".format(
                    label, relative_error, args.relative_tolerance))

    interior_difference = float(np.max(np.abs(
        sponge["res-inner"] - control["res-inner"])))
    interior_scale = max(
        float(np.max(np.abs(control["res-inner"]))),
        args.interior_absolute_tolerance)
    interior_fraction = interior_difference / interior_scale
    print(
        "interior_max_difference={:.8e} interior_fraction={:.8e}".format(
            interior_difference, interior_fraction))
    if (
        interior_difference >= args.interior_absolute_tolerance
        and interior_fraction >= args.interior_relative_tolerance
    ):
        failures.append(
            "interior difference exceeds both absolute and relative gates")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
