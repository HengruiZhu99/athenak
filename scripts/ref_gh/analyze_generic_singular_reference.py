#!/usr/bin/env python3
"""Classify generic singular-reference maxima as log, log2, or power behavior."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def relative_rms(actual, fitted):
    scale = max(float(np.max(np.abs(actual))), np.finfo(float).tiny)
    return float(np.sqrt(np.mean((actual - fitted) ** 2)) / scale)


def classify(values):
    values = sorted(values)
    h = np.array([row[0] for row in values], dtype=float)
    y = np.array([row[1] for row in values], dtype=float)
    ell = np.log(1.0 / h)
    candidates = {}
    for name, matrix in (
        ("log", np.column_stack((np.ones_like(ell), ell))),
        ("log2", np.column_stack((np.ones_like(ell), ell, ell * ell))),
    ):
        coefficients, *_ = np.linalg.lstsq(matrix, y, rcond=None)
        candidates[name] = {
            "relative_rms": relative_rms(y, matrix @ coefficients),
            "coefficients": [float(value) for value in coefficients],
        }
    positive = np.all(y > 0.0)
    if positive:
        matrix = np.column_stack((np.ones_like(ell), ell))
        coefficients, *_ = np.linalg.lstsq(matrix, np.log(y), rcond=None)
        fitted = np.exp(matrix @ coefficients)
        candidates["power"] = {
            "relative_rms": relative_rms(y, fitted),
            "coefficients": [float(value) for value in coefficients],
            "power": float(coefficients[1]),
        }
    else:
        candidates["power"] = {
            "relative_rms": math.inf,
            "coefficients": [],
            "power": None,
        }
    best = min(candidates, key=lambda name: candidates[name]["relative_rms"])
    # A quadratic polynomial in log(h) has one extra degree of freedom.  Only
    # prefer it when it materially improves on the two-parameter alternatives.
    simpler = min(("log", "power"),
                  key=lambda name: candidates[name]["relative_rms"])
    if best == "log2" and not (
            candidates["log2"]["relative_rms"]
            < 0.35 * candidates[simpler]["relative_rms"]):
        best = simpler
    if best == "log2":
        _, linear, quadratic = candidates["log2"]["coefficients"]
        if abs(quadratic) * float(np.max(ell)) < 0.05 * abs(linear):
            best = "log"
    return {
        "classification": best,
        "candidates": candidates,
        "coarsest": float(y[-1]),
        "finest": float(y[0]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    groups = defaultdict(list)
    with args.input.open() as stream:
        reader = csv.DictReader(
            (line for line in stream if not line.startswith("#")),
            delimiter="\t",
            fieldnames=("mode", "tau_M", "R_G_M", "h_M", "measure",
                        "maximum", "radius_M", "radius_over_h"),
        )
        for row in reader:
            key = (row["mode"], float(row["tau_M"]),
                   float(row["R_G_M"]), row["measure"])
            groups[key].append((float(row["h_M"]), float(row["maximum"])))

    results = []
    for (mode, tau, width, measure), values in sorted(groups.items()):
        result = classify(values)
        result.update({"mode": mode, "tau_M": tau, "R_G_M": width,
                       "measure": measure})
        results.append(result)
    unexpected = [
        {"mode": result["mode"], "tau_M": result["tau_M"],
         "R_G_M": result["R_G_M"], "measure": result["measure"],
         "power": result["candidates"]["power"]["power"]}
        for result in results
        if result["mode"] == "dynamic"
        and result["classification"] == "power"
        and result["candidates"]["power"]["power"] is not None
        and result["candidates"]["power"]["power"] > 0.35
    ]
    payload = {"input": str(args.input), "groups": results,
               "unexpected_dynamic_algebraic_growth": unexpected,
               "prescribed_q_gate": "FAIL" if unexpected else "PASS"}
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    for result in results:
        if result["mode"] == "dynamic" and result["tau_M"] == 8.0 \
                and result["R_G_M"] == 3.0:
            power = result["candidates"]["power"]["power"]
            print("{}: {} coarse={:.8e} fine={:.8e} power={}".format(
                result["measure"], result["classification"],
                result["coarsest"], result["finest"],
                "n/a" if power is None else "{:.6f}".format(power)))
    print("prescribed_q_gate={}".format(payload["prescribed_q_gate"]))
    print("unexpected_dynamic_algebraic_growth={}".format(len(unexpected)))


if __name__ == "__main__":
    main()
