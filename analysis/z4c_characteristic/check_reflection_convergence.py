#!/usr/bin/env python3
"""Check resolution convergence of far-control-subtracted pulse reflection."""

import argparse
import math
import pathlib
import re


RATIO = re.compile(r"\bratio=([+\-0-9.eE]+)")


def load(root):
    values = {}
    for result in sorted(root.glob("axis*_side*_*/interior_reflection.txt")):
        match = RATIO.search(result.read_text())
        if match is None:
            raise SystemExit("{}: missing ratio".format(result))
        value = float(match.group(1))
        if not math.isfinite(value) or value <= 0.0:
            raise SystemExit("{}: invalid ratio {}".format(result, value))
        values[result.parent.name] = value
    if not values:
        raise SystemExit("{}: no reflection measurements".format(root))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("coarse", type=pathlib.Path)
    parser.add_argument("medium", type=pathlib.Path)
    parser.add_argument("fine", type=pathlib.Path)
    parser.add_argument("--minimum-order", type=float, default=1.8)
    parser.add_argument("--maximum-fine-ratio", type=float, default=0.02)
    args = parser.parse_args()

    coarse = load(args.coarse)
    medium = load(args.medium)
    fine = load(args.fine)
    if not (set(coarse) == set(medium) == set(fine)):
        raise SystemExit("convergence roots do not contain identical cases")

    failures = []
    for name in sorted(coarse):
        order_cm = math.log(
            coarse[name] / medium[name], 2.0)
        order_mf = math.log(
            medium[name] / fine[name], 2.0)
        print(
            "{} coarse={:.8e} medium={:.8e} fine={:.8e} "
            "order_cm={:.6f} order_mf={:.6f}".format(
                name, coarse[name], medium[name], fine[name],
                order_cm, order_mf))
        if fine[name] >= args.maximum_fine_ratio:
            failures.append(
                "{} fine ratio {:.6e} exceeds {:.6e}".format(
                    name, fine[name], args.maximum_fine_ratio))
        if min(order_cm, order_mf) < args.minimum_order:
            failures.append(
                "{} convergence order {:.6f}/{:.6f} is below {:.6f}".format(
                    name, order_cm, order_mf, args.minimum_order))

    print(
        "SUMMARY cases={} fine_max={:.8e} minimum_observed_order={:.6f}".format(
            len(fine), max(fine.values()),
            min(
                min(
                    math.log(coarse[name] / medium[name], 2.0),
                    math.log(medium[name] / fine[name], 2.0))
                for name in fine)))
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
