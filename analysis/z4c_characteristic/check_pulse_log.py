#!/usr/bin/env python3
"""Check reflected incoming/outgoing characteristic amplitudes in a run log."""

import argparse
import math
import pathlib
import re


FIELD = re.compile(r"([A-Za-z_]+)=([+\-0-9.eE]+)")


def parse_records(path):
    records = []
    for line in path.read_text().splitlines():
        if not line.startswith("Z4C_CHARACTERISTIC_CPBC "):
            continue
        values = {key: float(value) for key, value in FIELD.findall(line)}
        records.append(values)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=pathlib.Path)
    parser.add_argument("sector", choices=("gauge", "constraint", "radiation"))
    parser.add_argument("--maximum-ratio", type=float, default=0.02)
    parser.add_argument("--peak-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-outgoing", type=float, default=1.0e-10)
    args = parser.parse_args()

    records = parse_records(args.log)
    outgoing_key = f"outgoing_{args.sector}"
    if not records or any(outgoing_key not in record for record in records):
        raise SystemExit(f"{args.log}: no usable characteristic diagnostics")
    if any(
        not all(math.isfinite(value) for value in record.values())
        for record in records
    ):
        raise SystemExit(f"{args.log}: nonfinite diagnostic")

    peak_outgoing = max(record[outgoing_key] for record in records)
    if peak_outgoing < args.minimum_outgoing:
        raise SystemExit(
            f"{args.log}: outgoing pulse was not observed "
            f"({peak_outgoing:.6e} < {args.minimum_outgoing:.6e})"
        )
    selected = [
        record for record in records
        if record[outgoing_key] >= args.peak_fraction * peak_outgoing
    ]
    peak_incoming = max(record[args.sector] for record in selected)
    ratio = peak_incoming / peak_outgoing
    print(
        f"sector={args.sector} outgoing={peak_outgoing:.8e} "
        f"incoming={peak_incoming:.8e} ratio={ratio:.8e}"
    )
    if ratio >= args.maximum_ratio:
        raise SystemExit(
            f"{args.log}: reflected ratio {ratio:.6e} exceeds "
            f"{args.maximum_ratio:.6e}"
        )


if __name__ == "__main__":
    main()
