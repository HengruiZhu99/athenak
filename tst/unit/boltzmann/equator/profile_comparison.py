"""Compare ring observables at common physical times and record diagnostic sampling errors."""
import argparse
import csv
from pathlib import Path

import numpy as np

from diagnostics import profile


def history(folder, mass, *, quadrature=False):
    rows = []
    for path in sorted((folder/"bin").glob("*.tmunu.*.bin")):
        row = profile(path, mass)
        if quadrature:
            coarse = profile(path, mass, stride=2)
            for key in ("E", "J", "vertical_stress", "mean_Rcirc_M", "p90_Rcirc_M"):
                row[f"quadrature_{key}"] = abs(coarse[key] - row[key])
        rows.append(row)
    if not rows or any(b["t_M"] <= a["t_M"] for a,b in zip(rows, rows[1:])):
        raise ValueError(f"missing or unordered profiles in {folder}")
    return rows


def compare_profiles(reference, trial, mass, output):
    ref = history(reference, mass, quadrature=True)
    run = history(trial, mass)
    times = np.array([row["t_M"] for row in ref])
    fields = ("E", "J", "vertical_stress", "mean_Rcirc_M", "p90_Rcirc_M",
              "energy_fraction_dr00625", "outer_decile_mean_dx")
    values = {key:np.array([row[key] for row in ref]) for key in ref[0]}
    with output.open("w") as f:
        writer = None
        for row in run:
            t = row["t_M"]
            if t < times[0] - 1.e-12 or t > times[-1] + 1.e-12:
                continue
            nearest = np.argmin(abs(times-t))
            record = dict(t_M=t, nearest_ref_dt_M=times[nearest]-t)
            for key in fields:
                v = values[key]
                interp = np.interp(t, times, v)
                record[key] = row[key]
                record[f"delta_{key}"] = row[key]-interp
                if f"quadrature_{key}" in values:
                    record[f"ref_quadrature_{key}"] = np.interp(t, times, values[f"quadrature_{key}"])
                    # Largest neighboring one-sided change over one output interval.
                    near = v[max(nearest-1,0):min(nearest+2,len(v))]
                    record[f"ref_cadence_{key}"] = max(abs(near-v[nearest]))
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=record)
                writer.writeheader()
            writer.writerow(record)
        if writer is None:
            raise ValueError("no overlapping physical times")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference",type=Path)
    p.add_argument("trial",type=Path)
    p.add_argument("mass",type=float)
    p.add_argument("output",type=Path)
    args = p.parse_args()
    compare_profiles(args.reference, args.trial, args.mass, args.output)
