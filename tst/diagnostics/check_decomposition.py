"""Compare every native history column across decompositions, including NaN coverage."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vis/python"))
import athena_read


def compare(reference, other, *, restart=False, stationary_restart=False):
    ref = athena_read.hst(reference)
    trial = athena_read.hst(other)
    if ref.keys() != trial.keys():
        raise ValueError("History schemas differ")
    if restart:
        # Separate restart directories can omit the checkpoint row or begin there.
        start = np.flatnonzero(
            np.isclose(ref["time"], trial["time"][0], rtol=1e-12, atol=1e-14)
        )
        if len(start) != 1:
            raise ValueError("Restart begins outside reference time series")
        ix = slice(start[0], None)
    else:
        ix = slice(None)
    worst = 0.0
    for key in ref:
        a, b = ref[key][ix], trial[key]
        if key == "diagnostic_metric_time" and stationary_restart:
            # Fixed-mesh stationary restart initializes fresh ADM at the checkpoint.
            assert restart and np.all(a == a[0]) and np.all(b == b[0])
            assert a[0] <= b[0] <= trial["time"][0]
            continue
        np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b), err_msg=key)
        np.testing.assert_allclose(
            a, b, atol=2e-10, rtol=2e-10, equal_nan=True, err_msg=key
        )
        if np.isfinite(a).any():
            worst = max(worst, float(np.nanmax(np.abs(a - b))))
    return {
        "columns": len(ref),
        "rows": len(trial["time"]),
        "max_abs_difference": worst,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("reference", type=Path)
    p.add_argument("other", type=Path)
    p.add_argument("--restart", action="store_true")
    p.add_argument("--stationary-restart", action="store_true")
    args = p.parse_args()
    print(
        json.dumps(
            compare(
                args.reference,
                args.other,
                restart=args.restart or args.stationary_restart,
                stationary_restart=args.stationary_restart,
            )
        )
    )
