"""Compare two completed embedded-secondary runs with native AthenaK histories."""

import argparse
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vis/python"))
from athena_read import hst


def inspect_run(directory):
    log = (directory / "log").read_text(errors="replace")
    if "Terminating on time limit" not in log:
        raise ValueError(f"Incomplete matched interval in {directory}")
    seconds = float(re.search(r"cpu time used\s*=\s*(\S+)", log)[1])
    histfile = directory / "embedded.user.hst"
    with histfile.open() as stream:
        omega = float(
            re.search("frame_omega=([^ ]+)", next(s for s in stream if "frame_omega=" in s))[1]
        )
    h = hst(histfile)
    mhd = hst(directory / "embedded.mhd.hst")
    for key in ("ct_divB_relative", "ct_face_B_max", "ct_nonfinite", "zoom_zone", "zoom_event"):
        if not np.isfinite(h[key]).all():
            raise ValueError(f"Nonfinite monitor {key} in {directory}")
    if not all(np.isfinite(a).all() for a in mhd.values()) or np.any(mhd["mass"] <= 0):
        raise ValueError(f"Nonphysical global history in {directory}")
    tim, cycle = re.search(r"\ntime=(\S+) cycle=(\d+)\n", log).groups()
    if not math.isclose(h["time"][-1], float(tim), rel_tol=2e-6):
        raise ValueError("Incomplete history")
    if np.any(np.diff(h["time"]) <= 0):
        raise ValueError(f"Non-increasing history time in {directory}")
    np.testing.assert_array_equal(h["time"], mhd["time"])
    duration = h["time"][-1] - h["time"][0]
    # The StellarAI pilot pins one MPI rank to each allocated B200.
    ranks = int(re.search(r"Number of parallel ranks = (\d+)", log)[1])
    magnetic_energy = sum(mhd[f"{i}-ME"] for i in (1, 2, 3))
    return h, {
        "physical_duration_M": duration,
        "binary_orbits": duration * omega / (2 * np.pi),
        "loop_seconds": seconds,
        "ranks_one_per_gpu": ranks,
        "initial_blocks": int(re.search(r"Total number of MeshBlocks = (\d+)", log)[1]),
        "history_samples": len(h["time"]),
        "frame_omega": omega,
        "integration_steps": int(cycle),
        "projected_B200_hours_per_binary_orbit": ranks
        * seconds
        / 3600
        * 2
        * np.pi
        / (omega * duration),
        "max_relative_mass_change": float(np.max(np.abs(mhd["mass"] / mhd["mass"][0] - 1))),
        "peak_abs_face_field": float(np.max(h["ct_face_B_max"])),
        "max_relative_magnetic_energy_change": float(
            np.max(np.abs(magnetic_energy / magnetic_energy[0] - 1))
        ),
        "peak_relative_divergence": float(np.max(h["ct_divB_relative"])),
        "max_nonfinite_faces": int(np.max(h["ct_nonfinite"])),
        "qualified_finest_samples": {
            surf: int(np.count_nonzero((h["zoom_zone"] == 0) & (h["valid_" + surf] == 1)))
            for surf in ("H2", "BH2_Kerr")
        },
        "full_returns": len(set(h["zoom_event"][h["zoom_zone"] == 0]) - {0}),
        "median_timestep_by_zone": {
            int(z): float(np.median(h["dt"][h["zoom_zone"] == z]))
            for z in np.unique(h["zoom_zone"])
        },
    }


def compare(directory, timing_only=False):
    c, cyclic = inspect_run(directory / "cyclic")
    f, fixed = inspect_run(directory / "fixed")
    np.testing.assert_allclose(c["time"][[0, -1]], f["time"][[0, -1]], rtol=1e-10, atol=1e-12)
    for key in ("initial_blocks", "ranks_one_per_gpu", "frame_omega"):
        if cyclic[key] != fixed[key]:
            raise ValueError(f"Unmatched {key}")
    for key in ("phiB_H2", "phiB_BH2_Kerr", "M0_net_in_moving_H2"):
        np.testing.assert_allclose(c[key][0], f[key][0], atol=1e-13, rtol=1e-10)
    if cyclic["full_returns"] < 3 or np.any(f["zoom_zone"] != 0):
        raise ValueError("Expected at least three complete zoom cycles and a static control")
    results = {
        "cyclic": cyclic,
        "fixed": fixed,
        "speedup": fixed["loop_seconds"] / cyclic["loop_seconds"],
    }
    if timing_only:
        return results
    returns = []
    for event in sorted(set(c["zoom_event"][c["zoom_zone"] == 0]) - {0}):
        mask = (c["zoom_event"] == event) & (c["zoom_zone"] == 0)
        row = {"event": int(event)}
        for surface, key in [
            ("BH2_Kerr", "phiB_BH2_Kerr"),
            ("H2", "phiB_H2"),
            ("H2", "M0_net_in_moving_H2"),
        ]:
            if not np.all(c["valid_" + surface][mask] == 1) or not np.all(
                f["valid_" + surface] == 1
            ):
                raise ValueError(f"Unqualified matched sample for {surface}")
            times = c["time"][mask]
            if len(times) < 2:
                raise ValueError("Too few simultaneous samples")
            signal = c[key][mask]
            reference = np.interp(times, f["time"], f[key])
            if not np.all(np.isfinite(signal)) or not np.all(np.isfinite(reference)):
                raise ValueError(f"Unexpected invalid measurement {key}")
            weights = np.diff(times) / (2 * (times[-1] - times[0]))
            actual, control = [
                float(np.sum(weights * (v[1:] + v[:-1]))) for v in (signal, reference)
            ]
            row.update(t_start=times[0], t_end=times[-1])
            row[key] = {
                "cyclic_mean": actual,
                "fixed_mean": control,
                "fractional_difference": actual / control - 1 if control else None,
            }
        returns.append(row)
    results["matched_fine_visits"] = returns
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--timing-only", action="store_true", help="Measure cost without forming surface averages"
    )
    args = parser.parse_args()
    print(json.dumps(compare(args.directory, args.timing_only), indent=2, allow_nan=False))
