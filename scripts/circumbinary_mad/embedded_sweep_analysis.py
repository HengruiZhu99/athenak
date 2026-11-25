"""Measure matched finest visits, timestep limits and disk profiles in the 100:1 sweeps."""

import argparse
import json
from pathlib import Path

import numpy as np

from embedded_comparison import inspect_run, hst
from bin_convert import read_binary


def visit_errors(history, reference):
    rows = []
    for event in np.unique(history["zoom_event"][history["zoom_zone"] == 0]):
        if not event:
            continue
        sub = (history["zoom_event"] == event) & (history["zoom_zone"] == 0)
        t = history["time"][sub]
        row = dict(event=int(event), t0=t[0], t1=t[-1], samples=len(t))
        for surf in ("BH1_Kerr", "H1", "BH2_Kerr", "H2", "R40"):
            for prefix in ("phiB_", "M0_net_in_moving_"):
                k = prefix + surf
                v = history[k][sub]
                r = np.interp(
                    t,
                    reference["time"],
                    np.where(reference["valid_" + surf] == 1, reference[k], np.nan),
                    left=np.nan,
                    right=np.nan,
                )
                good = (history["valid_" + surf][sub] == 1) & np.isfinite(v) & np.isfinite(r)
                intervals = good[:-1] & good[1:]
                dt = np.diff(t)[intervals]
                if not dt.size:
                    continue
                a, b = (
                    float(np.sum(dt * (arr[:-1][intervals] + arr[1:][intervals]) * 0.5) / dt.sum())
                    for arr in (v, r)
                )
                row[k] = dict(
                    cyclic=a,
                    reference=b,
                    delta=a - b,
                    rel=a / b - 1 if b else None,
                    qualified_time_M=float(dt.sum()),
                )
        rows.append(row)
    return rows


def history_stats(directory):
    h, m = inspect_run(directory)
    for surf in ["BH1_Kerr", "BH2_Kerr", "H1", "H2", "R40"]:
        for name, mask in [
            ("fine", h["zoom_zone"] == 0),
            ("outer", h["zoom_zone"] == max(h["zoom_zone"])),
        ]:
            m[f"{name}_valid_{surf}"] = [
                int(np.count_nonzero(h["valid_" + surf][mask])),
                int(np.sum(mask)),
            ]
    outer = h["zoom_zone"] == np.max(h["zoom_zone"])
    # Count only intervals whose endpoints sampled the same outer visit.
    q = outer[:-1] & outer[1:] & (h["zoom_event"][:-1] == h["zoom_event"][1:])
    m["sampled_coarse_dwell_fraction"] = np.sum(np.diff(h["time"])[q]) / (
        h["time"][-1] - h["time"][0]
    )
    return h, m


def matched_transfer_rates(case, reference, stop=52.0):
    """Integrate rates over shared time segments with valid, unchanged stencils."""
    histories = [hst(path / "embedded.user.hst") for path in (case, reference)]
    edges = np.unique(np.concatenate(([0, stop], *(h["time"] for h in histories))))
    edges = edges[(edges >= 0) & (edges <= stop)]
    left, right = edges[:-1], edges[1:]
    mid = 0.5 * (left + right)
    widths = right - left
    summary = {}
    for surf in ("H1", "H2", "BH1_Kerr", "BH2_Kerr"):
        for channel in ("M0_net_in_moving_", "phiB_"):
            key = channel + surf
            keep = np.ones(len(mid), dtype=bool)
            signals = []
            for h in histories:
                t, value = h["time"], h[key]
                i = np.clip(np.searchsorted(t, mid) - 1, 0, len(t) - 2)
                valid = (h["valid_" + surf] == 1) & np.isfinite(value)
                keep &= valid[i] & valid[i + 1] & (h["zoom_event"][i] == h["zoom_event"][i + 1])
                keep &= (t[i] <= left) & (right <= t[i + 1])
                # Midpoint quadrature exactly integrates each linear interpolant.
                signals.append(np.interp(mid, t, value))
            exposure = widths[keep].sum()
            entry = {"common_exposure_M": float(exposure), "until_M": stop}
            if exposure:
                a, b = (float(np.sum(v[keep] * widths[keep]) / exposure) for v in signals)
                entry.update(cyclic=a, reference=b, delta=a - b, rel=a / b - 1 if b else None)
            summary[key] = entry
    return summary


def probe(case):
    p = Path(case)
    h = hst(p / "embedded.user.hst")
    rows = []
    for f in sorted((p / "bin").glob("*.bin")):
        s = read_binary(str(f))
        g = s["mb_geometry"]
        dx = (g[:, 1] - g[:, 0]) / s["nx1_mb"]
        idx = np.argmin(np.abs(h["time"] - s["time"]))
        cells = lambda x: float(
            dx[(g[:, 0] <= x) & (g[:, 1] > x) & (g[:, 2] <= 0) & (g[:, 3] > 0)][0]
        )
        rows.append(
            dict(
                zone=int(h["zoom_zone"][idx]),
                step=int(s["cycle"]),
                t=float(s["time"]),
                dt=float(h["dt"][idx]),
                primary_dx=cells(25 / 101),
                secondary_dx=cells(-2500 / 101),
                disk_dx=cells(28),
                midplane_blocks=s["n_mbs"],
            )
        )
    return rows


def midplane_difference(case, reference, interpolate=False):
    from embedded_figure import raster

    files = sorted((case / "bin").glob("*.bin"))
    control = read_binary(str(sorted((reference / "bin").glob("*.bin"))[-1]))
    time = control["time"]
    if interpolate:
        samples = {}
        for path in files:
            with path.open("rb") as stream:
                stream.readline()
                stream.readline()
                stamp = stream.readline().decode().strip()
            if not stamp.startswith("time="):
                raise ValueError(f"Missing physical time in {path}")
            samples[float(stamp.split("=")[1])] = path
        lo = max(t for t in samples if t <= time)
        hi = min(t for t in samples if t >= time)
        if hi - lo > 1.7:
            raise ValueError("Missing bracketing slices for the 1.5M output cadence")
        history = hst(case / "embedded.user.hst")
        # Include the native samples immediately before and after the slice times.
        left = max(0, np.searchsorted(history["time"], lo) - 1)
        right = min(len(history["time"]), np.searchsorted(history["time"], hi) + 1)
        if np.ptp(history["zoom_event"][left:right]) != 0:
            raise ValueError("Cannot interpolate across a zoom transition")
        fraction = (time - lo) / (hi - lo) if hi > lo else 0.0
        first = read_binary(str(samples[lo]))
        second = read_binary(str(samples[hi])) if hi > lo else first
    else:
        first = read_binary(str(files[-1]))
        lo = hi = first["time"]
        if not np.isclose(lo, time, atol=1e-7, rtol=1e-8):
            raise ValueError("Unmatched output times")
        fraction = 0.0
    radius = 40
    n = 640
    v = radius * ((np.arange(n) + 0.5) * 2 / n - 1)
    R = np.hypot(v[None, :], v[:, None])
    r2 = np.hypot(v[None, :] + 2500 / 101, v[:, None])
    r1 = np.hypot(v[None, :] - 25 / 101, v[:, None])
    regions = {
        "inner_disk": (R >= 12) & (R < 24),
        "dense_disk": (R >= 24) & (R < 40),
        "secondary_neighborhood": (r2 > 2.5) & (r2 < 5),
        "primary_transfer_neighborhood": (r1 > 3) & (r1 < 6),
    }
    out = {"time_M": time, "comparison_slice_times_M": [lo, hi], "reference": reference.name}
    for k in ["dens", "bcc3", "press"]:
        a = raster(first, 0, radius, k, n=n)
        if fraction:
            a += fraction * (raster(second, 0, radius, k, n=n) - a)
        b = raster(control, 0, radius, k, n=n)
        for label, reg in regions.items():
            weights = abs(b[reg]).sum()
            out.setdefault(label, {})[k] = (
                float(np.sum(abs(a[reg] - b[reg])) / weights) if weights > 0 else None
            )
    return out


def report(root, original_fixed):
    results = {"grid_probes": {c: probe(root / c) for c in ("single-probe", "dual-probe")}}
    modes = (
        "live3",
        "live4",
        "fine-control",
        "return-check",
        "relax-secondary",
        "relax-both",
        "relax-compact",
        "relax-long",
    )
    results["runs"] = runs = {}
    for c in modes:
        h, stats = history_stats(root / c)
        runs[c] = stats
        if c.startswith("live") or c == "return-check":
            control = root / "fine-control" if c == "return-check" else original_fixed
            f, fixed = inspect_run(control)
            np.testing.assert_allclose(h["time"][[0, -1]], f["time"][[0, -1]])
            if stats["initial_blocks"] != fixed["initial_blocks"]:
                raise ValueError("Unmatched finest mesh")
            stats["matched_speedup"] = fixed["loop_seconds"] / stats["loop_seconds"]
            stats["fine_visits"] = visit_errors(h, f)
            stats["midplane_L1_relative_to_fixed"] = midplane_difference(root / c, control)
    for control in ("relax-secondary", "relax-compact"):
        runs["relax-both"]["matched_rates_relative_to_" + control] = matched_transfer_rates(
            root / "relax-both", root / control
        )
        runs["relax-both"]["midplane_L1_relative_to_" + control] = midplane_difference(
            root / "relax-both", root / control
        )
    for control in ("relax-both", "relax-compact"):
        runs["relax-long"]["matched_rates_relative_to_" + control] = matched_transfer_rates(
            root / "relax-long", root / control
        )
        runs["relax-long"]["midplane_L1_relative_to_" + control] = midplane_difference(
            root / "relax-long", root / control, interpolate=True
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, help="One subdirectory per sweep, named as in embedded_sweeps.py"
    )
    parser.add_argument("original_fixed", type=Path, help="Original M/8 primary static control")
    args = parser.parse_args()
    print(json.dumps(report(args.root, args.original_fixed), indent=2, allow_nan=False))
