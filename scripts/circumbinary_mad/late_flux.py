"""Compare relaxed returns and a continuous finest replay on common physical intervals."""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

from disk_quality import measure
from flux_summary import athena_read, summarize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from magnetic_smooth_state import checkpoint_state

SETTLE = 4 * math.sqrt(5**3 / 0.5)
SURFACES = ("H1", "H2", "BH1_Kerr", "BH2_Kerr")
CHANNELS = (
    "M0_net_in_moving_H1",
    "M0_net_in_moving_H2",
    "phiB_H1",
    "phiB_H2",
    "phiB_BH1_Kerr",
    "phiB_BH2_Kerr",
)


def matched(fine, replay, anchor, levels):
    cycle_events = 2 * (levels - 1)
    # Exclude the finest visit both branches share before the first outbound transition.
    shared = anchor["id"] + (-anchor["id"]) % cycle_events
    start, end = max(fine["time"][0], replay["time"][0], anchor["time"]), min(
        fine["time"][-1], replay["time"][-1]
    )
    if start >= end:
        raise ValueError("The branches have no overlapping output")
    tf, tr = fine["time"], replay["time"]
    change = np.r_[True, np.diff(replay["zoom_event"]) != 0]
    entries = np.maximum.accumulate(np.where(change, tr, -np.inf))
    times = np.unique(
        np.concatenate(
            (
                [start, end],
                tf[(tf > start) & (tf < end)],
                tr[(tr > start) & (tr < end)],
                (entries[change] + SETTLE),
            )
        )
    )
    times = times[(times >= start) & (times <= end)]
    mid = (times[1:] + times[:-1]) * 0.5
    dt = np.diff(times)
    eligible = np.ones(len(mid), dtype=bool)
    values = {}
    for name, h in (("continuous_finest", fine), ("ordinary_zoom", replay)):
        t = h["time"]
        i = np.searchsorted(t, mid) - 1
        sampled = h["zoom_zone"] == 0
        for key in SURFACES:
            sampled &= h["valid_" + key] == 1
        for key in CHANNELS:
            sampled &= np.isfinite(h[key])
        width = t[i + 1] - t[i]
        eligible &= sampled[i] & sampled[i + 1] & (width <= 3.0)
        eligible &= h["zoom_event"][i] == h["zoom_event"][i + 1]
        if name == "ordinary_zoom":
            eligible &= (h["zoom_event"][i] >= shared + cycle_events) & (mid >= entries[i] + SETTLE)
        frac = (mid - t[i]) / width
        values[name] = {k: h[k][i] + frac * (h[k][i + 1] - h[k][i]) for k in CHANNELS}
    exposure = float(np.sum(dt[eligible]))
    if not exposure:
        raise ValueError("No qualified post-divergence relaxed finest overlap")
    out = {
        "start": float(times[:-1][eligible][0]),
        "end": float(times[1:][eligible][-1]),
        "exposure": exposure,
        "branches": {},
    }
    for name, columns in values.items():
        means = {
            k: float(np.average(v[eligible], weights=dt[eligible])) for k, v in columns.items()
        }
        mdot = sum(means[k] for k in CHANNELS[:2])
        if mdot <= 0:
            raise ValueError("Matched total mass rate must be positive for normalization")
        means.update(
            {
                f"phi_common_{s}": math.sqrt(4 * math.pi)
                * means["phiB_" + s]
                / (0.5 * math.sqrt(mdot))
                for s in SURFACES
            }
        )
        out["branches"][name] = means
    return out


def cycle_means(h, start, end):
    t = h["time"]
    reentry = np.flatnonzero(np.r_[False, (np.diff(h["zoom_event"]) != 0)] & (h["zoom_zone"] == 0))
    runs = []
    for first, next_return in zip(reentry, reentry[1:]):
        if t[first] < start or t[next_return] > end:
            continue
        sl = slice(first, next_return + 1)
        result = summarize({k: v[sl] for k, v in h.items()}, t[first], t[next_return])
        rates = [result["holes"][i]["mean_5M_mdot"] for i in (1, 2)]
        flux = [result["holes"][i]["post_return_relaxed"]["half_unsigned_HL"] for i in (1, 2)]
        if any(v is None for v in rates + flux):
            raise ValueError("Missing mass or relaxed horizon exposure in a late cycle")
        runs.append([*rates, *flux])
    if len(runs) < 6:
        raise ValueError("Fewer than six complete late cycles with qualified horizons")
    x = np.array(runs)

    def statistics(block):
        means = block.mean(axis=-2)
        total = means[..., 0] + means[..., 1]
        if np.any(total <= 0):
            raise ValueError("Nonpositive cycle-batch accretion rate")
        return np.stack(
            (
                total,
                *(math.sqrt(4 * math.pi) * means[..., k] / (0.5 * np.sqrt(total)) for k in (2, 3)),
            ),
            axis=-1,
        )

    left, right = np.array_split(x, 2)

    def difference(a, b):
        return 2 * (b - a) / (np.abs(a) + np.abs(b))

    drift = difference(statistics(left), statistics(right))
    # Resample neighboring pairs of cycles to retain short-lag dependence.
    rng = np.random.default_rng(25835)

    def moving_pairs(part):
        indices = rng.integers(len(part) - 1, size=(4000, (len(part) + 1) // 2))
        sample = np.stack((indices, indices + 1), axis=-1).reshape(4000, -1)[:, : len(part)]
        return statistics(part[sample])

    interval = np.quantile(
        difference(moving_pairs(left), moving_pairs(right)), [0.025, 0.975], axis=0
    )
    return {
        "complete_cycles": len(x),
        "columns": ["total_mdot", "common_phi_horizon_1", "common_phi_horizon_2"],
        "cycle_average": statistics(x).tolist(),
        "half_to_half_fractional_drift": drift.tolist(),
        "descriptive_95pct_drift_interval": interval.tolist(),
        "cycles_mdot1_mdot2_flux1_flux2": runs,
    }


def analyze(directory, sep):
    period = 2 * math.pi * sep**1.5
    seq = directory / f"seq{sep}"
    h = athena_read.hst(seq / f"mad{sep}.user.hst")
    replay = athena_read.hst(directory / f"seq{sep}-replay" / f"mad{sep}.user.hst")
    anchor = checkpoint_state(seq / "comparison-start.rst")
    levels = int(anchor["parameters"]["amr_criterion1/nlevels"])
    late = summarize(h, 150 * period, 200 * period)
    extended = summarize(h, h["time"][-1] - 4 * period, h["time"][-1])
    for i in (1, 2):
        if extended["holes"][i]["horizon"]["coverage"] < 0.98:
            raise ValueError("Last four orbits lack continuous resolved horizon measurements")
    report = {
        "separation": sep,
        "late_cyclic": late,
        "last_four_finest_orbits": extended,
        "late_cycles": cycle_means(h, 150 * period, 200 * period),
        "matched_return": matched(h, replay, anchor, levels),
    }

    # Evaluate evolved MRI and thickness on the final three paired disk snapshots.
    files = sorted((seq / "bin").glob(f"mad{sep}.midplane.*.bin"))[-3:]
    if not files:
        raise ValueError("Missing evolved disk slices")
    report["evolved_disk"] = [
        measure(f, f.with_name(f.name.replace(".midplane.", ".meridional."))) for f in files
    ]
    if sep == 25:
        # Rounded only for presentation in the paper; these values use the
        # JSON figure source and the shared total mass-transfer denominator.
        reference = {
            "coordinate_5M": [31.75622947, 31.32406446],
            "horizon": [17.51264651, 17.58195878],
        }
        report["paper_fractional_difference"] = {
            surf: [extended["holes"][i][surf]["phi_common"] / values[i - 1] - 1 for i in (1, 2)]
            for surf, values in reference.items()
        }
    return report


def qualification(report):
    # A practical equivalence check at the requested fractional precision.
    # Compare late halves and twin branches independently of the paper value.
    intervals = np.array(report["late_cycles"]["descriptive_95pct_drift_interval"])
    reasons = []
    for name, bounds in zip(report["late_cycles"]["columns"], intervals.T):
        if min(bounds) < -0.30 or max(bounds) > 0.30:
            reasons.append(f"{name} late-cycle drift interval {bounds.tolist()} exceeds ±30%")
    matched = report["matched_return"]
    branches = matched["branches"]
    for key in ["phi_common_H1", "phi_common_H2", "phi_common_BH1_Kerr", "phi_common_BH2_Kerr"]:
        base = branches["continuous_finest"][key]
        if base <= 0 or abs(branches["ordinary_zoom"][key] - base) > 0.30 * base:
            reasons.append(f"The ordinary and continuous runs differ by more than 30% in {key}")
    return reasons


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path, nargs="?")
    p.add_argument("separation", type=int, choices=[25, 15, 35], nargs="?")
    p.add_argument("--check-report", type=Path)
    args = p.parse_args()
    if args.check_report:
        reasons = qualification(json.loads(args.check_report.read_text()))
        if reasons:
            raise SystemExit("Scientific review required:\n" + "\n".join(reasons))
        print("Passed late-cycle and matched-return comparisons")
    else:
        if args.campaign is None or args.separation is None:
            p.error("Provide campaign directory and separation")
        print(json.dumps(analyze(args.campaign, args.separation), indent=2, allow_nan=False))
