"""Time-weighted half-unsigned fluxes and two paper normalizations over a specified window."""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vis/python"))
import athena_read


def summarize(h, start, end, settle_time=4 * math.sqrt(5**3 / 0.5), max_gap=3.0):
    t = h["time"]
    if start >= end or end > t[-1] + 1e-7 or start < t[0]:
        raise ValueError("Analysis window must be covered by the history")
    dt = np.diff(t)
    if not np.all(dt > 0):
        raise ValueError("Overlapping or decreasing history times")
    left, right = np.maximum(t[:-1], start), np.minimum(t[1:], end)
    same_event = np.diff(h["zoom_event"]) == 0
    sampled = same_event & (dt <= max_gap)

    def mean(key, valid, phase_left=None, phase_right=None):
        x = h[key]
        mask = valid & np.isfinite(x)
        lo = left if phase_left is None else np.maximum(left, phase_left)
        hi = right if phase_right is None else np.minimum(right, phase_right)
        weights = np.maximum(hi - lo, 0)
        use = (weights > 0) & sampled & mask[1:] & mask[:-1]
        duration = float(np.sum(weights[use]))
        if not duration:
            return None, 0.0
        # Integrate the piecewise linear interpolant, including fractional end intervals.
        frac = ((lo[use] + hi[use]) / 2 - t[:-1][use]) / dt[use]
        val = x[:-1][use] + frac * (x[1:][use] - x[:-1][use])
        return float(np.sum(weights[use] * val) / duration), duration / (end - start)

    def good(surface):
        return h["valid_" + surface] == 1

    transfer = good("H1") & good("H2")
    transfer &= np.isfinite(h["M0_net_in_moving_H1"]) & np.isfinite(h["M0_net_in_moving_H2"])
    rates = [mean(f"M0_net_in_moving_H{i}", transfer) for i in (1, 2)]
    total = sum(x for x, c in rates) if all(x is not None for x, c in rates) else None
    results = {"start": start, "end": end, "mean_5M_mdot_total": total, "holes": {}}
    finest = h["zoom_zone"] == 0
    # A first recorded sample after a transition can lag the event by <= one
    # history interval. Use it as a conservative origin for relaxation cuts.
    entry = np.maximum.accumulate(np.where(np.r_[True, ~same_event], t, -np.inf))
    cut = entry[:-1] + settle_time

    def normalized(flux, mdot):
        if flux is None or mdot is None or mdot <= 0:
            return None
        return math.sqrt(4 * math.pi) * flux / (0.5 * math.sqrt(mdot))

    for i in (1, 2):
        rate, coverage = rates[i - 1]
        out = {"mean_5M_mdot": rate, "rate_coverage": coverage}
        surfaces = [
            ("coordinate_5M", f"H{i}", transfer),
            ("horizon", f"BH{i}_Kerr", finest & good("BH1_Kerr") & good("BH2_Kerr")),
        ]
        if f"valid_BH{i}_KS5" in h:
            surfaces.append(("kerr_radius_5M", f"BH{i}_KS5", good("BH1_KS5") & good("BH2_KS5")))
        for key, surface, valid in surfaces:
            flux, fraction = mean("phiB_" + surface, valid)

            out[key] = {
                "half_unsigned_HL": flux,
                "coverage": fraction,
                "phi_common": normalized(flux, total),
                "phi_per_hole": normalized(flux, rate),
            }
        qualified = finest & good("BH1_Kerr") & good("BH2_Kerr")
        key = f"phiB_BH{i}_Kerr"
        for label, lo, hi in [("post_return_early", None, cut), ("post_return_relaxed", cut, None)]:
            flux, fraction = mean(key, qualified & (h["zoom_event"] > 0), lo, hi)
            out[label] = {
                "half_unsigned_HL": flux,
                "coverage": fraction,
                "phi_common": normalized(flux, total),
                "phi_per_hole": normalized(flux, rate),
            }
        results["holes"][i] = out
    results["max_sample_gap"] = max_gap
    results["post_return_settle_time"] = settle_time
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("--separation", type=float, required=True)
    parser.add_argument("--start-orbit", type=float, required=True)
    parser.add_argument("--end-orbit", type=float, required=True)
    args = parser.parse_args()
    period = 2 * math.pi * args.separation**1.5
    h = athena_read.hst(args.history)
    start = period * args.start_orbit
    end = period * args.end_orbit
    result = summarize(h, start, end)
    # Equal physical-time batches expose secular trends and cycle-to-cycle scatter.
    # Nested summaries report their own accretion denominators and valid exposure.
    edges = np.linspace(start, end, 5)
    result["time_batches"] = [summarize(h, a, b) for a, b in zip(edges[:-1], edges[1:])]
    print(json.dumps(result, indent=2, allow_nan=False))
