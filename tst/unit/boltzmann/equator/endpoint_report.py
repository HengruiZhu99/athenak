"""Summarize a completed full-domain and trial run without promoting it."""
import csv
from pathlib import Path

import numpy as np
from performance import summarize


def write_report(result, reference, trial, output):
    cost = [summarize(p) for p in (reference, trial)]
    with (output/"timed-profiles.csv").open() as stream:
        profiles = list(csv.DictReader(stream))
    lines = ["# Matched photon-ring evolution", "",
             "| Run | End time (code) | Time-mean blocks | Steps | Seconds | GPU-hours | Peak MiB/GPU |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for side, snapshot in zip(("Reference","Trial"), cost):
        c = snapshot
        lines.append(f"| {side} | {c['end_time']:.8g} | {c['time_mean_blocks']:.3f} | "
                     f"{c['steps']} | {c['wall_seconds']:.2f} | {c['GPU_hours']:.5f} | "
                     f"{c['sampled_peak_MiB']} |")
    lines += ["", f"GPU-hour saving over the matched interval was "
              f"{100*(1-cost[1]['GPU_hours']/cost[0]['GPU_hours']):.2f}%. "
              f"Hosts were {cost[0]['host']} and {cost[1]['host']}.",
              "", "| Quantity | Final trial | Final trial minus reference | Largest absolute difference |",
              "|---|---:|---:|---:|"]
    for key in ("mean_Rcirc_M","p90_Rcirc_M","vertical_stress","E","J","outer_decile_mean_dx"):
        v = np.array([float(r["delta_"+key]) for r in profiles])
        lines.append(f"| {key} | {float(profiles[-1][key]):.8g} | {v[-1]:.5g} | {max(abs(v)):.5g} |")
    lines += ["", "Differences in the table use reference values interpolated in physical time. "
              "The accompanying timed profiles record quadrature and cadence scales.", ""]
    for side in ("full","half"):
        v = result[side]
        lines.append(f"**{side}** relative energy and angular-momentum budget residuals were "
                     f"{v['energy_budget_relative']:.4g} and {v['J_budget_relative']:.4g}.")
        surface = v.get("first_verified_in_window")
        if surface:
            outer = surface["enclosing_fit"]
            lines.append(f"The first strictly trapped witness in the searched window occurred at "
                         f"{surface['t_M']:.7f} M and spanned {surface['zones']:.2f} radial cells.")
            if outer["accepted"]:
                lines.append(f"The independently fitted enclosing marginal surface had area "
                             f"{outer['area']:.8g}, mass {outer['surface_mass']:.8g}, "
                             f"and spin {outer['surface_spin_ratio']:.8f}.")
        elif v["first_online_marginal_cycle"] is None:
            lines.append("The online search reported no marginal surface through the final output.")
        else:
            lines.append(f"The online search first reported a marginal surface on cycle "
                         f"{v['first_online_marginal_cycle']}. Strict trapping remains unverified.")
        lines.append("")
    (output/"report.md").write_text("\n".join(lines)+"\n")
