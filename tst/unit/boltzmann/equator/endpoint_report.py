"""Summarize a completed full-domain and trial run without promoting it."""
import csv
from pathlib import Path

import numpy as np
from performance import summarize


def write_report(result, reference, trial, output):
    cost = [summarize(p) if "Terminating on time limit" in (p/"stdout").read_text() else None
            for p in (reference, trial)]
    with (output/"timed-profiles.csv").open() as stream:
        profiles = list(csv.DictReader(stream))
    lines = ["# Matched photon-ring evolution", "",
             "| Run | End time (code) | Time-mean blocks | Steps | Seconds | GPU-hours | Peak MiB/GPU |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for side, snapshot in zip(("Reference","Trial"), cost):
        c = snapshot
        if c is None:
            history = result["full" if side=="Reference" else "half"]
            lines.append(f"| {side} (incomplete) | {history['end']:.8g} | — | — | — | — | — |")
            continue
        lines.append(f"| {side} | {c['end_time']:.8g} | {c['time_mean_blocks']:.3f} | "
                     f"{c['steps']} | {c['wall_seconds']:.2f} | {c['GPU_hours']:.5f} | "
                     f"{c['sampled_peak_MiB']} |")
    if all(cost) and abs(cost[0]["end_time"]-cost[1]["end_time"]) < 1e-10:
        lines += ["", f"GPU-hour saving over the matched interval was "
                  f"{100*(1-cost[1]['GPU_hours']/cost[0]['GPU_hours']):.2f}%. "
                  f"Hosts were {cost[0]['host']} and {cost[1]['host']}."]
    else:
        lines += ["", "At least one evolution stopped before the requested endpoint. "
                  "GPU-hours over the complete matched interval remain unavailable."]
    lines += ["", "| Quantity | Final trial | Final trial minus reference | Largest absolute difference |",
              "|---|---:|---:|---:|"]
    for key in ("mean_Rcirc_M","p90_Rcirc_M","vertical_stress","E","J","outer_decile_mean_dx"):
        v = np.array([float(r["delta_"+key]) for r in profiles])
        lines.append(f"| {key} | {float(profiles[-1][key]):.8g} | {v[-1]:.5g} | {max(abs(v)):.5g} |")
    lines += ["", "Differences in the table use reference values interpolated in physical time. "
              "The accompanying timed profiles record quadrature and cadence scales.", ""]
    with (output/"comparison-fields.csv").open() as stream:
        fields = list(csv.DictReader(stream))
    for group in ("adm_", "tmunu_", "con_"):
        worst = max((r for r in fields if r["field"].startswith(group)),key=lambda r:float(r["Linf"]))
        lines.append(f"The largest {group[:-1]} field difference on the common spatial lattice was "
                     f"{float(worst['Linf']):.5g} in {worst['field']} at "
                     f"{float(worst['t_full_M']):.6g} M. The paired times differed by "
                     f"{abs(float(worst['delta_t_M'])):.3g} M.")
    dcon = np.array(result["half"]["final_constraint_sums"])-result["full"]["final_constraint_sums"]
    lines.append("Terminal trial-minus-reference squared constraint integrals (C, H, M, Z) were "
                 + ", ".join(f"{x:.5g}" for x in dcon) + ".")
    lines.append("")
    first = [result[side].get("first_verified_in_window") for side in ("full","half")]
    if all(m and m["enclosing_fit"]["accepted"] for m in first):
        a, b = first
        dt = abs(a["t_M"]-b["t_M"])
        cadence = max(a.get("previous_interval_M",0), b.get("previous_interval_M",0))
        lines.append(f"First verified times differed by {dt:.5g} M. "
                     f"The larger preceding sampling interval was {cadence:.5g} M.")
        for key in ("area","surface_mass","surface_spin_ratio"):
            diff = abs(a["enclosing_fit"][key]-b["enclosing_fit"][key])
            if "geometric_sampling_change" in a and "geometric_sampling_change" in b:
                spatial = a["geometric_sampling_change"][key]+b["geometric_sampling_change"][key]
                lines.append(f"The enclosing {key} differed by {diff:.5g}. The sum of the "
                             f"two measured spatial sampling changes was {spatial:.5g}.")
            if "next_cadence_change" in a:
                lines.append(f"The reference {key} changed by {a['next_cadence_change'][key]:.5g} "
                             "over the next saved interval.")
        lines.append("")
    for side in ("full","half"):
        v = result[side]
        lines.append(f"**{side}** relative energy and angular-momentum budget residuals were "
                     f"{v['energy_budget_relative']:.4g} and {v['J_budget_relative']:.4g}.")
        if "R17_charges" in v:
            initial, last = v["R17_charges"]
            lines.append(f"At coordinate radius 17 the finite-sphere mass changed from "
                         f"{initial['M_sphere']:.8g} to {last['M_sphere']:.8g}, and axial J "
                         f"from {initial['J_sphere']:.8g} to {last['J_sphere']:.8g}.")
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
            if "witness_frames" in v:
                usable = len(v["witness_frames_with_whole_surface_samples"])
                attempted = len(v["witness_frames"])
                lines.append(f"The independent search sampled a whole candidate on {usable} of "
                             f"{attempted} selected slices. It verified no strictly trapped witness.")
        else:
            lines.append(f"The online search first reported a marginal surface on cycle "
                         f"{v['first_online_marginal_cycle']}. Strict trapping remains unverified.")
        lines.append("")
    (output/"report.md").write_text("\n".join(lines)+"\n")
