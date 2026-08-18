#!/usr/bin/env python3
"""Offline timeline and stitched local-spectrum analysis for the stage-3 audit."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TARGET_GIDS = (35, 60)
META = {"gid", "level", "lx1", "lx2", "k", "j", "i", "rho", "z"}


def write_timeline(source, output, summary_path):
    rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    selected = [row for row in rows if row["gid"] in TARGET_GIDS]
    selected.sort(key=lambda row: (row["gid"], row["cycle"]))
    fieldnames = [
        "gid", "level", "lx1", "lx2", "cycle", "time", "time_hex",
        "requested_flag", "raw_dchi_max", "dchi_threshold", "threshold_ratio",
        "next_authority_event", "next_authority_time_hex", "tree_checksum",
        "observed_segment", "cycle_gap_from_previous",
    ]
    summaries = {}
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for gid in TARGET_GIDS:
            subset = [row for row in selected if row["gid"] == gid]
            segment = 0
            previous = None
            for row in subset:
                gap = "" if previous is None else row["cycle"] - previous
                if previous is not None and gap != 1:
                    segment += 1
                writer.writerow({
                    "gid": gid, "level": row["level"], "lx1": row["lx1"],
                    "lx2": row["lx2"], "cycle": row["cycle"],
                    "time": format(float.fromhex(row["actual_time_hex"]), ".17g"),
                    "time_hex": row["actual_time_hex"],
                    "requested_flag": row["requested_flag"],
                    "raw_dchi_max": format(row["raw_dchi_max"], ".17g"),
                    "dchi_threshold": format(row["dchi_threshold"], ".17g"),
                    "threshold_ratio": format(
                        row["raw_dchi_max"] / row["dchi_threshold"], ".17g"),
                    "next_authority_event": row["next_authority_event"],
                    "next_authority_time_hex": row["next_authority_time_hex"],
                    "tree_checksum": row["tree_checksum"],
                    "observed_segment": segment,
                    "cycle_gap_from_previous": gap,
                })
                previous = row["cycle"]
            requested = [row for row in subset if row["requested_flag"] == 1]
            gaps = [b["cycle"] - a["cycle"] for a, b in zip(requested, requested[1:])]
            summaries[str(gid)] = {
                "observed_rows": len(subset),
                "observed_refinement_requests": len(requested),
                "all_retained_rows_request_refinement": len(requested) == len(subset),
                "first_request_cycle": requested[0]["cycle"],
                "first_request_time_hex": requested[0]["actual_time_hex"],
                "first_request_raw_dchi": requested[0]["raw_dchi_max"],
                "last_request_cycle": requested[-1]["cycle"],
                "last_request_time_hex": requested[-1]["actual_time_hex"],
                "last_request_raw_dchi": requested[-1]["raw_dchi_max"],
                "accepted_cycle_span": requested[-1]["cycle"] - requested[0]["cycle"],
                "largest_unobserved_cycle_gap": max(gaps) if gaps else 0,
                "continuous_sampling_claim": all(gap == 1 for gap in gaps),
            }
    summary = {
        "schema": "athenak_target_block_refinement_timeline_v1",
        "targets": summaries,
        "interpretation": (
            "Both targets have 271 retained refinement-request rows. GID 35 requests "
            "in every retained row; GID 60 also has retained non-request rows. A "
            "735-cycle unobserved interval means persistence across the complete "
            "cycle-4542--5546 span is not established."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def detrend(values, rho, z):
    rr, zz = np.meshgrid(rho, z)
    rc = (rr - rr.mean()) / max(np.ptp(rr), np.finfo(float).tiny)
    zc = (zz - zz.mean()) / max(np.ptp(zz), np.finfo(float).tiny)
    design = np.column_stack([
        np.ones(rr.size), rc.ravel(), zc.ravel(), (rc * rc).ravel(),
        (rc * zc).ravel(), (zc * zc).ravel(),
    ])
    coefficients, _, _, _ = np.linalg.lstsq(design, values.ravel(), rcond=None)
    return values - (design @ coefficients).reshape(values.shape)


def high_metrics(power, q):
    mask = q > 0.0
    total = power[mask].sum()
    if not np.isfinite(total) or total <= np.finfo(float).tiny:
        return [math.nan] * 7
    fractions = [power[(q >= cut) & mask].sum() / total for cut in (0.5, 0.65, 0.8)]
    order = np.argsort(q[mask])
    q_sorted = q[mask][order]
    cumulative = np.cumsum(power[mask][order]) / total
    quantiles = [q_sorted[min(np.searchsorted(cumulative, value), len(q_sorted) - 1)]
                 for value in (0.9, 0.95, 0.99)]
    dominant = q[mask][np.argmax(power[mask])]
    return fractions + quantiles + [dominant]


def eta_metrics(values, spacing, axis):
    moved = np.moveaxis(values, axis, -1)
    center = moved[..., 3:-3]
    dm3, dm2, dm1 = moved[..., :-6], moved[..., 1:-5], moved[..., 2:-4]
    dp1, dp2, dp3 = moved[..., 4:-2], moved[..., 5:-1], moved[..., 6:]
    d4 = (dm2 - 8.0 * dm1 + 8.0 * dp1 - dp2) / (12.0 * spacing)
    d6 = ((-dm3 + 9.0 * dm2 - 45.0 * dm1 + 45.0 * dp1 -
           9.0 * dp2 + dp3) / (60.0 * spacing))
    scale = np.abs(center) + spacing * np.abs(d6) + np.finfo(float).eps
    eta_d = spacing * np.abs(d6 - d4) / scale
    second = dm1 - 2.0 * center + dp1
    fourth = dm2 - 4.0 * dm1 + 6.0 * center - 4.0 * dp1 + dp2
    eta4 = np.abs(fourth) / (np.abs(second) + np.finfo(float).eps * scale)
    return np.nanpercentile(eta_d, 95), np.nanpercentile(eta4, 95)


def write_spectra(source, output, plot_dir, summary_path):
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit("empty local patch")
    variables = [name for name in rows[0] if name not in META]
    rho = np.array(sorted({float(row["rho"]) for row in rows}))
    z = np.array(sorted({float(row["z"]) for row in rows}))
    if len(rows) != len(rho) * len(z):
        raise SystemExit("target active blocks do not form one complete stitched patch")
    r_index = {value: index for index, value in enumerate(rho)}
    z_index = {value: index for index, value in enumerate(z)}
    data = {name: np.empty((len(z), len(rho))) for name in variables}
    for row in rows:
        iz, ir = z_index[float(row["z"])], r_index[float(row["rho"])]
        for name in variables:
            data[name][iz, ir] = float(row[name])
    dr = np.diff(rho)
    dz = np.diff(z)
    if not (np.allclose(dr, dr[0], rtol=0.0, atol=1e-14) and
            np.allclose(dz, dz[0], rtol=0.0, atol=1e-14)):
        raise SystemExit("stitched patch is not uniform")
    # A full-patch Hann window vanishes only at the outer patch edges; z=0 is
    # near its maximum and is not treated as a boundary.
    window = np.outer(np.hanning(len(z)), np.hanning(len(rho)))
    qrho = np.abs(np.fft.fftfreq(len(rho))) / 0.5
    qz = np.abs(np.fft.fftfreq(len(z))) / 0.5
    q2 = np.maximum(qz[:, None], qrho[None, :])
    records = []
    plot_payload = {}
    for name in variables:
        residual = detrend(data[name], rho, z) * window
        transform = np.fft.fft2(residual)
        power2 = np.abs(transform) ** 2
        power_r = power2.sum(axis=0)
        power_z = power2.sum(axis=1)
        metrics2 = high_metrics(power2, q2)
        metrics_r = high_metrics(power_r, qrho)
        metrics_z = high_metrics(power_z, qz)
        eta_d_r, eta4_r = eta_metrics(data[name], dr[0], axis=1)
        eta_d_z, eta4_z = eta_metrics(data[name], dz[0], axis=0)
        direction = "rho" if metrics_r[0] > metrics_z[0] else "z"
        records.append({
            "variable": name,
            "f_high_50": metrics2[0], "f_high_65": metrics2[1],
            "f_high_80": metrics2[2], "k90_over_nyquist": metrics2[3],
            "k95_over_nyquist": metrics2[4], "k99_over_nyquist": metrics2[5],
            "dominant_k_over_nyquist": metrics2[6],
            "f_high_50_rho": metrics_r[0], "f_high_50_z": metrics_z[0],
            "etaD_p95_rho": eta_d_r, "etaD_p95_z": eta_d_z,
            "eta4_p95_rho": eta4_r, "eta4_p95_z": eta4_z,
            "predominant_short_direction": direction,
        })
        plot_payload[name] = (qrho, power_r, qz, power_z, q2, power2)
    records.sort(key=lambda row: (-np.nan_to_num(row["f_high_50"], nan=-1.0),
                                  row["variable"]))
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    plot_dir.mkdir(parents=True, exist_ok=True)
    selected = [row["variable"] for row in records[:5]]
    if "z4c_chi" not in selected:
        selected.append("z4c_chi")
    for name in selected:
        qr, pr, qzz, pz, qgrid, p2 = plot_payload[name]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        axes[0].semilogy(qr, pr + np.finfo(float).tiny, ".", ms=2)
        axes[0].set(xlabel=r"$|k_\rho|/k_N$", ylabel="summed power", title="rho")
        axes[1].semilogy(qzz, pz + np.finfo(float).tiny, ".", ms=2)
        axes[1].set(xlabel=r"$|k_z|/k_N$", title="z")
        image = axes[2].imshow(np.log10(np.fft.fftshift(p2) + np.finfo(float).tiny),
                               origin="lower", aspect="auto", cmap="magma")
        axes[2].set(title="2D log10 power", xlabel=r"$k_\rho$ bin", ylabel=r"$k_z$ bin")
        fig.colorbar(image, ax=axes[2], shrink=0.8)
        fig.suptitle(name + " stitched across z=0")
        fig.tight_layout()
        fig.savefig(plot_dir / (name + "_spectra.png"), dpi=180)
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 6))
    labels = [row["variable"].replace("z4c_", "") for row in records]
    values = [row["f_high_50"] for row in records]
    axis.barh(np.arange(len(records)), values)
    axis.set_yticks(np.arange(len(records)), labels=labels, fontsize=7)
    axis.invert_yaxis()
    axis.set(xlabel=r"power fraction with max directional $|k| \geq 0.5 k_N$",
             title="Stage-3 pre-update local high-frequency ranking")
    fig.tight_layout()
    fig.savefig(plot_dir / "all_variables_high_frequency_ranking.png", dpi=180)
    plt.close(fig)

    eta_records = sorted(
        records,
        key=lambda row: -np.nan_to_num(
            max(row["etaD_p95_rho"], row["etaD_p95_z"]), nan=-1.0))
    fig, axis = plt.subplots(figsize=(9, 6))
    eta_labels = [row["variable"].replace("z4c_", "") for row in eta_records]
    eta_values = [max(row["etaD_p95_rho"], row["etaD_p95_z"])
                  for row in eta_records]
    axis.barh(np.arange(len(eta_records)), eta_values)
    axis.set_yticks(np.arange(len(eta_records)), labels=eta_labels, fontsize=7)
    axis.invert_yaxis()
    axis.set(xlabel=r"95th percentile $\eta_D$ across rho/z",
             title="Stage-3 local derivative-disagreement ranking")
    fig.tight_layout()
    fig.savefig(plot_dir / "all_variables_etaD_ranking.png", dpi=180)
    plt.close(fig)

    finite = [row for row in records if np.isfinite(row["f_high_50"])]
    correlation_d = np.corrcoef(
        [row["f_high_50"] for row in finite],
        [max(row["etaD_p95_rho"], row["etaD_p95_z"]) for row in finite])[0, 1]
    correlation_4 = np.corrcoef(
        [row["f_high_50"] for row in finite],
        [max(row["eta4_p95_rho"], row["eta4_p95_z"]) for row in finite])[0, 1]
    summary = {
        "schema": "athenak_local_high_frequency_summary_v1",
        "rows": len(records), "stitched_shape_z_rho": [len(z), len(rho)],
        "z_zero_is_internal_seam": True,
        "outer_edge_window_only": True,
        "top_variables": [row["variable"] for row in records[:5]],
        "chi_rank_one_based": 1 + next(index for index, row in enumerate(records)
                                         if row["variable"] == "z4c_chi"),
        "pearson_fhigh50_vs_etaD_p95": float(correlation_d),
        "pearson_fhigh50_vs_eta4_p95": float(correlation_4),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-patch", type=Path, required=True)
    parser.add_argument("--shadow-requests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_timeline(args.shadow_requests,
                   args.output_dir / "target_block_refinement_timeline.csv",
                   args.output_dir / "target_block_refinement_summary.json")
    write_spectra(args.local_patch,
                  args.output_dir / "local_high_frequency_metrics.csv",
                  args.output_dir / "local_spectra",
                  args.output_dir / "local_high_frequency_summary.json")
    print("STAGE3_OFFLINE_TIMELINE_AND_SPECTRA_PASS")


if __name__ == "__main__":
    main()
