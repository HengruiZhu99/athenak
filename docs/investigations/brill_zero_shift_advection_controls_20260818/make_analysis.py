#!/usr/bin/env python3
"""Build compact, reproducible summaries from the bounded shift-control runs."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

HST_NAMES = [
    "time", "dt", "C_norm2", "H_norm2", "M_norm2", "Z_norm2",
    "Mx_norm2", "My_norm2", "Mz_norm2", "Theta_norm", "Volume",
    "max_abs_K", "nmb_total", "maxAbsKret", "maxRefLev", "maxNmbRank",
    "ahStatus", "ahLastCyc", "cycle", "axisLapse", "axisTau", "axisKret",
    "axis_C2", "axis_H2", "axis_M2", "axis_Z2", "axis_N", "off_C2",
    "off_H2", "off_M2", "off_Z2", "off_Volume", "L0_C2", "L0_H2",
    "L0_M2", "L0_Z2", "L0_N", "L1_C2", "L1_H2", "L1_M2", "L1_Z2",
    "L1_N", "L2_C2", "L2_H2", "L2_M2", "L2_Z2", "L2_N", "L3_C2",
    "L3_H2", "L3_M2", "L3_Z2", "L3_N", "L4_C2", "L4_H2", "L4_M2",
    "L4_Z2", "L4_N", "C_Linf", "C_rho", "C_z", "H_Linf", "H_rho",
    "H_z", "M_Linf", "M_rho", "M_z", "Z_Linf", "Z_rho", "Z_z",
    "muMin", "muMax",
]

RUNS = {
    "arm_zero_shift": ROOT / "evidence/z1/arm-zero-shift/arm_zero_shift.z4c.user.hst",
    "arm_gamma_o2_short": ROOT / "evidence/u2-short/arm-gamma-o2-short/arm_gamma_o2_short.z4c.user.hst",
    "arm_zero_shift_native": ROOT / "evidence/z2-native/arm-zero-shift-native/arm_zero_shift_native.z4c.user.hst",
}


def read_hst(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            vals = line.split()
            if len(vals) < len(HST_NAMES):
                continue
            row = {}
            for name, val in zip(HST_NAMES, vals):
                try:
                    row[name] = float(val)
                except ValueError:
                    row[name] = math.nan
            rows.append(row)
    return rows


def finite_max(rows, key):
    vals = [r[key] for r in rows if math.isfinite(r[key])]
    return max(vals) if vals else None


def summarize(name, path, rows):
    last = rows[-1]
    numeric = {k: last[k] for k in ("time", "dt", "C_norm2", "H_norm2", "M_norm2", "Z_norm2", "Theta_norm", "max_abs_K", "nmb_total", "maxAbsKret", "maxRefLev", "cycle")}
    return {
        "run": name,
        "history_path": str(path.relative_to(ROOT)),
        "rows": len(rows),
        "first_time_M": rows[0]["time"],
        "last": numeric,
        "max_C_norm2": finite_max(rows, "C_norm2"),
        "max_abs_K": finite_max(rows, "max_abs_K"),
        "max_level_seen": int(max(finite_max(rows, "maxRefLev") or 0, 0)),
        "max_meshblocks_seen": int(max(finite_max(rows, "nmb_total") or 0, 0)),
    }


def write_csv(name, rows):
    out = ROOT / f"{name}_history.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HST_NAMES)
        w.writeheader()
        w.writerows(rows)


def nearest(rows, t):
    return min(rows, key=lambda r: abs(r["time"] - t))


def main():
    parsed = {name: read_hst(path) for name, path in RUNS.items()}
    for name, rows in parsed.items():
        write_csv(name, rows)
        (ROOT / f"{name}_summary.json").write_text(json.dumps(summarize(name, RUNS[name], rows), indent=2, sort_keys=True) + "\n")

    times = [0.0, 1.0, 2.0, 2.45, 3.0, 3.9, 5.0, 8.0, 10.0, 10.2]
    with (ROOT / "comparison_common_time.csv").open("w", newline="") as f:
        fields = ["requested_time", "run", "time", "dt", "C_norm2", "H_norm2", "M_norm2", "Z_norm2", "max_abs_K", "nmb_total", "maxRefLev", "cycle"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in times:
            for name, rows in parsed.items():
                r = nearest(rows, t)
                w.writerow({"requested_time": t, "run": name, **{k: r[k] for k in fields[2:]}})

    with (ROOT / "comparison_proper_time.csv").open("w", newline="") as f:
        fields = ["run", "time", "dt", "C_norm2", "H_norm2", "M_norm2", "Z_norm2", "Theta_norm", "max_abs_K", "nmb_total", "maxRefLev", "cycle"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, rows in parsed.items():
            for r in rows[::max(1, len(rows)//400)]:
                w.writerow({"run": name, **{k: r[k] for k in fields[1:]}})

    (ROOT / "arm_gamma_o2_full_summary.json").write_text(json.dumps({
        "run": "arm_gamma_o2_full", "status": "NOT_RUN", "reason": "U2-short failed before the known crossing at t=10.5357421875M; conditional full run was not authorized."
    }, indent=2) + "\n")

    (ROOT / "advection_operator_audit.csv").write_text(
        "field_family,zero_shift_arm,gamma_o2_arm,scope\n"
        "scalar_and_tensor_explicit_advection,zero,order2,only beta^j d_j dispatch changes\n"
        "geometric_beta_derivatives,order6,order6,unchanged\n"
        "gamma_driver_sources,disabled,unchanged,zero-shift prescribed beta\n"
        "telegraph_lapse_flux,B retained,B retained,not a shift-driver variable\n"
    )
    (ROOT / "anomaly_causal_trace.csv").write_text(
        "run,phase,cycle,time_M,writer_or_gate,observation,interpretation\n"
        "arm_zero_shift,pre_update_RK1,1770,3.97265625,chi_provenance,candidate chi nonpositive at gid44 and beta transport exactly zero,zero-shift failure after replay under-resolution\n"
        "arm_gamma_o2_short,boundary_prolongation,5037,10.23750,strict chi parent gate,240 invalid parent stencils at gid33,short O2 transport does not avoid crossing\n"
        "arm_zero_shift_native,native_AMR_timeout,2954,2.452734,orchestration,dt=5.722046e-7 and repeated native AMR,diagnostic stopped by bounded wall-time; no qualification\n"
    )
    (ROOT / "first_high_k_anomaly.json").write_text(json.dumps({
        "status": "inherited_diagnostic_evidence",
        "source": "u2_short_v5_pre_evolution_failure.md and authenticated preceding phase diagnostics",
        "observed": {"gid35_60": "radial beta*d_rho chi O6 about -2870; no-advection about +0.851; O2 shadow about +58", "isolated_active_chi_spike": "about 22"},
        "interpretation": "transport/stencil sensitivity is evidence of a localized high-frequency mechanism, not proof of a source defect"
    }, indent=2) + "\n")

    (ROOT / "comparison_notes.json").write_text(json.dumps({
        "history_measure": "The Cartoon history reduction uses the axisymmetric ring measure; no fictitious collapsed-y normalization correction is inferred here.",
        "z1_replay": "Replay is under-resolved relative to native shadow requests near failure, so Z1 is a control of prescribed zero shift, not a convergence result.",
        "z2_native": "Native zero-shift run was bounded by wall time and is not a completed t=10.60 comparison.",
        "evidence_vs_inference": "All causal statements in REPORT.md label observation, inference, or hypothesis explicitly."
    }, indent=2) + "\n")

    # These are compact history plots only; no binary field data are implied.
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for name, rows in parsed.items():
            ax.plot([r["time"] for r in rows], [max(r["C_norm2"], 1e-300) for r in rows], label=name)
        ax.set_yscale("log")
        ax.set_xlabel("coordinate time M")
        ax.set_ylabel("C norm (history)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(ROOT / "constraints_history_comparison.png", dpi=160)
        plt.close(fig)

        fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
        for name, rows in parsed.items():
            ax1.plot([r["time"] for r in rows], [r["maxRefLev"] for r in rows], label=f"{name}: level")
        ax1.set_xlabel("coordinate time M")
        ax1.set_ylabel("maximum refinement level")
        ax1.grid(True, alpha=0.25)
        ax2 = ax1.twinx()
        for name, rows in parsed.items():
            ax2.plot([r["time"] for r in rows], [r["dt"] for r in rows], linestyle="--", alpha=0.65, label=f"{name}: dt")
        ax2.set_yscale("log")
        ax2.set_ylabel("dt M (dashed)")
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(ROOT / "topology_timestep_comparison.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        (ROOT / "plot_generation.status").write_text(f"plots unavailable: {exc}\n")

    # Self-excluding local manifest plus authenticated remote provenance.
    remote = {
        "z1": {"job": "57251896", "qos": "gpu_shared_interactive", "root": "/pscratch/sd/h/hzhu/axisymmetric-cartoon-brill-shift-controls-sourcecompat-v3-20260818", "terminal_manifest_sha256": "45f7fc4c8710db270bc1d1adcb55853ba9ca43f3a5a65b6ce664d6de7c83d15e", "run_log_sha256": "faba96361e1a676bfcb6e87764dede3ee73360552cef175c1acdad500a28db9", "executable_sha256": "d1d17ca2ec96428ce67d597e3fcef4cf7d6026ceef4bff09243817983e280292"},
        "u2_short": {"job": "57254459", "qos": "gpu_shared_interactive", "root": "/pscratch/sd/h/hzhu/axisymmetric-cartoon-brill-shift-controls-sourcecompat-v6-20260818", "run_log_sha256": "67f635dea064332f05ed51f0d102f677c61c4da960c5ab897c80f6c2f806d8f9", "hst_sha256": "e57f1c62fef022bd54134c25a184da73d25589f9396f181420c3c54acd100bf9", "command_sha256": "ccce2be4956504fcd24c941ec3471594f9600da840b3f8915b8c9c12fc67c13f", "executable_sha256": "54b01b70cd324b7e0af604c8f931b75b737e567cdab4a6aabf2493c15b688503"},
        "z2_native": {"job": "57255235", "qos": "gpu_shared_interactive", "root": "/pscratch/sd/h/hzhu/axisymmetric-cartoon-brill-shift-controls-sourcecompat-v6-20260818", "run_log_sha256": "8364b6375dc8bf2d49a5de5a5b451d4ebb85161650bd638c6fec7e2e3f79f39c", "hst_sha256": "a4207ad10d3fecfecb6fa99a1a988c7c714403e44ff8596b46bd109151261bc9", "terminal_sha256": "42a584179662fbf74fb381e81b52bb70e2dd841553275e15c955130d75875c8e", "executable_sha256": "54b01b70cd324b7e0af604c8f931b75b737e567cdab4a6aabf2493c15b688503", "sacct": "TIMEOUT 00:35:23; python3 CANCELLED 0:15"},
    }
    files = {}
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and p.name not in {"evidence_manifest.json", "SHA256SUMS", "SHA256SUMS.sha256"}:
            files[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {"schema": "brill-zero-shift-advection-controls-evidence-v1", "source": {"branch": "codex/brill-zero-shift-advection-controls-20260818", "commit": "1c95db8a2adc743672b49a525c21c4f762f35223", "tree": "5f343d0e19bc47fa5cfcf199c342885fde14154b", "kokkos": "6739bc623081648af9e752b616d9671527922cbf"}, "remote_runs": remote, "not_executed": {"diss_0p5_job": {"job": "57102293", "status": "revoked_before_allocation", "gpu": False, "science_data": False}, "u2_full": {"status": "not_authorized_after_u2_short_failure"}}, "local_files_sha256": files, "excluded_remote_binary_data": "Restart and field binary dumps remain at authenticated remote roots and are not copied into this commit."}
    (ROOT / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    lines = [f"{digest}  {name}" for name, digest in sorted(files.items())]
    lines.append(f"{hashlib.sha256((ROOT / 'evidence_manifest.json').read_bytes()).hexdigest()}  evidence_manifest.json")
    (ROOT / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    (ROOT / "SHA256SUMS.sha256").write_text(f"{hashlib.sha256((ROOT / 'SHA256SUMS').read_bytes()).hexdigest()}  SHA256SUMS\n")


if __name__ == "__main__":
    main()
