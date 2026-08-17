#!/usr/bin/env python3
"""Same-state O2/O4/O6 constraint audit for a zero-PDE Cartoon AMR event."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cartoon_amr_jump_analyze as base
import cartoon_amr_transfer_compare as comparison


SCHEMA = "athenak_z4c_amr_derivative_order_audit_v1"
ORDERS = ("O2", "O4", "O6")
FILES = {"O2": "constraints_o2.bin", "O4": "constraints_o4.bin",
         "O6": "constraints_o6.bin"}
PAIRS = (("O2", "O4"), ("O4", "O6"), ("O2", "O6"))


class AuditError(RuntimeError):
    pass


def strict_dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def load_named_views(event_roots: list[Path], phase: str, filename: str,
                     children: set[int]) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    metadata_by_gid: dict[int, dict[str, Any]] = {}
    for event in event_roots:
        root = event / phase
        metadata = base.strict_load(root / "phase.json")
        if metadata.get("derivative_order_audit") is not True:
            raise AuditError(f"derivative-order audit is not authenticated in {root}")
        shape = metadata.get("constraint_shape")
        if not isinstance(shape, list) or len(shape) != 5 or not all(
                isinstance(item, int) and item > 0 for item in shape):
            raise AuditError(f"invalid constraint shape in {root}")
        path = root / filename
        if not path.is_file() or path.stat().st_size != math.prod(shape) * 8:
            raise AuditError(f"missing or malformed derivative-order file {path}")
        view = np.fromfile(path, dtype="<f8").reshape(shape)
        if not np.all(np.isfinite(view)):
            raise AuditError(f"nonfinite derivative-order constraints in {path}")
        rows = base.read_csv(root / "topology.csv")
        if len(rows) != shape[0]:
            raise AuditError(f"topology/view mismatch in {root}")
        for row in rows:
            gid, local_m = int(row["gid"]), int(row["local_m"])
            if gid in result:
                raise AuditError(f"duplicate ownership for gid {gid}")
            if int(row["owner_rank"]) != int(metadata["rank"]):
                raise AuditError(f"owner/rank mismatch for gid {gid}")
            result[gid] = view[local_m]
            metadata_by_gid[gid] = metadata
    if not children.issubset(result):
        raise AuditError("derivative-order files omit selected refined children")
    return {gid: base.active_slice(result[gid], metadata_by_gid[gid])
            for gid in sorted(children)}


def scalar_field(constraints: np.ndarray, family: str) -> np.ndarray:
    values = constraints[comparison.FAMILIES[family]]
    return values if family == "H" else np.sqrt(np.maximum(values, 0.0))


def pairwise_disagreement(
        fields: dict[str, dict[int, np.ndarray]], adm: dict[int, np.ndarray],
        topology: dict[int, dict[str, str]], children: set[int]) -> tuple[
            list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for low, high in PAIRS:
        for family in comparison.FAMILIES:
            accum = {
                (measure, region): {"integral": 0.0, "volume": 0.0,
                                    "cell_count": 0, "max_abs": -math.inf,
                                    "max_rho": None, "max_z": None}
                for measure in comparison.MEASURES
                for region in comparison.REGIONS + ["GLOBAL"]
            }
            for gid in sorted(children):
                left = scalar_field(fields[low][gid], family)
                right = scalar_field(fields[high][gid], family)
                delta = left - right
                metric, topo = adm[gid], topology[gid]
                ny, nx = delta.shape
                dx1 = (float(topo["x1max"]) - float(topo["x1min"])) / nx
                dx2 = (float(topo["x2max"]) - float(topo["x2min"])) / ny
                detg = base.determinant(metric)
                for j in range(ny):
                    z = float(topo["x2min"]) + (j + 0.5) * dx2
                    for i in range(nx):
                        rho = float(topo["x1min"]) + (i + 0.5) * dx1
                        region = comparison.region_for_cell(
                            gid, i, j, nx, ny, topology)
                        value = float(delta[j, i])
                        coordinate = 2.0 * math.pi * rho * dx1 * dx2
                        weights = {
                            "coordinate_ring": coordinate,
                            "proper_ring": coordinate * math.sqrt(float(detg[j, i])),
                        }
                        for measure, weight in weights.items():
                            for target_region in (region, "GLOBAL"):
                                target = accum[(measure, target_region)]
                                target["integral"] += value * value * weight
                                target["volume"] += weight
                                target["cell_count"] += 1
                                if abs(value) > target["max_abs"]:
                                    target["max_abs"] = abs(value)
                                    target["max_rho"], target["max_z"] = rho, z
                        if (low, high) == ("O2", "O6"):
                            cells.append({"family": family, "gid": gid, "i": i,
                                          "j": j, "rho": rho, "z": z,
                                          "region": region, "difference": value,
                                          "absolute_difference": abs(value)})
            for (measure, region), values in accum.items():
                if values["cell_count"] == 0:
                    if values["volume"] != 0.0 or values["integral"] != 0.0:
                        raise AuditError("inconsistent empty pairwise region")
                    rms, maximum = None, None
                else:
                    if values["volume"] <= 0.0:
                        raise AuditError("populated pairwise region has invalid volume")
                    rms = math.sqrt(values["integral"] / values["volume"])
                    maximum = values["max_abs"]
                rows.append({"low_order": low, "high_order": high,
                             "family": family, "measure": measure,
                             "region": region,
                             "l2": math.sqrt(values["integral"]),
                             "rms": rms, **values, "max_abs": maximum})
    return rows, cells


def spectra(fields: dict[str, dict[int, np.ndarray]],
            children: set[int]) -> list[dict[str, Any]]:
    accumulated: dict[tuple[str, str, int], list[float]] = {}
    for order in ORDERS:
        for family in comparison.FAMILIES:
            for gid in sorted(children):
                values = scalar_field(fields[order][gid], family)
                fluctuation = values - float(np.mean(values))
                transform = np.fft.fft2(fluctuation)
                power = np.abs(transform) ** 2 / float(values.size ** 2)
                ky = np.fft.fftfreq(values.shape[0]) * values.shape[0]
                kx = np.fft.fftfreq(values.shape[1]) * values.shape[1]
                for j, mode_y in enumerate(ky):
                    for i, mode_x in enumerate(kx):
                        mode = int(math.floor(math.hypot(mode_x, mode_y)))
                        accumulated.setdefault((order, family, mode), []).append(
                            float(power[j, i]))
    return [{"order": order, "family": family, "mode_bin": mode,
             "sample_count": len(values), "power_sum": sum(values),
             "power_mean": sum(values) / len(values)}
            for (order, family, mode), values in sorted(accumulated.items())]


def plots(budgets: list[dict[str, Any]], spectra_rows: list[dict[str, Any]],
          cells: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for axis, family in zip(axes, comparison.FAMILIES):
        for order in ORDERS:
            rows = [row for row in budgets if row["order"] == order and
                    row["family"] == family and row["measure"] == "proper_ring" and
                    row["region"] in comparison.REGIONS]
            axis.plot(comparison.REGIONS, [row["integral"] for row in rows], "o-",
                      label=order)
        axis.set_yscale("log")
        axis.set_title(f"{family} proper-ring regional integral")
        axis.tick_params(axis="x", rotation=25)
        axis.legend()
    fig.savefig(output / "derivative_order_regional_integrals.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, family in zip(axes, comparison.FAMILIES):
        for order in ORDERS:
            rows = [row for row in spectra_rows if row["order"] == order and
                    row["family"] == family and row["mode_bin"] > 0]
            axis.semilogy([row["mode_bin"] for row in rows],
                          [row["power_sum"] for row in rows], label=order)
        axis.set_title(f"{family} block-local fluctuation spectrum")
        axis.set_xlabel("integer mode-radius bin")
        axis.set_ylabel("summed power")
        axis.legend()
    fig.savefig(output / "derivative_order_spectra.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    for axis, family in zip(axes, comparison.FAMILIES):
        rows = [row for row in cells if row["family"] == family]
        values = np.asarray([row["difference"] for row in rows])
        limit = float(np.max(np.abs(values)))
        scatter = axis.scatter([row["rho"] for row in rows],
                               [row["z"] for row in rows], c=values, s=3,
                               cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(f"O2 - O6 {family}")
        axis.set_xlabel(r"$\rho/M$")
        axis.set_ylabel(r"$z/M$")
        axis.set_aspect("equal")
        fig.colorbar(scatter, ax=axis)
    fig.savefig(output / "derivative_order_o2_minus_o6_maps.png", dpi=180)
    plt.close(fig)


def analyze(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    events, schema, proposal = comparison.event_roots(
        args.diagnostic_root.resolve(), args.ranks, args.level_before,
        args.level_after, args.expected_cycle)
    if schema.get("amr_transfer") != "high_order" or \
            schema.get("derivative_order_audit") is not True:
        raise AuditError("diagnostic root is not the requested high-order audit arm")
    provenance = comparison.validate_provenance(
        args.provenance.resolve(), args.diagnostic_root.resolve(),
        "high_order", args.ranks)
    children = comparison.selected_children(proposal)
    phase = "t5_00_ADM_OR_CONSTRAINT_RECOMPUTATION"
    topology = base.phase_topology(events, base.PHASES[-1])
    adm = comparison.active_views(events, phase, "adm", children)
    production = comparison.active_views(events, phase, "constraints", children)
    fields = {order: load_named_views(events, phase, filename, children)
              for order, filename in FILES.items()}
    if any(not np.array_equal(production[gid], fields["O6"][gid])
           for gid in children):
        raise AuditError("diagnostic O6 constraints differ from production T5 bytes")

    budget_rows: list[dict[str, Any]] = []
    for order in ORDERS:
        rows, _ = comparison.budgets(
            fields[order], adm, topology, children, order, "T5")
        for row in rows:
            row["order"] = row.pop("arm")
        budget_rows.extend(rows)
    disagreement_rows, cells = pairwise_disagreement(
        fields, adm, topology, children)
    spectra_rows = spectra(fields, children)

    global_proper = {
        order: {family: next(row["integral"] for row in budget_rows
                             if row["order"] == order and row["family"] == family and
                             row["measure"] == "proper_ring" and
                             row["region"] == "GLOBAL")
                for family in comparison.FAMILIES}
        for order in ORDERS
    }
    ratios = {
        family: {f"{low}_over_{high}":
                 global_proper[low][family] / global_proper[high][family]
                 for low, high in PAIRS}
        for family in comparison.FAMILIES
    }
    verdict = {
        "schema": SCHEMA,
        "qualification_claim": False,
        "event_cycle": args.expected_cycle,
        "level_before": args.level_before,
        "level_after": args.level_after,
        "refined_child_count": len(children),
        "production_o6_byte_exact": True,
        "global_proper_ring_integrals": global_proper,
        "global_integral_ratios": ratios,
        "correctable_operator_seam_isolated": False,
        "disposition": "inconclusive_parent_resolution_or_derivative_sensitivity",
        "provenance": provenance,
        "limitations": [
            "All three constraint fields were recomputed from one byte-identical T5 "
            "evolved state; no PDE update or production derivative was changed.",
            "Derivative-order disagreement characterizes sensitivity but does not by "
            "itself identify a defective source operator or establish convergence.",
            "Block-local spectra are descriptive fluctuation spectra on the selected "
            "level-3 child MeshBlocks, not a global Fourier decomposition.",
        ],
    }

    for name, rows in (("derivative_order_regional_budgets.csv", budget_rows),
                       ("derivative_order_pairwise_disagreement.csv",
                        disagreement_rows),
                       ("derivative_order_spectra.csv", spectra_rows),
                       ("derivative_order_o2_minus_o6_cells.csv", cells)):
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    strict_dump(output / "verdict.json", verdict)
    plots(budget_rows, spectra_rows, cells, output)
    report = (
        "# Same-state Cartoon Z4c derivative-order audit\n\n"
        f"Disposition: `{verdict['disposition']}`; `qualification_claim=false`.\n\n"
        "O2, O4, and O6 constraints were evaluated from the same accepted T5 evolved "
        "state. The independently recomputed O6 bytes equal the production T5 "
        "constraint bytes. Regional proper-ring/coordinate-ring budgets, extrema, "
        "pairwise disagreements, and block-local spectra are retained. No correctable "
        "operator seam is isolated from order sensitivity alone.\n")
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    products = sorted(path for path in output.iterdir() if path.is_file())
    with (output / "SHA256SUMS").open("w", encoding="utf-8") as stream:
        for path in products:
            stream.write(f"{base.sha256(path)}  {path.name}\n")


def self_test() -> None:
    constraints = np.zeros((3, 1, 2, 2), dtype=np.float64)
    constraints[0, 0] = 4.0
    constraints[1, 0] = [[-1.0, 1.0], [-2.0, 2.0]]
    constraints[2, 0] = 9.0
    if not np.array_equal(scalar_field(constraints, "C"), np.full((1, 2, 2), 2.0)):
        raise AuditError("C magnitude conversion failed")
    if not np.array_equal(scalar_field(constraints, "M"), np.full((1, 2, 2), 3.0)):
        raise AuditError("M magnitude conversion failed")
    if not np.array_equal(scalar_field(constraints, "H"), constraints[1]):
        raise AuditError("H signed field conversion failed")
    print("cartoon_amr_derivative_order_audit self-test: PASS")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--diagnostic-root", type=Path, required=True)
    analyze_parser.add_argument("--provenance", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--ranks", type=int, required=True)
    analyze_parser.add_argument("--level-before", type=int, required=True)
    analyze_parser.add_argument("--level-after", type=int, required=True)
    analyze_parser.add_argument("--expected-cycle", type=int, required=True)
    sub.add_parser("self-test")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "self-test":
        self_test()
    else:
        analyze(args)


if __name__ == "__main__":
    main()
