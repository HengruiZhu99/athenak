#!/usr/bin/env python3
"""Recompute the bound, single-allocation communication comparison; no execution."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from validate_samples import validate

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROUTES = ("device", "host", "staged")
METRICS = ("operation_seconds", "mpi_seconds", "sequence_seconds", "producer_fence_seconds",
           "barrier_seconds", "d2h_seconds", "h2d_seconds", "completion_fence_seconds")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(name, data):
    (HERE/name).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def distribution(values):
    values = np.asarray(values, dtype=float)
    assert len(values) and np.isfinite(values).all()
    q = np.quantile(values, [0, .25, .5, .75, .95, 1])
    return dict(zip(("minimum", "q25", "median", "q75", "p95", "maximum"), map(float, q)))


def verify():
    compressed = (HERE/"communication.report.json.gz").read_bytes()
    raw = gzip.decompress(compressed)
    report = json.loads(raw)
    worker, binding, plan = (read(PARENT/name) for name in
                              ("worker-status.json", "binding.json", "plan.json"))
    case = next(c for c in worker["cases"] if c["name"] == "communication")
    command = next(c for c in plan["commands"] if c["name"] == "communication")
    identity = case["artifacts"][command["report"]]
    assert identity["sha256"] == sha(raw) and identity["bytes"] == len(raw)
    assert sha((PARENT/"plan.json").read_bytes()) == binding["plan_sha256"] == worker["plan_sha256"]
    assert binding["source_manifest_sha256"] == worker["source_manifest_sha256"]
    assert binding["binaries"] == worker["binaries"]
    assert case["status"] == "passed" and case["returncode"] == 0 and not case["timed_out"]
    assert case["storage_error"] is None and all(c["passed"] for c in case["checks"])
    assert worker["status"] == "passed"
    assert report["backend"] == "SYCL" and report["mpi_size"] == 12
    assert report["device_memory_space"] == "SYCLDeviceUSM"
    assert report["repeats"] == 24 and report["warmup"] == 3 and report["mix_repeats"] == 6
    assert report["mode"] == "both"
    # This helper is an exact retained copy of the helper in the frozen source.
    manifest = read(PARENT/"source-v10-non-kokkos-manifest.json")
    helper_key = "source-v10/tst/hyperbolic_shift_3d/test_communication.py"
    assert sha((HERE/"validate_samples.py").read_bytes()) == manifest[helper_key]
    validate(report)
    assert len(report["groups"]) == 18
    assert len({r["host"] for r in report["ranks"]}) == 1
    for rank in report["ranks"]:
        assert rank["environment"]["ZE_AFFINITY_MASK"] == f"{rank['rank']//2}.{rank['rank']%2}"
        assert rank["environment"]["ZE_FLAT_DEVICE_HIERARCHY"] == "COMPOSITE"
    checked = sum(t["checked_values"] for g in report["groups"] for s in g["samples"]
                  for t in s["rank_timings"])
    provenance = dict(
        report_raw_sha256=sha(raw), report_raw_bytes=len(raw),
        report_gzip_sha256=sha(compressed), report_gzip_bytes=len(compressed),
        worker_sha256=sha((PARENT/"worker-status.json").read_bytes()),
        binding_sha256=sha((PARENT/"binding.json").read_bytes()), plan_sha256=binding["plan_sha256"],
        source_manifest_sha256=binding["source_manifest_sha256"],
        executable=command["executable"], binary_sha256=binding["binaries"][command["executable"]],
        job_id=worker["job_id"], backend=report["backend"], memory_space=report["device_memory_space"],
        host_memory_space=report["host_memory_space"], host_pinned_allocation_claimed=False,
        mpi_library=report["mpi_library"], compiler=report["compiler"],
        kokkos_version=report["kokkos_version"], case_elapsed_seconds=case["elapsed_seconds"],
        ranks=report["ranks"], layout=report["layouts"], validated_values=int(checked),
        raw_sample_count=sum(len(g["samples"]) for g in report["groups"]),
        rank_sample_count=sum(len(s["rank_timings"]) for g in report["groups"] for s in g["samples"]),
        exhaustive_validation_mismatches=0, all_original_report_summaries_rechecked=True,
        analysis_sha256=sha(Path(__file__).read_bytes()),
        validation_helper_sha256=sha((HERE/"validate_samples.py").read_bytes()),
        numpy_version=np.__version__, matplotlib_version=matplotlib.__version__)
    return report, provenance


def summarize(group):
    result = dict(n=group["n"], workload=group["workload"], mode=group["mode"],
                  rounds=group["repeats"], call_sequence=group["call_sequence"], routes={})
    maxima = {}
    permutations = []
    for round_ in range(group["repeats"]):
        batch = sorted((s for s in group["samples"] if s["round"] == round_),
                       key=lambda s: s["order_position"])
        permutations.append(tuple(s["route"] for s in batch))
    assert len(set(permutations)) == 6
    assert all(permutations.count(p) == group["repeats"]//6 for p in set(permutations))
    result["permutations"] = [{"order": list(p), "count": permutations.count(p)}
                              for p in sorted(set(permutations))]
    for route in ROUTES:
        samples = [s for s in group["samples"] if s["route"] == route]
        fields = {metric: distribution([max(t[metric] for t in s["rank_timings"])
                                       for s in samples]) for metric in METRICS}
        for metric in ("operation_seconds", "mpi_seconds", "sequence_seconds"):
            for key, old in group["max_over_ranks_then_quantile"][route][metric].items():
                assert np.isclose(fields[metric][key], old, rtol=1e-12, atol=1e-14)
        maxima[route] = [max(t["operation_seconds"] for t in s["rank_timings"]) for s in samples]
        # Components on the same slowest-operation rank are retained without
        # adding independently maximized or independently medianed components.
        critical = [max(s["rank_timings"], key=lambda t: t["operation_seconds"]) for s in samples]
        result["routes"][route] = dict(
            maximum_over_ranks=fields,
            position_operation_medians=[
                statistics.median(max(t["operation_seconds"] for t in s["rank_timings"])
                                  for s in samples if s["order_position"] == position)
                for position in range(3)],
            per_rank_operation_medians=[
                statistics.median(s["rank_timings"][rank]["operation_seconds"] for s in samples)
                for rank in range(12)],
            critical_rank_component_means={metric: statistics.mean(t[metric] for t in critical)
                                           for metric in METRICS},
            critical_rank_counts={str(rank): sum(t["rank"] == rank for t in critical)
                                  for rank in range(12)})
    for candidate in ("host", "staged"):
        ratios = np.asarray(maxima[candidate])/np.asarray(maxima["device"])
        result[candidate+"_over_device_paired"] = dict(
            distribution=distribution(ratios), samples=list(map(float, ratios)),
            candidate_faster_rounds=int(np.count_nonzero(ratios < 1)),
            ratio_of_medians=statistics.median(maxima[candidate])/statistics.median(maxima["device"]))
    return result


def plot(groups):
    colors = dict(device="#0072B2", host="#555555", staged="#D55E00")
    labels = dict(device="Direct device", host="Host only (copies excluded)", staged="Explicit staging")
    titles = dict(volume="One volume gather", surface="One surface gather",
                  rk_mix="Four volume + fourteen surface gathers")
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7), constrained_layout=True)
    for row, mode in enumerate(("barrier", "no_barrier")):
        for col, workload in enumerate(("volume", "surface", "rk_mix")):
            ax = axes[row, col]
            subset = sorted((g for g in groups if g["mode"] == mode and g["workload"] == workload),
                            key=lambda g: g["n"])
            for route in ROUTES:
                stats = [g["routes"][route]["maximum_over_ranks"]["operation_seconds"] for g in subset]
                mid = np.array([s["median"] for s in stats])*1000
                low = mid-np.array([s["q25"] for s in stats])*1000
                high = np.array([s["q75"] for s in stats])*1000-mid
                ax.errorbar([g["n"] for g in subset], mid, yerr=[low, high], color=colors[route],
                            marker="o", markersize=4, capsize=3, linewidth=1.4, label=labels[route])
            if workload != "surface":
                ax.set_yscale("log")
            ax.set_xticks([16, 24, 32]); ax.set_xlim(14.5, 33.5)
            ax.set_xlabel("Cells per block edge, n")
            ax.set_ylabel("Operation time [ms]")
            ax.grid(alpha=.2, which="both")
            ax.set_title(titles[workload]+"\n"+("Barrier before operation" if mode == "barrier"
                                                else "No added barrier within group"), fontsize=10)
    handles, legends = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Aurora: direct device MPI remains the fastest mixed route\n"
                 "Median of per-sample maximum over 12 ranks; whiskers show the interquartile range",
                 fontsize=12)
    fig.savefig(HERE/"communication-comparison.png", dpi=180)
    fig.savefig(HERE/"communication-comparison.pdf",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    report, provenance = verify()
    groups = [summarize(g) for g in report["groups"]]
    write("provenance.json", provenance)
    write("analysis.json", dict(scope="One Aurora allocation; no evolution or inter-node throughput inference.",
                               recommendation="Keep existing direct device MPI unchanged.",
                               groups=groups))
    fields = ["n", "workload", "mode", "route", "rounds", "operation_median_ms",
              "operation_q25_ms", "operation_q75_ms", "operation_p95_ms", "mpi_median_ms",
              "sequence_median_ms", "staged_over_device_paired_median"]
    with (HERE/"summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for group in groups:
            for route in ROUTES:
                stats = group["routes"][route]["maximum_over_ranks"]
                writer.writerow(dict(n=group["n"], workload=group["workload"], mode=group["mode"],
                    route=route, rounds=group["rounds"],
                    operation_median_ms=stats["operation_seconds"]["median"]*1000,
                    operation_q25_ms=stats["operation_seconds"]["q25"]*1000,
                    operation_q75_ms=stats["operation_seconds"]["q75"]*1000,
                    operation_p95_ms=stats["operation_seconds"]["p95"]*1000,
                    mpi_median_ms=stats["mpi_seconds"]["median"]*1000,
                    sequence_median_ms=stats["sequence_seconds"]["median"]*1000,
                    staged_over_device_paired_median=group["staged_over_device_paired"]["distribution"]["median"]))
    lines = ["# Complete route table", "",
             "Times are milliseconds: median of each sample's maximum across twelve ranks. "
             "Device and staged include their required completion/copy costs; host excludes copies.", "",
             "| n | Workload | Barrier mode | Device | Host only | Staged | Paired staged/device median | Staged faster rounds |",
             "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for group in groups:
        med = [group["routes"][route]["maximum_over_ranks"]["operation_seconds"]["median"]*1000
               for route in ROUTES]
        ratio = group["staged_over_device_paired"]
        lines.append(f"| {group['n']} | {group['workload']} | {group['mode']} | "
                     + " | ".join(f"{value:.6g}" for value in med)
                     + f" | {ratio['distribution']['median']:.6g} | {ratio['candidate_faster_rounds']}/{group['rounds']} |")
    (HERE/"summary.md").write_text("\n".join(lines)+"\n")
    plot(groups)
    print(json.dumps(dict(report_sha256=provenance["report_raw_sha256"],
                          rank_samples=provenance["rank_sample_count"],
                          mismatches=provenance["exhaustive_validation_mismatches"],
                          groups=len(groups), recommendation="keep direct device MPI")))


if __name__ == "__main__":
    main()
