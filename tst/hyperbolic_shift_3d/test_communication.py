#!/usr/bin/env python3
"""Bounded native communication correctness/CLI checks; no performance claims."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import tempfile


def validate(report):
    assert report["status"] == "passed" and report["real_bytes"] == 8
    ranks = report["mpi_size"]
    assert [r["rank"] for r in report["ranks"]] == list(range(ranks))
    assert all(r["shared_node_ranks"] == ranks for r in report["ranks"])
    layouts = {}
    for item in report["layouts"]:
        n, kind = item["n"], item["kind"]
        points = 19*n**3 if kind == "volume" else 6*n*n
        assert item["points"] == points and item["global_values"] == 31*points
        assert item["global_receive_bytes"] == 8*31*points
        assert item["counts"] == [31*(points*(r+1)//ranks-points*r//ranks)
                                   for r in range(ranks)]
        assert item["displacements"] == [31*(points*r//ranks) for r in range(ranks)]
        layouts[n, kind] = item
    if report["layout_only"]:
        assert not report["groups"]
        return
    for group in report["groups"]:
        workload, mode = group["workload"], group["mode"]
        calls = group["call_sequence"]
        expected = {"volume": [0], "surface": [1],
                    "rk_mix": [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]}
        assert calls == expected[workload]
        repeats = report["mix_repeats"] if workload == "rk_mix" else report["repeats"]
        assert group["repeats"] == repeats and len(group["samples"]) == 3*repeats
        sample_orders = []
        for round_ in range(repeats):
            batch = group["samples"][3*round_:3*round_+3]
            assert [s["round"] for s in batch] == [round_]*3
            assert [s["order_position"] for s in batch] == [0, 1, 2]
            assert set(s["route"] for s in batch) == {"device", "host", "staged"}
            sample_orders.append(tuple(s["route"] for s in batch))
        if repeats > 1:
            assert sample_orders[0] == tuple(reversed(sample_orders[1]))
        if repeats >= 6:
            assert len(set(sample_orders[:6])) == 6
        for sample in group["samples"]:
            assert [t["rank"] for t in sample["rank_timings"]] == list(range(ranks))
            expected_checked = sum(layouts[group["n"], kind]["global_values"]
                                   for edge, kind in [(0, "volume"), (1, "surface")]
                                   if edge in calls)
            if sample["route"] == "staged":
                expected_checked *= 2
            for timing in sample["rank_timings"]:
                assert timing["mismatches"] == 0 and timing["checked_values"] == expected_checked
                assert all(math.isfinite(v) and v >= 0 for v in timing.values())
                assert math.isclose(timing["mpi_seconds"],
                                    timing["volume_mpi_seconds"]+timing["surface_mpi_seconds"],
                                    rel_tol=1e-12, abs_tol=1e-12)
                pieces = sum(timing[k] for k in
                             ["mpi_seconds", "d2h_seconds", "h2d_seconds", "completion_fence_seconds"])
                assert timing["operation_seconds"]+1e-12 >= pieces
                assert timing["sequence_seconds"]+1e-12 >= (
                    timing["operation_seconds"]+timing["barrier_seconds"]+timing["producer_fence_seconds"])
                if mode == "no_barrier":
                    assert timing["barrier_seconds"] == 0
                if sample["route"] != "staged":
                    assert timing["d2h_seconds"] == timing["h2d_seconds"] == 0
                if sample["route"] != "device":
                    assert timing["completion_fence_seconds"] == 0
            for name, value in sample["maximum_over_ranks"].items():
                assert value == max(t[name] for t in sample["rank_timings"])
        for route in ["device", "host", "staged"]:
            for name in ["mpi_seconds", "operation_seconds", "sequence_seconds"]:
                maxima = [s["maximum_over_ranks"][name] for s in group["samples"] if s["route"] == route]
                summary = group["max_over_ranks_then_quantile"][route][name]
                assert math.isclose(summary["median"], statistics.median(maxima), rel_tol=1e-14)
                assert summary["minimum"] == min(maxima) and summary["maximum"] == max(maxima)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--mpiexec", type=Path)
    parser.add_argument("--mpi-arg", action="append", default=[])
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--n", action="append", type=int)
    parser.add_argument("--cli-tests", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    options = parser.parse_args()
    binary = options.binary.resolve()
    prefix = ([str(options.mpiexec), *options.mpi_arg, "-n", str(options.ranks)]
              if options.mpiexec else [])
    assert options.mpiexec or options.ranks == 1
    temporary = tempfile.TemporaryDirectory(prefix="communication-test-")
    directory = options.output_dir or Path(temporary.name)
    directory.mkdir(parents=True, exist_ok=True)
    assert not any(directory.iterdir()), "test evidence directory must be empty"
    records = []

    def run(name, args, success=True):
        command = [*prefix, str(binary), *args]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=180)
        (directory/f"{name}.log").write_text(result.stdout)
        records.append({"name": name, "command": command, "returncode": result.returncode,
                        "expected_success": success})
        assert (result.returncode == 0) == success, (name, result.stdout)
        return result

    run("help", ["--help"])
    layout = directory/"layout.json"
    run("layout", ["--output", str(layout), "--layout-only", "--expect-ranks", str(options.ranks)])
    validate(json.loads(layout.read_text()))
    measured = directory/"measured.json"
    grids = options.n or [16]
    args = ["--output", str(measured), "--repeats", "2", "--warmup", "1",
            "--mix-repeats", "1", "--expect-ranks", str(options.ranks)]
    for n in grids:
        args += ["--n", str(n)]
    run("measured", args)
    report = json.loads(measured.read_text())
    validate(report)
    assert len(report["groups"]) == 6*len(grids)
    if options.cli_tests:
        run("overwrite", ["--output", str(layout), "--layout-only"], success=False)
        invalid = [
            [], ["--repeats", "0"], ["--repeats", "201"], ["--repeats", "1.5"],
            ["--repeats", "-1"], ["--warmup", "21"], ["--mix-repeats", "21"],
            ["--n", "18"], ["--n", "16", "--n", "16"], ["--n", "999999999999999999999"],
            ["--mode", "unknown"], ["--unknown"], ["--repeats"],
            ["--expect-ranks", str(options.ranks+1)], ["--expect-backend", "nonexistent"],
            ["--repeats", "1", "--repeats", "2"],
        ]
        for i, args in enumerate(invalid):
            output = directory/f"invalid-{i}.json"
            # The first case specifically checks the missing required output argument.
            command = args if i == 0 else ["--output", str(output), *args]
            run(f"invalid-{i}", command, success=False)
            assert not output.exists()
    files = [{"path": p.name, "bytes": p.stat().st_size,
              "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(directory.iterdir()) if p.is_file()]
    summary = {"status": "passed", "scope": "Local communication correctness and CLI validation; no GPU performance inference.",
               "backend": report["backend"], "mpi_size": report["mpi_size"],
               "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
               "commands": records, "artifacts": files}
    (directory/"verification.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"status": "passed", "backend": report["backend"], "ranks": options.ranks,
                      "commands": len(records), "evidence": str(directory)}))


if __name__ == "__main__":
    main()
