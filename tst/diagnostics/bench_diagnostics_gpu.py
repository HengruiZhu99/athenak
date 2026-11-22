"""Paired CUDA timings and data comparisons on a single allocated GPU."""

import argparse
import json
from pathlib import Path
import os
import re
import statistics
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Kokkos labels identify the production kernel. Synchronize just that kernel
# during separate profiling runs. Normal throughput runs have no tool attached.
PROBE = r"""
#include <cuda_runtime_api.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
static std::chrono::steady_clock::time_point start;
extern "C" void kokkosp_begin_parallel_for(const char* name,uint32_t,uint64_t* id) {
  *id = !std::strcmp(name,"torque");
  if (*id) { cudaDeviceSynchronize(); start=std::chrono::steady_clock::now(); }
}
extern "C" void kokkosp_end_parallel_for(uint64_t id) {
  if (id) {
    cudaDeviceSynchronize();
    std::printf("TORQUE_US %.6f\n",std::chrono::duration<double,std::micro>(
        std::chrono::steady_clock::now()-start).count());
  }
}
"""


def check_torque_output(candidate, path, base, launcher=()):
    import sys

    sys.path.insert(0, str(ROOT / "vis/python"))
    import bin_convert

    # A second run compares all components against the full-volume output on
    # the same hierarchy, including an intersecting line and ghost-zone mask.
    path.mkdir()
    selections = [
        ("volume", ""),
        ("line", "slice_x1=0.125\nslice_x2=0.25"),
        ("block", "gid=0\nghost_zones=true"),
    ]
    text = base.split("<output2>")[0].replace("nlim = 8", "nlim = 1")
    for n, (name, options) in enumerate(selections, 2):
        text += f"\n<output{n}>\nfile_type=bin\nvariable=torque\nid={name}\ndcycle=1\n{options}\n"
    (path / "input").write_text(text)
    with (path / "run.log").open("w") as log:
        subprocess.run(
            [*launcher, candidate, "-i", path / "input", "-d", path],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

    def dump(name):
        return bin_convert.read_binary(str(sorted((path / "bin").glob(f"*.{name}.*.bin"))[-1]))

    full = dump("volume")
    indices = {tuple(loc): i for i, loc in enumerate(full["mb_logical"])}
    for name in ("line", "block"):
        sample = dump(name)
        for m, loc in enumerate(sample["mb_logical"]):
            idx = indices[tuple(loc)]
            bounds = sample["mb_index"][m].reshape(3, 2)[::-1]
            for key in full["mb_data"]:
                a = full["mb_data"][key][idx]
                b = sample["mb_data"][key][m]
                expected = np.full(b.shape, np.nan, dtype=a.dtype)
                lower = np.maximum(bounds[:, 0], 0)
                upper = np.minimum(bounds[:, 1] + 1, a.shape)
                src = tuple(slice(lo, hi) for lo, hi in zip(lower, upper))
                dst = tuple(
                    slice(lo - offset, hi - offset)
                    for lo, hi, offset in zip(lower, upper, bounds[:, 0])
                )
                expected[dst] = a[src]
                np.testing.assert_array_equal(b, expected)


def benchmark(baseline, candidate, root):
    base = (ROOT / "tst/diagnostics/binary_shell.athinput").read_text().split("<output2>")[0]
    base = base.replace("flux_shell_budget = true", "flux_shell_budget = false")
    base = base.replace("flux_ks_radius1 = 1.4", "flux_ks_radius1 = 0")
    base = base.replace("flux_ks_radius2 = 1.2", "flux_ks_radius2 = 0")
    base = base.replace("<output1>", "<output1>\nuser_hist_only=true")
    base = base.replace("nlim = 2", "nlim = 8")
    base = base.replace("b_inf = 0", "b_inf = 0.002")
    base += "\n<output2>\nfile_type=bin\nvariable=torque\nslice_x3=0.25\ndcycle=1\n"
    src = root / "probe.cu"
    src.write_text(PROBE)
    lib = root / "probe.so"
    subprocess.run(["nvcc", "-shared", "-Xcompiler=-fPIC", src, "-o", lib], check=True)

    def measure(name, exe, rep, params, profile=False):
        path = root / f"{name}-{rep}"
        path.mkdir()
        (path / "input").write_text(params)
        env = os.environ.copy()
        if profile:
            env["KOKKOS_TOOLS_LIBS"] = str(lib)
        result = subprocess.run(
            [exe, "-i", path / "input", "-d", path], env=env, capture_output=True, text=True
        )
        (path / "run.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        return dict(
            name=name,
            rep=rep,
            seconds=float(re.search(r"cpu time used\s*=\s*([\d.eE+-]+)", result.stdout)[1]),
            torque_us=[float(x) for x in re.findall(r"TORQUE_US ([\d.]+)", result.stdout)],
        )

    records = []
    for rep in range(4):
        for name, exe in (
            (("baseline", baseline), ("candidate", candidate))
            if rep % 2 == 0
            else (("candidate", candidate), ("baseline", baseline))
        ):
            records.append(measure(name, exe, rep, base, rep == 3))
    summary = {}
    for name in ("baseline", "candidate"):
        series = [r for r in records if r["name"] == name]
        times = series[-1]["torque_us"][1:]
        assert len(times) >= 7, "Torque timing probe did not observe the production kernel"
        summary[name] = {
            "run_s": statistics.median(r["seconds"] for r in series[:3]),
            "torque_us": statistics.median(times),
        }
    # Common full-volume output and the incremental cost of all optional shell budgets.
    for name, exe in (("baseline", baseline), ("candidate", candidate)):
        full = measure(name + "-volume", exe, 0, base.replace("slice_x3=0.25", ""), True)
        records.append(full)
        summary[name]["full_torque_us"] = statistics.median(full["torque_us"][1:])
    extras = base.replace("flux_ks_radius1 = 0", "flux_ks_radius1 = 1.4").replace(
        "flux_ks_radius2 = 0", "flux_ks_radius2 = 1.2"
    )
    for rep in range(3):
        for name, params in (
            ("ks", extras),
            ("ks-shells", extras.replace("flux_shell_budget = false", "flux_shell_budget = true")),
        ):
            records.append(measure(name, candidate, rep, params))
    for name in ("ks", "ks-shells"):
        summary[name] = {
            "run_s": statistics.median(r["seconds"] for r in records if r["name"] == name)
        }
    # Compare complete saved spatial torque slices from the same cycle.
    import sys

    sys.path.insert(0, str(ROOT / "vis/python"))
    import bin_convert

    def read_slice(name):
        path = sorted((root / f"{name}-0" / "bin").glob("*.bin"))[-1]
        return bin_convert.read_binary(str(path))["mb_data"]

    reference = read_slice("baseline")
    latest = read_slice("candidate")
    assert len(reference) == 3 and set(reference) == set(latest)
    differences = {}
    for key in reference:
        x, y = np.asarray(reference[key]), np.asarray(latest[key])
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
        np.testing.assert_allclose(x, y, rtol=3e-6, atol=2e-10)
        differences[key] = float(np.max(np.abs(x - y)))
    check_torque_output(candidate, root / "selection", base)
    summary["torque_slice_max_abs_difference"] = differences
    (root / "performance.json").write_text(
        json.dumps(dict(runs=records, summary=summary), indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("baseline", "candidate", "output"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    benchmark(a.baseline.resolve(), a.candidate.resolve(), a.output.resolve())
