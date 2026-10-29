#!/usr/bin/env python3
"""Read-only replay of frozen scri interpolation; never loads/restarts an evolution.

Requires NumPy, the original raw payloads, and a configured local serial Kokkos
build. Output must be a new directory. Only small JSON/text outputs are retained;
synthetic control payloads and the audit executable live in a temporary folder.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
FROZEN = "113aaff783008204e1ff8cbf2fb8bb5f2af2204c"
MANIFEST = "3f367265e2e627aa770d0ae78be4fac912f4c0980a09471382cf9b295c677d7f"
BINARY = "6e7553b556dd9c4f8b56f81eb0f43c15cf0517a0b81776320a9a6c6d3c29ccef"
HEADERS = ["src/athena.hpp", "src/mesh/cubed_sphere/map.hpp",
           "src/mesh/cubed_sphere/composite.hpp",
           "src/mesh/cubed_sphere/composite_difference.hpp",
           "src/z4c/hyperboloidal/hyperbolic_composite_exchange.hpp",
           "src/z4c/hyperboloidal/conformal_constraints.hpp"]
CASES = [("original-n16", "stability-v1", "minkowski", 16),
         ("n24", "resolution-control-v2", "minkowski-n24", 24),
         ("n16-half-dt", "resolution-control-v2", "minkowski-n16-half-dt", 16)]
HEADER = struct.Struct("<16s4i3Q2d")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(command, timeout=410):
    start = time.monotonic()
    p = subprocess.run([str(x) for x in command], cwd=ROOT,
                       capture_output=True, text=True, timeout=timeout)
    record = dict(command=[str(x) for x in command], returncode=p.returncode,
                  seconds=time.monotonic() - start, stdout=p.stdout, stderr=p.stderr)
    if p.returncode:
        raise RuntimeError(json.dumps(record, indent=2))
    return record


def metric(fields):
    g = np.zeros(fields.shape[:-1] + (3, 3))
    for f, (i, j) in enumerate([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)], 1):
        g[..., i, j] = g[..., j, i] = fields[..., f] + (i == j)
    return g


def checkpoint_surface(path, n):
    with path.open("rb") as f:
        magic, size, nf, kind, admitted, cells, ns, steps, t, dt = HEADER.unpack(f.read(HEADER.size))
        if (magic.rstrip(b"\0"), size, nf, kind, admitted, cells, ns) != (
                b"HYP3D_MPI_V1", n, 31, 0, 0, 19*n**3, 31*6*n*n):
            raise ValueError("checkpoint must match unadmitted original Minkowski schema")
        if path.stat().st_size != HEADER.size + 16*(31*cells + ns):
            raise ValueError("checkpoint size mismatch")
        f.seek(HEADER.size + 16*31*cells)
        surface = np.fromfile(f, dtype="<f8", count=ns).reshape(-1, 31)
    return surface, dict(magic="HYP3D_MPI_V1", n=n, nf=nf, kind=kind,
                         admitted=admitted, steps=steps, accepted_time=t, dt=dt,
                         failed_stage_abscissa="not recorded", attempted_interval=[t, t+dt])


def interpolate(surface, query):
    ids = query["donor_surface_ids"]
    anchor = surface[ids[0]].copy()
    value = anchor.copy()
    for i, w in zip(ids, query["weights"]):
        value += w * (surface[i] - anchor)
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, default=ROOT/"output/hyperbolic-core-shell/aurora")
    ap.add_argument("--build-dir", type=Path, default=ROOT/"build-hyperbolic-mpi")
    ap.add_argument("--compiler", default="/usr/bin/c++")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    provenance = dict(frozen_numerical_commit=FROZEN, source_manifest_sha256=MANIFEST,
                      original_sycl_binary_sha256=BINARY, source_comparison=[], inputs=[])
    for relative in HEADERS:
        historical = subprocess.check_output(["git", "show", f"{FROZEN}:{relative}"], cwd=ROOT)
        if historical != (ROOT/relative).read_bytes():
            raise ValueError(f"source differs from frozen run: {relative}")
        provenance["source_comparison"].append(dict(path=relative, sha256=digest(ROOT/relative),
                                                    identical_to_frozen=True))
    for filename in ["run_audit.py", "scri_interpolation_audit.cpp"]:
        provenance[filename + "_sha256"] = digest(HERE/filename)
    build = args.build_dir.resolve()
    flags_path = build/"tst/hyperbolic_shift_3d/CMakeFiles/hyperbolic_shift_3d_mpi.dir/flags.make"
    flags = {}
    for line in flags_path.read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            flags[key] = shlex.split(value)
    provenance["flags_file"] = dict(path=str(flags_path), sha256=digest(flags_path))
    summaries = []
    with tempfile.TemporaryDirectory(prefix="scri-positivity-audit-") as tmpname:
        tmp = Path(tmpname)
        executable = tmp/"audit"
        command = ([args.compiler] + flags["CXX_DEFINES"] + flags["CXX_INCLUDES"] + flags["CXX_FLAGS"]
                   + [HERE/"scri_interpolation_audit.cpp", "-o", executable])
        for part in ["containers", "algorithms", "core", "simd"]:
            command += [build/f"kokkos/{part}/src/libkokkos{part}.a"]
        write(output/"build.json", run(command, timeout=120))
        # Controlled constant fields exercise exact rejection branches. They are
        # temporary arrays, not evolved data or modified scientific payloads.
        controls = []
        for name in ["positive_identity", "negative_chi", "indefinite_metric", "nonfinite_auxiliary"]:
            n = 6
            surface = np.zeros((6*n*n, 31), dtype="<f8")
            surface[:, 0] = 1
            if name == "negative_chi":
                surface[:, 0] = -1
            elif name == "indefinite_metric":
                surface[:, 1] = -2
            elif name == "nonfinite_auxiliary":
                surface[:, 30] = np.nan
            payload, result = tmp/(name+".bin"), tmp/(name+".json")
            with payload.open("wb") as f:
                f.seek(8*31*19*n**3)
                surface.tofile(f)
            execution = run([executable, n, payload, result])
            data = json.loads(result.read_text())
            expected = {k: 0 for k in data["rejection_counts"]}
            if name != "positive_identity":
                field = dict(negative_chi="nonpositive_chi", indefinite_metric="metric_invalid",
                             nonfinite_auxiliary="nonfinite_field")[name]
                expected[field] = data["unique_global_boundary_queries"]
            if data["rejection_counts"] != expected:
                raise AssertionError((name, data["rejection_counts"], expected))
            controls.append(dict(name=name, passed=True, counts=expected,
                                 seconds=execution["seconds"]))
        write(output/"controls.json", controls)
        for label, campaign, name, n in CASES:
            package = HERE.parent/campaign
            worker_file = package/"worker-status.json"
            worker = json.loads(worker_file.read_text())
            if worker["source_manifest_sha256"] != MANIFEST or set(worker["binaries"].values()) != {BINARY}:
                raise ValueError("worker source/binary binding mismatch")
            case = next(c for c in worker["cases"] if c["name"] == name)
            paths = {}
            for suffix in [".failed-stage.bin", ".checkpoint"]:
                relative = f"campaigns/{campaign}/output/{name}{suffix}"
                path = args.raw_root.resolve()/campaign/"output"/(name+suffix)
                binding = case["artifacts"].get(relative)
                authority = dict(kind="worker", path=str(worker_file.relative_to(ROOT)), sha256=digest(worker_file))
                if binding is None:
                    if (label, suffix) != ("original-n16", ".failed-stage.bin"):
                        raise ValueError("missing worker artifact")
                    supplement = package/"analysis/failed-stage-supplement.json"
                    binding = json.loads(supplement.read_text())
                    if binding["path"] != relative:
                        raise ValueError("supplement path mismatch")
                    authority = dict(kind="post-collection supplemental hash; not original worker-bound",
                                     path=str(supplement.relative_to(ROOT)), sha256=digest(supplement))
                actual = dict(sha256=digest(path), bytes=path.stat().st_size)
                if actual != {k: binding[k] for k in ["sha256", "bytes"]}:
                    raise ValueError(f"input hash/size mismatch: {path}")
                provenance["inputs"].append(dict(case=label, path=str(path), **actual, authority=authority))
                paths[suffix] = path
            result = output/(label+".json")
            write(output/(label+".execution.json"), run([executable, n, paths[".failed-stage.bin"], result]))
            data = json.loads(result.read_text())
            saved, header = checkpoint_surface(paths[".checkpoint"], n)
            raw = np.memmap(paths[".failed-stage.bin"], mode="r", dtype="<f8",
                            offset=8*31*19*n**3, shape=(6*n*n, 31))
            minimum_node_eigenvalue = float(np.linalg.eigvalsh(metric(raw)).min())
            details = []
            for query in data["rejected"]:
                weights = np.array(query["weights"])
                value = interpolate(raw, query)
                if not np.allclose(value, query["interpolated_fields"], rtol=0, atol=2e-13):
                    raise AssertionError("independent interpolation replay mismatch")
                previous = interpolate(saved, query)
                g = metric(previous)
                details.append(dict(query_id=query["query_id"], boundary_xyz=query["boundary_xyz"],
                    donor_block=query["donor_block"], donor_face=query["donor_face"], donor_local=query["donor_local"],
                    target_ghosts_block_i_j_k=query["targets"],
                    interpolated_check=query["check"],
                    eigenvalues=np.linalg.eigvalsh(np.array(query["check"]["metric"])).tolist(),
                    minimum_donor_eigenvalue=float(np.linalg.eigvalsh(metric(raw[query["donor_surface_ids"]])).min()),
                    donor_chi_range=[query["donor_field_min"][0],query["donor_field_max"][0]],
                    minimum_donor_determinant=query["donor_min_determinant"],
                    weights=dict(sum=float(weights.sum()), l1=float(abs(weights).sum()),
                                 minimum=float(weights.min()), maximum=float(weights.max())),
                    rollback_same_query=dict(chi=float(previous[0]), determinant=float(np.linalg.det(g)),
                                             eigenvalues=np.linalg.eigvalsh(g).tolist())))
            summaries.append(dict(case=label, header=header, nodal_surface=data["nodal_surface"],
                nodal_minimum_eigenvalue=minimum_node_eigenvalue,
                unique_global_boundary_queries=data["unique_global_boundary_queries"],
                rejected_exact_coordinate_queries=data["rejected_queries"],
                distinct_failed_directions_at_12_decimals=len({tuple(round(x,12) for x in q["boundary_xyz"]) for q in details}),
                rejection_counts=data["rejection_counts"], interpolant_ranges=data["interpolants"], rejected=details))
        for item in provenance["inputs"]:
            path = Path(item["path"])
            item["unchanged_after_replay"] = digest(path) == item["sha256"] and path.stat().st_size == item["bytes"]
            if not item["unchanged_after_replay"]:
                raise AssertionError("input changed during read-only replay")
    write(output/"provenance.json", provenance)
    write(output/"summary.json", dict(scope="Read-only rejection localization, not identification of the first growing mode",
        exact_failed_rk_substep_time_recorded=False, no_checkpoint_admission_or_restart=True,
        q_zero_for_invalid_metric="Geometry returns before constructing its inverse; q=0 is consequent, not independent",
        query_count_scope="global exact-coordinate deduplication; not sum of rank-local invalid counters",
        host_replay="Frozen stencil and Geometry on CPU binary64; not a bitwise SYCL replay",
        cases=summaries))
    print(json.dumps(dict(output=str(output), cases=len(summaries), controls_passed=len(controls))))


if __name__ == "__main__":
    main()
