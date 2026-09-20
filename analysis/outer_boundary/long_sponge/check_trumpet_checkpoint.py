#!/usr/bin/env python3
"""Check a complete uniform M=R0=1 trumpet checkpoint, including ghost metrics.

Supports a serial run or a fixed contiguous MPI partition with multiple uniform blocks per rank.
This validates saved state, not stability or suitability for a stellar restart.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np

regression_path = os.environ.get("ATHENA_REGRESSION_PATH")
if regression_path is None:
    regression_path = next((str(p / "tst/regression") for p in Path(__file__).resolve().parents
                            if (p / "tst/regression/z4c_background_restart.py").is_file()), None)
if regression_path is None:
    raise RuntimeError("Set ATHENA_REGRESSION_PATH to the AthenaK tst/regression directory")
sys.path.insert(0, regression_path)
from z4c_background_restart import checkpoint, cohort


def parameter_bool(parameters, name, default):
    """Read the supported boolean spellings, rejecting ambiguous metadata."""
    value = parameters.get(name)
    if value is None:
        return default
    value = value.lower()
    if value in ("true", "1"):
        return True
    if value in ("false", "0"):
        return False
    raise ValueError(f"Unsupported boolean {name}={value!r}")


def validate(run, ranks, cycle=None):
    candidates = list((run / "rst/rank_00000000").glob("*.rst"))
    assert candidates, "No checkpoints"
    if cycle is None:
        cycle = max(checkpoint(p)["cycle"] for p in candidates)
    first, records = cohort(run, ranks, cycle)
    q = records[0]
    owners = [rank for rank, rec in enumerate(records) for _ in rec["state"]]
    assert len(owners) == q["total"]
    raw = first.read_bytes()
    end = raw.index(b"<par_end>\n") + len(b"<par_end>\n")
    params, block = {}, ""
    for line in raw[:end].decode().splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("<"):
            block = line[1:-1]
            params.setdefault(block, {})
        elif "=" in line:
            k, v = line.split("=", 1)
            params[block][k.strip()] = v.strip()
    p = params["problem"]
    assert p["bh_background"] == "schwarzschild_trumpet"
    assert p["use_direct_z4c_background"] == "true"
    assert float(p["bh_mass"]) == 1 and float(p["bh_spin"]) == 0
    assert all(float(p[f"bh_center_x{a}"]) == 0 for a in (1, 2, 3))
    assert params["z4c"]["use_analytic_background"] == "true"
    assert float(params["z4c"]["chi_psi_power"]) == -4, "Only chi=psi^-4 is supported"
    # Match Z4c's constructor defaults and ReconstructFullState: stored lapse
    # residuals contribute only when evolution or preservation is enabled.
    # This checker always reconstructs alpha_bg+delta_alpha, so reject the
    # unsupported frozen-lapse interpretation instead of reporting its metric.
    z4c = params["z4c"]
    evolve_gauge = parameter_bool(z4c, "evolve_gauge_residual", True)
    evolve_lapse = parameter_bool(z4c, "evolve_lapse_residual", evolve_gauge)
    preserve_lapse = parameter_bool(z4c, "preserve_lapse_residual", False)
    if not (evolve_lapse or preserve_lapse):
        raise ValueError("Checker requires evolve_lapse_residual or "
                         "preserve_lapse_residual for alpha_bg+delta_alpha")
    assert params["mesh_refinement"]["refinement"] == "none"
    ng, nx, ny, nz = struct.unpack_from("<19i", raw, end + 8 + 72 + 76)[:4]
    start = end + 8 + 72 + 2 * 76 + 20
    locations = [struct.unpack_from("<4i", raw, start + 16 * i)
                 for i in range(q["total"])]
    assert all(loc[3] == q["level"] for loc in locations), "Nonuniform mesh"
    arrays = np.asarray([s for rec in records for s in rec["state"]]).reshape(
        -1, 25, nz + 2 * ng, ny + 2 * ng, nx + 2 * ng)
    assert arrays.shape[0] == q["total"]
    mesh = params["mesh"]
    lo = [float(mesh[f"x{a}min"]) for a in (1, 2, 3)]
    hi = [float(mesh[f"x{a}max"]) for a in (1, 2, 3)]
    ns = [int(mesh[f"nx{a}"]) for a in (1, 2, 3)]
    minima = dict.fromkeys(["alpha", "chi", "gxx", "second_minor", "determinant"], np.inf)
    bad, samples = 0, []
    for b, loc in enumerate(locations):
        axes = [lo[a] + (loc[a] * n + np.arange(n + 2 * ng) - ng + .5)
                * (hi[a] - lo[a]) / ns[a] for a, n in enumerate((nx, ny, nz))]
        z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
        r = np.sqrt(x*x + y*y + z*z)
        assert np.all(r > 0), "Cell coincides with puncture"
        u = arrays[b]
        alpha, chi = r/(r+1) + u[18], (r/(r+1))**2 + u[0]
        xx, xy, xz, yy, yz, zz = 1+u[1], u[2], u[3], 1+u[4], u[5], 1+u[6]
        minor = xx*yy - xy*xy
        det = xx*yy*zz + 2*xy*xz*yz - xx*yz*yz - yy*xz*xz - zz*xy*xy
        vals = dict(alpha=alpha, chi=chi, gxx=xx, second_minor=minor, determinant=det)
        invalid = np.zeros_like(alpha, dtype=bool)
        for key, value in vals.items():
            invalid |= ~np.isfinite(value) | (value <= 0)
            minima[key] = min(minima[key], float(np.min(value)))
        bad += int(np.count_nonzero(invalid))
        for k, j, i in np.argwhere(invalid)[:max(0, 16-len(samples))]:
            ijk = (int(i), int(j), int(k))
            samples.append(dict(
                rank=owners[b], global_block=b,
                logical_level=loc[3], relative_level=loc[3]-q["level"],
                xyz_M=[float(axes[a][ijk[a]]) for a in range(3)],
                ghost_depth=[max(ng-ijk[a], ijk[a]-(ng+n-1), 0)
                             for a, n in enumerate((nx, ny, nz))],
                values={key: float(value[k, j, i]) for key, value in vals.items()}))
    payload_start = start + 20*q["total"] + 16 + 8
    files = []
    payload_finite = True
    for rank in range(ranks):
        path = run / "rst" / f"rank_{rank:08d}" / first.name
        data = path.read_bytes()
        finite = bool(np.isfinite(np.frombuffer(data[payload_start:], dtype="<f8")).all())
        payload_finite &= finite
        files.append(dict(rank=rank, name=str(path), bytes=len(data),
                          sha256=hashlib.sha256(data).hexdigest(), payload_finite=finite))
    return dict(time_M=q["time"], cycle=q["cycle"], blocks=q["total"], ranks=ranks,
                matching_headers=True, all_payload_finite=payload_finite,
                invalid_metric_cells_including_ghosts=bad,
                invalid_metric_samples=samples, minimum=minima, files=files,
                passed=payload_finite and bad == 0,
                note="Saved checkpoint validity only; not a perturbation-stability result.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--cycle", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert args.ranks > 0
    result = validate(args.run.resolve(), args.ranks, args.cycle)
    output = args.output or args.run / "checkpoint-validity.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
