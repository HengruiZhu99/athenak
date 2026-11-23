"""Resolution series for physical binary inflow and a two-center cyclic-zoom control."""

import argparse
import json
from pathlib import Path
import numpy as np
from check_binary_shells import ROOT, run


def reference_volumes():
    # Equal-time Kerr ellipsoid volume contracts by the inverse orbital Lorentz factor.
    factor = 4 * np.pi / 3
    central = factor * 3.75**3
    for mass, chi in ((1 / 1.75, 0.3), (0.75 / 1.75, 0.2)):
        r_plus = mass * (1 + np.sqrt(1 - chi**2))
        speed = 4 * (1 - mass) * 0.125
        central -= factor * r_plus * (r_plus**2 + (mass * chi)**2) * np.sqrt(1 - speed**2)
    return {"R3.75": central, "R4.25": factor * (4.25**3 - 3.75**3),
            "R4.75": factor * (4.75**3 - 4.25**3)}


def campaign(exe, out, zoom_only=False, sizes=(48, 64, 96)):
    base = (ROOT / "tst/diagnostics/binary_shell.athinput").read_text()
    base = base.split("<output2>")[0]
    report = {}
    volumes = reference_volumes()
    if any(n < 48 or n % 16 for n in sizes):
        raise ValueError("This control requires at least 48 cells and complete 16-cell blocks")
    for n in (() if zoom_only else sorted(set(sizes))):
        params = base.replace(" = 48\n", f" = {n}\n")
        params = params.replace("flux_ntheta = 24", f"flux_ntheta = {n//2}")
        params = params.replace("nlim = 2", f"nlim = {max(100, n)}").replace(
            "tlim = 1\n", "tlim = 0.02\n"
        )
        params = params.replace("cfl_number = 0.0001", "cfl_number = 0.04")
        params = params.replace("b_inf = 0", "b_inf = 0.002")
        runs = {}
        for mode in ("rot", "inert"):
            text = (
                params
                if mode == "rot"
                else params.replace("corotating = true", "corotating = false").replace(
                    "frame_omega = 0.125", "frame_omega = 0"
                )
            )
            h = run(exe, out / f"{mode}-{n}", text)
            runs[mode] = h
            np.testing.assert_allclose(h["time"][-1], 0.02)
            vals = {}
            for region in ("R3.75", "R4.25", "R4.75"):
                pre = "shell_" + region + "_"
                assert np.all(h[pre + "volume_valid"] == 1)
                volume_error = abs(h[pre + "sampled_volume"][-1] / volumes[region] - 1)
                vals[region + "_volume_relative_error"] = float(volume_error)
                if n == 512:
                    assert volume_error < 1e-6, (mode, region, volume_error)
                for comp in ("Px", "Py", "Pz", "Jz"):
                    r = h[pre + comp + "_interval_residual"][1:]
                    assert np.isfinite(r).all()
                    vals[region + "_" + comp] = float(sum(abs(r)) / 0.02)
                    if comp == "Pz":
                        assert vals[region + "_" + comp] < 1e-9
                    else:
                        total = sum(
                            np.abs(h[pre + comp + "_interval_" + term][1:]).sum()
                            for term in ("source", "boundary", "other")
                        )
                        vals[region + "_" + comp + "_fraction"] = float(np.abs(r).sum() / total)

            report[f"{mode}-{n}"] = vals
        theta = 0.125 * 0.02
        diff = {}
        for face in ("BH1_Kerr", "BH2_Kerr"):
            for part in ("f", "em"):
                I, R = runs["inert"], runs["rot"]

                def vec(h):
                    return np.array(
                        [h[k + "_" + part + "_in_moving_" + face][-1] for k in ("Px", "Py", "Pz")]
                    )

                v = vec(I)
                ref = [
                    np.cos(theta) * v[0] + np.sin(theta) * v[1],
                    -np.sin(theta) * v[0] + np.cos(theta) * v[1],
                    v[2],
                ]
                diff[face + part + "_P_rel"] = float(
                    np.linalg.norm(vec(R) - ref) / np.linalg.norm(v)
                )
                label = "Jz_" + part + "_in_moving_" + face
                diff[face + part + "_J_rel"] = float(abs(R[label][-1] / I[label][-1] - 1))
        assert max(diff.values()) < 0.01, diff
        # Check the rotating vector basis and its time derivative for the entire
        # shell, including the -Omega x P term in the translation source.
        rotation = np.array(
            [[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        )
        shell_differences = {}
        for region in ("R3.75", "R4.25", "R4.75"):
            pre = "shell_" + region + "_"

            def vector(h, kind):
                return np.array([h[pre + q + "_" + kind][-1] for q in ("Px", "Py", "Pz")])

            for kind in ("stored", "boundary_in", "source"):
                expected = rotation @ vector(I, kind)
                if kind == "source":
                    expected -= np.cross([0, 0, 0.125], rotation @ vector(I, "stored"))
                delta = np.linalg.norm(vector(R, kind) - expected) / np.linalg.norm(expected)
                j = pre + "Jz_" + kind
                # The outer Jz source is a small difference between much larger terms.
                # For it, express the difference against total transport plus source.
                scale = abs(I[j][-1]) + (
                    abs(I[pre + "Jz_boundary_in"][-1]) if kind == "source" else 0
                )
                jdelta = abs(R[j][-1] - I[j][-1]) / scale
                shell_differences[region + "_" + kind] = [float(delta), float(jdelta)]
                if n == 96:
                    assert delta < 0.01 and jdelta < 0.01, (region, kind, delta, jdelta)
        report[f"shell-frame-{n}"] = shell_differences
        report[f"frame-{n}"] = diff
        if n in (96, 512) and 48 in sizes:
            for mode in ("rot", "inert"):
                fine, coarse = report[f"{mode}-{n}"], report[f"{mode}-48"]
                for region in ("R3.75", "R4.25", "R4.75"):
                    for group in (("Px", "Py"), ("Jz",)):
                        assert sum(fine[region + "_" + q] for q in group) < sum(
                            coarse[region + "_" + q] for q in group
                        ), (mode, region, group)
                for q in ("Px", "Py", "Jz"):
                    assert fine["R3.75_" + q + "_fraction"] < 0.02, fine

        (out / "resolution.json").write_text(json.dumps(report, indent=2) + "\n")
    zoom = (ROOT / "tst/diagnostics/zoom_smoke.athinput").read_text()
    zoom = zoom[zoom.index("<mesh_refinement>") : zoom.index("<time>")]
    zoom = zoom.replace("ncenters = 1", "ncenters = 2\nx1c = 2\nx1c2 = -2")
    zoom = zoom.replace("trun_fac = 0.15", "trun_fac = 0.005")
    params = base.replace(" = 48\n", " = 32\n").replace("q = 0.75", "q = 1")
    params = params.replace("nlim = 2", "nlim = 12").replace("tlim = 1\n", "tlim = 0.05\n")
    params = params.replace("cfl_number = 0.0001", "cfl_number = 0.05")
    params = params.replace("<time>", zoom + "<time>").replace("dynamic = true", "dynamic = false")
    h = run(exe, out / "binary-zoom", params)
    for face in ("BH1_Kerr", "BH2_Kerr"):
        reason = h["reason_" + face].astype(int)
        frozen = np.flatnonzero(reason & 32)
        assert frozen.size >= 2 and reason[0] == 0, reason
        assert np.any(reason[frozen[0] + 1 :] == 0), reason
        assert h["M0_unavailable_duration_" + face][-1] > 0
    assert np.any(h["shell_R3.75_rejected_volume"] > 0)
    assert h["shell_R3.75_grid_jump_unavailable_events"][-1] > 0
    rejected = h["shell_R3.75_volume_valid"] == 0
    assert np.isnan(h["shell_R3.75_Jz_stored"][rejected]).all()
    assert np.isfinite(h["shell_R3.75_Jz_partial_stored"][rejected]).all()
    report["zoom"] = {
        "times": h["time"].tolist(),
        "BH1_reason": h["reason_BH1_Kerr"].tolist(),
        "BH2_reason": h["reason_BH2_Kerr"].tolist(),
        "unavailable_grid_events": float(h["shell_R3.75_grid_jump_unavailable_events"][-1]),
        "rejected_volume": h["shell_R3.75_rejected_volume"].tolist(),
    }
    (out / "resolution.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("executable", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--zoom-only", action="store_true")
    p.add_argument("--sizes", nargs="+", type=int, default=(48, 64, 96))
    a = p.parse_args()
    a.output.mkdir(exist_ok=True)
    campaign(a.executable.resolve(), a.output.resolve(), a.zoom_only, a.sizes)
