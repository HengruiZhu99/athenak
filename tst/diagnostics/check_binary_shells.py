"""Run matched no-disk binary calculations, verify transformation and shell balances."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vis/python"))
import athena_read


def run(exe, directory, text):
    directory.mkdir()
    inp = directory / "input"
    inp.write_text(text)
    with (directory / "run.log").open("w") as log:
        subprocess.run(
            [exe, "-i", inp, "-d", directory], stdout=log, stderr=subprocess.STDOUT, check=True
        )
    return athena_read.hst(directory / "matched_bondi.user.hst")


def check(exe, output):
    # Identical initial data, equal physical end time, no disk relaxation.
    base = (ROOT / "tst/diagnostics/binary_shell.athinput").read_text()
    base = base.split("<output2>")[0].replace("tlim = 1\n", "tlim = 0.00002\n")
    base = base.replace("nlim = 2", "nlim = 8")
    rot = run(exe, output / "rot", base)
    inert = run(
        exe,
        output / "inert",
        base.replace("corotating = true", "corotating = false").replace(
            "frame_omega = 0.125", "frame_omega = 0"
        ),
    )
    metrics = {}
    for mode, h in (("rot", rot), ("inert", inert)):
        np.testing.assert_allclose(h["time"][-1], 2e-5, rtol=1e-12)
        for radius in ("R3.75", "R4.25", "R4.75"):
            pre = "shell_" + radius + "_"
            assert np.all(h[pre + "volume_valid"] == 1)
            for comp in ("Px", "Py", "Pz", "Jz"):
                key = pre + comp
                terms = [
                    h[key + "_interval_" + suffix][1:]
                    for suffix in ("boundary", "source", "other", "residual")
                ]
                np.testing.assert_allclose(
                    np.diff(h[key + "_stored"]), sum(terms), atol=5e-10, rtol=1e-6
                )
                metrics[f"{mode}_{radius}_{comp}_max_residual_rate"] = float(
                    np.max(np.abs(terms[-1] / h[pre + "interval_dt"][1:]))
                )
    # At t=0 the bases coincide. In rotating coordinates the translation
    # generator contributes -Omega x P to the integrated momentum source.
    for n, radius in enumerate(("R3.75", "R4.25", "R4.75")):
        prefix = "shell_" + radius + "_"
        p = np.array([inert[prefix + q + "_stored"][0] for q in ("Px", "Py", "Pz")])
        g = np.array([inert[prefix + q + "_source"][0] for q in ("Px", "Py", "Pz")])
        expected = g - np.cross([0, 0, 0.125], p)
        measured = [rot[prefix + q + "_source"][0] for q in ("Px", "Py", "Pz")]
        np.testing.assert_allclose(measured, expected, rtol=1e-8, atol=5e-10)
        np.testing.assert_allclose(
            rot[prefix + "Jz_source"][0], inert[prefix + "Jz_source"][0], rtol=1e-8, atol=5e-10
        )
        for h in (inert, rot):
            for q in ("Px", "Py", "Pz", "Jz"):

                def inward(face):
                    return sum(h[q + "_" + part + "_in_moving_" + face] for part in ("f", "em"))

                holes = ("BH1_Kerr", "BH2_Kerr") if n == 0 else (("R3.75", "R4.25")[n - 1],)
                np.testing.assert_allclose(
                    h[prefix + q + "_boundary_in"],
                    inward(radius) - sum(map(inward, holes)),
                    atol=2e-12,
                )
    for face in [k[6:] for k in rot if k.startswith("valid_BH")]:
        assert np.all(rot["valid_" + face] == 1) and np.all(inert["valid_" + face] == 1)
        t = rot["time"][-1]
        for qty in (
            "M0_net_in_moving",
            "Px_f_in_moving",
            "Py_f_in_moving",
            "Pz_f_in_moving",
            "Jz_f_in_moving",
        ):
            key = qty + "_" + face
            # Rotating the interpolated shift and velocity separately yields dx-errors.
            np.testing.assert_allclose(rot[key][0], inert[key][0], rtol=2e-4, atol=2e-5)
            expected = inert[key][-1]
            if qty.startswith(("Px", "Py")):
                px = inert["Px_f_in_moving_" + face][-1]
                py = inert["Py_f_in_moving_" + face][-1]
                theta = 0.125 * t
                expected = (
                    np.cos(theta) * px + np.sin(theta) * py
                    if qty.startswith("Px")
                    else -np.sin(theta) * px + np.cos(theta) * py
                )
            np.testing.assert_allclose(rot[key][-1], expected, rtol=3e-4, atol=3e-5)
            metrics[key + "_end_abs_difference"] = float(abs(rot[key][-1] - expected))
    (output / "shells.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("executable", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(exist_ok=True)
    check(a.executable.resolve(), a.output.resolve())
