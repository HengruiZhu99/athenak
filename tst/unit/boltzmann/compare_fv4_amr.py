"""Compare the deterministic six-step AMR pulse across CPU, CUDA and MPI.

Pass the CPU direct run (from evolution.restart_amr with order=4) and
the qualification/amr-JOB directory made by stellar_fv4_amr.sh.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evolution import check_energy


def active(folder):
    final = sorted((folder / "rst").glob("*.rst"))[-1].read_bytes()
    cells, vertices, bands, blocks = 24, 25, 144, 4
    phase = bands * cells**2
    stride = phase + 25 * vertices**2
    data = np.frombuffer(final, dtype=np.float64, offset=len(final) - blocks * stride * 8)
    ordinates = data.reshape(blocks, stride)[:, :phase].reshape(blocks, bands, cells, cells)
    return ordinates[:, :, 4:-4, 4:-4]


def compare(cpu, gpu):
    reference = active(cpu)
    std = np.loadtxt(cpu / "ring.user.hst")
    evidence = {}
    assert reference.shape == (4, 144, 16, 16)
    for name in ("ranks-1", "ranks-2", "resumed"):
        folder = gpu / name
        log = (folder / "run.log").read_text()
        values = active(folder)
        history = np.loadtxt(folder / "ring.user.hst", ndmin=2)
        ranks = 2 if name == "ranks-2" else 1
        devices = [json.loads(line) for line in log.splitlines() if line.startswith('{"host"')]
        assert len({(dev["host"], dev["pci"]) for dev in devices}) == ranks
        if name == "ranks-2":
            assert "8 communicated for load balancing" in log
        if name != "resumed":
            assert "3 MeshBlocks created, 3 deleted" in log
            np.testing.assert_allclose(history, std, rtol=1e-11, atol=2e-13)
        else:
            np.testing.assert_allclose(history[-1], std[-1], rtol=1e-11, atol=2e-13)
        np.testing.assert_allclose(values, reference, rtol=2e-12, atol=1e-16)
        assert np.min(values) >= 0
        # The accumulated work/escape history spans the checkpoint.
        rows = history if name != "resumed" else np.vstack((std[0], history))
        balance = check_energy(rows, ranks, reference.size, log)
        evidence[name] = dict(max_bin_difference=float(np.max(np.abs(values - reference))),
                              n_active=int(values.size), min_bin=float(np.min(values)),
                              relative_energy_residual=balance, end_time=history[-1, 0],
                              devices=devices)
    evidence["two_gpu_to_one_restart_max_bin_difference"] = float(
        np.max(np.abs(active(gpu / "resumed") - active(gpu / "ranks-2"))))
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu_direct", type=Path)
    parser.add_argument("gpu_cases", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.cpu_direct, args.gpu_cases), indent=2))
