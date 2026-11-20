"""Measure FV4 energy closure, CPU/GPU/MPI parity and one-node strong scaling."""

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np


def active_checkpoint(path):
    files = sorted((path / "rst").glob("*.rst"))
    nb, nang, cells, vertices = 4, 12 * 16, 32, 33
    stride = nang * cells**2 + 25 * vertices**2
    data = np.frombuffer(files[-1].read_bytes()[-nb * stride * 8 :], dtype=np.float64)
    return data.reshape(nb, stride)[:, : nang * cells**2].reshape(nb, nang, cells, cells)[
        :, :, 4:-4, 4:-4
    ]


def table(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ["host", "scaling", "transport", "output"]:
        parser.add_argument(k, type=Path)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    perf = []
    for case in sorted(a.scaling.glob("bench-*")):
        _, ranks, order, n, bands = case.name.split("-")
        ranks, order, n, bands = map(int, (ranks, order, n, bands))
        content = (case / "run.log").read_text()
        devices = [json.loads(s) for s in content.splitlines() if s.startswith('{"host"')]
        assert len({r["pci"] for r in devices}) == ranks
        cycles = {
            int(c): (float(w), float(t))
            for w, c, t in re.findall(r"elapsed=(\S+) cycle=(\d+) time=(\S+)", content)
        }
        secs = cycles[40][0] - cycles[10][0]
        physical = cycles[40][1] - cycles[10][1]
        phase = n * (n // 2) * bands * (bands * 4 // 3)
        stages = 10 if order == 4 else 2
        perf.append(
            dict(
                ranks=ranks,
                order=order,
                grid=f"{n}x{n//2}",
                sky=f"{bands}x{bands*4//3}",
                job=a.scaling.name,
                seconds=secs,
                steps=30,
                physical_time=physical,
                simulated_per_second=physical / secs,
                phase_cell_stages_per_second=30 * stages * phase / secs,
                host=devices[0]["host"],
                gpu_pci=" ".join(sorted(r["pci"] for r in devices)),
            )
        )
    assert {(p["ranks"], p["order"]) for p in perf} == {
        (r, o) for r in [1, 2, 4, 8] for o in [2, 4]
    }
    for p in perf:
        one = next(q for q in perf if q["ranks"] == 1 and q["order"] == p["order"])
        p["speedup"] = one["seconds"] / p["seconds"]
        p["parallel_efficiency"] = p["speedup"] / p["ranks"]
    table(a.output / "scaling.csv", perf)
    checks = []
    for case in sorted(a.transport.iterdir()):
        if not case.is_dir():
            continue
        mode, ranks, order, n, bands = case.name.split("-")
        h = np.loadtxt(case / "torus.user.hst", ndmin=2)
        energy = h[:, 2] + h[:, 4] - h[:, 6]
        spin = h[:, 3] + h[:, 5]
        balance = max(abs(energy - energy[0])) / energy[0]
        assert balance < 2e-12 and h[:, -1].min() >= 0 and np.isfinite(h).all(), (
            case,
            balance,
            h[:, -1].min(),
        )
        checks.append(
            dict(
                mode=mode,
                ranks=ranks,
                order=order,
                n=n,
                nmu=bands,
                job=a.transport.name,
                time=h[-1, 0],
                initial_E=h[0, 2],
                final_E=h[-1, 2],
                escaped_E=h[-1, 4],
                metric_work=h[-1, 6],
                max_energy_balance=balance,
                minimum=h[:, -1].min(),
                initial_J=spin[0],
                final_J=spin[-1],
                relative_spin_change=(spin[-1] / spin[0] - 1),
            )
        )
    table(a.output / "budgets.csv", checks)
    host = a.host / "serial"
    bins = active_checkpoint(host)
    history = np.loadtxt(host / "torus.user.hst")
    parity = []
    for gpu in sorted(a.transport.glob("coupled-*")):
        field = active_checkpoint(gpu)
        np.testing.assert_allclose(field, bins, rtol=1e-10, atol=2e-13 * bins.max())
        gpu_history = np.loadtxt(gpu / "torus.user.hst")
        np.testing.assert_allclose(gpu_history, history, rtol=2e-12, atol=2e-14)
        parity.append(
            dict(
                case=gpu.name,
                active_bins=bins.size,
                max_bin_difference=float(abs(field - bins).max()),
                max_bin_relative_to_peak=float(abs(field - bins).max() / bins.max()),
                max_history_difference=float(abs(history - gpu_history).max()),
            )
        )
    (a.output / "backend.json").write_text(json.dumps(parity, indent=2) + "\n")


if __name__ == "__main__":
    main()
