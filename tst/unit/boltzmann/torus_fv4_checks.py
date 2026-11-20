"""Check FV4 energy balance, spin reflection, MPI and restart on imported tori."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from evolution import check_energy

ROOT = Path(__file__).resolve().parents[3]


def run(exe, family, root, label, ranks, sign="strong", restart=None, longer=False):
    path = root / label
    path.mkdir()
    deck = (ROOT / "inputs/boltzmann/photon_torus.athinput").read_text()
    deck += "\n<output2>\nfile_type = rst\ndcycle = 2\n"
    (path / "input").write_text(deck)
    args = ["mpiexec", "-n", str(ranks), str(exe)]
    if restart:
        args += ["-r", str(restart)]
    else:
        args += [
            "-i",
            "input",
            f"problem/torus_file={family / (sign + '.adm')}",
            "boltzmann/transport_order=4",
            "time/integrator=ssprk104",
            "boltzmann/nmu_half=12",
            "boltzmann/npsi=16",
        ]
        if sign == "negative":
            args.append("boltzmann/sign=retrograde")
        if longer:
            args.extend(["time/tlim=0.5", "time/nlim=4"])
    (path / "command.json").write_text(json.dumps(args, indent=2))
    with (path / "run.log").open("w") as output:
        subprocess.run(args, cwd=path, stdout=output, stderr=output, check=True)
    rows = np.loadtxt(path / "torus.user.hst", ndmin=2)
    residual = (
        check_energy(rows, ranks, 48**2 * 12 * 16, (path / "run.log").read_text())
        if not restart
        else None
    )
    return rows, residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["exe", "family", "output"]:
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    exe, family, root = (x.resolve() for x in (args.exe, args.family, args.output))
    root.mkdir(parents=True)
    one, b1 = run(exe, family, root, "serial", 1)
    two, b2 = run(exe, family, root, "mpi", 2)
    np.testing.assert_allclose(one, two, atol=4e-14, rtol=1e-13)
    pos, bp = run(exe, family, root, "positive", 2, longer=True)
    neg, bn = run(exe, family, root, "negative", 2, "negative", longer=True)
    parity = np.array([1, 1, 1, -1, 1, -1, 1, -1, 1])
    np.testing.assert_allclose(pos, neg * parity, atol=4e-13, rtol=2e-11)
    mid = root / "positive/rst/torus.00001.rst"
    resumed, _ = run(exe, family, root, "resumed", 1, restart=mid)
    np.testing.assert_allclose(pos[-1], resumed[-1], rtol=1e-12, atol=4e-14)

    def active(directory):
        # Four 24x24 blocks, four ghost layers, 25 native Z4c fields per block.
        nb, nang, cells, vertices = 4, 12 * 16, 32, 33
        stride = nang * cells**2 + 25 * vertices**2
        checkpoint = sorted((root / directory / "rst").glob("*.rst"))[-1]
        data = np.frombuffer(checkpoint.read_bytes()[-nb * stride * 8 :], dtype=np.float64)
        return data.reshape(nb, stride)[:, : nang * cells**2].reshape(nb, nang, cells, cells)[
            :, :, 4:-4, 4:-4
        ]

    direct, cont = active("positive"), active("resumed")
    np.testing.assert_allclose(direct, cont, rtol=1e-12, atol=3e-15)
    result = dict(
        cpu_balances=[b1, b2, bp, bn],
        cpu_rank_max=float(abs(one - two).max()),
        reflection_max=float(abs(pos - neg * parity).max()),
        restart_max_bin=float(abs(direct - cont).max()),
        minimum=float(min(pos[:, 8].min(), neg[:, 8].min())),
        restart_time=float(pos[-1, 0]),
    )
    (root / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
