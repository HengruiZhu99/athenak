"""Compare an excised rotating-torus MPI restart with uninterrupted evolution."""

import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "scripts/circumbinary_mad"))
sys.path.insert(0, str(root / "vis/python"))
from inputs import read_header, write_blocks
from athena_read import hst


def check(exe, deck, directory, ranks):
    b = read_header(deck)
    for d in (1, 2, 3):
        b["mesh"][f"nx{d}"] = 16
        b["meshblock"][f"nx{d}"] = 8
    b["mesh_refinement"] = {"refinement": "static"}
    for level in range(1, 9):
        for hole, center in enumerate((-12.5, 12.5)):
            b[f"refined_region{2*level+hole}"] = dict(
                level=level,
                x1min=center - 0.95,
                x1max=center + 0.95,
                x2min=-0.95,
                x2max=0.95,
                x3min=-0.95,
                x3max=0.95,
            )
    b["problem"].update(
        flux_modeled_horizons="true", flux_min_cells_per_radius=2, flux_ntheta=16, flux_nphi=32
    )
    b["time"].update(nlim=3, cfl_number=0.02)
    b["output1"] = {"file_type": "hst", "dcycle": 1, "data_format": "%24.16e"}
    b["output2"] = {"file_type": "rst", "dcycle": 2}
    directory.mkdir(parents=True)
    path = directory / "torus.athinput"
    path.write_text(write_blocks(b))

    def run(name, args):
        work = directory / name
        work.mkdir()
        with (work / "log").open("w") as log:
            subprocess.run(
                ["mpiexec", "-n", str(ranks), exe, *args, "-d", work],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return hst(work / "mad25.user.hst")

    full = run("full", ["-i", path])
    checkpoints = sorted((directory / "full/rst").glob("*.rst"))
    restart = run("restarted", ["-r", checkpoints[1]])
    for k in full:
        if k in ("diagnostic_metric_time", "dt"):
            continue
        np.testing.assert_allclose(
            restart[k][-1], full[k][-1], atol=1e-10, rtol=1e-9, equal_nan=True, err_msg=k
        )
    for k in ("valid_BH1_Kerr", "valid_BH2_Kerr"):
        assert full[k][-1] == 1, k
    print(f"Matched restart and uninterrupted histories at t={full['time'][-1]:.9g}.")


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("executable", type=Path)
    a.add_argument("rotating_ic", type=Path)
    a.add_argument("output", type=Path)
    a.add_argument("--ranks", type=int, default=2)
    p = a.parse_args()
    check(p.executable.resolve(), p.rotating_ic, p.output.resolve(), p.ranks)
