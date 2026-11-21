"""Run a reflected smooth angular pulse through a two-rank equatorial AMR junction."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BLOCK = 16
NG = 4
SECTORS = 12
BINS = 6 * SECTORS


def run(exe, root, name, half, restart=None, coupled=False, scale=1):
    directory = root / name
    directory.mkdir(parents=True)
    ranks = 1 if restart else 2
    if os.environ.get("EQUATOR_SRUN") == "1":
        args = ["srun", "--exact", "-N1", f"-n{ranks}", "-c4", f"--gres=gpu:{ranks}",
                "--gpus-per-task=1", "--gpu-bind=single:1", "--cpu-bind=none", "--mpi=pmix",
                "--kill-on-bad-exit=1", sys.executable,
                str(ROOT / "tst/unit/boltzmann/stellar_photon_rank.py"), "--rank-bound", str(exe)]
    else:
        args = ["mpiexec", "-n", str(ranks), str(exe)]
    if restart:
        args += ["-r", str(restart)]
    else:
        args += ["-i", str(ROOT / "inputs/boltzmann/ring.athinput"),
                 "time/nlim=6", "time/integrator=ssprk104", "boltzmann/transport_order=4",
                 "boltzmann/source_order=4", "boltzmann/nmu_half=6",
                 f"boltzmann/backreaction={str(coupled).lower()}",
                 f"boltzmann/fixed_metric={str(not coupled).lower()}",
                 "fastflow/num_horizons=0", "problem/exercise_deterministic_amr=true",
                 "problem/ring_radius=8", "problem/angular_width=0.65",
                 "amr_criterion0/method=user", "mesh_refinement/ncycle_check=1",
                 "mesh_refinement/refinement_interval=1", "mesh_refinement/max_nmb_per_rank=16",
                 "output1/dcycle=1", "output2/dcycle=1", "problem/amr_target_lx2=0"]
        if half:
            args += ["mesh/nx2=16", "mesh/x2min=0", "mesh/ix2_bc=reflect"]
        else:
            args += ["problem/amr_target_count=2", "problem/amr_target1_lx1=0",
                     "problem/amr_target1_lx2=1"]
        if coupled:
            args = [a for a in args if not a.startswith(("time/nlim=", "mesh/nx2="))]
            args += ["time/nlim=-1", "time/tlim=0.25", f"mesh/nx1={32*scale}",
                     f"mesh/nx2={(16 if half else 32)*scale}",
                     f"meshblock/nx1={16*scale}", f"meshblock/nx2={16*scale}"]
    if not restart:
        # Athena accepts CLI overrides only for keys already present in the input.
        deck = Path(args[args.index("-i")+1]).read_text()
        for override in args[args.index("-i")+2:]:
            field, value = override.split("=", 1)
            section, key = field.split("/")
            start = deck.index("<"+section+">") + len(section) + 2
            end = deck.find("\n<", start)
            if end < 0:
                end = len(deck)
            if not re.search(rf"(?m)^{key}\s*=", deck[start:end]):
                deck = deck[:start] + f"\n{key} = {value}" + deck[start:]
        path = directory / "input"
        path.write_text(deck)
        args[args.index("-i")+1] = str(path)
    with (directory / "log").open("w") as log:
        subprocess.run(args, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)
    return np.loadtxt(directory / "ring.user.hst", ndmin=2)


def checkpoint(root, name, count, scale=1):
    path = sorted((root / name / "rst").glob("*.rst"))[-1]
    data = path.read_bytes()
    side = scale*BLOCK + 2 * NG
    phase = BINS * side**2
    stride = phase + 25 * (side + 1)**2
    records = np.frombuffer(data, dtype=np.float64, offset=len(data)-count*stride*8)
    blocks = records.reshape(count, stride)
    bins = blocks[:, :phase].reshape(count, 6, SECTORS, side, side)
    geometry = blocks[:, phase:].reshape(count, 25, side+1, side+1)
    return bins[..., NG:-NG, NG:-NG], geometry[..., NG:-NG, NG:-NG]


def coupled_comparison(exe, work):
    # Determine the tolerance from the full-domain spatial refinement at
    # fixed physical time and angular resolution. Last step lands at t=0.25.
    results, fields = {}, {}
    for scale in (1, 2):
        for half in (False, True):
            name = f"coupled-{scale}-{'half' if half else 'full'}"
            h = run(exe, work, name, half, coupled=True, scale=scale)
            con = np.loadtxt(work / name / "ring.z4c.user.hst", ndmin=2)
            fields[scale, half] = checkpoint(work, name, 2 if half else 4, scale)
            results[scale, half] = np.r_[h[-1, [2, 3, 6]], con[-1, 2:6]]
            assert h[-1, 0] == 0.25
    trunc = abs(results[2, False] - results[1, False])
    for scale in (1, 2):
        delta = abs(results[scale, False] - results[scale, True])
        print("coupled resolution", scale, "half/full differences", delta)
        assert np.all(delta <= trunc + 2e-13)
    print("full-domain refinement differences in E, J, work, C2, H2, M2, Z2", trunc)
    coarse_u, coarse_g = fields[1, False]
    fine_u, fine_g = fields[2, False]
    mean = fine_u.reshape(4,6,SECTORS,BLOCK,2,BLOCK,2).mean(axis=(4,6))
    trunc_u = np.max(abs(mean-coarse_u))
    trunc_g = np.max(abs(fine_g[...,::2,::2]-coarse_g))
    parity = np.ones(25)
    parity[[2,5,9,12,15,20,23]] = -1
    for scale in (1,2):
        u,g = fields[scale,True]
        uf,gf = fields[scale,False]
        delta_u = max(np.max(abs(u-uf[2:])),np.max(abs(u-uf[:2,:,::-1,::-1])))
        delta_g = max(np.max(abs(g-gf[2:])),
                      np.max(abs(g-gf[:2,:,::-1,:]*parity[None,:,None,None])))
        print("coupled reflected ordinate/geometry Linf",scale,delta_u,delta_g,
              "full refinement",trunc_u,trunc_g)
        assert delta_u < trunc_u and delta_g < trunc_g
        assert np.all(g[:,parity<0,0,:] == 0)
    name = "coupled-1-half"
    resumed = run(exe, work, "coupled-resumed", True, work/name/"rst/ring.00001.rst")
    direct = np.loadtxt(work / name / "ring.user.hst", ndmin=2)
    np.testing.assert_allclose(resumed[-1], direct[-1], rtol=3e-12, atol=5e-16)
    for resumed_field, direct_field in zip(checkpoint(work, "coupled-resumed", 2), fields[1,True]):
        np.testing.assert_allclose(resumed_field,direct_field,rtol=3e-12,atol=5e-16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe", type=Path)
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    exe, work = args.exe.resolve(), args.work.resolve()
    full = run(exe, work, "full", False)
    half = run(exe, work, "half", True)
    for name, changes in [("full", 6), ("half", 3)]:
        log = (work / name / "log").read_text()
        assert f"{changes} MeshBlocks created, {changes} deleted" in log
        if name == "half":
            assert re.search(r"[1-9]\d* communicated for load balancing", log)
    resumed = run(exe, work, "resumed", True, work / "half/rst/ring.00001.rst")
    np.testing.assert_allclose(resumed[-1], half[-1], rtol=4e-12, atol=5e-16)
    for name, hist in [("full", full), ("half", half)]:
        error = np.max(abs(hist[:, 2] + hist[:, 4] - hist[:, 6] - hist[0, 2])) / hist[0, 2]
        print(name, "relative energy budget residual", error)
        assert error < 4e-13
    a, b = checkpoint(work, "full", 4)[0], checkpoint(work, "half", 2)[0]
    print("upper / reflected lower / restart max ordinate errors",
          np.max(abs(a[2:] - b)), np.max(abs(a[:2, :, ::-1, ::-1] - b)),
          np.max(abs(checkpoint(work, "resumed", 2)[0] - b)))
    np.testing.assert_allclose(checkpoint(work, "resumed", 2)[0], b, rtol=3e-12, atol=1e-16)
    np.testing.assert_allclose(a[2:], b, rtol=3e-11, atol=1e-16)
    # The full solver itself has a small mirror defect at the coarse-fine junction.
    mirror_defect = np.max(abs(a[:2, :, ::-1, ::-1] - a[2:]))
    np.testing.assert_allclose(a[:2, :, ::-1, ::-1], b, rtol=0, atol=mirror_defect+1e-16)
    np.testing.assert_allclose(half, full, rtol=4e-11, atol=5e-16)
    coupled_comparison(exe, work)


if __name__ == "__main__":
    main()
