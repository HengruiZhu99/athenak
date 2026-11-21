"""Write a deck and check every leaf intersecting the initial refined-ring rectangle."""
import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def make(half, coarse, square):
    zmax = 36 if square else 18
    if coarse and not square:
        raise ValueError("24-cell blocks require the 36-unit polar extent for the coarse pair")
    text = (ROOT / "inputs/boltzmann/refined_ring.athinput").read_text()
    spacing = .5 if coarse else .25
    def setval(section, key, value):
        nonlocal text
        block = re.search(rf"(?ms)^<{section}>\n(.*?)(?=^<|\Z)", text)
        body, count = re.subn(rf"(?m)^{key}\s*=.*$", f"{key} = {value}", block[1])
        assert count == 1
        text = text[:block.start(1)] + body + text[block.end(1):]
    for key, val in dict(nx1=int(36/spacing), nx2=int((1 if half else 2)*zmax/spacing),
                         x2min=0 if half else -zmax, x2max=zmax,
                         ix2_bc="reflect" if half else "outflow").items():
        setval("mesh", key, val)
    if half:
        for region in (1, 2):
            setval(f"refined_region{region}", "x2min", 0)
    if coarse:
        # Refining the same physical inner and ring boxes enforces the 2:1 collar
        # automatically. No permanent refined exterior region is necessary.
        setval("mesh_refinement", "num_levels", 4)
        for region in (1, 2):
            setval(f"refined_region{region}", "level", region+1)
    return text


def verify(path, half):
    blocks = 0
    covered = 0.
    finest = set()
    for block in path.read_text().split("#MeshBlock")[1:]:
        coords = [list(map(float, line.split())) for line in block.splitlines()[2:7]]
        r0, z0 = coords[0]
        r1, z1 = coords[2]
        blocks += 1
        if (r1-r0)/24 == (z1-z0)/24 == .0625:
            finest.add((r0,r1,z0,z1))
            if half:
                finest.add((r0,r1,-z1,-z0))
        area = max(0, min(r1, 6.90)-max(r0, 5.10))*max(0, min(z1, .90)-max(z0, -.90))
        if area:
            assert (r1-r0)/24 == (z1-z0)/24 == .0625, coords
            covered += area
    # Whole physical footprint of the original finest level, including the
    # complete blocks around the nominal ring box.
    expected = {(r,r+1.5,z,z+1.5) for r in (3,4.5,6,7.5) for z in (-3,-1.5,0,1.5)}
    if finest != expected:
        raise ValueError(f"changed original finest ring blocks: {finest ^ expected}")
    return blocks, covered


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--half", action="store_true")
    p.add_argument("--coarse", action="store_true")
    p.add_argument("--square", action="store_true")
    p.add_argument("--check", type=Path, metavar="ATHENA")
    a = p.parse_args()
    a.output.write_text(make(a.half, a.coarse, a.square))
    if a.check:
        subprocess.run([a.check.resolve(), "-i", a.output.resolve(), "-m", "mesh_refinement/max_nmb_per_rank=256"],
                       cwd=a.output.parent, check=True, stdout=subprocess.DEVNULL)
        blocks, covered = verify(a.output.parent / "mesh_structure.dat", a.half)
        assert abs(covered - (1.62 if a.half else 3.24)) < 1e-12
        print(f"{blocks} leaves, ring rectangle area {covered:g}, dr=dz=0.0625")
