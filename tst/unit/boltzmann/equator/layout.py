"""Write a deck and check every leaf intersecting the initial refined-ring rectangle."""
import argparse
import math
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def make(half, coarse, square, *, extent=None, root_blocks=None, block_cells=None, axial_block_cells=None):
    if root_blocks is None:
        root_blocks = 3 if extent is None else 2
    if block_cells is None:
        block_cells = 24 if extent is None else 36
    axial_cells = block_cells if axial_block_cells is None else axial_block_cells
    radial_extent = 36 if extent is None else extent
    zmax = radial_extent if (square or extent is not None) else 18
    if coarse and not square and extent is None:
        raise ValueError("24-cell blocks require the 36-unit polar extent for the coarse pair")
    text = (ROOT / "inputs/boltzmann/refined_ring.athinput").read_text()
    spacing = .5 if coarse else .25
    if extent is not None:
        if root_blocks < 1 or extent <= 0 or any(n < 36 or n % 2 for n in (block_cells, axial_cells)):
            raise ValueError("AMR requires positive extent, positive root count and even block sizes >=36 on both spatial axes")
        spacing = extent/(root_blocks*block_cells)
        if (root_blocks*block_cells) % axial_cells:
            raise ValueError("axial root blocks must tile the positive square extent")
    elif (root_blocks, block_cells, axial_cells) != (3, 24, 24):
        raise ValueError("set the square extent when changing root or block dimensions")
    finest_level = round(math.log2(spacing/.0625))
    if finest_level < 2 or .0625 * 2**finest_level != spacing:
        raise ValueError(f"root spacing {spacing:g} cannot refine dyadically to 0.0625")
    def setval(section, key, value):
        nonlocal text
        block = re.search(rf"(?ms)^<{section}>\n(.*?)(?=^<|\Z)", text)
        body, count = re.subn(rf"(?m)^{key}\s*=.*$", f"{key} = {value}", block[1])
        assert count == 1
        text = text[:block.start(1)] + body + text[block.end(1):]
    for key, val in dict(x1max=radial_extent, nx1=int(radial_extent/spacing), nx2=int((1 if half else 2)*zmax/spacing),
                         x2min=0 if half else -zmax, x2max=zmax,
                         ix2_bc="reflect" if half else "outflow").items():
        setval("mesh", key, val)
    if half:
        for region in (1, 2):
            setval(f"refined_region{region}", "x2min", 0)
    setval("meshblock", "nx1", block_cells)
    setval("meshblock", "nx2", axial_cells)
    setval("mesh_refinement", "num_levels", finest_level+1)
    for region in (1, 2):
        setval(f"refined_region{region}", "level", finest_level-2+region)
    if (block_cells, axial_cells) != (24,24):
        # Different block edges must still enclose every original finest cell.
        # Use interior cell centers to avoid marking an adjacent block at a face.
        pad = .0625/2
        for key, val in dict(x1min=3+pad, x1max=9-pad,
                             x2min=0 if half else -3+pad, x2max=3-pad).items():
            setval("refined_region2", key, val)
    return text


def leaves(path):
    """Active leaf bounds from Athena's 2D mesh dump."""
    for block in path.read_text().split("#MeshBlock")[1:]:
        rows = block.splitlines()
        r0, z0 = map(float, rows[2].split())
        r1, z1 = map(float, rows[4].split())
        yield r0, r1, z0, z1


def verify(path, half, block_cells=24, axial_block_cells=None):
    axial_cells = block_cells if axial_block_cells is None else axial_block_cells
    blocks = 0
    covered = 0.
    finest = set()
    for r0, r1, z0, z1 in leaves(path):
        blocks += 1
        if (r1-r0)/block_cells == (z1-z0)/axial_cells == .0625:
            finest.add((r0,r1,z0,z1))
            if half:
                finest.add((r0,r1,-z1,-z0))
        area = max(0, min(r1, 6.90)-max(r0, 5.10))*max(0, min(z1, .90)-max(z0, -.90))
        if area:
            assert (r1-r0)/block_cells == (z1-z0)/axial_cells == .0625, (r0, r1, z0, z1)
            covered += area
    # Whole physical footprint of the original finest level, including the
    # complete blocks around the nominal ring box.
    expected = {(r,r+1.5,z,z+1.5) for r in (3,4.5,6,7.5) for z in (-3,-1.5,0,1.5)}
    # For alternate block sizes the same fine cells can span more area.
    area = sum(max(0, min(9,r1)-max(3,r0))*max(0, min(3,z1)-max(-3,z0))
               for r0,r1,z0,z1 in finest)
    if area != 36 or ((block_cells, axial_cells) == (24,24) and finest != expected):
        raise ValueError(f"changed original finest ring blocks: {finest ^ expected}")
    return blocks, covered


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--half", action="store_true")
    p.add_argument("--coarse", action="store_true")
    p.add_argument("--square", action="store_true")
    p.add_argument("--extent", type=int, help="distance to both outer faces in code units")
    p.add_argument("--root-blocks", type=int, help="root blocks along the positive radial axis")
    p.add_argument("--block-cells", type=int, help="radial cells per block (36 with --extent, otherwise 24)")
    p.add_argument("--axial-block-cells", type=int, help="axial cells per block (defaults to radial size)")
    p.add_argument("--check", type=Path, metavar="ATHENA")
    a = p.parse_args()
    a.output.write_text(make(a.half, a.coarse, a.square, extent=a.extent,
                             root_blocks=a.root_blocks, block_cells=a.block_cells, axial_block_cells=a.axial_block_cells))
    if a.check:
        subprocess.run([a.check.resolve(), "-i", a.output.resolve(), "-m", "mesh_refinement/max_nmb_per_rank=256"],
                       cwd=a.output.parent, check=True, stdout=subprocess.DEVNULL)
        radial_cells = a.block_cells or (36 if a.extent is not None else 24)
        blocks, covered = verify(a.output.parent / "mesh_structure.dat", a.half, radial_cells, a.axial_block_cells)
        assert abs(covered - (1.62 if a.half else 3.24)) < 1e-12
        print(f"{blocks} leaves, ring rectangle area {covered:g}, dr=dz=0.0625")
