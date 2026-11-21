"""Full-volume radiation, circumferential centroid and outgoing shell resolution."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]/"vis/python"))
from bin_convert import read_binary


def profile(path, mass):
    f = read_binary(path)
    adm = read_binary(path.with_name(path.name.replace(".tmunu.", ".adm.")))
    np.testing.assert_array_equal(adm["mb_geometry"], f["mb_geometry"])
    assert adm["cycle"] == f["cycle"]
    # Require the declared equatorial boundary, not just an output covering z>=0.
    block = ""
    copies = 1
    for line in f["header"]:
        if line.startswith("<"):
            block = line
        elif block == "<mesh>" and line.partition("=")[0].strip() == "ix2_bc":
            copies = 2 if line.partition("=")[2].strip() == "reflect" else 1
    weights, radii, sizes = [], [], []
    angular = vertical = 0.
    for b, bounds in enumerate(f["mb_geometry"]):
        matter = f["mb_data"]
        E = matter["tmunu_E"][b][0].astype(float)
        nz, nr = E.shape
        dr, dz = (bounds[1]-bounds[0])/nr, (bounds[3]-bounds[2])/nz
        rho = bounds[0] + (np.arange(nr)+.5)*dr
        metric = []
        for name in ("xx","xy","xz","yy","yz","zz"):
            v = adm["mb_data"]["adm_g"+name][b][0].astype(float)
            assert v.shape == (nz+1, nr+1)
            metric.append((v[:-1,:-1]+v[1:,:-1]+v[:-1,1:]+v[1:,1:])*.25)
        xx, xy, xz, yy, yz, zz = metric
        vol = copies*2*np.pi*rho*dr*dz*np.sqrt(xx*yy*zz+2*xy*xz*yz-xx*yz**2-yy*xz**2-zz*xy**2)
        w = vol * E
        weights.append(w.ravel())
        radii.append((rho*np.sqrt(zz)).ravel())
        sizes.append(np.full(w.size,max(dr,dz)))
        angular += (vol*rho*matter["tmunu_Sz"][b][0]).sum()
        # Stress along the coordinate z tangent of unit proper length.
        vertical += (vol*matter["tmunu_Syy"][b][0]/yy).sum()
    w, r, h = map(np.concatenate, (weights,radii,sizes))
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(r)) and w.sum() > 0):
        raise ValueError(f"invalid radius or proper energy in {path}")
    indices = np.argsort(r)
    q90 = r[indices[np.searchsorted(np.cumsum(w[indices]), .9*w.sum())]]
    outer = r >= q90
    return dict(cycle=f["cycle"], t_M=f["time"]/mass,
                E=w.sum(), J=angular, vertical_stress=vertical,
                mean_Rcirc_M=w@r/w.sum()/mass, p90_Rcirc_M=q90/mass,
                energy_fraction_dr00625=w[h<=.0625].sum()/w.sum(),
                outer_decile_mean_dx=w[outer]@h[outer]/w[outer].sum())


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mass", type=float)
    cli.add_argument("files", type=Path, nargs="+")
    args = cli.parse_args()
    writer = None
    for path in sorted(args.files):
        row = profile(path, args.mass)
        if writer is None:
            writer = csv.DictWriter(sys.stdout, fieldnames=row)
            writer.writeheader()
        writer.writerow(row)
