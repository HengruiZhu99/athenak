"""Compare photon bins and native Z4c fields in the 48² FV4 coupled overlap."""
import argparse
import json
from pathlib import Path

import numpy as np

from evolution import check_energy
from qualify_fv4 import active_checkpoint


def geometry(folder):
    nb, nang, cells, vertices = 4, 192, 32, 33
    phases = nang*cells**2
    stride = phases+25*vertices**2
    path = sorted((folder/"rst").glob("*.rst"))[-1]
    data = np.frombuffer(path.read_bytes()[-nb*stride*8:], dtype=np.float64)
    return data.reshape(nb,stride)[:,phases:].reshape(nb,25,vertices,vertices)[:,:,4:-4,4:-4]


def differences(ref, case):
    bins = active_checkpoint(case)
    vertices = geometry(case)
    old_bins, old_vertices = active_checkpoint(ref), geometry(ref)
    np.testing.assert_allclose(bins, old_bins, rtol=2e-12, atol=1e-15)
    np.testing.assert_allclose(vertices, old_vertices, rtol=2e-11, atol=2e-14)
    return dict(active_bins=bins.size, geometry_values=vertices.size,
                max_bin_difference=float(np.max(abs(bins-old_bins))),
                max_geometry_difference=float(np.max(abs(vertices-old_vertices))))


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cpu", type=Path, help="serial run from torus_fv4_checks.py")
    cli.add_argument("gpu", type=Path, help="directory containing runs 1 and 2")
    cli.add_argument("--legacy", type=Path, nargs=2, metavar=("ORIGINAL","RESUMED"))
    args=cli.parse_args()
    records={}
    for n in (1,2):
        case=args.gpu/str(n)
        r=differences(args.cpu,case)
        hist=np.loadtxt(case/"torus.user.hst",ndmin=2)
        log=(case/"run.log").read_text()
        devices=[json.loads(l) for l in log.splitlines() if l.startswith('{"host"')]
        assert len({(dev["host"],dev["pci"]) for dev in devices})==n
        r.update(devices=devices,relative_energy_balance=check_energy(hist,n,r["active_bins"],log),
                 min_bin=float(active_checkpoint(case).min()),end_time=hist[-1,0])
        records[n]=r
    if args.legacy:
        ref,case=args.legacy
        r=differences(ref,case)
        checkpoint=sorted((case/"rst").glob("*.rst"))[-1]
        setting=next(l.decode().strip() for l in checkpoint.read_bytes()[:30000].splitlines()
                     if l.startswith(b"source_order"))
        assert setting.split("=")[1].split()[0]=="2"
        r["source_setting"]=setting
        records["legacy"]=r
    print(json.dumps(records,indent=2))
