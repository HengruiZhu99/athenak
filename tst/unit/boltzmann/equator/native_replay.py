"""Run the production meridional correction on archived tensors with block-local cubic sampling."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from replay import NativePatch, measure, read_binary, upper_only
from witness import LocalPolynomial


def export(path, patch, outer, inner, degree):
    n = round(patch.bound/patch.dx)
    z = np.arange(-n,n+1)*patch.dx
    r = np.arange(n+1)*patch.dx
    # Convert the stored (rho,z,phi) components to the finder's physical (x,y,z).
    order = (0,2,1,5,4,3)
    def grid(offset=0, **deriv):
        return np.stack([patch.splines[offset+k](z,r,**deriv) for k in order],axis=-1)
    g, kr = grid(), grid(6)
    gz, gr = grid(dx=1), grid(dy=1)
    # Use the same fourth-order centered derivative in the resolved patch.
    # The outer two rows lack external native geometry in this saved crop.
    for k, component in enumerate(order):
        spline = patch.splines[component]
        h = patch.dx
        gz[2:-2,:,k] = (g[:-4,:,k]-8*g[1:-3,:,k]+8*g[3:-1,:,k]-g[4:,:,k])/(12*h)
        x = r[:-2]
        gr[:,:-2,k] = (spline(z,x-2*h)-8*spline(z,x-h)+8*spline(z,x+h)-spline(z,x+2*h))/(12*h)
    def rotate(a):
        xx,xy,xz,yy,yz,_ = np.moveaxis(a,-1,0)
        return np.stack((-2*xy,xx-yy,-yz,2*xy,xz,np.zeros_like(xx)),axis=-1)
    gy = rotate(g)
    gy[:,1:] /= r[None,1:,None]
    gy[:,0] = rotate(gr[:,0])
    with path.open("wb") as f:
        np.asarray([n,degree],dtype=np.int32).tofile(f)
        np.asarray([patch.dx],dtype=np.float64).tofile(f)
        np.concatenate((g,kr,gr,gy,gz),axis=-1).astype(np.float64).tofile(f)
        for coeff in (outer,inner):
            a = np.zeros(degree+1)
            a[:min(len(a),len(coeff))] = coeff[:len(a)]
            a.tofile(f)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("exe",type=Path)
    p.add_argument("reference",type=Path)
    p.add_argument("snapshots",type=Path,nargs="+")
    p.add_argument("--degree",type=int,default=24)
    a = p.parse_args()
    ref = json.loads(a.reference.read_text())
    out = csv.writer(sys.stdout,lineterminator="\n")
    out.writerow(["file","time","folded","target","guess","converged","direct_rms","area",
                  "mass","spin","milliseconds","finder_status","verified_on_archive","max_abs_hThetaPlus",
                  "max_hThetaPlus","max_hThetaMinus"])
    for frame in sorted(a.snapshots):
        native = read_binary(frame)
        patch = NativePatch(native)
        interps = (patch,LocalPolynomial(patch,3),LocalPolynomial(patch,5))
        theta = np.r_[0.,np.linspace(1.e-5,np.pi-1.e-5,4097),np.pi]
        with tempfile.TemporaryDirectory() as tmp:
            gridpath = Path(tmp)/"sample"
            lines = []
            for method, geometry in (("full",patch),("folded",NativePatch(upper_only(native)))):
                export(gridpath,geometry,ref["enclosing_horizon"]["coeff"],ref["coeff"],a.degree)
                proc = subprocess.run([a.exe.resolve(),gridpath,method],text=True,capture_output=True,check=True)
                lines.extend(proc.stdout.splitlines())
        for row in csv.reader(lines):
            coeff = np.fromstring(row.pop(),sep=" ")
            check = ["","","",""]
            if row[3] == "1" or float(row[1]) < 0:
                peaks = []
                for metric in interps:
                    try:
                        h, plus, minus, _, _ = measure(metric,coeff,theta)
                    except (ValueError, np.linalg.LinAlgError):
                        break
                    peaks.append([max(abs(h*plus)),max(h*plus),max(h*minus)])
                if len(peaks) == len(interps):
                    residual, plus, minus = np.max(peaks,axis=0)
                    ok = (residual <= .006 and minus <= 0) if float(row[1])==0 else max(plus,minus)<0
                    check = [int(ok),residual,plus,minus]
            out.writerow([frame.name,native["time"],*row,*check])
