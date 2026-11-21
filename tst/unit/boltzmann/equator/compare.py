"""Compare matched-cycle outputs. Keep time offsets and field centering explicit."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

from diagnostics import profile, read_binary

ODD = {"tmunu_Sxy", "tmunu_Syz", "tmunu_Sy", "adm_gxy", "adm_gyz", "adm_Kxy",
       "adm_Kyz", "adm_betay", "con_My"}


def raster(frame, key, h):
    # Sample the native data on common cell centers. Use piecewise constants
    # for finite-volume fields and bilinear interpolation for nodal outputs.
    bounds = np.array(frame["mb_geometry"])
    z0 = bounds[:,2].min()
    dims = (round((bounds[:,3].max()-z0)/h), round(bounds[:,1].max()/h))
    out = np.full(dims, np.nan)
    for block, a in zip(bounds, frame["mb_data"][key]):
        r0,r1,y0,y1 = block[:4]
        a = a[0]
        cell = key.startswith("tmunu_")
        nx, ny = a.shape[1]-(not cell), a.shape[0]-(not cell)
        ix = round(r0/h)
        iy = round((y0-z0)/h)
        width, height = round((r1-r0)/h), round((y1-y0)/h)
        x = (np.arange(width)+.5)*nx/width
        y = (np.arange(height)+.5)*ny/height
        i, j = x.astype(int), y.astype(int)
        values = a[j[:,None],i]
        if not cell:
            fx, fy = x-i, (y-j)[:,None]
            values = (1-fy)*((1-fx)*values+fx*a[j[:,None],i+1]) + fy*(
                       (1-fx)*a[j[:,None]+1,i]+fx*a[j[:,None]+1,i+1])
        out[iy:iy+height, ix:ix+width] = values
    if not np.isfinite(out).all():
        raise ValueError(f"incomplete or nonfinite raster for {key}")
    return out


def compare(full, half, mass, prefix):
    scalar_rows, field_rows = [], []
    for p in sorted((half/"bin").glob("*.tmunu.*.bin")):
        q = full/"bin"/p.name
        if not q.is_file():
            continue
        a,b = profile(q,mass),profile(p,mass)
        if a["cycle"] != b["cycle"]:
            # A forced terminal dump may have its own independently clipped timestep.
            if abs(a["t_M"]-b["t_M"]) > 1.e-10:
                continue
        row = dict(cycle_full=a["cycle"],cycle_half=b["cycle"],t_full_M=a["t_M"],
                   delta_t_M=b["t_M"]-a["t_M"])
        scalar_rows.append(dict(row, **{key:b[key]-a[key] for key in a if key not in ("cycle","t_M")}))
        for kind in ("tmunu","adm","con"):
            ff = read_binary(q.with_name(q.name.replace(".tmunu.",f".{kind}.")))
            hf = read_binary(p.with_name(p.name.replace(".tmunu.",f".{kind}.")))
            fb,hb = np.array(ff["mb_geometry"]),np.array(hf["mb_geometry"])
            np.testing.assert_allclose([fb[:,1].max(),fb[:,3].max(),-fb[:,2].min()],
                                       [hb[:,1].max(),hb[:,3].max(),hb[:,3].max()])
            assert hb[:,2].min() == 0
            key0 = ff["var_names"][0]
            nx = ff["mb_data"][key0][0].shape[-1]-(kind != "tmunu")
            h = min(np.min(fb[:,1]-fb[:,0]),np.min(hb[:,1]-hb[:,0]))/nx
            for key in ff["var_names"]:
                value = raster(ff,key,h)
                upper = raster(hf,key,h)
                lower = upper[::-1] * (-1 if key in ODD else 1)
                error = np.concatenate((lower,upper))-value
                rho = (np.arange(value.shape[1])+.5)*h
                weights = np.broadcast_to(rho,value.shape)
                field_rows.append(dict(row, field=key, Linf=np.max(abs(error)),
                                       rms=np.sqrt(np.average(error**2, weights=weights))))
    for suffix, rows in [("profiles",scalar_rows),("fields",field_rows)]:
        if not rows:
            raise ValueError("no comparable outputs")
        with Path(f"{prefix}-{suffix}.csv").open("w") as stream:
            writer=csv.DictWriter(stream,fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("full",type=Path)
    cli.add_argument("half",type=Path)
    cli.add_argument("mass",type=float)
    cli.add_argument("prefix",type=Path)
    args=cli.parse_args()
    compare(args.full,args.half,args.mass,args.prefix)
