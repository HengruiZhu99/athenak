"""Compare matched-cycle outputs. Keep time offsets and field centering explicit."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

from diagnostics import profile, read_binary

ODD = {"tmunu_Sxy", "tmunu_Syz", "tmunu_Sy", "adm_gxy", "adm_gyz", "adm_Kxy",
       "adm_Kyz", "adm_betay", "con_My"}


def raster(frame, key, h, stride=1):
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
        if stride > 1:
            if cell:
                nz, nr = a.shape
                a = a.reshape(nz//stride,stride,nr//stride,stride).mean(axis=(1,3))
            else:
                a = a[::stride,::stride]
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


def stamp(path):
    """Read the time and cycle without loading block data."""
    with path.open("rb") as stream:
        if not stream.readline().startswith(b"Athena binary output"):
            raise ValueError(f"invalid Athena output {path}")
        count = int(stream.readline().split(b"=")[-1])
        rows = dict(stream.readline().strip().split(b"=",1) for _ in range(count-1))
    return float(rows[b"time"]), int(rows[b"cycle"])


def compare(full, half, mass, prefix):
    scalar_rows, field_rows = [], []
    frames = sorted((full/"bin").glob("*.tmunu.*.bin"))
    if not frames:
        raise ValueError(f"no radiation snapshots in {full}")
    reference_times = np.array([stamp(f)[0] for f in frames])
    if np.any(np.diff(reference_times) <= 0):
        raise ValueError(f"unordered snapshot times in {full}")
    for p in sorted((half/"bin").glob("*.tmunu.*.bin")):
        t, _ = stamp(p)
        if t < reference_times[0]-1.e-12 or t > reference_times[-1]+1.e-12:
            continue
        nearest = int(np.argmin(abs(reference_times-t)))
        q = frames[nearest]
        other = nearest
        if abs(reference_times[nearest]-t) > 1.e-12:
            other = min(max(nearest+(1 if reference_times[nearest]<t else -1),0),len(frames)-1)
        fraction = ((t-reference_times[nearest])/(reference_times[other]-reference_times[nearest])
                    if other != nearest else 0.)
        a,b = profile(q,mass),profile(p,mass)
        row = dict(cycle_full=a["cycle"],cycle_half=b["cycle"],t_full_M=a["t_M"],
                   delta_t_M=b["t_M"]-a["t_M"])
        scalar_rows.append(dict(row, **{key:b[key]-a[key] for key in a if key not in ("cycle","t_M")}))
        for kind in ("tmunu","adm","con"):
            ff = read_binary(q.with_name(q.name.replace(".tmunu.",f".{kind}.")))
            hf = read_binary(p.with_name(p.name.replace(".tmunu.",f".{kind}.")))
            next_frame = None
            if other != nearest:
                neighbor = frames[other]
                next_frame = read_binary(neighbor.with_name(neighbor.name.replace(".tmunu.",f".{kind}.")))
            fb,hb = np.array(ff["mb_geometry"]),np.array(hf["mb_geometry"])
            np.testing.assert_allclose(fb[:,[1,3]].max(axis=0), hb[:,[1,3]].max(axis=0))
            fold = fb[:,2].min() < 0 and hb[:,2].min() == 0
            if fold:
                np.testing.assert_allclose(-fb[:,2].min(), hb[:,3].max())
            else:
                np.testing.assert_allclose(fb[:,2].min(), hb[:,2].min())
            key0 = ff["var_names"][0]
            nx = ff["mb_data"][key0][0].shape[-1]-(kind != "tmunu")
            h = min(np.min(fb[:,1]-fb[:,0]),np.min(hb[:,1]-hb[:,0]))/nx
            for key in ff["var_names"]:
                value = raster(ff,key,h)
                upper = raster(hf,key,h)
                if fold:
                    lower = upper[::-1] * (-1 if key in ODD else 1)
                    upper = np.concatenate((lower,upper))
                error = upper-value
                rho = (np.arange(value.shape[1])+.5)*h
                weights = np.broadcast_to(rho,value.shape)
                spatial = raster(ff,key,h,stride=2)-value
                shift = fraction*(raster(next_frame,key,h)-value) if next_frame is not None else 0.
                aligned = error-shift
                field_rows.append(dict(row, field=key, Linf=np.max(abs(error)),
                                       rms=np.sqrt(np.average(error**2, weights=weights)),
                                       matched_time_Linf=np.max(abs(aligned)),
                                       matched_time_rms=np.sqrt(np.average(aligned**2,weights=weights)),
                                       reference_sampling_Linf=np.max(abs(spatial)),
                                       reference_sampling_rms=np.sqrt(np.average(spatial**2,weights=weights)),
                                       reference_time_shift_Linf=np.max(abs(shift))))
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
