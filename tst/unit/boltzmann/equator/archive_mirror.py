"""Measure the existing full solver's own reflection defect on saved meridional meshes."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from compare import ODD, raster, stamp
from diagnostics import profile_frames, read_binary
from replay import upper_only


def defects(frame):
    bounds = np.asarray(frame["mb_geometry"])
    lookup = {tuple(v[:4]):i for i,v in enumerate(bounds)}
    pairs = []
    for i,(r0,r1,z0,z1,*_) in enumerate(bounds):
        if z0 < 0:
            continue
        pairs.append((i,lookup.get((r0,r1,-z1,-z0))))
    unmatched = sum(j is None for i,j in pairs)
    worst, field = -1., ""
    for key in frame["var_names"]:
        sign = -1 if key in ODD else 1
        if unmatched:
            h = min(bounds[:,1]-bounds[:,0])/(frame["mb_data"][key][0].shape[-1]-
                                               (not key.startswith("tmunu_")))
            grid = raster(frame,key,h)
            south,north = np.split(grid,2)
            e = np.max(abs(north-sign*south[::-1]))
        else:
            data = frame["mb_data"][key]
            e = max(np.max(abs(data[i][0]-sign*data[j][0,::-1])) for i,j in pairs)
        if e > worst:
            worst, field = float(e),key
    return unmatched,worst,field


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mass",type=float)
    p.add_argument("segments",type=Path,nargs="+")
    p.add_argument("--every",type=int,default=20)
    a = p.parse_args()
    # Restart segments can contain a forced dump at the same physical time.
    frames = {stamp(path)[0]:path for segment in a.segments
              for path in sorted((segment/"bin").glob("*.tmunu.*.bin"))}
    frames = [v for _,v in sorted(frames.items())]
    selected = frames[::a.every]+[frames[-1]]
    selected += [v for v in frames if 14.32 < stamp(v)[0]/a.mass < 14.6]
    out = None
    for path in sorted(set(selected),key=lambda v:stamp(v)[0]):
        matter=read_binary(path)
        adm=read_binary(path.with_name(path.name.replace(".tmunu.",".adm.")))
        con=read_binary(path.with_name(path.name.replace(".tmunu.",".con.")))
        full = profile_frames(matter,adm,a.mass)
        half = profile_frames(upper_only(matter),upper_only(adm),a.mass)
        row = dict(cycle=full["cycle"],t_M=full["t_M"],centroid_M=full["mean_Rcirc_M"],
                   outer_decile_dx=full["outer_decile_mean_dx"])
        for key in ("mean_Rcirc_M","vertical_stress","E","J","p90_Rcirc_M"):
            row["north_minus_full_"+key]=half[key]-full[key]
        for label, frame in (("matter",matter),("adm",adm),("constraints",con)):
            unmatched, error, key = defects(frame)
            row["unmatched_northern_leaves"]=unmatched
            row[label+"_mirror_Linf"]=error
            row[label+"_worst_field"]=key
        if out is None:
            out=csv.DictWriter(sys.stdout,fieldnames=row,lineterminator="\n")
            out.writeheader()
        out.writerow(row)
        sys.stdout.flush()
