"""Compare the large-domain block choices from one completed --wide allocation."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
from diagnostics import profile
from performance import summarize


def record(folder, mass):
    row=summarize(folder)
    text=(folder/"stdout").read_text()
    deck=(folder/"input").read_text().split("<meshblock>")[1].split("<")[0]
    bx,bz=[int(re.search(rf"(?m)^nx{i}\s*=\s*(\d+)",deck)[1]) for i in (1,2)]
    elapsed={int(n):float(t) for t,n in re.findall(r"(?m)^elapsed=([.\de+\-]+) cycle=(\d+)",text)}
    row.update(block_cells=[bx,bz],mean_active_cells=bx*bz*row["time_mean_blocks"],
               cell_updates_per_second=bx*bz*row["blocks_per_step"]*row["steps"]/row["wall_seconds"])
    # Identical first 50 steps allow a per-step comparison before regridding.
    changed=[int(n) for n in re.findall(r"CARTOON_AMR_TRANSACTION cycle=(\d+)",text)]
    if 0 in elapsed and 50 in elapsed and all(n>=50 for n in changed):
        row["pre_regrid_50_steps_s"]=elapsed[50]-elapsed[0]
    profiles=[profile(f,mass) for f in sorted((folder/"bin").glob("*.tmunu.*.bin"))]
    row["initial"] = profiles[0]
    row["final"] = profiles[-1]
    history=np.loadtxt(folder/"torus.user.hst",ndmin=2)
    e,j,escaped,escaped_j,work=history[:,2:7].T
    row["max_rel_E_budget"]=float(np.max(np.abs(e+escaped-work-e[0]))/e[0])
    row["max_rel_J_budget"]=float(np.max(np.abs(j+escaped_j-j[0]))/abs(j[0]))
    return row


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("allocation",type=Path)
    cli.add_argument("metadata",type=Path,help="initial ring.adm.json")
    args=cli.parse_args()
    mass=json.loads(args.metadata.read_text())["M_adm"]
    rows=[record(folder,mass) for folder in sorted(args.allocation.iterdir()) if folder.is_dir()]
    if len({(r["host"],tuple(r["pci_devices"]),r["executable_sha256"],r["initial_data_sha256"],
             r["start_time"],r["end_time"]) for r in rows})!=1:
        raise ValueError("expected the same physical devices, binary, initial data and time interval")
    print(json.dumps(rows,indent=2))
