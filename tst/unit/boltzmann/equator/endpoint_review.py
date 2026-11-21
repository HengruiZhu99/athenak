"""Reduce a matched pair and look for strictly trapped witnesses near the first marginal detection."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from compare import compare


def summarize_run(folder):
    history=np.loadtxt(folder/"torus.user.hst", ndmin=2)
    constraints=np.loadtxt(folder/"torus.z4c.user.hst", ndmin=2)
    energy, angular, escaped, escaped_j, work=history[:,2:7].T
    # Rotational symmetry removes axial torque; energy includes metric work.
    return dict(start=float(history[0,0]),end=float(history[-1,0]),
                energy_budget_relative=float(np.max(abs(energy+escaped-work-energy[0]))/energy[0]),
                J_budget_relative=float(np.max(abs(angular+escaped_j-angular[0]))/abs(angular[0])),
                history_rows=len(history), final_constraint_sums=constraints[-1,2:6].tolist())


def window(folder):
    log=folder/"torus.cartoon_m0_horizon_0.txt"
    rows=[line.split() for line in log.read_text().splitlines() if line and not line.startswith("#")]
    found=next((int(row[0]) for row in rows if row[3]=="1"),None)
    if found is None:
        return None, []
    files=sorted((folder/"bin").glob("*.adm.*.bin"))
    selected=[]
    for f in files:
        with f.open("rb") as stream:
            for _ in range(10):
                line=stream.readline()
                if line.startswith(b"  cycle=") or line.strip().startswith(b"cycle="):
                    cycle=int(line.split(b"=")[1])
                    break
            else:
                raise ValueError(f"missing cycle in {f}")
        if cycle >= found-2*120:
            selected.append(f)
    return found, selected


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("full",type=Path)
    p.add_argument("half",type=Path)
    p.add_argument("metadata",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--seed",type=Path)
    args=p.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    mass=json.loads(args.metadata.read_text())["M_adm"]
    compare(args.full,args.half,mass,args.output/"comparison")
    result={}
    for side, folder in (("full",args.full),("half",args.half)):
        result[side]=summarize_run(folder)
        first, selected=window(folder)
        result[side]["first_online_marginal_cycle"]=first
        if selected and args.seed:
            accepted=[]
            for order, expansion in [(40,-.002),(48,-.0005)]:
                target=args.output/f"{side}-l{order}-witnesses.json"
                subprocess.run([sys.executable,str(Path(__file__).with_name("witness.py")),
                                str(args.seed),str(target),*map(str,selected),
                                f"--target={expansion}",f"--order={order}"],check=True)
                accepted.extend(v for v in json.loads(target.read_text()) if v["all_negative"])
            if accepted:
                v=min(accepted,key=lambda t:t["time"])
                measured={k:v[k] for k in
                    ("file","cycle","time","area","surface_mass","surface_spin_ratio","zones","checks")}
                measured["t_M"]=v["time"]/mass
                native=(folder/"torus.cartoon_m0_horizon_0.txt").read_text().splitlines()
                candidates=[line.split() for line in native if line.startswith(str(v["cycle"])+" ")]
                enclosing=[r for r in candidates if r[3]=="1" and r[2]=="origin"]
                if enclosing:
                    r=enclosing[0]
                    measured["online_origin_marginal"]=dict(area=float(r[7]),mass=float(r[9]),
                                                spin=float(r[10])/float(r[9])**2,
                                                direct_residual=float(r[13]))
                result[side]["first_verified_in_window"]=measured
    (args.output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
