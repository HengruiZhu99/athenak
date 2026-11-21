"""Reduce a matched pair and look for strictly trapped witnesses near the first marginal detection."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from compare import compare, stamp
from endpoint_report import write_report
from profile_comparison import compare_profiles
from witness import LocalPolynomial, NativePatch, measure, read_binary, solve_surface


def enclosing_surface(frame, interior, seed, stride=1):
    """Select a resolved outer marginal branch enclosing the verified interior everywhere."""
    native = read_binary(frame)
    try:
        patch = NativePatch(native, stride)
        shape = solve_surface(patch, seed[0], 0., 48, 70, seed)
    except (ValueError, np.linalg.LinAlgError) as error:
        return dict(accepted=False, failure=str(error))
    theta = np.r_[0., np.linspace(1.e-5, np.pi-1.e-5, 4097), np.pi]
    inner_radius = np.polynomial.legendre.legval(np.cos(theta), interior)
    checks = {}
    for name, metric in (("spline", patch), ("cubic", LocalPolynomial(patch, 3)),
                         ("quintic", LocalPolynomial(patch, 5))):
        h, plus, minus, _, _ = measure(metric, shape["coeff"], theta)
        checks[name] = dict(residual=float(np.max(abs(h*plus))), max_in=float(max(h*minus)),
                            min_gap=float(min(h-inner_radius)))
    # Use the production fastflow tolerance, retain the actual residual for comparison.
    accepted = all(v["residual"] < .006 and v["max_in"] < 0 and v["min_gap"] > 0
                   for v in checks.values())
    return dict(accepted=accepted, dx=patch.dx, checks=checks, **shape)



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
        _, cycle = stamp(f)
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
    compare_profiles(args.full,args.half,mass,args.output/"timed-profiles.csv")
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
                seed = json.loads(args.seed.read_text())["enclosing_horizon"]["coeff"]
                measured["enclosing_fit"] = enclosing_surface(folder/"bin"/v["file"], v["coeff"], seed)
                fine = measured["enclosing_fit"]
                if fine["accepted"]:
                    coarse = enclosing_surface(folder/"bin"/v["file"], v["coeff"], fine["coeff"], 2)
                    measured["coarse_geometry_enclosing_fit"] = coarse
                    if coarse["accepted"]:
                        keys = ("area", "surface_mass", "surface_spin_ratio")
                        measured["geometric_sampling_change"] = {
                            key:abs(coarse[key]-fine[key]) for key in keys}
                    earlier = [stamp(f)[0] for f in selected if stamp(f)[0] < v["time"]]
                    if earlier:
                        measured["previous_interval_M"] = (v["time"]-max(earlier))/mass
                    following = next((f for f in selected if stamp(f)[0] > v["time"]), None)
                    if following:
                        neighbor = enclosing_surface(following, v["coeff"], fine["coeff"])
                        if neighbor["accepted"]:
                            measured["next_cadence_change"] = {
                                key:abs(neighbor[key]-fine[key])
                                for key in ("area","surface_mass","surface_spin_ratio")}
                result[side]["first_verified_in_window"]=measured
    (args.output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    write_report(result, args.full, args.half, args.output)
