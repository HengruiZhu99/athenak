"""Reduce a matched pair and look for strictly trapped witnesses near the first marginal detection."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from adm_spheres import charges
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
                energy_budget_relative=float(np.max(abs(energy+escaped-work-(energy[0]+escaped[0]-work[0])))/energy[0]),
                J_budget_relative=float(np.max(abs(angular+escaped_j-(angular[0]+escaped_j[0])))/abs(angular[0])),
                history_rows=len(history), final_constraint_sums=constraints[-1,2:6].tolist())


def check_online_interiors(folder, frames, mass):
    """Verify the first reported interior on each signed branch with native geometry."""
    path = folder/"torus.cartoon_m0_inner_0.txt"
    if not path.exists():
        return []
    by_cycle = {stamp(f)[1]:f for f in frames}
    theta = np.r_[0., np.linspace(1.e-5,np.pi-1.e-5,4097), np.pi]
    verified, trials = set(), []
    for line in path.read_text().splitlines():
        row = line.split()
        if not row or row[0].startswith("#") or row[4] != "1" or row[2] in verified:
            continue
        cycle, branch, center = int(row[0]), row[2], float(row[5])
        record = dict(cycle=cycle, branch=branch, center_z=center, t_M=float(row[1])/mass,
                      verified=False)
        trials.append(record)
        if cycle not in by_cycle:
            record["failure"] = "no matching native geometry"
            continue
        coeff = np.array(row[10:], dtype=float)
        coeff *= np.sqrt((2*np.arange(coeff.size)+1)/(4*np.pi))
        try:
            patch = NativePatch(read_binary(by_cycle[cycle]))
            checks = []
            for metric in (patch, LocalPolynomial(patch,3), LocalPolynomial(patch,5)):
                h, plus, minus, _, _ = measure(metric, coeff, theta, center)
                checks.append([float(max(h*plus)), float(max(h*minus))])
            record["max_expansions"] = checks
            record["verified"] = all(max(pair)<0 for pair in checks)
            if record["verified"]:
                verified.add(branch)
        except (ValueError, np.linalg.LinAlgError) as error:
            record["failure"] = str(error)
    return trials


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
    seed = args.seed or args.output.parent/"j09508-trapped.json"
    mass=json.loads(args.metadata.read_text())["M_adm"]
    compare(args.full,args.half,mass,args.output/"comparison")
    compare_profiles(args.full,args.half,mass,args.output/"timed-profiles.csv")
    with (args.output/"timed-profiles.csv").open() as stream:
        profiles = list(csv.DictReader(stream))
    result={}
    for side, folder in (("full",args.full),("half",args.half)):
        result[side]=summarize_run(folder)
        frames=sorted((folder/"bin").glob("*.adm.*.bin"))
        result[side]["R17_charges"] = [charges(read_binary(f),17) for f in (frames[0],frames[-1])]
        result[side]["reported_interiors"] = check_online_interiors(folder, frames, mass)
        first, selected=window(folder)
        result[side]["first_online_marginal_cycle"]=first
        if first is None and seed.is_file():
            # Check closest approach on the actual centroid and the final slice
            # even when no online marginal candidate crossed its tolerance.
            def radius(row):
                mean = float(row["mean_Rcirc_M"])
                return mean-float(row["delta_mean_Rcirc_M"]) if side=="full" else mean
            closest = mass*float(min(profiles,key=radius)["t_M"])
            frames = sorted((folder/"bin").glob("*.adm.*.bin"))
            selected = sorted(set(sorted(frames,key=lambda path:abs(stamp(path)[0]-closest))[:5]
                                  + frames[-1:]))
            result[side]["witness_window"] = "closest approach and terminal frame"
        if selected and seed.is_file():
            result[side]["witness_seed"] = str(seed)
            result[side]["witness_frames"] = [f.name for f in selected]
            accepted=[]
            sampled_frames=set()
            for order, expansion in [(40,-.002),(48,-.0005)]:
                target=args.output/f"{side}-l{order}-witnesses.json"
                subprocess.run([sys.executable,str(Path(__file__).with_name("witness.py")),
                                str(seed),str(target),*map(str,selected),
                                f"--target={expansion}",f"--order={order}"],check=True)
                fits = json.loads(target.read_text())
                accepted.extend(v for v in fits if v["all_negative"])
                sampled_frames.update(v["file"] for v in fits if "checks" in v)
            result[side]["witness_frames_with_whole_surface_samples"] = sorted(sampled_frames)
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
                outer_seed = json.loads(seed.read_text())["enclosing_horizon"]["coeff"]
                measured["enclosing_fit"] = enclosing_surface(folder/"bin"/v["file"], v["coeff"], outer_seed)
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
