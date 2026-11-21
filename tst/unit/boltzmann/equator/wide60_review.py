"""Classify the separate wide-domain 1.0600 evolution using its restart ancestry."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from compare import stamp
from diagnostics import profile
from endpoint_review import check_online_interiors


def review(final, mass, output, seed):
    chain = [final.resolve()]
    while (chain[-1]/"restart.path").is_file():
        checkpoint = Path((chain[-1]/"restart.path").read_text().strip()).resolve()
        parent = checkpoint.parent.parent
        if parent in chain or not (parent/"stdout").is_file():
            raise ValueError(f"broken restart ancestry at {checkpoint}")
        chain.append(parent)
    chain.reverse()
    frames, samples, branches = {}, {}, {}
    marginal, interiors = set(), []
    for folder in chain:
        geometry = []
        for path in sorted((folder/"bin").glob("*.tmunu.*.bin")):
            adm = path.with_name(path.name.replace(".tmunu.",".adm."))
            time, cycle = stamp(path)
            if stamp(adm) != (time,cycle):
                raise ValueError(f"unmatched geometry and radiation {path}")
            geometry.append(adm)
            if cycle in samples and abs(samples[cycle]["t_M"]-time/mass)>1.e-9:
                raise ValueError(f"overlapping restart at different times on cycle {cycle}")
            frames[cycle], samples[cycle] = adm, profile(path,mass)
        interiors.extend(check_online_interiors(folder,geometry,mass))
        for line in (folder/"torus.cartoon_m0_horizon_0.txt").read_text().splitlines():
            fields=line.split()
            if fields and not fields[0].startswith("#"):
                cycle=int(fields[0])
                branches.setdefault(cycle,set()).add(fields[2])
                if fields[3]=="1":
                    marginal.add(cycle)

    cycles=sorted(samples)
    if not cycles or samples[cycles[0]]["t_M"] != 0:
        raise ValueError("missing initial slice")
    output.mkdir(parents=True,exist_ok=True)
    with (output/"centroid.csv").open("w") as out:
        writer=csv.DictWriter(out,samples[cycles[0]],lineterminator="\n")
        writer.writeheader()
        writer.writerows(samples[c] for c in cycles)

    closest=min(cycles,key=lambda c:samples[c]["mean_Rcirc_M"])
    centers={closest, cycles[-1]}
    if marginal:
        centers.add(min(marginal))
    selected=sorted({c for center in centers for c in cycles if abs(c-center)<=240})
    accepted=[]
    tested=set()
    for degree,target in [(40,-.002),(48,-.0005)]:
        # The cadence and file numbering can restart across directories. Preserve cycle order.
        # witness.py writes per-frame expansions and checks the poles and equator.
        path=output/f"centered-degree{degree}.json"
        subprocess.run([sys.executable,str(Path(__file__).with_name("witness.py")),str(seed),
                        str(path),*[str(frames[c]) for c in selected],
                        f"--order={degree}",f"--target={target}"],check=True)
        fits=json.loads(path.read_text())
        accepted.extend(row["cycle"] for row in fits if row["all_negative"])
        tested.update(row["cycle"] for row in fits if "checks" in row)

    last=samples[cycles[-1]]
    # The input's fastflow/find_interval_0 is 120 and all three branches are enabled.
    missing=[c for c in range(120,cycles[-1]+1,120)
             if branches.get(c,set()) != {"origin","plus","minus"}]
    positive=[v["cycle"] for v in interiors if v["verified"]] + accepted
    if positive:
        outcome="collapse"
    elif abs(last["t_M"]-60)>1.e-7 or missing or marginal or interiors:
        outcome="unresolved"
    elif last["mean_Rcirc_M"]>2.5:
        outcome="dispersal"
    else:
        outcome="indeterministic"
    result=dict(outcome=outcome,t_M=last["t_M"],mean_Rcirc_M=last["mean_Rcirc_M"],
                ancestry=[str(f) for f in chain],scheduled_searches=len(branches),
                missing_search_cycles=missing,online_marginal_cycles=sorted(marginal),
                online_interiors=interiors,independent_centered_cycles_attempted=selected,
                independent_centered_cycles_with_whole_surface_checks=sorted(tested),
                earliest_verified_cycle=min(positive,default=None))
    (output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(outcome,last["t_M"],last["mean_Rcirc_M"])


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("final",type=Path);cli.add_argument("metadata",type=Path)
    cli.add_argument("output",type=Path);cli.add_argument("seed",type=Path)
    args=cli.parse_args()
    review(args.final,json.loads(args.metadata.read_text())["M_adm"],args.output,args.seed)
