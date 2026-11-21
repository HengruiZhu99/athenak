"""Summarize completed StellarAI intervals. GPU-hours include initialization and I/O."""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


def summarize(folder):
    text = (folder/"stdout").read_text()
    if "Terminating on time limit" not in text:
        raise ValueError(f"incomplete evolution in {folder}")
    devices = [json.loads(row) for row in text.splitlines() if row.startswith('{"host"')]
    hardware = {(row["host"], row["pci"]) for row in devices}
    if len(hardware) != len(devices) or not hardware:
        raise ValueError("missing or duplicated GPUs")
    steps = int(re.search(r"(?m)^time=[^ ]+ cycle=(\d+)", text)[1])
    blocksteps = int(re.search(r"MeshBlock-cycles = (\d+)", text)[1])
    seconds = float((folder/"wall.seconds").read_text())
    with (folder/"gpu-memory.csv").open() as stream:
        samples = [row for row in csv.reader(stream) if len(row)==3 and row[-1].strip().isdigit()]
    history = np.loadtxt(folder/"torus.z4c.user.hst", ndmin=2)
    with (folder/"z4c_timestep_contract.csv").open() as stream:
        times = {int(v["cycle"]):float(v["time"]) for v in csv.DictReader(stream)}
    times[steps] = history[-1,0]
    count = int(re.search(r"Total number of MeshBlocks = (\d+)", text)[1])
    changes = {int(c):int(add)-int(drop) for c,add,drop in re.findall(
        r"CARTOON_AMR_TRANSACTION cycle=(\d+).*?created_leaves=(\d+) deleted_leaves=(\d+)",text)}
    weighted, check = 0., 0
    for cycle in range(steps):
        count += changes.get(cycle,0)
        weighted += count*(times[cycle+1]-times[cycle])
        check += count
    if check != blocksteps:
        raise ValueError(f"block census {check} differs from driver {blocksteps}")
    time_weighted = weighted / (times[steps]-times[0])
    return dict(layout=folder.name, host=devices[0]["host"], GPUs=len(hardware), steps=steps,
                end_time=history[-1,0], blocks_per_step=blocksteps/steps,
                time_mean_blocks=time_weighted, wall_seconds=seconds,
                GPU_hours=seconds*len(hardware)/3600,
                sampled_peak_MiB=max(int(row[-1]) for row in samples))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    for run in args.runs:
        print(json.dumps(summarize(run)))
