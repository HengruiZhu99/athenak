"""Continue one StellarAI sequence from a successfully closed native checkpoint."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from magnetic_smooth_state import checkpoint_state


def final_target(state, period):
    """Cover a full ordinary sweep and most of the following resolved visit."""
    t = state["time"]
    zone = state["zone"]
    dwell = state["parameters"]
    outer = int(dwell["amr_criterion1/nlevels"]) - 1
    bridge = sum(float(dwell[f"amr_criterion1/trun_fac_{z}"]) for z in range(1, outer))
    span = 2 * bridge + sum(float(dwell[f"amr_criterion1/trun_fac_{z}"]) for z in (0, outer))
    span += 0.55 * period + 10
    if zone == 0:
        return t + span
    direction = state["direction"]
    arrival = max(t, state["next_time"])
    while True:
        zone += direction
        if zone == 0:
            break
        if zone == outer:
            direction = -1
        arrival += float(dwell[f"amr_criterion1/trun_fac_{zone}"])
    # Each of at most nine time-triggered transitions can overshoot by one step.
    # Audit the actual finest exposure from the history after completion.
    return arrival + span


def run(sep, predecessor, nodes, paired):
    base = Path.cwd()
    physical = base / f"seq{sep}"
    out = base / (f"seq{sep}-replay" if paired else f"seq{sep}")
    first_replay = paired and not out.exists()
    if first_replay:
        (out / "rst").mkdir(parents=True)
        os.link(physical / "comparison-start.rst", out / "rst" / f"mad{sep}.00000.rst")
    log = (physical if first_replay else out) / f"slurm-{predecessor}.log"
    if "Terminating on " not in log.read_text():
        raise RuntimeError(f"No clean time-limit exit in {log}")
    checkpoints = sorted((out / "rst").glob(f"mad{sep}.[0-9][0-9][0-9][0-9][0-9].rst"))
    checkpoint = checkpoints[-1]
    state = checkpoint_state(checkpoint)
    print(f"Continuing {checkpoint} at t={state['time']} zone={state['zone']}", flush=True)
    period = 2 * math.pi * sep**1.5
    params = state["parameters"]
    options = []
    target = float(params["time/tlim"])
    ordinary_end = 200 * period
    if state["time"] >= target * (1 - 1e-12):
        if target > ordinary_end + 1:
            raise RuntimeError("Sequence already reached its terminal checkpoint")
        target = final_target(state, period)
        options.append(f"time/tlim={target:.17g}")
        if not paired:
            anchor = physical / "comparison-start.rst"
            if not anchor.exists():
                os.link(checkpoint, anchor)
            elif not os.path.samefile(checkpoint, anchor):
                raise RuntimeError(f"Unexpected existing comparison checkpoint {anchor}")
            options.append(f"amr_criterion1/trun_fac_0={16*period:.17g}")
            if state["zone"] == 0:
                options.append("cyclic_zoom/restart_reset_finest_dwell=true")
    # Finalize writes every output at wall time. Restore the ordinary physical
    # cadence so short segments cannot progressively postpone scheduled dumps.
    for key, value in params.items():
        if key.startswith("output") and key.endswith("/dt") and float(value) > 0:
            dt = float(value)
            options.append(f"{key[:-2]}last_time={math.floor(state['time']/dt)*dt:.17g}")
    # These files are rolling campaign scratch. Preserve two closed checkpoints
    # and the separately linked scientific anchor. Only rotate after a clean exit.
    for old in checkpoints[:-2]:
        old.unlink()
    subprocess.run(
        ["bash", "./run_stellar.sh", str(sep), *options],
        check=True,
        env={**os.environ, "ATHENA_RUN_DIR": str(out)},
    )
    newest = sorted((out / "rst").glob(f"mad{sep}.*.rst"))[-1]
    end = checkpoint_state(newest)
    if end["cycle"] <= state["cycle"] or end["time"] <= state["time"]:
        raise RuntimeError("No forward checkpoint progress")
    print(json.dumps({k: end[k] for k in ("time", "cycle", "zone", "id")}), flush=True)
    done = end["time"] >= target * (1 - 1e-12) and target > ordinary_end + 1
    if done and paired:
        print("Completed ordinary-cycle replay of the extended finest interval", flush=True)
        scripts = Path(__file__).resolve().parent
        subprocess.run(
            [
                "sbatch",
                "--parsable",
                f"--dependency=afterok:{os.environ['SLURM_JOB_ID']}",
                "-N1",
                "-n1",
                "-c1",
                "--mem=12G",
                "-p",
                "cpu",
                "-A",
                "fpretori",
                "-t",
                "00:30:00",
                "-J",
                f"mad{sep}-results",
                str(scripts / "qualify_stellar.sh"),
                str(scripts),
                str(sep),
            ],
            check=True,
        )
        return
    if done:
        if end["zone"] != 0:
            raise RuntimeError("Final resolved interval ended on a coarse zone")
        print("Completed extended finest exposure, scheduling ordinary replay", flush=True)
    submit(sep, nodes, paired or done)


def submit(sep, nodes, paired):
    job = os.environ["SLURM_JOB_ID"]
    cmd = [
        "sbatch",
        "--parsable",
        f"--dependency=afterok:{job}",
        f"--nodes={nodes}",
        "--ntasks-per-node=8",
        "-c4",
        "--mem=450G",
        "--gres=gpu:8",
        "-p",
        "gpu",
        "--qos=gpu-medium",
        "-A",
        "fpretori",
        "-t",
        "12:00:00",
        "--export=ALL,ATHENA_WALL_LIMIT=11:48:00",
        "-J",
        f"mad{sep}-continue",
        "./segment.sh",
        str(sep),
        job,
        str(nodes),
    ]
    if paired:
        cmd.append("--paired")
    print("Successor", subprocess.check_output(cmd, text=True).strip(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("separation", type=int, choices=[25, 15, 35])
    parser.add_argument("predecessor", type=int)
    parser.add_argument("nodes", type=int, choices=[1, 2, 3])
    parser.add_argument("--paired", action="store_true")
    args = parser.parse_args()
    run(args.separation, args.predecessor, args.nodes, args.paired)
