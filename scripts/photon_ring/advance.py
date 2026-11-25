"""Advance one Slurm campaign. Run periodically on a login node."""

import argparse
from decimal import Decimal
import fcntl
import getpass
import json
import re
import subprocess

from campaign import CODE, ROOT, case_source, load
from smr_checkpoint import validate_restart
from survey import propose

STAGES = {
    "gpu": ("evolve", "worker.py"),
    "cal": ("calibrate", "calibrate.py"),
    "plan": ("fit", "replay.py"),
    "reduce": ("fit", "replay.py"),
    "replay": ("evolve", "replay.py"),
}


def publish(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def submit(key, stage, scan, profile, queued):
    case = ROOT / "cases" / key
    # Preserve the job names and journals of campaigns started with the width dispatcher.
    prefix = scan["name"]
    stage_prefix = prefix if stage == "gpu" else f"{prefix}-{stage}"
    if scan.get("legacy_width_names") and stage != "gpu":
        stage_prefix = "w" + stage
    name = f"{stage_prefix}-{key}"
    if name in queued:
        return
    log = case / ("jobs.txt" if stage == "gpu" else f"{stage}-jobs.txt")
    if log.exists() and log.stat().st_size:
        last = log.read_text().splitlines()[-1].split()[-1]
        states = subprocess.check_output(
            ["sacct", "-X", "-j", last, "-n", "-P", "-o", "State"], text=True
        ).splitlines()
        state = states[0].strip() if states else ""
        if not (
            state == "COMPLETED"
            and stage == "gpu"
            or state in {"TIMEOUT", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}
        ):
            print("Await accounting or inspect allocation", last, stage, state)
            return
    resource, script = STAGES[stage]
    slurm = profile["slurm"]
    cmd = [
        "sbatch",
        "--parsable",
        *slurm["common"],
        *slurm[resource],
        "-J",
        name,
        "-o",
        str(case / f"{stage}-%j.log"),
        str(CODE / "stage.sh"),
        str(ROOT),
        str(CODE / script),
        key,
    ]
    if script == "replay.py":
        cmd.append("evolve" if stage == "replay" else stage)
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode and any(
        s in run.stderr for s in ("MaxSubmitJob", "AssocMaxSubmit", "QOSGrpSubmit")
    ):
        print("Slurm submission limit, will retry", name)
        return
    if run.returncode:
        raise RuntimeError(f"{name}: {run.stderr.strip()}")
    job = run.stdout.strip().split(";")[0]
    if not job.isdecimal():
        raise ValueError(f"unexpected sbatch response {run.stdout!r}")
    with log.open("a") as stream:
        print(stage, job, file=stream)
    queued.add(name)
    print("submitted", name, job)


def advance(dry_run=False):
    scan, profile = load("scan.json"), load("cluster.json")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", scan["name"]):
        raise ValueError("use a distinct short Slurm job prefix in scan.name")
    if scan["axis"] not in ("width", "spin") or not set(scan["objectives"]) <= {
        "threshold",
        "onset spin",
    }:
        raise ValueError("unsupported scan axis or objective")
    path = ROOT / "targets.json"
    targets = (
        load("targets.json") if path.exists() else {k: "coarse" for k in scan["coarse"]}
    )
    if any(not re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d+)?", k) for k in targets):
        raise ValueError("case keys must be decimal numbers in ordinary notation")
    outcomes, spins, summary, invalid = {}, {}, {}, set()
    stages = []
    for k in targets:
        value = Decimal(k)
        case = ROOT / "cases" / k
        source = case_source(k)
        if (case / "invalid.json").exists():
            invalid.add(value)
            continue
        if (source / "state.json").exists():
            state = json.loads((source / "state.json").read_text())
            mass = json.loads((source / "ring.adm.json").read_text())["M_adm"]
            validate_restart(
                source / f"segment-{state['segment']:03d}/final.rst",
                state["t_M"] * mass,
                state["cycle"],
            )
            summary[k] = {s: state[s] for s in ("status", "t_M", "event")}
            if state["status"] != "active":
                outcomes[value] = state["status"]
            result = case / "replay/result.json"
            if result.exists():
                r = json.loads(result.read_text())
                summary[k]["replay"] = r
                if r["status"] == "verified" and outcomes.get(value) == "collapse":
                    spins[value] = r["chi"], r["chi_error"]
            elif (state.get("event") or {}).get("kind") == "collapse":
                replay = case / "replay"
                stage = (
                    "reduce"
                    if (replay / "dense-complete.json").exists()
                    else "replay"
                    if (replay / "window.json").exists()
                    else "plan"
                )
                # A published history must include the completed post-trapping tail.
                if stage != "reduce" or state["status"] == "collapse":
                    stages.append((k, stage))
        elif k in scan.get("archived", {}):
            raise FileNotFoundError(source / "state.json")

    proposals, reason = propose(outcomes, spins, invalid, scan)
    for value, role in proposals.items():
        if not any(
            r == role and Decimal(k) not in outcomes and Decimal(k) not in invalid
            for k, r in targets.items()
        ) and value not in map(Decimal, targets):
            targets[format(value.normalize(), "f")] = role
    for k in targets:
        if (
            Decimal(k) not in outcomes
            and Decimal(k) not in invalid
            and k not in scan.get("archived", {})
        ):
            calibrated = (ROOT / "cases" / k / "coverage.json").is_file()
            stages.append((k, "gpu" if calibrated else "cal"))
    status = dict(reason=reason, cases=summary)
    if dry_run:
        print(json.dumps(dict(targets=targets, stages=stages, **status), indent=2))
        return
    publish(path, targets)
    publish(ROOT / "survey-state.json", status)
    queued = set(
        subprocess.check_output(
            ["squeue", "-u", getpass.getuser(), "-h", "-o", "%j"], text=True
        ).splitlines()
    )
    for k, stage in sorted(stages, key=lambda item: item[1] not in {"cal", "gpu"}):
        (ROOT / "cases" / k).mkdir(parents=True, exist_ok=True)
        submit(k, stage, scan, profile, queued)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.dry_run:
        advance(True)
    else:
        with (ROOT / "dispatch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            advance()
