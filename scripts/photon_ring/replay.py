"""Use a separate replay directory and validated checkpoints to refine onset in slice time."""

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from campaign import ROOT, angular_grid, case_source, solver_command
from onset import qualify, fold_fit
from formation_witness import fit_witness
from formation_branches import read_binary
from compare import stamp
from smr_checkpoint import validate_restart


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def archive(source):
    return sorted(
        ((*stamp(f), f) for f in source.glob("segment-*/torus.adm.*.bin")),
        key=lambda p: p[:2],
    )


def backwards(rows, seeds, mass, cache, witness=False):
    track = []
    first_strict = None
    missed = 0
    before = None
    for t, cycle, file in reversed(rows):
        if cycle in {v["cycle"] for v in track}:
            continue
        record = cache / f"fit-{cycle}.json"
        if record.exists():
            p = json.loads(record.read_text())
        else:
            p = qualify(file, seeds, mass)
            if p and witness:
                p["witness"] = fit_witness(
                    read_binary(file), p["outer"]["coeff"], p["inner"]["coeff"]
                )
            save(record, p)
        if p:
            before = None
            seeds = [p["outer"]["coeff"], p["inner"]["coeff"]]
            track.append(p)
            if (p.get("witness") or {}).get("accepted"):
                first_strict = p
            missed = 0
        elif track:
            if missed == 0:
                before = t / mass
            missed += 1
            if missed >= 3:
                break
    return list(reversed(track)), first_strict, before


def launch(directory, checkpoint, t_end, mass, dense, source):
    published = directory
    stampfile = directory / "finished.json"
    if stampfile.exists():
        saved = json.loads(stampfile.read_text())
        rst = directory / saved["restart"]
        validate_restart(rst, t_end * mass, saved["cycle"])
        return rst
    directory = Path(
        tempfile.mkdtemp(dir=published.parent, prefix=published.name + "-work-")
    )
    link = directory / "ring.adm"
    if not link.exists():
        link.symlink_to(source / "ring.adm")
    start_cycle = validate_restart(checkpoint)[1] if checkpoint else 0
    params = (
        ["-i", str(ROOT / "smr.in")]
        if checkpoint is None
        else ["-r", str(checkpoint), "-i", str(ROOT / "finder-restart.in")]
    )
    cadence = 5 if dense else 50
    meta = json.loads((source / "ring.adm.json").read_text())
    sign = angular_grid(meta)
    cmd = solver_command(
        [
            *params,
            f"boltzmann/sign={sign}",
            f"time/tlim={t_end * mass:.17g}",
            f"fastflow/find_interval_0={cadence}",
            f"output3/dcycle={cadence if dense else 10000000}",
            "output2/dcycle=10000000",
            "output4/dcycle=10000000",
            "output5/dcycle=2400",
        ]
    )
    with (directory / "stdout").open("w") as log:
        subprocess.run(
            cmd, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    if "Terminating on time limit" not in (directory / "stdout").read_text():
        raise RuntimeError("incomplete replay")
    final = sorted((directory / "rst").glob("*.rst"))[-1]
    _, cycle = stamp(sorted((directory / "bin").glob("*.adm.*.bin"))[-1])
    validate_restart(final, t_end * mass, cycle)
    relative = final.relative_to(directory)
    save(
        directory / "finished.json",
        dict(restart=str(relative), start_cycle=start_cycle, cycle=cycle),
    )
    directory.rename(published)
    return published / relative


def main(case, source):
    work = case / "replay"
    work.mkdir(exist_ok=True)
    mass = json.loads((source / "ring.adm.json").read_text())["M_adm"]
    event = json.loads((source / "state.json").read_text())["event"]["t_M"]
    rows = archive(source)
    request = work / "window.json"
    stage = sys.argv[2]
    if stage == "plan":
        trigger = [r for r in rows if abs(r[0] / mass - event) < 1e-7]
        if not trigger:
            raise ValueError("coarse event geometry missing")
        t, cycle, file = trigger[0]
        strict = [
            r
            for f in file.parent.glob("witness-l*.json")
            for r in json.loads(f.read_text())
            if r["cycle"] == cycle and r["all_negative"]
        ]
        seed = (strict[0] if strict else json.loads((ROOT / "seed.json").read_text()))[
            "coeff"
        ]
        seeds = [np.array(seed) * 1.06, np.array(seed) * 0.92]
        coarse = work / "coarse"
        coarse.mkdir(exist_ok=True)
        track, _, before = backwards(
            [r for r in rows if r[1] <= cycle], seeds, mass, coarse
        )
        first = track[0]["t_M"] if track else event
        if before is None:
            before = max(0, first - 0.25)
        start = max(0, min(before, first) - 0.05)
        end = max(first + 0.15, event + 0.03)
        # Starting at an existing segment boundary preserves its timestep history.
        save(
            request,
            dict(
                start=float(math.floor(start)),
                end=end,
                seeds=[v.tolist() for v in seeds],
                track=track,
                event_t_M=event,
            ),
        )
        return
    plan = json.loads(request.read_text())
    start, end, track, seeds = plan["start"], plan["end"], plan["track"], plan["seeds"]
    if stage == "evolve":
        restarts = []
        for rst in source.glob("segment-*/final.rst"):
            state = json.loads((rst.parent / "state.json").read_text())
            if state["t_M"] <= start:
                validate_restart(rst, state["t_M"] * mass, state["cycle"])
                restarts.append((state["t_M"], rst))
        epoch, checkpoint = max(restarts) if restarts else (0, None)
        save(
            work / "restart-source.json",
            dict(checkpoint=str(checkpoint), checkpoint_t_M=epoch),
        )
        # Preserve the production segment boundaries when regenerating a missing earlier checkpoint.
        while epoch + 1 < start - 1e-8:
            epoch += 1
            checkpoint = launch(
                work / f"prefix-{epoch:.8f}", checkpoint, epoch, mass, False, source
            )
        if start - epoch > 1e-8:
            checkpoint = launch(
                work / "prefix-final", checkpoint, start, mass, False, source
            )
        epoch = start
        chunks = []
        while epoch < end - 1e-8:
            stop = min(end, math.floor(epoch + 1e-8) + 1.0)
            path = work / f"dense-{stop:.8f}"
            checkpoint = launch(path, checkpoint, stop, mass, True, source)
            chunks.append(path.name)
            epoch = stop
        save(work / "dense-complete.json", dict(chunks=chunks))
        return
    if stage != "reduce":
        raise ValueError(stage)
    chunks = json.loads((work / "dense-complete.json").read_text())["chunks"]
    distinct = {}
    due = set()
    previous = None
    for name in chunks:
        directory = work / name
        meta = json.loads((directory / "finished.json").read_text())
        start_cycle, end_cycle = meta["start_cycle"], meta["cycle"]
        if previous is not None and start_cycle != previous:
            raise ValueError(f"discontinuous restart history at {directory}")
        previous = end_cycle
        scheduled = set(range((start_cycle // 5 + 1) * 5, end_cycle + 1, 5))
        frames = {}
        for f in sorted((directory / "bin").glob("*.adm.*.bin")):
            t, c = stamp(f)
            frames[c] = (t, c, f)
        found_cycles = {}
        logfile = directory / "torus.cartoon_m0_horizon_0.txt"
        if scheduled or logfile.exists():
            for line in logfile.read_text().splitlines():
                if line.strip() and not line.startswith("#"):
                    fields = line.split()
                    if fields[2] == "origin":
                        found_cycles[int(fields[0])] = float(fields[1])
        if scheduled - frames.keys() or scheduled - found_cycles.keys():
            raise ValueError(
                f"incomplete five-cycle geometry or finder history in {directory}"
            )
        if any(abs(found_cycles[c] - frames[c][0]) > 1.0e-9 for c in scheduled):
            raise ValueError(f"finder and ADM timestamps disagree in {directory}")
        due.update(scheduled)
        distinct.update(frames)
    if not due:
        raise ValueError("dense window contains no scheduled finder calls")
    dense = sorted(distinct.values())
    fits = work / "dense-fits"
    fits.mkdir(exist_ok=True)
    effective = [track[-1][k]["coeff"] for k in ["outer", "inner"]] if track else seeds
    found, strict, before = backwards(dense, effective, mass, fits, True)
    result = dict(
        status="needs_review",
        cadence=5,
        verified_finder_epochs=len(due),
        window=[start, end],
        previous_without_pair_M=before,
    )
    turning = {}
    if found:
        result.update(first_pair_M=found[0]["t_M"], fold=fold_fit(found[:20]))
    if found:
        from dense_fold import dense_fold

        turning = dense_fold(dense, found, mass)
        save(work / "turning-point.json", turning)
        result["turning_point_status"] = turning["status"]
    if strict:
        index = found.index(strict)
        # Match the coarse event on this replay too. Different restart endpoint timesteps
        # can change cycles, so compare the nearest dense geometry in physical time.
        near = min(found, key=lambda p: abs(p["t_M"] - event))
        old = next((r for r in track if abs(r["t_M"] - event) < 1e-7), None)
        cadence_dt = float(np.median(np.diff(sorted(set(v[0] / mass for v in dense)))))
        consistent = old is not None and abs(near["t_M"] - event) <= 1.5 * cadence_dt
        chi = strict["outer"]["chi"]
        neighbors = [p for p in found[max(0, index - 1) : index + 2] if p is not strict]
        errors = [abs(chi - strict["order40"]["chi"])] + [
            abs(chi - p["outer"]["chi"]) for p in neighbors
        ]
        if old is not None:
            errors.append(abs(near["outer"]["chi"] - old["outer"]["chi"]))
        temporal_ok = (
            before is not None and found[0]["t_M"] - before <= 3.5 * cadence_dt
        )
        complete = (
            temporal_ok
            and len(neighbors) > 0
            and consistent
            and strict["outer"]["min_cells"] >= 20
        )
        result.update(
            status="verified" if complete else "needs_review",
            first_strict_M=strict["t_M"],
            chi=chi,
            chi_error=max(errors),
            zeta=strict["outer"]["zeta"],
            comparison_time_M=near["t_M"],
            comparison_coarse_chi=(old or {}).get("outer", {}).get("chi"),
            inner_chi=strict["inner"]["chi"],
        )
    from horizon_history import export

    if json.loads((source / "state.json").read_text())["status"] != "collapse":
        raise ValueError("production tail is incomplete")
    result["formation_history"] = export(
        work, mass, dense, found, strict, rows, turning, effective
    )
    save(work / "result.json", result)


if __name__ == "__main__":
    case = ROOT / "cases" / sys.argv[1]
    source = case_source(sys.argv[1])
    main(case, source)
