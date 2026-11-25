"""Publish signed extremality and stability for measured and reconstructed MOTSs."""

import csv
from onset import continued_outer, qualify


def measurements(row, mass):
    return {
        **{k: row[k] for k in ("area", "J", "zeta", "chi", "min_cells")},
        "lambda0": row["lambda0"],
        "lambda0_Madm2": row["lambda0"] * mass**2,
        "max_abs_hTheta_plus": max(r[0] for r in row["checks"]),
        "max_hTheta_minus": max(r[1] for r in row["checks"]),
    }


def sampled(pair, mass):
    if pair is None:
        return None
    return dict(
        t_M=pair["t_M"],
        cycle=pair["cycle"],
        outer=measurements(pair["outer"], mass),
        inner=measurements(pair["inner"], mass),
    )


def export(
    directory,
    mass,
    dense,
    pairs,
    first_strict,
    production,
    turning,
    seeds,
    window_source="replay",
):
    # Keep the replay and production epochs identifiable at their handoff. Segment
    # boundaries can write the same state twice. Cycles can differ between evolutions.
    history = []
    by_cycle = {p["cycle"]: p for p in pairs}
    dense_end = max(t for t, _, _ in dense)
    frames = [(window_source, r) for r in dense] + [
        ("production", r) for r in production if r[0] > dense_end + 1.0e-10 * mass
    ]
    outer_seed, inner_seed = seeds
    # Continue backward from the earliest paired outer surface. If no pair
    # survived, start at the late-time enclosing seed and still look for the outer.
    early = {}
    seed = pairs[0]["outer"]["coeff"] if pairs else outer_seed
    earlier = [r for r in dense if not pairs or r[1] < pairs[0]["cycle"]]
    misses = 0
    anchored = bool(pairs)
    for _, cycle, file in reversed(earlier):
        row = continued_outer(file, seed)
        early[cycle] = row
        if row:
            seed = row["coeff"]
            misses = 0
            anchored = True
        else:
            misses += 1
            if anchored and misses >= 3:
                break
    seen = set()
    for source, (t, c, file) in frames:
        if (source, c) in seen:
            continue
        seen.add((source, c))
        pair = (
            by_cycle.get(c)
            if source == window_source
            else qualify(file, [outer_seed, inner_seed], mass)
        )
        outer = pair["outer"] if pair else None
        inner = pair["inner"] if pair else None
        prior = source == window_source and (not pairs or c < pairs[0]["cycle"])
        if outer is None:
            outer = early.get(c) if prior else continued_outer(file, outer_seed)
        if outer:
            outer_seed = outer["coeff"]
        if inner:
            inner_seed = inner["coeff"]
        for label, surface in (("outer", outer), ("inner", inner)):
            record = dict(
                source=source,
                cycle=c,
                t_M=t / mass,
                branch=label,
                status="resolved" if surface else "unresolved",
                strict_witness=((pair or {}).get("witness") or {}).get("accepted"),
            )
            if prior and c not in early and not surface and label == "outer":
                record["status"] = "not_searched"
            if surface:
                record.update(measurements(surface, mass))
                if surface.get("barriers"):
                    record["strict_witness"] = True
                    b = surface["barriers"]
                    record["inside_max_hTheta_plus"] = -b["inside"][
                        "min_signed_outgoing"
                    ]
                    record["outside_min_hTheta_plus"] = b["outside"][
                        "min_signed_outgoing"
                    ]
            history.append(record)
    cols = [
        "source",
        "cycle",
        "t_M",
        "branch",
        "status",
        "strict_witness",
        "area",
        "J",
        "zeta",
        "chi",
        "min_cells",
        "lambda0",
        "lambda0_Madm2",
        "max_abs_hTheta_plus",
        "max_hTheta_minus",
        "inside_max_hTheta_plus",
        "outside_min_hTheta_plus",
    ]
    path = directory / "horizon-history.csv"
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, cols)
        writer.writeheader()
        writer.writerows(history)
    temp.replace(path)
    variants = turning.get("variants", [])
    fold = variants[0].get("fold") if variants else None
    return dict(
        first_strict_outer=next(
            (
                r
                for r in history
                if r["branch"] == "outer" and r["strict_witness"] is True
            ),
            None,
        ),
        first_resolved_outer=next(
            (
                r
                for r in history
                if r["branch"] == "outer" and r["status"] == "resolved"
            ),
            None,
        ),
        first_resolved_pair=sampled(pairs[0], mass) if pairs else None,
        first_verified_strict=sampled(first_strict, mass),
        reconstructed_fold={
            k: fold[k] for k in ("t_M", "area", "J", "zeta", "chi", "lambda_M2")
        }
        if fold
        else None,
        file=path.name,
        last_sample_M=history[-1]["t_M"],
        resolved={
            label: sum(
                r["branch"] == label and r["status"] == "resolved" for r in history
            )
            for label in ("outer", "inner")
        },
    )
