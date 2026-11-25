"""Choose threshold and onset-spin probes from the classified samples on one axis."""

from decimal import Decimal as D


def propose(outcomes, spins, invalid, config):
    coarse = tuple(map(D, config["coarse"]))
    tolerance = D(config["tolerance"])
    if len(set(coarse)) != len(coarse) or len(coarse) < 2 or tolerance <= 0:
        raise ValueError(
            "specify at least two distinct coarse samples and a positive tolerance"
        )
    if any(x not in outcomes and x not in invalid for x in coarse):
        return {}, "coarse survey in progress"
    ordered = sorted(set(outcomes) | set(invalid))
    sides = {"collapse", "dispersal"}
    plan = {}
    if "threshold" in config["objectives"]:
        brackets = [
            (b - a, (a + b) / 2)
            for a, b in zip(ordered, ordered[1:])
            if {outcomes.get(a), outcomes.get(b)} == sides and b - a > tolerance
        ]
        if brackets:
            plan[max(brackets)[1]] = "threshold"

    if "onset spin" in config["objectives"] and spins:
        best = max(value - error for value, error in spins.values())
        peaks = []
        for x, (value, error) in spins.items():
            if value + error < best:
                continue
            i = ordered.index(x)
            for y in ordered[max(0, i - 1) : i] + ordered[i + 1 : i + 2]:
                distance = abs(x - y)
                if outcomes.get(y) not in sides or distance <= tolerance:
                    continue
                if outcomes[y] == "collapse" and y not in spins:
                    continue
                if (
                    y in spins
                    and abs(value - spins[y][0]) <= error + spins[y][1]
                    and distance <= (max(coarse) - min(coarse)) / 32
                ):
                    continue
                trial = x + (y - x) * D(".38196601125")
                peaks.append((distance, trial.quantize(D(".00000001"))))
        if peaks:
            plan.setdefault(max(peaks)[1], "onset spin")

    if config.get("extend_width") and config["axis"] == "width":
        measured = sorted(outcomes)
        if len(measured) < 2:
            return plan, "insufficient classified samples"
        lo, hi = measured[0], measured[-1]
        for edge, neighbor, sign in [(lo, measured[1], -1), (hi, measured[-2], 1)]:
            threshold = "threshold" in config["objectives"] and all(
                outcomes[v] == "collapse" for v in measured
            )
            rising = (
                "onset spin" in config["objectives"]
                and edge in spins
                and neighbor in spins
                and sum(spins[edge]) >= best
                and spins[edge][0] - spins[edge][1] > sum(spins[neighbor])
            )
            if not threshold and not rising:
                continue
            step = D(".03125") if sign < 0 and threshold else D(".0546875")
            trial = edge + sign * step
            rejected = [v for v in invalid if (v - edge) * sign > 0]
            if rejected:
                trial = (edge + min(rejected, key=lambda v: abs(v - edge))) / 2
            if (
                abs(trial - edge) > tolerance
                and float(trial) * 2 / 0.0625 * 0.6398 >= 8
            ):
                role = "extend thin side" if sign < 0 else "extend thick side"
                plan[trial] = role if threshold else "edge maximum"
    return plan, "survey"
