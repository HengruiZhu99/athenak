#!/usr/bin/env python3
"""Check second-order convergence of an edge/corner response diagnostic.

The continuum response of a planar L=1 condition at oblique incidence need
not vanish.  For a factor-two resolution sequence, eliminate that unknown
limit from R(h)=R(0)+a h^p and estimate

    p = log2(abs((R(h)-R(h/2))/(R(h/2)-R(h/4)))).
"""

import argparse
import math
import pathlib
import re


RATIO_PATTERN = re.compile(r"\bratio=(?P<ratio>\S+)")
GEOMETRY_PATTERN = re.compile(r"\bgeometry=(?P<geometry>edge|corner)\b")


def load(path):
    text = path.read_text(encoding="utf-8")
    ratio_matches = list(RATIO_PATTERN.finditer(text))
    geometry_matches = list(GEOMETRY_PATTERN.finditer(text))
    if len(ratio_matches) != 1 or len(geometry_matches) != 1:
        raise SystemExit(
            "{}: expected exactly one geometry and ratio".format(path))
    ratio = float(ratio_matches[0].group("ratio"))
    if not math.isfinite(ratio) or ratio < 0.0:
        raise SystemExit("{}: invalid response ratio".format(path))
    return geometry_matches[0].group("geometry"), ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("coarse", type=pathlib.Path)
    parser.add_argument("medium", type=pathlib.Path)
    parser.add_argument("fine", type=pathlib.Path)
    parser.add_argument("--minimum-order", type=float, default=1.8)
    args = parser.parse_args()

    rows = [load(path) for path in (args.coarse, args.medium, args.fine)]
    geometries = {geometry for geometry, _ in rows}
    if len(geometries) != 1:
        raise SystemExit("input diagnostics use different geometries")
    values = [value for _, value in rows]
    coarse_difference = values[0] - values[1]
    fine_difference = values[1] - values[2]
    if (
        coarse_difference == 0.0
        or fine_difference == 0.0
        or coarse_difference * fine_difference <= 0.0
    ):
        raise SystemExit(
            "response sequence is not monotone toward one continuum limit: "
            "{}".format(values))
    order = math.log2(abs(coarse_difference / fine_difference))
    factor = 2.0**order
    continuum = (factor * values[2] - values[1]) / (factor - 1.0)
    print(
        "geometry={} coarse={:.8e} medium={:.8e} fine={:.8e} "
        "order={:.8e} extrapolated_limit={:.8e}".format(
            rows[0][0], values[0], values[1], values[2], order, continuum))
    if not math.isfinite(order) or order < args.minimum_order:
        raise SystemExit(
            "oblique convergence order {:.6g} is below {:.6g}".format(
                order, args.minimum_order))


if __name__ == "__main__":
    main()
