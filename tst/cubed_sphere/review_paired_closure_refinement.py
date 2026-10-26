#!/usr/bin/env python3
"""Combine two three-grid constraint reviews and matched field-difference reports.

This is a necessary short-time screen, never full physical acceptance.
All input files are recorded by SHA256. Exact-zero differences are reported
separately; nonzero differences receive no invented roundoff exemption.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("first_review", type=Path)
    p.add_argument("second_review", type=Path)
    p.add_argument("coarse_difference", type=Path)
    p.add_argument("medium_difference", type=Path)
    p.add_argument("fine_difference", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--bulk-scales", nargs=3, type=int, default=[24, 32, 40])
    a = p.parse_args()
    paths = [a.first_review, a.second_review, a.coarse_difference,
             a.medium_difference, a.fine_difference]
    first, second, *differences = [json.loads(x.read_text()) for x in paths]
    assert len({x["time"] for x in differences}) == 1
    # Original closure-only reports predate the variation field.
    assert all(x.get("variation", "closure") == "closure" and
               x["closure_signatures"][0] != x["closure_signatures"][1]
               for x in differences)
    assert a.bulk_scales[0] < a.bulk_scales[1] < a.bulk_scales[2]
    assert all(x["regions"].keys() == differences[0]["regions"].keys()
               for x in differences)
    for review in [first, second]:
        for name in ["coarse", "medium", "fine", "fine-halfstep"]:
            case = review["cases"][name]
            assert case["outcome"] == "complete"
            assert float(case["last_rows"][-1]["time"]) == differences[0]["time"]
    regional = {}
    failed = []
    for region in differences[0]["regions"]:
        regional[region] = []
        for component in range(25):
            values = []
            for d in differences:
                v = d["regions"][region]["components"][component]
                assert v["component"] == component
                values.append(v["l2"])
            assert all(math.isfinite(x) and x >= 0 for x in values)
            zero = all(x == 0 for x in values)
            decreasing = values[0] > values[1] > values[2]
            orders = [math.log(values[i]/values[i+1]) /
                      math.log(a.bulk_scales[i+1]/a.bulk_scales[i])
                      if values[i] > 0 and values[i+1] > 0 else None
                      for i in range(2)]
            row = dict(component=component, l2=values,
                       identically_zero=zero, strictly_decreases=decreasing,
                       orders=orders)
            regional[region].append(row)
            if not (zero or decreasing):
                failed.append(dict(region=region, **row))
    out = dict(
        endpoint=differences[0]["time"],
        input_sha256={str(x): hashlib.sha256(x.read_bytes()).hexdigest()
                      for x in paths},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        short_time_constraint_screens=[x["short_time_screen"] for x in [first, second]],
        regional_component_differences=regional,
        nondecreasing_nonzero_components=failed,
        closure_difference_necessary_screen_passes=not failed,
        complete_physical_acceptance=False,
        limitations=[
            "Source, parameter, geometry and minimum-Omega matching must be audited in the input comparison provenance.",
            "Orders describe within-grid closure differences across changing cell-center samples; no cross-grid solution interpolation.",
            "No measured noise floor: tiny nonzero failures remain unresolved, never silently exempted.",
            "Regional maxima, temporal field differences, late-time behavior, incoming boundary consistency and both physical targets remain separate requirements."])
    a.output.write_text(json.dumps(out, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(endpoint=out["endpoint"],
                          nondecreasing_nonzero_components=len(failed),
                          complete_physical_acceptance=False)))


if __name__ == "__main__":
    main()
