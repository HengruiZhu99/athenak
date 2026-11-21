"""Benchmark independent centered generating-curve solves on identical saved slices."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from replay import NativePatch, read_binary, upper_only
from ring_surface import solve_surface


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("shape", type=Path)
    p.add_argument("snapshot", type=Path)
    p.add_argument("--target", type=float, default=0.)
    p.add_argument("--order", type=int, default=24)
    p.add_argument("--evaluations", type=int, default=30)
    a = p.parse_args()
    frame = read_binary(a.snapshot)
    archive = json.loads(a.shape.read_text())
    for coverage, f in (("full",frame),("folded",upper_only(frame))):
        patch = NativePatch(f)
        for factor, spherical in ((1.,False),(.9,False),(1.,True)):
            shape = archive if a.target < 0 else archive["enclosing_horizon"]
            coeff=np.asarray(shape["coeff"])*factor
            start=perf_counter()
            try:
                row=solve_surface(patch, coeff[0], a.target, a.order, a.evaluations,
                                  None if spherical else coeff)
                row.pop("coeff")
                row["accepted_marginal"] = a.target == 0 and max(abs(row["min_rTheta"]), abs(row["max_rTheta"])) < .006 and row["max_rThetaMinus"] <= 0
            except (ValueError,np.linalg.LinAlgError) as error:
                row=dict(accepted_marginal=False, failure=str(error))
            print(json.dumps(dict(file=a.snapshot.name,coverage=coverage,seed_scale=factor,
                                  spherical=spherical,seconds=perf_counter()-start,**row)),flush=True)
if __name__ == "__main__":
    main()
