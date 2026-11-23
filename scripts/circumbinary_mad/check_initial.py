"""Compare the physical phase-zero torus on coincident inertial and rotating grids."""

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vis/python"))
from bin_convert import read_binary


def compare(first, second):
    a, b = (read_binary(str(p)) for p in (first, second))
    assert a["time"] == b["time"] == 0
    np.testing.assert_equal(a["mb_logical"], b["mb_logical"])
    np.testing.assert_equal(a["mb_geometry"], b["mb_geometry"])
    for k in a["var_names"]:
        x, y = (np.asarray(r["mb_data"][k]) for r in (a, b))
        assert np.isfinite(x).all() and np.isfinite(y).all(), k
        np.testing.assert_allclose(y, x, atol=1e-13, rtol=2e-7, err_msg=k)
    assert np.max(a["mb_data"]["dens"]) > 0.1
    assert np.max(a["mb_data"]["bcc3"]) > 0
    return a["var_names"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inertial", type=Path)
    parser.add_argument("rotating", type=Path)
    args = parser.parse_args()
    print("Matched", ", ".join(compare(args.inertial, args.rotating)))
