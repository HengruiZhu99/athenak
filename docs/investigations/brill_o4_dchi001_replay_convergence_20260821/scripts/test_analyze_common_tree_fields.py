#!/usr/bin/env python3
"""Focused algebra tests for common-tree field interpolation."""

import importlib.util
from pathlib import Path
import tempfile
import numpy as np


SCRIPT = Path(__file__).with_name("analyze_common_tree_fields.py")
spec = importlib.util.spec_from_file_location("field_analysis", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def polynomial(x, y):
    return 1 + 2*x - 3*y + 0.5*x*x + x*y - 0.25*y**3 + 0.1*x**4


for count in (16, 32, 64):
    x = (np.arange(count) + 0.5) / count
    y = (np.arange(count) + 0.5) / count
    xx, yy = np.meshgrid(x, y, indexing="xy")
    array = polynomial(xx, yy)[None, :, :]
    data = {
        "mb_geometry": np.array([[0., 1., 0., 1., -.5, .5]]),
        "mb_logical": np.array([[0, 0, 0, 0]]),
        "nx1_out_mb": count, "nx2_out_mb": count,
        "mb_data": {name: [array.copy()] for name in module.BASE_FIELDS.values()},
    }
    points = np.array([[0.03, 0.37, 0.97], [0.12, 0.51, 0.89]])
    rho = points[0]; zed = points[1]
    sampled, masks = module.sample_snapshot(data, rho, zed)
    np.testing.assert_allclose(sampled["chi"], polynomial(rho, zed), rtol=0, atol=2e-12)
    assert np.isinf(masks["cf_distance"]).all()

nodes = np.array([0.0, 0.3, 0.8, 1.4])
target = 0.55
weights = module.lagrange_weights(nodes, target)
for degree in range(4):
    np.testing.assert_allclose(np.sum(weights * nodes**degree), target**degree,
                               rtol=0, atol=2e-14)

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "profile.csv.gz"
    fields = {case: {name: np.ones((1, 2)) for name in module.OUTPUT_FIELDS}
              for case in module.CASES}
    module.write_profile(path, np.array([[0.1, 0.2]]), np.array([[0.0, 0.0]]), fields)
    assert path.is_file() and path.stat().st_size > 0

print("COMMON_TREE_FIELD_ANALYZER_TEST_PASS")
