from fo_gh.matched_hyperbolicity_audit import (
    audit_state,
    characteristic_subspace_condition,
    minkowski_state,
    sample_directions,
    weak_random_state,
    wormhole_state,
)

import numpy as np


def test_minkowski_transformed_symmetrizer():
    directions = sample_directions(np.random.default_rng(1), 8)
    result = audit_state(minkowski_state(), directions)
    assert result["minimum_H"] > 0.0
    assert result["symmetry_residual"] < 2.0e-6
    assert result["maximum_imaginary_eigenvalue"] < 2.0e-7


def test_weak_random_transformed_symmetrizer():
    rng = np.random.default_rng(2)
    directions = sample_directions(rng, 8)
    for _ in range(4):
        result = audit_state(weak_random_state(rng), directions)
        assert result["minimum_H"] > 0.0
        assert result["symmetry_residual"] < 2.0e-6
        assert result["maximum_imaginary_eigenvalue"] < 2.0e-7


def test_wormhole_characteristics_expose_nonuniform_conditioning():
    coarse = characteristic_subspace_condition(wormhole_state(0.25), 1.0e-5)
    fine = characteristic_subspace_condition(wormhole_state(0.0625), 1.0e-5)
    assert coarse > 1.0e5
    assert fine > 1.0e10
    assert fine > 1.0e4 * coarse
