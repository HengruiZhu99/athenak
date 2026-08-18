from fo_gh.matched_driver_pullback_audit import (
    high_precision_scan,
    verify_conditioning,
    verify_matched_driver,
    verify_old_map_independently,
    verify_target,
)


def test_old_map_obstruction_is_independently_reproduced():
    inverse_error, power = verify_old_map_independently(1000)
    assert inverse_error < 2.0e-13
    assert power < 0.0


def test_matched_pullback_matches_dense_matrix_oracle():
    inverse_error, rhs_error = verify_matched_driver(1000)
    assert inverse_error < 2.0e-13
    assert rhs_error < 3.0e-13


def test_matched_target_recovers_moving_puncture_gauge():
    assert verify_target(1000) < 2.0e-13


def test_normalized_map_conditioning_is_finite():
    condition, inverse_error = verify_conditioning(1000)
    assert condition < 100.0
    assert inverse_error < 2.0e-14


def test_every_scanned_production_intermediate_is_nonnegative_power():
    powers, relative_error = high_precision_scan(64)
    assert powers
    assert min(powers.values()) >= 0
    assert relative_error < 1.0e-12
