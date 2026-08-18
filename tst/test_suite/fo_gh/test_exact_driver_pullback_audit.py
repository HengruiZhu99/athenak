from fo_gh.exact_driver_pullback_audit import (
    conditioning_sequence,
    trumpet_power_audit,
    verify_dense_oracle,
    verify_regular_gauge_target,
)


def test_exact_weighted_driver_component_oracle():
    assert verify_dense_oracle(samples=256) < 2.0e-14


def test_regular_moving_puncture_target_identities():
    assert verify_regular_gauge_target(samples=256) < 3.0e-15


def test_required_driver_weights_fail_trumpet_regularity_gate():
    bad_power, required_z_perp_power = trumpet_power_audit()
    assert bad_power < 0.0
    assert required_z_perp_power < 0.0
    rows = conditioning_sequence(maximum_n=64)
    assert rows[-1][2] > rows[0][2]
    assert rows[-1][4] < 1.0e-12
