from fo_gh.matched_einstein_map_audit import (
    verify_round_trip,
    verify_tangent_round_trip,
)


def test_exact_58d_regular_parent_round_trip():
    regular, parent, constraints = verify_round_trip(1000)
    assert regular < 2.0e-12
    assert parent < 2.0e-12
    assert constraints < 2.0e-13


def test_exact_58d_tangent_map_round_trip():
    identity, condition = verify_tangent_round_trip(8)
    assert identity < 2.0e-6
    assert condition > 0.0
