"""Device-side unit tests for canonical scalar-field algebra."""

import test_suite.testutils as testutils


def test_scalar_field_algebra():
    """Run potential, matter, charge, and accumulator checks."""
    testutils.run("inputs/ut_scalar_field.athinput")
