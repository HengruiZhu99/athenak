"""GPU tests for the FO-GH primary-field RHS."""

import test_suite.testutils as testutils


def test_fo_gh_rhs():
    testutils.run("inputs/fo_gh_rhs_unit.athinput")
