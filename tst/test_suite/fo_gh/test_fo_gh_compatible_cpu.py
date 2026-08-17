"""CPU compatible-gradient regression for regularized vacuum FO-GH."""

import test_suite.testutils as testutils


def test_fo_gh_compatible_gradient():
    testutils.run("inputs/fo_gh_compatible_unit.athinput")
