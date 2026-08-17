"""GPU dynamic Minkowski regression for regularized vacuum FO-GH."""

import test_suite.testutils as testutils


def test_fo_gh_minkowski():
    testutils.run("inputs/fo_gh_minkowski.athinput")
