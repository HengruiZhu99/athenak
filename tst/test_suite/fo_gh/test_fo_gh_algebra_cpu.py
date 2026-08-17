"""CPU tests for FO-GH pointwise algebra and conversion maps."""

import test_suite.testutils as testutils


def test_fo_gh_algebra():
    testutils.run("inputs/fo_gh_algebra_unit.athinput")
