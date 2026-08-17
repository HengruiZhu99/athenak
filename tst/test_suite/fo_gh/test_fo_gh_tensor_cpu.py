"""CPU test for FO-GH spacetime and mixed-dimension tensor storage."""

import test_suite.testutils as testutils


def test_fo_gh_tensor():
    testutils.run("inputs/fo_gh_tensor_unit.athinput")
