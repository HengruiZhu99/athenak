"""CPU dynamic Minkowski regression for reference-frame FO-GH."""

import test_suite.testutils as testutils


def test_ref_gh_minkowski():
    testutils.run("inputs/ref_gh_minkowski.athinput")
