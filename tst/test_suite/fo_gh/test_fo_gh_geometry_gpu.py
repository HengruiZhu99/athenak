"""GPU tests for FO-GH conformal geometry and vacuum constraints."""

import test_suite.testutils as testutils


def test_fo_gh_geometry():
    testutils.run("inputs/fo_gh_geometry_unit.athinput")
