"""Short no-floor/no-excision FO-GH puncture evolution smoke test."""

import test_suite.testutils as testutils


def test_fo_gh_puncture_evolution():
    testutils.run("inputs/fo_gh_puncture_evolution.athinput")
