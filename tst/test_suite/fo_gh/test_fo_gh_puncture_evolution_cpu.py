"""Short no-floor/no-excision FO-GH puncture evolution smoke test."""

from pathlib import Path

import numpy as np
import test_suite.testutils as testutils


def test_fo_gh_puncture_evolution():
    history = Path("fo_gh_puncture_evolution.fo_gh.hst")
    try:
        testutils.run("inputs/fo_gh_puncture_evolution.athinput")
        data = np.loadtxt(history)
        checkpoint = np.loadtxt("fo_gh_puncture_evolution-checkpoint.dat")
        assert data.shape[1] == 15
        assert np.all(np.isfinite(data))
        assert checkpoint.shape == (41,)
        assert int(checkpoint[3]) == 1
        assert np.all(np.isfinite(checkpoint))
    finally:
        history.unlink(missing_ok=True)
        testutils.cleanup()
