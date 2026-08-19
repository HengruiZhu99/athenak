"""Compiled flat-reference regression for the production covariant Ref-GH source."""

import test_suite.testutils as testutils


def test_ref_gh_source_unit():
    assert testutils.run("inputs/ref_gh/ref_gh_source_unit.athinput")
