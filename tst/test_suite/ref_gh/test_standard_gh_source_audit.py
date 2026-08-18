"""Regression wrapper for the standard-GH source oracle."""

from .standard_gh_source_audit import run_audit, validate


def test_standard_gh_source_audit():
    validate(run_audit())
