"""Regression for the table-faithful stationary source-sector diagnosis."""

from .binary64_stationary_source_audit import run_audit, validate


def test_binary64_stationary_source_audit():
    validate(run_audit())
