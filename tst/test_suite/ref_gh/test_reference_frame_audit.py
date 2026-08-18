"""Regression wrapper for the standalone reference-frame algebra audit."""

from .reference_frame_audit import run_audit, validate


def test_reference_frame_audit():
    validate(run_audit())
