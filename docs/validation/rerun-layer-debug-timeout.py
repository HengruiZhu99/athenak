"""Resource-only rerun: retain assertions and extend the subprocess timeout.

Run from the checkout root with ATHENA_OVERHAUL_EXE and ATHENA_HYP_PATCH_EXE
pointing to the sanitizer Debug executables.
"""
import subprocess
import pytest

run = subprocess.run


def with_timeout(*args, **kwargs):
    if 'timeout' in kwargs and kwargs['timeout'] is not None:
        kwargs['timeout'] = max(600, kwargs['timeout'])
    return run(*args, **kwargs)


subprocess.run = with_timeout
raise SystemExit(pytest.main([
    '-q', 'tst/test_suite/z4c/test_hyperboloidal_native_cpu.py::'
    'test_fifth_degree_requires_interior_donors']))
