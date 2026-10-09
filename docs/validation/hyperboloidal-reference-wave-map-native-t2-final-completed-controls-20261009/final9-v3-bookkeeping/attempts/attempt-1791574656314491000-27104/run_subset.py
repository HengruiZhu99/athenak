#!/usr/bin/env python3
"""HELD outer wrapper; captures authorization/report failures in fresh receipts."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    # Establish a source-owned fresh receipt destination before auth parsing.
    attempt = HERE / 'attempts' / ('attempt-' + str(time.time_ns()) + '-' + str(os.getpid()))
    attempt.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    pins = {}
    receipt = {'passed_saved_subset_bookkeeping': False,
               'partial_matrix_diagnostic_only': True, 't6_admission': False, 't12_admission': False,
               'command': list(sys.argv), 'cwd': str(Path.cwd()), 'attempt_directory': str(attempt),
               'new_scientific_queries': 0, 'new_native_steps': 0,
               'scope': 'Source/receipt/saved scalar JSON only; no arrays, logs, kernel or subprocess.'}
    result = None
    try:
        own = Path(__file__).resolve()
        pins[str(own)] = sha(own)
        (attempt / 'run_subset.py').write_bytes(own.read_bytes())
        if len(sys.argv) != 2:
            raise ValueError('Expected exactly one authorization JSON path')
        authorization = Path(sys.argv[1]).resolve()
        pins[str(authorization)] = sha(authorization)
        (attempt / 'authorization.json').write_bytes(authorization.read_bytes())
        def invalid(value):
            raise ValueError('Non-finite authorization constant: ' + value)
        auth = json.loads(authorization.read_text(), parse_constant=invalid)
        if auth['completed_subset_comparison_authorized'] is not True:
            raise ValueError('Source-only preparation is not authorized for execution')
        module = HERE / 'compare_subset.py'
        if auth['source_sha256']['run_subset.py'] != pins[str(own)]:
            raise ValueError('Wrapper differs from authorization')
        pins[str(module)] = sha(module)
        if auth['source_sha256']['compare_subset.py'] != pins[str(module)]:
            raise ValueError('Comparator differs from authorization')
        (attempt / 'compare_subset.py').write_bytes(module.read_bytes())
        # Import only the already hash-authorized source, inside failure guard.
        import compare_subset as observer
        if Path(observer.__file__).resolve() != module:
            raise ValueError('Unexpected comparator import origin')
        # Repeat the strict full JSON finite walk before processing any evidence.
        auth = observer.load(authorization)
        result = observer.compare(auth, pins)
        for path, digest in pins.items():
            if sha(path) != digest:
                raise ValueError('Protected input drift: ' + path)
        dump(attempt / 'comparison.json', result)
        receipt.update(passed_saved_subset_bookkeeping=True,
                       comparison_sha256=sha(attempt / 'comparison.json'),
                       protected_before_after_equal=True)
    except BaseException as error:
        # KeyboardInterrupt is retained too; an OS kill/filesystem failure still
        # needs the caller's independent stdout/stderr capture.
        receipt.update(error_type=type(error).__name__, error=str(error),
                       protected_before_after_equal=None)
        (attempt / 'failure.stderr').write_text(traceback.format_exc())
        receipt['failure_stderr_sha256'] = sha(attempt / 'failure.stderr')
    drift = {}
    for path, digest in pins.items():
        try:
            current = sha(path)
            if current != digest:
                drift[path] = {'expected': digest, 'actual': current}
        except Exception as error:
            drift[path] = {'expected': digest, 'read_error': repr(error)}
    if drift:
        receipt.update(passed_saved_subset_bookkeeping=False,
                       protected_before_after_equal=False, input_drift=drift)
    dump(attempt / 'input-pins.json', pins)
    receipt.update(input_pins_sha256=sha(attempt / 'input-pins.json'),
                   seconds=time.monotonic() - started, python_version=sys.version)
    dump(attempt / 'receipt.json', receipt)
    print(json.dumps({'passed': receipt['passed_saved_subset_bookkeeping'],
                      'receipt': str(attempt / 'receipt.json'),
                      'receipt_sha256': sha(attempt / 'receipt.json'),
                      't6_admission': False, 'error': receipt.get('error')}, allow_nan=False))
    return 0 if receipt['passed_saved_subset_bookkeeping'] else 1


if __name__ == '__main__':
    sys.exit(main())
