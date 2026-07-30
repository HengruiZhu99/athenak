#!/usr/bin/env python3
"""Check exact-background residual and CPBC diagnostics against roundoff gates."""

import math
import pathlib
import re
import sys


def fail(message):
    raise SystemExit(f"FAIL: {message}")


if len(sys.argv) not in (3, 4):
    fail("usage: check_pure_background.py USER_HST STDOUT_LOG [TOL=1e-12]")

history_path = pathlib.Path(sys.argv[1])
log_path = pathlib.Path(sys.argv[2])
tolerance = float(sys.argv[3]) if len(sys.argv) == 4 else 1.0e-12
if not math.isfinite(tolerance) or tolerance <= 0.0:
    fail("tolerance must be finite and positive")

history_lines = history_path.read_text(encoding="utf-8").splitlines()
if len(history_lines) < 3:
    fail(f"history file is incomplete: {history_path}")

labels = {}
for match in re.finditer(r"\[(\d+)\]=(\S+)", history_lines[1]):
    labels[match.group(2)] = int(match.group(1)) - 1

residual_labels = ("res-inner", "res-ramp", "res-outer")
missing = [label for label in residual_labels if label not in labels]
if missing:
    fail(f"history is missing residual shell columns: {', '.join(missing)}")

max_residual = 0.0
row_count = 0
for line in history_lines[2:]:
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    row = [float(value) for value in line.split()]
    row_count += 1
    for label in residual_labels:
        value = row[labels[label]]
        if not math.isfinite(value):
            fail(f"nonfinite {label} value")
        max_residual = max(max_residual, abs(value))
if row_count == 0:
    fail("history contains no data rows")

diagnostic_pattern = re.compile(
    r"Z4C_CHARACTERISTIC_CPBC .*?"
    r"gauge=(?P<gauge>\S+) .*?"
    r"constraint=(?P<constraint>\S+) .*?"
    r"radiation=(?P<radiation>\S+) .*?"
    r"enforcement=(?P<enforcement>\S+)"
)
max_incoming = 0.0
max_enforcement = 0.0
diagnostic_count = 0
log_text = log_path.read_text(encoding="utf-8")
for match in diagnostic_pattern.finditer(log_text):
    diagnostic_count += 1
    for sector in ("gauge", "constraint", "radiation"):
        value = float(match.group(sector))
        if not math.isfinite(value):
            fail(f"nonfinite {sector} characteristic diagnostic")
        max_incoming = max(max_incoming, abs(value))
    enforcement = float(match.group("enforcement"))
    if not math.isfinite(enforcement):
        fail("nonfinite enforcement diagnostic")
    max_enforcement = max(max_enforcement, abs(enforcement))
if diagnostic_count == 0:
    fail("no characteristic diagnostics found")

if max_residual > tolerance:
    fail(f"max residual {max_residual:.8e} exceeds {tolerance:.8e}")
if max_incoming > tolerance:
    fail(f"max incoming amplitude {max_incoming:.8e} exceeds {tolerance:.8e}")
if max_enforcement > tolerance:
    fail(f"max enforcement error {max_enforcement:.8e} exceeds {tolerance:.8e}")

print(
    f"PASS residual={max_residual:.8e} "
    f"incoming={max_incoming:.8e} enforcement={max_enforcement:.8e}"
)
