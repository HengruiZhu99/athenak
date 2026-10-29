#!/usr/bin/env python3
"""Regenerate this comparison from the separately bound portable campaigns."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

here = Path(__file__).resolve().parent
aurora = here.parent
analysis = aurora / "resolution-control-v1/analyze.py"
expected = json.loads((here / "analysis.json").read_text())["analysis_script"]["sha256"]
if hashlib.sha256(analysis.read_bytes()).hexdigest() != expected:
    raise SystemExit("Analysis script differs from the recorded comparison; restore its bound revision first")
command = [sys.executable, str(analysis),
           "--baseline-dir", str(aurora / "stability-v1/output"),
           "--new-dir", str(aurora / "resolution-control-v1/output"),
           "--case-dir", "minkowski-n24", str(aurora / "resolution-control-v2/output"),
           "--case-dir", "minkowski-n16-half-dt", str(aurora / "resolution-control-v2/output"),
           "--out", str(here), "--require-all"]
subprocess.run(command, check=True)
