"""Capture the separately authorized actual single-matrix reference command."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "immutable-expm-diagnosis-pade13-synthetic-20261009/pade13-candidate/pade13_einsum.py"
assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == "0654c3ba8884dc467aa1c4665fc4758a3abddf58189e2702f7e8cfc90d2e79b7"
shutil.copyfile(FROZEN, HERE / "pade13_einsum.py")
OUT = HERE / "reference-001"
assert not OUT.exists()
OUT.mkdir()
python = ("/Users/hz0693/Documents/Codex/2026-10-06/"
          "referenced-chatgpt-conversation-this-is-an/work/venv/bin/python")
command = [python, str(HERE / "check_mpmath.py"), "--output", str(OUT / "results")]
env = os.environ.copy()
env.pop("PYTHONPATH", None)
env.update(OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
record = {"command": command, "cwd": str(HERE), "source_sha256": {
    name: sha(HERE / name) for name in ("PLAN.md", "check_mpmath.py", "run_reference.py", "pade13_einsum.py")},
    "environment_overrides": {"PYTHONPATH": None, "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}}
(OUT / "command.json").write_text(json.dumps(record, indent=2) + "\n")
start = time.monotonic()
with (OUT / "stdout.txt").open("w") as stdout, (OUT / "stderr.txt").open("w") as stderr:
    process = subprocess.run(command, cwd=HERE, env=env, stdout=stdout, stderr=stderr)
record.update(returncode=process.returncode, seconds=time.monotonic() - start,
              stdout_sha256=sha(OUT / "stdout.txt"), stderr_sha256=sha(OUT / "stderr.txt"))
(OUT / "command-receipt.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"returncode": process.returncode, "seconds": record["seconds"],
                  "stdout": (OUT / "stdout.txt").read_text(), "stderr": (OUT / "stderr.txt").read_text()}))
sys.exit(process.returncode)
