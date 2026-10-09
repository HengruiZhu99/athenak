"""Capture the bounded synthetic-only command without overwriting an attempt."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
OUT = HERE / "synthetic-001"
assert not OUT.exists()
OUT.mkdir()
python = ("/Users/hz0693/Documents/Codex/2026-10-06/"
          "referenced-chatgpt-conversation-this-is-an/work/venv/bin/python")
command = [python, str(HERE / "check_synthetic.py"), "--output", str(OUT / "results")]
env = os.environ.copy()
env.pop("PYTHONPATH", None)
env.update(OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
record = {"command": command, "cwd": str(HERE), "source_sha256": {
    name: sha(HERE / name) for name in ("pade13_einsum.py", "check_synthetic.py",
                                     "run_synthetic.py", "PLAN.md")},
          "executable_sha256": sha(Path(python).resolve()),
          "environment_overrides": {"PYTHONPATH": None, "OPENBLAS_NUM_THREADS": "1",
                                    "VECLIB_MAXIMUM_THREADS": "1"},
          "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
(OUT / "command.json").write_text(json.dumps(record, indent=2) + "\n")
start = time.monotonic()
with (OUT / "stdout.txt").open("w") as stdout, (OUT / "stderr.txt").open("w") as stderr:
    process = subprocess.run(command, cwd=HERE, env=env, stdout=stdout, stderr=stderr)
record.update(returncode=process.returncode, seconds=time.monotonic() - start,
              ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              stdout_sha256=sha(OUT / "stdout.txt"), stderr_sha256=sha(OUT / "stderr.txt"))
(OUT / "command-receipt.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"returncode": process.returncode, "seconds": record["seconds"],
                  "stdout": (OUT / "stdout.txt").read_text(),
                  "stderr": (OUT / "stderr.txt").read_text()}))
sys.exit(process.returncode)
