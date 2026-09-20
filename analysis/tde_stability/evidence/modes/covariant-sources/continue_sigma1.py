from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys

root = Path(__file__).resolve().parent
repo = root.parents[2]
sys.path.insert(0, str(repo / "analysis/tde_stability"))
from check_checkpoint import validate

source = root / "covariant_const10"
checked = validate(source, 1)
assert checked["passed"] and checked["time_M"] == 300.0
log = (source / "run.log").read_text()
assert "Terminating on time limit" in log
assert not any(s in log for s in ["Z4C_INVALID_STATE", "C2P_INVALID_ADM_INPUT",
                                 "An error occurred during the primitive solve"])
exe = root / "athena-covariant-sources"
sha = hashlib.sha256(exe.read_bytes()).hexdigest()
assert sha == "7456bbb2cbcef90cdfddbe6d1515220f5e4e02981b799c71e40adb81bf52cd8f"
target = root / "covariant_const10_continuation"
target.mkdir(exist_ok=False)
(target / "parent-checkpoint-validity.json").write_text(json.dumps(checked, indent=2) + "\n")
(target / "input.athinput").write_text((source / "input.athinput").read_text())
checkpoint_path = checked["files"][0]["name"]
cmd = [str(exe), "-r", checkpoint_path, "-i", str(target / "input.athinput"),
       "-d", str(target), "-t", "00:25:00", "time/tlim=1000"]
record = dict(command=cmd, status="running", start_utc=datetime.datetime.now(
    datetime.timezone.utc).isoformat(), binary_sha256=sha, OMP_NUM_THREADS=2,
    parent_time_M=300.0, parent_checkpoint=checked["files"][0])
(target / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
with (target / "run.log").open("w") as out:
    result = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT,
                            env=dict(os.environ, OMP_NUM_THREADS="2", OMP_PROC_BIND="false",
                                     OPENBLAS_NUM_THREADS="1"))
record.update(status="completed", exit_code=result.returncode,
              end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(target / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
(target / "exit_code.txt").write_text(str(result.returncode) + "\n")
print(json.dumps(record), flush=True)
