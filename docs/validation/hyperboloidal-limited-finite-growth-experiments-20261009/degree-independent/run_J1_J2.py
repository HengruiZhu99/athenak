"""Four saved-output readbacks, bound through the owner's immutable index."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
SOURCE = R / "boundary/total-j-finite-rb-degree-control-20261009"
INDEX = SOURCE / "immutable-J1-J2-finite-rb-degree-control-20261009/index.json"
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(INDEX) == "46ba57a7d8d0eb3be3f703826ab8036384512876e3a3651e5be4db4d640a5dfb"
assert sha(HERE / "verify_operator.py") == "855245e3e4429dc713d817285ec79f05dda112ec159b9b821c9983e95987d95b"
entries = {v["path"]: v for v in json.loads(INDEX.read_text())["files"]}
env = os.environ.copy()
env.update(PYTHONPATH=str(R / "boundary/python-deps"), OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
rows = []
for J in (1, 2):
    for N in (12, 16):
        folder = "J{}-N{}-sector-readback{:03d}".format(J, N, 1 if J == 1 else 2)
        source = SOURCE / folder
        rel = folder + "/matrix-metadata.json"
        meta_sha = entries[rel]["sha256"]
        assert sha(source / "matrix-metadata.json") == meta_sha
        meta = json.loads((source / "matrix-metadata.json").read_text())
        matrix_sha = meta["matrix_sha256"]
        assert sha(source / "operator.npz") == matrix_sha
        assert (meta["J"], meta["N"], meta["rb"]) == (J, N, .98)
        out = HERE / ("J{}-N{}-readback001".format(J, N))
        out.mkdir(exist_ok=False)
        command = ["/Library/Developer/CommandLineTools/usr/bin/python3", str(HERE / "verify_operator.py"),
                   "--npz", str(source / "operator.npz"), "--expect-sha256", matrix_sha,
                   "--metadata", str(source / "matrix-metadata.json"), "--expect-metadata-sha256", meta_sha,
                   "--output", str(out / "result.json")]
        record = {"command": command, "cwd": str(HERE), "immutable_index_sha256": sha(INDEX),
                  "environment_overrides": {k: env[k] for k in ("PYTHONPATH", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
                  "verifier_sha256": sha(HERE / "verify_operator.py")}
        (out / "command.json").write_text(json.dumps(record, indent=2) + "\n")
        start = time.monotonic()
        process = subprocess.run(command, cwd=HERE, env=env, capture_output=True, text=True)
        (out / "stdout.txt").write_text(process.stdout)
        (out / "stderr.txt").write_text(process.stderr)
        record.update(returncode=process.returncode, seconds=time.monotonic() - start,
                      stdout_sha256=hashlib.sha256(process.stdout.encode()).hexdigest(),
                      stderr_sha256=hashlib.sha256(process.stderr.encode()).hexdigest())
        (out / "command-receipt.json").write_text(json.dumps(record, indent=2) + "\n")
        rows.append({"J": J, "N": N, "returncode": process.returncode,
                     "stdout": process.stdout, "stderr": process.stderr})
print(json.dumps(rows, indent=2))
assert all(v["returncode"] == 0 and not v["stderr"] for v in rows)
