"""Reuse immutable generic verifier on two separately authorized saved matrices."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
VERIFIER = R / "continuum/immutable-finite-rb-independent-matrix-readback-20261009/verify_operator.py"
assert hashlib.sha256(VERIFIER.read_bytes()).hexdigest() == "855245e3e4429dc713d817285ec79f05dda112ec159b9b821c9983e95987d95b"
shutil.copyfile(VERIFIER, HERE / "verify_operator.py")
cases = (
    (12, "1df586b980dd7ce21555b762236dcd3ecac8cbeb2b5c848a9e8321e1d7da0aac",
     "cbc228937d6426be1f9e8973a4bc3af6c3fdde66b48939d7316b707bae76df63"),
    (16, "d54caf4494e0f7192de6de9124394dbaf6f7e039077683c0e72033ef1db87a52",
     "096eae51c3e6e2793c019e86c02218ff60e1a3f5f9910b2baa01f0ff67995ac2"),
)
env = os.environ.copy()
env.update(PYTHONPATH=str(R / "boundary/python-deps"), OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
rows = []
for N, matrix_sha, meta_sha in cases:
    source = R / ("boundary/total-j-finite-rb-degree-control-20261009/"
                  "J0-N{}-sector-readback001".format(N))
    out = HERE / ("N{}-readback001".format(N))
    out.mkdir(exist_ok=False)
    command = ["/Library/Developer/CommandLineTools/usr/bin/python3", str(HERE / "verify_operator.py"),
               "--npz", str(source / "operator.npz"), "--expect-sha256", matrix_sha,
               "--metadata", str(source / "matrix-metadata.json"),
               "--expect-metadata-sha256", meta_sha, "--output", str(out / "result.json")]
    record = {"command": command, "cwd": str(HERE),
              "environment_overrides": {k: env[k] for k in ("PYTHONPATH", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
              "verifier_sha256": hashlib.sha256((HERE / "verify_operator.py").read_bytes()).hexdigest()}
    (out / "command.json").write_text(json.dumps(record, indent=2) + "\n")
    start = time.monotonic()
    process = subprocess.run(command, cwd=HERE, env=env, capture_output=True, text=True)
    (out / "stdout.txt").write_text(process.stdout)
    (out / "stderr.txt").write_text(process.stderr)
    record.update(returncode=process.returncode, seconds=time.monotonic() - start,
                  stdout_sha256=hashlib.sha256(process.stdout.encode()).hexdigest(),
                  stderr_sha256=hashlib.sha256(process.stderr.encode()).hexdigest())
    (out / "command-receipt.json").write_text(json.dumps(record, indent=2) + "\n")
    rows.append({"N": N, "returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr})
print(json.dumps(rows, indent=2))
assert all(v["returncode"] == 0 and not v["stderr"] for v in rows)
