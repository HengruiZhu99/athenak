"""Actual snapshot live damping range/identity audit; no evolution or changed helpers."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess

import numpy as np


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
READER = ROOT / "build-layer-research/time-projection-controls/rst-reader-gate/restart_reader.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(READER) == "74c62c97afd113819438d8be42be84cc605902983bd974022ebb7e0747f3cb13"
    build_path = BASE / "native-build/build-receipt.json"
    assert sha(build_path) == "5ade5d3c80b68913301957a7093929a6793af5da8a9fc19c4215f140d7e49a72"
    build = json.loads(build_path.read_text())
    compile_receipt=json.loads((HERE/'compile-receipt.json').read_text())
    command=json.loads((HERE/'compile-command.json').read_text())
    assert compile_receipt['returncode']==0 and compile_receipt['command']==command
    assert compile_receipt['source_sha256']==sha(HERE/'check_snapshot.cpp')
    assert compile_receipt['executable_sha256']==sha(HERE/'check_snapshot')
    spec = importlib.util.spec_from_file_location("independent_pole_restart", READER)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    rows = []
    for run in ("reference", "short"):
        result = json.loads((BASE / run / "results.json").read_text())
        case, = result["cases"]
        casepath = BASE / run / case["name"]
        history = np.atleast_2d(np.loadtxt(casepath / "hyp.z4c.user.hst"))
        for rstpath in sorted((casepath / "rst").glob("*.rst")):
            checkpoint = reader.read_rst(rstpath)
            raw = HERE / (run + "-" + rstpath.stem + ".raw")
            ng = checkpoint["mb_indcs"]["ng"]
            steps = [checkpoint["mesh_size"][f"dx{a}"] for a in (1, 2, 3)]
            first = [checkpoint["mesh_size"][f"x{a}min"] + (.5-ng)*steps[a-1]
                     for a in (1, 2, 3)]
            with raw.open("wb") as stream:
                stream.write(struct.pack("<3i", *checkpoint["u"].shape[1:][::-1]))
                stream.write(struct.pack("<3d", *first))
                stream.write(struct.pack("<3d", *steps))
                stream.write(checkpoint["u"].astype("<f8").tobytes())
            args = [str(HERE / "check_snapshot"), str(raw)]
            process = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
            process.check_returncode()
            value = json.loads(process.stdout)
            nearest = int(np.argmin(abs(history[:, 0] - checkpoint["time"])))
            assert abs(history[nearest, 0] - checkpoint["time"]) < 1e-14
            assert value['active_cells']==6152
            if checkpoint['time']==0:
                assert value['kappa2_min']>-1 and value['kappa2_max']<=0
            bound=(-1<value['kappa2_min'] and value['kappa2_max']<=0)
            rows.append({"run": run, "time": checkpoint["time"],
                         "restart": str(rstpath.relative_to(ROOT)), "restart_sha256": sha(rstpath),
                         "raw_file": str(raw.relative_to(ROOT)), "raw_sha256": sha(raw),
                         "command": args, "computed": value,
                         "within_flat_frozen_coefficient_interval": bound,
                         "returncode": process.returncode, "stderr": process.stderr})
    assert len(rows) == 6
    receipt = {"status": "PASS", "scope": "Actual live kappa2 ranges and sigma identity at six binary64 snapshots; no future interval preservation, energy, scri or BH acceptance",
               "rows": rows, "compile_command": command, "compile_returncode": compile_receipt["returncode"],
               "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in
                                 (Path(__file__), HERE / "check_snapshot.cpp", READER)},
               "private_build_receipt_sha256": sha(build_path),
               "utility_executable_sha256": sha(HERE / "check_snapshot"),
               "compile_log_sha256": sha(HERE / "compile.log"),
               "actual_inputs_unchanged": all(sha(ROOT / name) == digest for name, digest in
                                               build["all_compiled_repository_dependencies_sha256"].items())}
    assert receipt["actual_inputs_unchanged"]
    (HERE / "snapshot-results.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print("PASS six binary64 snapshots; live coefficient helper/manual and stress identities")
    print(json.dumps([{'run':x['run'],'time':x['time'],'kappa2_min':x['computed']['kappa2_min'],
                       'kappa2_max':x['computed']['kappa2_max'],
                       'within_interval':x['within_flat_frozen_coefficient_interval']} for x in rows],indent=2))



if __name__ == "__main__":
    main()
