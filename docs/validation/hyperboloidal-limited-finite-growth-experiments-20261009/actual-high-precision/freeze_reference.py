"""Freeze the separate single-matrix high-precision check, preserving inputs."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "immutable-J0-N8-t6-expm-high-precision-20261009"
assert not OUT.exists()
OUT.mkdir()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


binding = json.loads((HERE / "prepared-001/receipt.json").read_text())
external = []
for path, expected in binding["source_sha256"].items():
    p = Path(path)
    assert sha(p) == expected
    if HERE not in p.parents:
        external.append({"path": str(p), "bytes": p.stat().st_size, "sha256": expected})
files, large = [], []
for p in sorted(HERE.rglob("*")):
    if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
        continue
    rel = str(p.relative_to(HERE))
    entry = {"path": rel, "origin": str(p), "bytes": p.stat().st_size, "sha256": sha(p)}
    if p.stat().st_size > 1048576:
        large.append(entry)
        continue
    target = OUT / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, target)
    assert sha(target) == sha(p) == entry["sha256"]
    files.append(entry)
index = {"kind": "Immutable actual J0 N8 t6 high-precision finite-matrix accuracy check",
         "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE.parents[2], text=True).strip(),
         "production_source_commit": "27c19d20696ea6dd4704032c51dfd026218f64f2",
         "source_directory": str(HERE), "files": files, "large_external_records": large,
         "external_input_records": external, "small_file_count": len(files),
         "small_bytes": sum(v["bytes"] for v in files),
         "scientific_scope": "One authorized finite-matrix t6 accuracy reference only; no PDE/native/nonlinear/scri/BH acceptance."}
(OUT / "index.json").write_text(json.dumps(index, indent=2, allow_nan=False) + "\n")
for row in files:
    assert sha(OUT / row["path"]) == sha(row["origin"]) == row["sha256"]
print(json.dumps({"path": str(OUT / "index.json"), "sha256": sha(OUT / "index.json"),
                  "files": len(files), "bytes": index["small_bytes"], "large": len(large),
                  "external_inputs": len(external)}))
