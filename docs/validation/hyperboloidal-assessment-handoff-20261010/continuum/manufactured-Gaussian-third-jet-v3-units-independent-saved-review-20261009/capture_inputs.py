"""Capture completed compact v3 unit evidence; no scientific imports."""
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OWNER = ROOT / "build-layer-research/continuum/manufactured-angular-Gaussian-third-jet-oracle-v3-held-20261009"
RELEASE = ROOT / "build-layer-research/Gaussian-third-jet-oracle-v3-root-release-20261009"
SOURCE_REVIEW = ROOT / "build-layer-research/continuum/manufactured-Gaussian-third-jet-v3-independent-source-review-20261009"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    require(sys.flags.optimize == 0 and sys.flags.isolated == 1 and sys.dont_write_bytecode,
            "explicit unoptimized -I -B required")
    require(os.environ.get("PYTHONOPTIMIZE") == "0", "explicit optimization environment required")
    require(not (HERE / "capture.json").exists(), "one-shot capture")
    literal = {OWNER / "source-index.json": "41a050d288076d6b54ff396b09851835dac03dd3518f74dca65890f6fd025a8a",
               OWNER / "recipe.json": "847cd099b7d98f16963fbbdfc6f8a0eb6f7913529b9a2d72dcabb909c287c8cf",
               OWNER / "attempts/units001/receipt.json": "d41b648021152f62c7972c483be08c2f70d6b949ad01cecabe7b621787f34e16",
               OWNER / "attempts/units001/result.json": "c569c2e3d8fe884e3a4f14723fc91ce7c25c8cf447b577cff6ddb8244c98db89",
               RELEASE / "units-outer001/receipt.json": "79e973af0a68a41db0da0b462d940ba9118c6ac90a05fc07d9deeff73388cd18",
               SOURCE_REVIEW / "index.json": "6773bbe3899c06db8e1486acf7a6cc964448983b950936462c0b017cf64f8988",
               SOURCE_REVIEW / "receipt.json": "bce50b583db35f3a9fb9728abf91213174554c23ada0324fd79662dd2330828e"}
    for path, expected in literal.items():
        require(sha(path) == expected, "literal input drift: " + str(path))
    paths = set(literal)
    paths.update(p for p in (OWNER / "attempts/units001").rglob("*") if p.is_file())
    paths.update(p for p in RELEASE.rglob("*") if p.is_file())
    paths.update(OWNER / name for name in ("run_oracle.py", "outer_once.py", "units.py"))
    rows = []
    for path in sorted(paths):
        require(path.suffix not in (".jsonl", ".npz", ".npy") and path.stat().st_size <= 1048576,
                "only compact completed source/scalar evidence is admitted")
        data = path.read_bytes()
        data.decode("utf-8")
        target = HERE / "captured" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        require(not target.exists(), "fresh capture target required")
        target.write_bytes(data)
        require(sha(path) == sha(target), "copy drift")
        rows.append({"origin": str(path), "copy": str(target), "bytes": len(data), "sha256": sha(path)})
    result = {"passed": True, "captured_inputs": len(rows), "inputs": rows,
              "command": ["python3", "-I", "-B", str(Path(__file__).resolve())],
              "runtime": {"executable": sys.executable, "resolved_executable": str(Path(sys.executable).resolve()),
                          "executable_sha256": sha(sys.executable), "version": sys.version, "flags": str(sys.flags),
                          "environment": {k: os.environ.get(k) for k in ("PYTHONOPTIMIZE", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}},
              "candidate_imports": False, "target_evaluation": False, "scientific_payload_decoding": False}
    (HERE / "capture.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "inputs": len(rows), "bytes": sum(row["bytes"] for row in rows),
                      "root_receipt_sha256": sha(RELEASE / "units-invocation001/receipt.json")}))


if __name__ == "__main__":
    main()
