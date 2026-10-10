"""Standard-library-only capture of the held source before content review."""
import hashlib
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
SOURCE = ROOT.parent / "manufactured-angular-Gaussian-third-jet-oracle-v3-held-20261009"
EXPECTED_INDEX = "41a050d288076d6b54ff396b09851835dac03dd3518f74dca65890f6fd025a8a"
EXPECTED_RECIPE = "847cd099b7d98f16963fbbdfc6f8a0eb6f7913529b9a2d72dcabb909c287c8cf"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    require(sys.flags.optimize == 0 and sys.flags.isolated == 1 and sys.dont_write_bytecode,
            "explicit -I -B and optimization disabled are required")
    require(not (ROOT / "capture.json").exists(), "capture is one shot")
    idx_bytes = (SOURCE / "source-index.json").read_bytes()
    require(sha(idx_bytes) == EXPECTED_INDEX, "held source index changed")
    idx = json.loads(idx_bytes)
    require(idx["source_only"] is True and idx["execution_admitted"] is False,
            "source-only held status required")
    require(idx["file_count"] == len(idx["files"]) == 58, "unexpected source count")
    files = dict(idx["files"])
    files[str(SOURCE / "source-index.json")] = EXPECTED_INDEX
    records = []
    for name, expected in sorted(files.items()):
        origin = pathlib.Path(name)
        rel = origin.relative_to(SOURCE)
        data = origin.read_bytes()
        require(len(data) <= 1024 * 1024, "unexpected large source input")
        data.decode("utf-8")
        require(sha(data) == expected, "input drift: " + name)
        target = ROOT / "inputs" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        require(not target.exists(), "capture target already exists")
        target.write_bytes(data)
        require(sha(origin.read_bytes()) == expected and sha(target.read_bytes()) == expected,
                "capture mismatch: " + name)
        records.append({"origin": name, "copy": str(target), "bytes": len(data), "sha256": expected})
    require(sha((ROOT / "inputs/recipe.json").read_bytes()) == EXPECTED_RECIPE,
            "held recipe changed")
    result = {"passed": True, "source_capture_only": True,
              "reviewed_source_index_sha256": EXPECTED_INDEX,
              "recipe_sha256": EXPECTED_RECIPE, "captured_file_count": len(records),
              "files": records, "command": ["python3", "-I", "-B", str(pathlib.Path(__file__).resolve())],
              "runtime": {"executable": sys.executable, "resolved_executable": str(pathlib.Path(sys.executable).resolve()),
                          "executable_sha256": sha(pathlib.Path(sys.executable).read_bytes()),
                          "version": sys.version, "flags": str(sys.flags),
                          "environment": {k: os.environ.get(k) for k in ["PYTHONOPTIMIZE", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]}},
              "limitations": ["Only held indexed UTF-8 source and metadata captured; no candidate import, arithmetic or execution.",
                              "The source index itself was read as metadata to select this capture."]}
    (ROOT / "capture.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "files": len(records), "bytes": sum(x["bytes"] for x in records)}))


if __name__ == "__main__":
    main()
