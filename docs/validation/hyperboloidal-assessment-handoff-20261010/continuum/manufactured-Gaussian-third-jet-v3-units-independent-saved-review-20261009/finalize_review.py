"""One-shot standard-library compact saved-unit review freeze."""
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write(p, value):
    with Path(p).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def main():
    require(sys.flags.isolated == 1 and sys.flags.optimize == 0 and sys.dont_write_bytecode and os.environ.get("PYTHONOPTIMIZE") == "0",
            "explicit unoptimized -I -B required")
    require(not (HERE / "index.json").exists() and not (HERE / "receipt.json").exists(), "one-shot fresh freeze")
    result = json.loads((HERE / "saved-readback.json").read_text())
    require(result["passed"] is True and result["inputs_unchanged"] is True and result["checks"] == 318 and result["failed"] == 0,
            "completed passing saved readback required")
    capture = json.loads((HERE / "capture.json").read_text())
    for row in capture["inputs"]:
        require(sha(row["origin"]) == row["sha256"] and sha(row["copy"]) == row["sha256"], "final original/copy drift")
    require((HERE / "review.stderr").stat().st_size == 0, "saved review execution failed")
    receipt = {"passed": True, "saved_only": True, "inputs_unchanged": True, "checks": 318, "failed_total": 0,
               "reviewed_source_index_sha256": result["source_index_sha256"], "source_index_sha256": result["source_index_sha256"],
               "recipe_sha256": result["recipe_sha256"], "driver_sha256": result["driver_sha256"],
               "child_receipt_sha256": result["child_receipt_sha256"], "result_sha256": result["result_sha256"],
               "unit_checks_sha256": result["unit_checks_sha256"], "outer_receipt_sha256": result["outer_receipt_sha256"],
               "root_receipt_sha256": result["root_receipt_sha256"], "root_process_group_cap_seconds": 60,
               "root_elapsed_seconds": result["root_seconds"], "child_elapsed_seconds": result["child_seconds"],
               "unique_protected_pins": result["unique_child_plus_root_protected_pins"], "captured_inputs": result["captured_inputs"],
               "review_sha256": sha(HERE / "REVIEW.md"), "saved_readback_sha256": sha(HERE / "saved-readback.json"),
               "disposition": "PASS independent compact saved 318-unit provenance and scalar arithmetic readback",
               "candidate_imported_or_targets_rerun": False, "scientific_payload_decoding": False,
               "timing_full_native_or_BH_admission": False,
               "finalization_command": ["python3", "-I", "-B", str(Path(__file__).resolve())],
               "interpreter": sys.executable, "interpreter_sha256": sha(sys.executable), "flags": str(sys.flags)}
    write(HERE / "receipt.json", receipt)
    files = []
    for p in sorted(HERE.rglob("*")):
        if not p.is_file() or p == HERE / "index.json":
            continue
        require(p.suffix not in (".jsonl", ".npz", ".npy") and p.stat().st_size <= 1048576, "compact review policy")
        p.read_bytes().decode("utf-8")
        files.append({"path": str(p.relative_to(HERE)), "bytes": p.stat().st_size, "sha256": sha(p)})
    write(HERE / "index.json", {"status": "immutable independent saved v3 Gaussian units review", "files": files,
                               "file_count": len(files), "total_bytes": sum(row["bytes"] for row in files)})
    print(json.dumps({"index_sha256": sha(HERE / "index.json"), "receipt_sha256": sha(HERE / "receipt.json"),
                      "files": len(files), "bytes": sum(row["bytes"] for row in files)}))


if __name__ == "__main__":
    main()
