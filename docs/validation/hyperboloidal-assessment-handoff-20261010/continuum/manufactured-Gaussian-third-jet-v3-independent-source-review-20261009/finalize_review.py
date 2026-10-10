"""One-shot standard-library finalization of this source-only review."""
import hashlib
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    with pathlib.Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    require(sys.flags.optimize == 0 and sys.flags.isolated == 1 and sys.dont_write_bytecode,
            "explicit unoptimized -I -B required")
    require(os.environ.get("PYTHONOPTIMIZE") == "0", "explicit optimization environment required")
    require(not (HERE / "receipt.json").exists() and not (HERE / "index.json").exists(), "fresh one-shot freeze required")
    result = json.loads((HERE / "static-readback.json").read_text())
    require(result["passed"] is True and result["inputs_unchanged"] is True, "successful unchanged static review required")
    before = json.loads((HERE / "pins-before.json").read_text())
    after = json.loads((HERE / "pins-after.json").read_text())
    require(before == after, "input snapshot drift")
    for name, digest in before.items():
        require(sha(name) == digest, "final external/source drift: " + name)
    require((HERE / "review.stderr").stat().st_size == 0, "review process failure must be retained and resolved")
    receipt = {"passed": True, "inputs_unchanged": True, "source_review_only": True,
               "reviewed_source_index_sha256": result["reviewed_source_index_sha256"],
               "recipe_sha256": result["recipe_sha256"], "driver_sha256": result["driver_sha256"],
               "diff_sha256": result["diff_sha256"], "review_sha256": sha(HERE / "REVIEW.md"),
               "static_readback_sha256": sha(HERE / "static-readback.json"),
               "captured_source_files": result["captured_source_files"],
               "protected_external_pins": result["protected_external_pins"],
               "unique_total_input_pins": result["unique_total_input_pins"],
               "disposition": "PASS narrow precision-only source/math/admission review; execution remains separately held",
               "execution_authorized": False, "candidate_imports": False, "target_evaluation": False,
               "scientific_payload_decoding": False, "kernel_native_compile_CAS_calls": False,
               "future_prerequisites": ["exact separate root units release with a 60-second process-group cap",
                                        "fresh same-index units PASS before timing release",
                                        "fresh same-index timing PASS and exact timing-bound source/cost review before full release"],
               "limitations": result["limitations"],
               "finalization_command": ["python3", "-I", "-B", str(pathlib.Path(__file__).resolve())],
               "finalization_runtime": {"executable": sys.executable, "resolved_executable": str(pathlib.Path(sys.executable).resolve()),
                                        "executable_sha256": sha(sys.executable), "version": sys.version, "flags": str(sys.flags)},
               "review_process_returncode": 0, "review_stderr_bytes": 0}
    save(HERE / "receipt.json", receipt)
    rows = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path == HERE / "index.json":
            continue
        require(path.suffix not in (".jsonl", ".npz", ".npy") and path.stat().st_size <= 1048576,
                "compact source review policy violated")
        path.read_bytes().decode("utf-8")
        rows.append({"path": str(path.relative_to(HERE)), "bytes": path.stat().st_size, "sha256": sha(path)})
    save(HERE / "index.json", {"status": "immutable independent source-only Gaussian v3 precision review",
                              "file_count": len(rows), "total_bytes": sum(row["bytes"] for row in rows), "files": rows})
    print(json.dumps({"index_sha256": sha(HERE / "index.json"), "receipt_sha256": sha(HERE / "receipt.json"),
                      "review_sha256": sha(HERE / "REVIEW.md"), "files": len(rows),
                      "bytes": sum(row["bytes"] for row in rows)}))


if __name__ == "__main__":
    main()
