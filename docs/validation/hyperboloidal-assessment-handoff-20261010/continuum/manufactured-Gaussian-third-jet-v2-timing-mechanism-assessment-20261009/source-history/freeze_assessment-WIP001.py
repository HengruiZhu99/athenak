"""One-shot stdlib-only compact saved association and source-note freeze.

Never imports the oracle or scientific runtime and never decodes either JSONL.
"""
from pathlib import Path
from decimal import Decimal
import hashlib
import json
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OWNER = ROOT / "build-layer-research/continuum/manufactured-angular-Gaussian-third-jet-oracle-v2-held-20261009"
RELEASE = ROOT / "build-layer-research/Gaussian-third-jet-oracle-root-release-20261009"
PEER = ROOT / "build-layer-research/continuum/manufactured-Gaussian-third-jet-v2-timing-failure-independent-saved-review-20261009"
EXPECTED = {
    str(OWNER / "source-index.json"): "92a412f36e668bb2e2414bd65e5dee381e42534d94e2df712fd7e9f3a52ce5d3",
    str(OWNER / "attempts/timing001/receipt.json"): "368ffb6f2435efedefac5e24d43ca6550ffe5b7737fcfeab055af1557358c0a1",
    str(OWNER / "attempts/timing001/result.json"): "9ea24c4ba44cad69548e75191178350ae28bdfaedaf4c3f56e51037f596f7901",
    str(RELEASE / "timing-outer001/stdout.log"): "5d8b86db5d64e755ea4e76d4f3d39733b3a80a4b448c16892f775ea4d472b20d",
    str(PEER / "index.json"): "a8d0797e9707c4108fbbc5b228eb9749921daed645abf0140212367115bc8448",
    str(PEER / "receipt.json"): "ef04dd1f1cf98a4ed93a57bbbb669feac7a575b81137c5671a744b943959b20f",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    path = Path(path)
    if path.suffix in (".jsonl", ".npy", ".npz") or path.stat().st_size > 1048576:
        raise RuntimeError("payload decode is outside this assessment")
    return json.loads(path.read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    require(not (HERE / "index.json").exists(), "assessment already frozen")
    require(sys.flags.optimize == 0, "optimized metadata reader is not admitted")
    before = dict(EXPECTED)
    for path, digest in before.items():
        require(sha(path) == digest, "typed source or saved-output pin mismatch: " + path)
    child = load(OWNER / "attempts/timing001/receipt.json")
    result = load(OWNER / "attempts/timing001/result.json")
    original = load(OWNER / "attempts/timing001/source-before.json")
    require(original == load(OWNER / "attempts/timing001/source-after.json"), "saved source sets differ")
    before.update(original)
    before.update(load(OWNER / "source-index.json")["files"])
    for relative, digest in child["output_hashes"].items():
        before[str(OWNER / "attempts/timing001" / relative)] = digest
    peer_index = load(PEER / "index.json")
    for relative, meta in peer_index.get("files", {}).items():
        path = Path(relative)
        path = path if path.is_absolute() else PEER / path
        before[str(path)] = meta if isinstance(meta, str) else meta["sha256"]
    for path, digest in before.items():
        require(sha(path) == digest, "protected metadata/source drift: " + path)
    require(child["completed"] is True and child["passed"] is False and child["sources_unchanged"] is True,
            "original timing classification mismatch")
    require((result["records"], result["identity_checks"], result["precision_checks"], result["failed_total"])
            == (20, 11816, 1692, 6), "saved result counts differ")
    expected_names = ["connection_conformal_" + suffix for suffix in ("011", "012", "013", "022", "023", "033")]
    require([row["name"] for row in result["failed"]] == expected_names, "saved failure names differ")
    require(all(row["branch"] == "bounded_outer_conformal" and
                Decimal(row["scaled"]) > Decimal("1e-55") for row in result["failed"]), "failure classification differs")
    events = [json.loads(line) for line in (RELEASE / "timing-outer001/stdout.log").read_text().splitlines() if line.strip()]
    completes = [row for row in events if row.get("event") == "complete"]
    require(len(completes) == 20 and [row["records"] for row in completes] == list(range(1, 21)), "progress coverage differs")
    previous = 0
    increments = []
    for row in completes:
        require(type(row["failed"]) is int and row["failed"] >= previous, "failure count is not monotone")
        if row["failed"] != previous:
            increments.append({**row, "new_failures": row["failed"] - previous})
        previous = row["failed"]
    require(len(increments) == 1 and increments[0]["records"] == 9 and increments[0]["digits"] == 110
            and increments[0]["new_failures"] == 6, "failure event association differs")
    require(all(row["failed"] == 6 for row in completes[9:]), "later failures were omitted")
    capture = HERE / "captured"
    capture.mkdir(exist_ok=False)
    selected = [OWNER / name for name in ("source-index.json", "recipe.json", "run_oracle.py", "diagnostics.py",
                "geometry.py", "taylor3.py", "oracle.py", "gaussian3.py", "reference3.py", "PLAN.md", "SCHEMA.md")]
    selected += [OWNER / "attempts/timing001" / name for name in ("receipt.json", "result.json")]
    selected += [RELEASE / "timing-outer001" / name for name in ("stdout.log", "receipt.json", "failure.txt")]
    selected += [PEER / "index.json", PEER / "receipt.json", PEER / "REVIEW.md"]
    copied = []
    for number, path in enumerate(selected):
        require(path.stat().st_size <= 1048576 and path.suffix != ".jsonl", "compact capture policy violated")
        target = capture / ("%02d-%s-%s" % (number, path.parent.name, path.name))
        shutil.copyfile(path, target)
        require(sha(target) == sha(path), "copy mismatch")
        copied.append({"original": str(path), "copy": str(target.relative_to(HERE)), "sha256": sha(target), "bytes": target.stat().st_size})
    metadata = [{"path": str(OWNER / "attempts/timing001" / name),
                 "sha256": child["output_hashes"][name], "bytes": (OWNER / "attempts/timing001" / name).stat().st_size,
                 "policy": "large_payload", "decoded": False, "copied": False}
                for name in ("oracle.jsonl", "precision-checks.jsonl")]
    after = {path: sha(path) for path in before}
    require(after == before, "input drift during assessment")
    save(HERE / "protected-inputs.json", before)
    save(HERE / "saved-association.json", dict(records=20, identity_checks=11816, precision_checks=1692,
         failed_total=6, failure_event=increments[0], names=expected_names,
         maximum_saved_scaled_error=max((row["scaled"] for row in result["failed"]), key=Decimal),
         no_new_150_digit_failures=True, original_timing_passed=False, scientific_payload_metadata=metadata))
    save(HERE / "receipt.json", dict(completed=True, assessment_passed=True, scientific_timing_passed=False,
         source_only_precision_proposal=True, implementation_admitted=False, execution_admitted=False,
         command=sys.argv, unique_protected_pins=len(before), inputs_unchanged=True,
         no_candidate_imports=True, no_target_evaluation=True, no_payload_decode=True,
         no_kernel_native_CAS_or_inverse_execution=True, copied_inputs=copied,
         launch_HEAD="284b4c21e09077ab86f0d0cbbbb5b3a11503cf58"))
    files = {str(p.relative_to(HERE)): {"bytes": p.stat().st_size, "sha256": sha(p), "policy": "source_or_receipt"}
             for p in sorted(HERE.rglob("*")) if p.is_file()}
    save(HERE / "index.json", dict(status="FROZEN_SOURCE_ONLY_MECHANISM_AND_PRECISION_PROPOSAL_HELD",
         original_v2_timing_passed=False, implementation_admitted=False, execution_admitted=False,
         file_count=len(files), bytes=sum(v["bytes"] for v in files.values()), files=files))
    print(json.dumps({"index_sha256": sha(HERE / "index.json"), "receipt_sha256": sha(HERE / "receipt.json"),
          "assessment_sha256": sha(HERE / "ASSESSMENT.md"), "proposal_sha256": sha(HERE / "precision-proposal.json"),
          "files": len(files), "protected_pins": len(before)}))


if __name__ == "__main__":
    main()
