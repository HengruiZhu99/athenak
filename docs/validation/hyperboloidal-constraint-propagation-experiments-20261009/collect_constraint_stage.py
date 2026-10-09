"""Copy completed, checked evidence once; never rewrite an existing archive."""

import hashlib
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
CONTROLS = ROOT / "build-layer-research/time-projection-controls"
DEST = ROOT / "docs/validation/hyperboloidal-constraint-propagation-experiments-20261009"
CATALOG = {}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def finite_json(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            finite_json(item)
    elif isinstance(value, list):
        for item in value:
            finite_json(item)


def add(source, target):
    source = Path(source)
    if not source.is_absolute():
        source = ROOT / source
    data = source.read_bytes()
    if source.suffix == ".json":
        finite_json(json.loads(data))
    assert target not in CATALOG, target
    path = DEST / target
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    assert path.read_bytes() == data
    CATALOG[target] = {
        "source": str(source.relative_to(ROOT)),
        "sha256": digest(data),
        "bytes": len(data),
    }


def frozen(folder, index, expected, target):
    base = ROOT / folder
    data = (base / index).read_bytes()
    assert digest(data) == expected, base
    record = json.loads(data)
    entries = record.get("files", record.get("small_files"))
    assert isinstance(entries, dict) and entries
    for name, spec in entries.items():
        source = ROOT / name if name.startswith("build-layer-research/") else base / name
        data = source.read_bytes()
        wanted = spec if isinstance(spec, str) else spec["sha256"]
        assert digest(data) == wanted, source
        if isinstance(spec, dict) and "bytes" in spec:
            assert len(data) == spec["bytes"], source
        relative = source.relative_to(base) if source.is_relative_to(base) else source.relative_to(ROOT)
        add(source, target + "/" + str(relative))
    add(base / index, target + "/" + index)


def native_run(name, target):
    run = CONTROLS / name
    results = json.loads((run / "results.json").read_text())
    # The driver records one or more completed cases under cases.
    cases = results["cases"]
    assert cases and all(case["exit_status"] == 0 for case in cases)
    for path in sorted(run.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in {".json", ".log", ".hst", ".athinput", ".patch"}:
            add(path, target + "/" + str(path.relative_to(run)))


def main():
    assert not DEST.exists(), "Refuse to mutate a previously collected archive"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert head == "b37a20f2d7a8ccc42148f1a814a80b2957e17b53", head
    DEST.mkdir(parents=True)
    gates = [
        ("continuum/constraint-propagation/immutable-constraint-propagation-20261009",
         "manifest.json", "1ef4735a5b1136c7836daf4fc8783a24f5b83e42eca8281fa63bff8735c4040a",
         "subsidiary"),
        ("continuum/discrete-bianchi/immutable-discrete-bulk-20261009", "index.json",
         "befb62b2d341b2dbe16b3f51540b073b21cb44eb1e2094381364af7c2386e2d4", "discrete-bulk"),
        ("continuum/tensor-discretization/immutable-tensor-negative-20261009", "index.json",
         "a66a6af44341612d4200960e3babe95c486f1a63ac2bd0a647458f6337a5ebff", "tensor-negative"),
        ("inner-trumpet-gate", "frozen-index.json",
         "6e052b27abdb49343afca46c4e3028ddd9d3a27bb88357bc7031ce6a14795135", "inner-trumpet"),
        ("time-projection-controls/rst-reader-gate", "frozen-index.json",
         "0fbb56d679b03343a599d0663fcda2249180f3d8920d07b97b0f66728d9b7687", "rst-reader"),
        ("time-projection-controls/composed-derivative-gate", "frozen-index.json",
         "fb6a7a051fabf72d3a47decb1b08e36098dee1b1dca9051ed86e85774fecbf11", "composed-gate"),
        ("boundary/full-tensor-stage-interim", "manifest.json",
         "81132e94cc9b15ac953cd7368ddd092bfbae2e06beea96f1367116eca7ba2ddc", "full-tensor-stage"),
        ("boundary/full-tensor-global-final", "manifest.json",
         "4f62819e2337c0cb20571071fe10d0e80d8b43895a928ccd4418faec0bc590d2", "full-tensor-global"),
        ("continuum/covariant-z4/immutable-C1-tensor-identity-20261009", "index.json",
         "6d54d686664ffdf3dc8a185075d5c7c34d505f275d58582e71f38ccbc30aea4f", "C1-identity"),
        ("time-projection-controls/small-and-control-review-frozen", "frozen-index.json",
         "1b80c5412200510725cffd781441a3a27d9b503d3a2f3ff77c8e075bbba3f9f3", "small-and-control-review"),
        ("time-projection-controls/composed-final-native-audit", "frozen-index.json",
         "45c173ae2e5192be418e411ee6eb234f3884df104769261b83c00704e5412c5d", "composed-final-native"),
    ]
    for folder, index, sha, target in gates:
        frozen("build-layer-research/" + folder, index, sha, target)

    for name in [
        "build_stage_projection.py", "audit_native_control.py",
        "N24-half-step.athinput", "N24-half-step-launch.json",
        "N24-half-step-t2-audit.json", "stage-projection-launch.json",
        "stage-projection-reference-preflight.json", "stage-projection-reference-audit.json",
        "stage-projection-t2-audit.json", "stage-projection-build/z4c_tasks.cpp",
        "stage-projection-build/build-receipt.json", "stage-projection-build/build.log",
        "build_composed_derivative.py", "composed-preflight-launch.json",
        "composed-evolution-launch.json", "composed-N24.athinput", "composed-N36.athinput",
        "audit_composed_native.py", "audit_small_native.py",
        "small-angular-N24.athinput", "small-angular-launch.json",
    ]:
        add(CONTROLS / name, "native-controls/" + name)

    for name in ["N24-half-step-t2", "stage-projection-t2", "stage-projection-reference",
                 "composed-reference", "composed-short", "small-angular-t2",
                 "composed-N24-t2", "composed-N36-t0.2"]:
        native_run(name, "native-runs/" + name)

    for path in sorted((CONTROLS / "composed-native-audit").rglob("*")):
        if path.is_file() and path.suffix in {".json", ".txt", ".py"}:
            add(path, "composed-native-audit/" + str(path.relative_to(CONTROLS / "composed-native-audit")))

    failure = CONTROLS / "composed-derivative-build-failed-rvalue"
    for path in sorted(failure.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".py", ".hpp", ".cpp", ".log"}:
            add(path, "failed-rvalue-build/" + str(path.relative_to(failure)))

    add(Path(__file__), "collect_constraint_stage.py")
    add(CONTROLS / "archive-README.md", "README.md")
    add(CONTROLS / "verify_constraint_archive.py", "verify_constraint_archive.py")
    catalog = {
        "scope": "Completed private experiments and negative gates; no stable pulse or regular scri claim.",
        "compiled_production_implementation": "27c19d20696ea6dd4704032c51dfd026218f64f2",
        "collection_head": head,
        "files": CATALOG,
        "excluded": "Executables, objects, CSR matrices, BIN/RST and other large outputs stay local. "
                    "Exact metadata/hashes and original commands are in the copied receipts.",
    }
    (DEST / "catalog.json").write_text(json.dumps(catalog, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"files": len(CATALOG), "bytes": sum(x["bytes"] for x in CATALOG.values()),
                      "catalog_sha256": digest((DEST / "catalog.json").read_bytes())}))


if __name__ == "__main__":
    main()
