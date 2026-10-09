"""Freeze small read-only attribution evidence; large arrays/exes by hash only."""
import hashlib
import json
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEST = HERE / "immutable-composed-diagnostic-attribution-20261009"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not DEST.exists()
    DEST.mkdir()
    receipt = json.loads((HERE / "receipt.json").read_text())
    assert receipt["status"] == "PASS" and len(receipt["rows"]) == 9
    files = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or DEST in path.parents:
            continue
        if path.suffix in (".raw", ".cells") or path.name in ("check_original", "check_composed"):
            continue
        files.append((path, path.relative_to(HERE)))
    external = []
    sources = [ROOT / name for name in receipt["source_sha256"]]
    sources += [ROOT / receipt["private_build"]["path"]]
    build = json.loads((ROOT / receipt["private_build"]["path"]).read_text())
    sources += [ROOT / name for name in build["private_source_sha256"]]
    sources += [ROOT / "build-layer-research/continuum/preferred/native-overlay/spatial-norm-family/native_injection.hpp",
                ROOT / "build-layer-research/continuum/preferred/native-overlay/spatial-norm-family/spatial_norm_control.hpp"]
    for row in receipt["rows"]:
        for kind in ("native_results", "native_input", "native_HST"):
            sources.append(ROOT / row[kind]["path"])
        case = (ROOT / row["native_input"]["path"]).parent
        manifest = case.parent / "source-at-launch/manifest.json"
        if manifest.exists():
            sources.append(manifest)
        for kind in ("restart", "raw", "cell_data"):
            external.append(row[kind])
    external += list(receipt["executables"].values())
    external.append({"path": build["executable"], "sha256": build["executable_sha256"],
                     "bytes": Path(build["executable"]).stat().st_size})
    seen = set()
    for source in sources:
        if source in seen or HERE in source.parents:
            continue
        seen.add(source)
        files.append((source, Path("inputs") / source.relative_to(ROOT)))
    for source, relative in files:
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
    listing = {str(path.relative_to(DEST)): {"sha256": sha(path), "bytes": path.stat().st_size}
               for path in sorted(DEST.rglob("*")) if path.is_file()}
    index = {"scope": "Frozen read-only composed constraint-functional attribution; no evolution or acceptance",
             "files": listing, "file_count": len(listing),
             "bytes": sum(entry["bytes"] for entry in listing.values()),
             "large_external_by_hash_only": {entry["path"]: entry for entry in external},
             "receipt_sha256": sha(HERE / "receipt.json"),
             "original_asbuilt_receipt_sha256": receipt["private_build"]["sha256"]}
    (DEST / "index.json").write_text(json.dumps(index, indent=2, allow_nan=False)+"\n")
    print(str(DEST.relative_to(ROOT)), sha(DEST / "index.json"))
    print(len(listing), "files", index["bytes"], "bytes", len(index["large_external_by_hash_only"]), "large external hashes")


if __name__ == "__main__":
    main()
