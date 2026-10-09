"""Freeze completed C1 native preflights without copying binaries/field arrays."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "immutable-native-C1-preflight-20261009"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUT.exists()
    spec = importlib.util.spec_from_file_location("accepted_C1_native_auditor", HERE / "audit_covariant_native.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    build = audit.verify_build()
    audit.load_helper()
    ref = json.loads((HERE / "audit/reference.json").read_text())
    short = json.loads((HERE / "audit/short.json").read_text())
    for name, receipt in [("reference", ref), ("short", short)]:
        assert receipt["status"] == "PASS"
        assert receipt["auditor_sha256"] == sha(HERE / "audit_covariant_native.py")
        assert receipt["build_verification"] == build
        launch = json.loads((HERE / (name + "-launch.json")).read_text())
        assert launch["native_build_receipt_sha256"] == audit.BUILD_RECEIPT
        assert launch["native_executable_sha256"] == audit.PRIVATE_EXE
        assert launch["override_sha256"] == sha(HERE / "native.athinput")
        assert launch["launch_script_sha256"] == sha(HERE / "launch_preflight.py")
        assert launch["gates"]["immutable-C1-stiffness-20261009"]["index_sha256"] == audit.V1
        assert launch["gates"]["immutable-C1-stiffness-v2-20261009"]["index_sha256"] == audit.V2
        assert launch["gates"]["immutable-C1-stiffness-20261009"]["receipt_sha256"] == launch["gates"]["immutable-C1-stiffness-v2-20261009"]["receipt_sha256"]
        command = launch["command"]
        assert Path(command[2]) == HERE / "native-build/athena-covariant-c1"
        assert Path(command[3]) == HERE / name
        assert Path(command[command.index("--overrides") + 1]) == HERE / "native.athinput"
        assert float(command[command.index("--duration") + 1]) == receipt["exact_final_comparison_time"]
        for path, row in receipt["all_run_files"].items():
            source = ROOT / path
            assert sha(source) == row["sha256"] and source.stat().st_size == row["bytes"]
    assert json.loads((HERE / "short-launch.json").read_text())["reference_audit_sha256"] == sha(HERE / "audit/reference.json")
    explicit = ["audit_covariant_native.py", "build_covariant_native.py", "launch_preflight.py",
                "freeze_preflight.py", "native.athinput", "reference-launch.json", "short-launch.json",
                "audit-reference.log", "audit-short.log", "audit/reference.json", "audit/short.json",
                "native-build/build-source.py", "native-build/build-receipt.json", "native-build/build.log"]
    sources = {HERE / name for name in explicit}
    sources |= set((HERE / "native-build").glob("attempt-*.json"))
    sources |= {p for p in (HERE / "native-build/include").rglob("*") if p.is_file()}
    large = set((HERE / "native-build").glob("*.o")) | set((HERE / "native-build").glob("*.d"))
    large.add(HERE / "native-build/athena-covariant-c1")
    for name in ("reference", "short"):
        for path in (HERE / name).rglob("*"):
            if not path.is_file():
                continue
            if path.suffix in (".bin", ".rst") or path.name == "athena-validation":
                large.add(path)
            else:
                sources.add(path)
    OUT.mkdir()
    files = {}
    for source in sorted(sources):
        name = str(source.relative_to(HERE))
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert source.read_bytes() == target.read_bytes()
        files[name] = {"source": str(source.relative_to(ROOT)), "bytes": target.stat().st_size,
                       "sha256": sha(target)}
    report = OUT / "REPORT.txt"
    report.write_text(
        "Independent repaired-C1 native preflight audit: data/build checks PASS.\n"
        "All369 original sources+4 norm overlays,3 private source headers,6 exact\n"
        "compile/object/dependency replacements,269 repository dependencies,\n"
        "182 unchanged original objects+4 libraries and actual link/executable hashes\n"
        "are verified. As-built gatev1 321439... is retained; launchgatev2 d8d137...\n"
        "corrects prose count370 and records the kappa5 positive leading pole. The\n"
        "numeric/source receipt and kappa10 candidate are identical betweenv1/v2.\n"
        "Native override is byte-identical to the original norm-gauge baseline.\n\n"
        "Reference t.05:3 binary64 snapshots,18.007seconds; maximum full-state drift\n"
        "1.49341317e-14. Initial fields/coordinates/masks bitwise equal baseline.\n"
        "Final H/M/Z8.42490469e-14/2.87268184e-14/1.34692860e-15.\n\n"
        "Finite angular pulse t.02:3 binary64 snapshots versus2 original snapshots;\n"
        "initial active arrays/coordinates/masks bitwise equal. Initial geometry\n"
        "also bitwise equals zero reference. Only generated-input differences are\n"
        "outputcadence.02 to.01. Exact endpoint H/M/Z=.00318248706/.00668864913/\n"
        ".00168289263; ratios1.02309991/1.36669030/1.40474897; Theta ratio1.28938735.\n"
        "C1 worsens the early diagnostic result and is not stabilization acceptance.\n\n"
        "All25 active evolved binary64 fields are finite, alpha/chi positive,\n"
        "metric SPD and det/trace errors below1e-12; all active binary32 BIN fields\n"
        "exactly match rounded RST fields bitwise. Historical physical_metric_eigen\n"
        "keys denote Penrose gtilde/chi eigenvalues, not physical magnitudes;\n"
        "positivity is equivalent forOmega>0. No live falloff or scri closure is\n"
        "imposed. No production change, BH action or long-pulse acceptance.\n\n"
        "The independent accepted auditor is hash-pinned; helper files unchanged.\n"
        "No binaries, objects, dependency files or BIN/RST arrays are copied here;\n"
        "their hashes and full snapshot metadata are preserved. No accepted or\n"
        "previously frozen evidence is rewritten.\n")
    files[report.name] = {"bytes": report.stat().st_size, "sha256": sha(report)}
    index = {"immutable": True, "native_or_scri_stability_accepted": False,
             "small_files": files,
             "large_files": {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size,
                             "sha256": sha(p)} for p in sorted(large)},
             "as_built_gate_v1": audit.V1, "launch_gate_v2": audit.V2,
             "build_receipt_sha256": audit.BUILD_RECEIPT,
             "independent_reference_audit_sha256": sha(HERE / "audit/reference.json"),
             "independent_short_audit_sha256": sha(HERE / "audit/short.json")}
    (OUT / "index.json").write_text(json.dumps(index, indent=2, allow_nan=False) + "\n")
    print("Frozen", len(files), "small files,", len(large), "large hashes", sha(OUT / "index.json"))


if __name__ == "__main__":
    main()
