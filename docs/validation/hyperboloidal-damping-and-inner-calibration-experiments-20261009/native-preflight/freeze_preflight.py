"""Freeze only completed C0 profile reference/short preflight evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "immutable-native-profile-preflight-20261009"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUT.exists(), "Never overwrite immutable evidence"
    spec = importlib.util.spec_from_file_location("accepted_profile_auditor", HERE / "audit_profiled_native.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    build = audit.verify_build()
    audit.load_helper()
    for name in ("reference", "short"):
        receipt = json.loads((HERE / "audit" / (name + ".json")).read_text())
        assert receipt["status"] == "PASS"
        assert receipt["auditor_sha256"] == sha(HERE / "audit_profiled_native.py")
        assert receipt["build_verification"] == build
        launch = json.loads((HERE / (name + "-launch.json")).read_text())
        assert launch["native_build_receipt_sha256"] == audit.BUILD_RECEIPT
        assert launch["native_executable_sha256"] == audit.PRIVATE_EXE
        assert launch["override_sha256"] == sha(HERE / "native.athinput")
        assert launch["launch_script_sha256"] == sha(HERE / "launch_preflight.py")
        assert launch["profile_gate_index_sha256"] == audit.PROFILE_INDEX
        cmd = launch["command"]
        assert Path(cmd[2]) == HERE / "native-build/athena-profiled-c0"
        assert Path(cmd[3]) == HERE / name
        assert Path(cmd[cmd.index("--overrides") + 1]) == HERE / "native.athinput"
        assert float(cmd[cmd.index("--duration") + 1]) == receipt["exact_final_comparison_time"]
        for path, row in receipt["all_run_files"].items():
            source = ROOT / path
            assert sha(source) == row["sha256"] and source.stat().st_size == row["bytes"]
    assert json.loads((HERE / "short-launch.json").read_text())["reference_audit_sha256"] == sha(HERE / "audit/reference.json")
    gated = json.loads((HERE / "GATED_BUILD.json").read_text())
    assert gated["native_build_receipt_sha256"] == audit.BUILD_RECEIPT
    assert gated["initial_hold_receipt_sha256"] == sha(HERE / "HOLD.json")
    failure = HERE / "prelaunch-index-format-failure"
    failed = json.loads((failure / "failure.json").read_text())
    assert failed["phase"] == "guard" and not failed["native_launch_created"]
    assert failed["launcher_sha256"] == sha(failure / "launch_preflight.py")
    explicit = ["audit_profiled_native.py", "build_profiled_native.py", "launch_preflight.py",
                "freeze_preflight.py", "native.athinput", "GATED_BUILD.json", "HOLD.json",
                "reference-launch.json", "short-launch.json", "audit-reference.log", "audit-short.log",
                "audit/reference.json", "audit/short.json", "native-build/build-source.py",
                "native-build/build-receipt.json", "native-build/build.log"]
    sources = {str(HERE / name): name for name in explicit}
    for p in (HERE / "native-build").glob("attempt-*.json"):
        sources[str(p)] = str(p.relative_to(HERE))
    for directory in (HERE / "native-build/include", failure):
        for p in directory.rglob("*"):
            if p.is_file():
                sources[str(p)] = str(p.relative_to(HERE))
    base_receipt = audit.FAMILY / "native-build-receipt.json"
    sources[str(base_receipt)] = "dependencies/original-norm-native-build-receipt.json"
    large = set((HERE / "native-build").glob("*.o")) | set((HERE / "native-build").glob("*.d"))
    large.add(HERE / "native-build/athena-profiled-c0")
    for name in ("reference", "short"):
        for p in (HERE / name).rglob("*"):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            if p.suffix in (".bin", ".rst") or p.name == "athena-validation":
                large.add(p)
            else:
                sources[str(p)] = str(p.relative_to(HERE))
    OUT.mkdir()
    files = {}
    for path, name in sorted(sources.items()):
        source, target = Path(path), OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert source.read_bytes() == target.read_bytes()
        files[name] = {"source": str(source.relative_to(ROOT)), "bytes": target.stat().st_size,
                       "sha256": sha(target)}
    report = OUT / "REPORT.txt"
    report.write_text(
        "Independent C0 kappa2-profile native reference/short preflight: PASS\n"
        "data/build integrity only. Production remains27c19d20; no stable pulse,\n"
        "scri closure, global energy or BH acceptance.\n\n"
        "Verified369 original sources plus4 norm overlays,2 private headers,6\n"
        "compile/object/dependency replacements,268 repository dependencies,182\n"
        "unchanged original objects plus4 libraries and exact link/executable.\n"
        "The original norm build receipt is copied byte-identically in dependencies.\n"
        "Profile gatev2 c9180b5b... and receipt d832017b... pin12 passing commands\n"
        "and378 unchanged inputs; historical v1 is preserved separately. Only4\n"
        "ConformalRHS kappa2 callsites plus an include change in the private patch,\n"
        "verified by restoring the entire production header byte-identically.\n"
        "Evolution, analytic reference residual and live/reference geometric pole\n"
        "diagnostics all use the same prescribed profile. Physical-P/gauge, ng3,\n"
        "ghosts, derivatives, KO and final-only projection are retained. No C1 or\n"
        "covector repair/new double pole is added.\n\n"
        "Reference t=.05:3 binary64 snapshots; runner wall17.562575875seconds.\n"
        "Maximum full-state drift1.3235296584782815e-14. Final H/M/Z=\n"
        "8.468871252039659e-14/2.7207503121927156e-14/1.2532372547752724e-15.\n"
        "Short angular pulse t=.02:3 binary64 snapshots versus2 original baseline;\n"
        "runner wall7.452683209seconds. Final H/M/Z=.0030791450287825455/\n"
        ".004895142602895229/.0011948044333206062; same-grid baseline ratios\n"
        ".9898777057170621/1.0002234802905825/.9973305876999488. Theta ratio\n"
        "1.000329439816572. This early mixed change is not finite-pulse acceptance.\n\n"
        "All25 active binary64 evolved fields are finite; lapse/chi positive,\n"
        "metric SPD,det/trace below1e-12; all active BIN fields equal exact binary32\n"
        "casts. Initial fields,coordinates,masks bitwise equal the original norm\n"
        "baseline; initial geometry also bitwise equals zero reference. Only short\n"
        "generated-input differences are output cadence .02 to .01. Historical\n"
        "physical_metric_eigen keys are Penrose gtilde/chi magnitudes; positivity\n"
        "is equivalent to physical SPD on Omega>0.\n\n"
        "The original HOLD and GATED_BUILD history and prelaunch-index-format\n"
        "failure are retained. That launcher guard expected dict records while\n"
        "the profile gate stores string hashes. It failed before any launch/RHS;\n"
        "corrected launch reads both formats without changing the as-built kernel.\n\n"
        "Frozen content is reference+short only. Root DRAFT, later live profiles,\n"
        "global/long results and separate redundant pole utility are excluded.\n"
        "Large exes/objects/dependencies/BIN/RST have hashes/metadata only. Prior\n"
        "accepted helpers and frozen gates remain unchanged.\n")
    files[report.name] = {"bytes": report.stat().st_size, "sha256": sha(report)}
    index = {"immutable": True, "native_global_or_scri_stability_accepted": False,
             "small_files": files,
             "large_files": {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size,
                              "sha256": sha(p)} for p in sorted(large)},
             "profile_gate_index_sha256": audit.PROFILE_INDEX,
             "build_receipt_sha256": audit.BUILD_RECEIPT,
             "reference_audit_sha256": sha(HERE / "audit/reference.json"),
             "short_audit_sha256": sha(HERE / "audit/short.json")}
    (OUT / "index.json").write_text(json.dumps(index, indent=2, allow_nan=False)+"\n")
    for name, row in files.items():
        assert sha(OUT / name) == row["sha256"] and (OUT / name).stat().st_size == row["bytes"]
    print("Frozen", len(files), "small files,", sum(row["bytes"] for row in files.values()),
          "bytes,", len(large), "large hashes", sha(OUT / "index.json"))


if __name__ == "__main__":
    main()
