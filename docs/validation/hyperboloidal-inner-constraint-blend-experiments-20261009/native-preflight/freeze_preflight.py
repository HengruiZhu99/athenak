"""Freeze completed blend native preflights; retain large arrays by hash only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "immutable-native-blend-preflight-20261009"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUT.exists(), "Never rewrite a frozen snapshot"
    spec = importlib.util.spec_from_file_location("accepted_blend_auditor", HERE / "audit_blended_native.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    build = audit.verify_build()
    audit.load_helper()
    for name in ("reference", "short"):
        receipt = json.loads((HERE / "audit" / (name + ".json")).read_text())
        assert receipt["status"] == "PASS"
        assert receipt["auditor_sha256"] == sha(HERE / "audit_blended_native.py")
        assert receipt["build_verification"] == build
        launch = json.loads((HERE / (name + "-launch.json")).read_text())
        assert launch["native_build_receipt_sha256"] == audit.BUILD_RECEIPT
        assert launch["native_executable_sha256"] == audit.PRIVATE_EXE
        assert launch["override_sha256"] == sha(HERE / "native.athinput")
        assert launch["launch_script_sha256"] == sha(HERE / "launch_preflight.py")
        assert launch["blend_gate_index_sha256"] == audit.BLEND_INDEX
        command = launch["command"]
        assert Path(command[2]) == HERE / "native-build/athena-blended-c1"
        assert Path(command[3]) == HERE / name
        assert Path(command[command.index("--overrides") + 1]) == HERE / "native.athinput"
        assert float(command[command.index("--duration") + 1]) == receipt["exact_final_comparison_time"]
        for path, row in receipt["all_run_files"].items():
            source = ROOT / path
            assert sha(source) == row["sha256"] and source.stat().st_size == row["bytes"]
    short_launch = json.loads((HERE / "short-launch.json").read_text())
    assert short_launch["reference_audit_sha256"] == sha(HERE / "audit/reference.json")
    pole = json.loads((HERE / "pole-audit/snapshot-results.json").read_text())
    assert pole["status"] == "PASS" and pole["compile_returncode"] == 0
    assert pole["actual_inputs_unchanged"] and len(pole["rows"]) == 6
    assert pole["private_build_receipt_sha256"] == audit.BUILD_RECEIPT
    assert sha(HERE / "pole-audit/check_snapshot") == pole["utility_executable_sha256"]
    assert sha(HERE / "pole-audit/compile.log") == pole["compile_log_sha256"]
    for path, digest in pole["source_sha256"].items():
        assert sha(ROOT / path) == digest
    for row in pole["rows"]:
        assert row["returncode"] == 0
        assert sha(ROOT / row["restart"]) == row["restart_sha256"]
        assert sha(ROOT / row["raw_file"]) == row["raw_sha256"]
        assert all(q["absolute_difference"] == 0 for q in row["native_HST_comparisons"].values())
        assert row["computed"]["synthetic_manual_vs_helper_max"] < 2e-15
        assert row["computed"]["synthetic_missing_double_pole_control"] > .01
    explicit = ["audit_blended_native.py", "build_blended_native.py", "launch_preflight.py",
                "freeze_preflight.py", "blend_injection.hpp", "native.athinput",
                "reference-launch.json", "short-launch.json", "audit-reference.log", "audit-short.log",
                "audit/reference.json", "audit/short.json", "native-build/build-source.py",
                "native-build/build-receipt.json", "native-build/build.log"]
    sources = {HERE / name for name in explicit}
    sources |= set((HERE / "native-build").glob("attempt-*.json"))
    sources |= {p for p in (HERE / "native-build/include").rglob("*") if p.is_file()}
    large = set((HERE / "native-build").glob("*.o")) | set((HERE / "native-build").glob("*.d"))
    large.add(HERE / "native-build/athena-blended-c1")
    for name in ("reference", "short", "pole-audit"):
        for path in (HERE / name).rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix in (".bin", ".rst", ".raw") or path.name in ("athena-validation", "check_snapshot"):
                large.add(path)
            else:
                sources.add(path)
    assert all(p.name not in ("DRAFT.md", "archive-README.md") for p in sources)
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
        "Independent inner blended-C1 native preflight and geometric pole audit.\n"
        "Data/build checks PASS; no stabilization, nonlinear closure, scri or BH acceptance.\n"
        "Production remains 27c19d20696ea6dd4704032c51dfd026218f64f2.\n"
        "369 base sources plus4 norm overlays,4 private headers,6 exact replacement\n"
        "compile/object/dependency chains,270 repository dependencies,182 unchanged\n"
        "original objects plus4 libraries and actual link/executable hashes verified.\n"
        "The coefficient C=1-W_gauge multiplies all mechanical C1 plus spatial-Z\n"
        "covector repair additions; compact support r<.85 has Omega>=.2775 here.\n"
        "Physical-P storage and gauge prescription unchanged; geometric P RHS has\n"
        "the required regular C1 trace addition. This is a lower-order blend.\n\n"
        "Reference t=.05:3 full binary64 snapshots,17.760430542seconds; maximum\n"
        "full-state drift1.3070271764715835e-14. Initial25 fields,coordinates,masks\n"
        "bitwise equal baseline. Final H/M/Z=8.43866403793709e-14/\n"
        "2.783653940764748e-14/1.2550557899618163e-15.\n\n"
        "Angular pulse t=.02:3 full binary64 snapshots versus2 original baseline\n"
        "snapshots,7.056620084seconds. Initial25 fields,coordinates,masks bitwise\n"
        "equal baseline; geometry also bitwise equals zero-reference geometry.\n"
        "Only generated-input differences are output cadence .02 to .01.\n"
        "Exact endpoint H/M/Z=.0031099659712307947/.0048944949479244595/\n"
        ".0011980012280368697; baseline ratios=.999785963858048/\n"
        "1.0000911450837373/.99999902536574. Early constraints essentially baseline.\n\n"
        "All25 evolved binary64 fields finite; lapse/chi positive,metric SPD,\n"
        "det/trace errors below1e-12; active binary32 BIN equals rounded RST bitwise.\n"
        "Historical physical_metric_eigen keys denote Penrose gtilde/chi magnitudes;\n"
        "physical metric positivity is equivalent on Omega>0.\n\n"
        "Independent actual snapshot jets reproduce H/M/Z/Theta and corrected\n"
        "geometric pole deviation exactly for all6 native snapshots. The pole\n"
        "numerator is C0 plus blended simple plus double/Omega, with identical\n"
        "analytic background assembly. It excludes regular and gauge terms.\n"
        "Shell2608 cells,296 overlapping addition support. Actual early correction\n"
        "is tiny (max6.661338147750939e-15), so7 nontrivial off-constraint point\n"
        "controls independently verify helper at1.3877787807814457e-17; omitting\n"
        "double/Omega produces .06294062732031455 error. Pole utility Release only.\n\n"
        "This index freezes completed preflight evidence only. Root DRAFT, archive\n"
        "README, later damping-profile work and global results are excluded.\n"
        "Large binaries/objects/dependencies/raw/BIN/RST are metadata and hashes\n"
        "only. Prior accepted helpers and immutable archives remain unchanged.\n")
    files[report.name] = {"bytes": report.stat().st_size, "sha256": sha(report)}
    index = {"immutable": True, "native_or_scri_stability_accepted": False,
             "small_files": files,
             "large_files": {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size,
                              "sha256": sha(p)} for p in sorted(large)},
             "blend_mathematical_gate_index_sha256": audit.BLEND_INDEX,
             "build_receipt_sha256": audit.BUILD_RECEIPT,
             "independent_reference_audit_sha256": sha(HERE / "audit/reference.json"),
             "independent_short_audit_sha256": sha(HERE / "audit/short.json"),
             "independent_geometric_pole_audit_sha256": sha(HERE / "pole-audit/snapshot-results.json")}
    (OUT / "index.json").write_text(json.dumps(index, indent=2, allow_nan=False) + "\n")
    for name, row in files.items():
        assert sha(OUT / name) == row["sha256"] and (OUT / name).stat().st_size == row["bytes"]
    print("Frozen", len(files), "small files,", sum(r["bytes"] for r in files.values()),
          "bytes,", len(large), "large hashes", sha(OUT / "index.json"))


if __name__ == "__main__":
    main()
