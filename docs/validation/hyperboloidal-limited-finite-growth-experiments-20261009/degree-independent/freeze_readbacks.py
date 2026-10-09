"""One saved-array algebra readback capsule; no generator or propagation."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "immutable-finite-rb-degree-independent-readback-20261009"
assert not OUT.exists()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


rows, external = [], {}
for path in sorted(HERE.glob("*-readback001/result.json")):
    data = json.loads(path.read_text())
    assert data["status"] == "PASS"
    parameter = data["parameters"]
    rows.append({"J": parameter["J"], "N": parameter["N"], "rb": parameter["rb"],
                 "result_sha256": sha(path), "incoming_rank": data["incoming_trace_rank"]["rank"],
                 "sector_ranks": {k: v["rank"] for k, v in data["sector_ranks"].items()},
                 "weak_strong": data["checks"]["weak_strong"],
                 "bulk_energy_identity": data["checks"]["bulk_energy_identity"],
                 "manufactured_E_norm_error": data["manufactured"]["exact_Xdot_E_norm_error"]})
    for pin in data["outputs"].values():
        assert sha(pin["path"]) == pin["sha256"]
        external[pin["path"]] = pin
    assert (path.parent / "stderr.txt").read_bytes() == b""
    command = json.loads((path.parent / "command-receipt.json").read_text())
    assert command["returncode"] == 0
assert len(rows) == 6
summary = {"status": "PASS_six_saved_matrices_only", "cases": sorted(rows, key=lambda v: (v["J"], v["N"])),
           "verifier_sha256": sha(HERE / "verify_operator.py"),
           "kernel_assembly_generator_eigs_or_propagation": False,
           "scope": "Saved algebra/SPD/solve/SAT/trace ranks and one supplied mixed manufactured field only. No new source/action/family/continuum or uniform-energy claim."}
(HERE / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
(HERE / "REPORT.md").write_text(
    "The unchanged frozen verifier855245 was applied to six separately pinned\n"
    "J=0/1/2,N=12/16,rb=.98 saved decorated outputs. All6 returned zero with\n"
    "empty stderr and PASS algebra/SPD/congruence/solve/weak-strong/volume/SAT\n"
    "checks. Measured incoming ranks are4/8/10 and sector ranks2+2+0,4+4+0,\n"
    "4+4+2 forJ0/J1/J2 at both degrees. Values/absolute-vs-scaled norms are\n"
    "retained in the individual results and compact summary. Only symmetric\n"
    "energy eigenvalues for positivity and singular values for rank were used.\n\n"
    "No point API, assembler, generator eigensolve or propagation was executed.\n"
    "The verifier reads the one supplied mixed manufactured field. Broader\n"
    "forcing-family/point-action/paired-quadrature source gates remain separate\n"
    "owner/root evidence; this capsule does not replace or extend them. It\n"
    "establishes no continuum, CPBC, uniform energy, scri, pulse or BH acceptance.\n"
)
OUT.mkdir()
files = []
for p in sorted(HERE.rglob("*")):
    if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
        continue
    rel = str(p.relative_to(HERE))
    assert p.stat().st_size <= 1048576
    target = OUT / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, target)
    row = {"path": rel, "origin": str(p), "sha256": sha(p), "bytes": p.stat().st_size}
    assert sha(target) == row["sha256"]
    files.append(row)
index = {"kind": "Immutable independent six-degree saved-matrix readback",
         "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE.parents[2], text=True).strip(),
         "production_source_commit": "27c19d20696ea6dd4704032c51dfd026218f64f2",
         "files": files, "large_external_records": list(external.values()),
         "small_file_count": len(files), "small_bytes": sum(x["bytes"] for x in files),
         "scope": summary["scope"]}
(OUT / "index.json").write_text(json.dumps(index, indent=2, allow_nan=False) + "\n")
print(json.dumps({"path": str(OUT / "index.json"), "sha256": sha(OUT / "index.json"),
                  "files": len(files), "bytes": index["small_bytes"], "external_records": len(external)}))
