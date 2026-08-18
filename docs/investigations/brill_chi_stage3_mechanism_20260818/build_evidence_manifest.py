#!/usr/bin/env python3
"""Build and strictly validate the stage-3 evidence manifest."""

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXCLUDED = {"evidence_manifest.json", "SHA256SUMS", "SHA256SUMS.sha256"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def finite(value, location="root"):
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"nonfinite JSON value at {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            finite(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite(item, f"{location}[{index}]")


def main() -> None:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if path.parent == ROOT and path.name in EXCLUDED:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"),
                                 parse_constant=lambda value: (_ for _ in ()).throw(
                                     ValueError(f"invalid JSON constant {value}")))
            finite(payload, relative)
        files.append({"path": relative, "bytes": path.stat().st_size,
                      "sha256": digest(path)})

    required = {
        "REPORT.md", "REMOTE_REVIEW_PROMPT.md", "conditional_control_summary.json",
        "target_block_refinement_timeline.csv",
        "local_high_frequency_metrics.csv",
        "rk_stage3_candidate_summary.json",
        "rk_accumulator_audit.csv",
        "chi_rhs_term_decomposition.csv",
        "chi_candidate_counterfactuals.csv",
        "local_stiffness_metrics.csv",
        "chi_stencil_values.csv",
        "chi_stencil_owner_comparison.csv",
        "derivative_order_comparison.csv",
        "ko_directional_comparison.csv",
        "focused-tests.log",
        "evidence/v14/run/conditional-control/run.log",
        "source/tracked.patch", "source/owned-untracked.tar.gz",
    }
    present = {item["path"] for item in files}
    missing = sorted(required - present)
    if missing:
        raise RuntimeError(f"required evidence missing: {missing}")

    manifest = {
        "schema": "athenak_brill_chi_stage3_evidence_manifest_v1",
        "final_disposition": "NOT_ESTABLISHED",
        "phase1_classification": "ADVECTION_DOMINATED_FAILURE",
        "production_qualification_claim": False,
        "convergence_claim": False,
        "figure3_reproduction_claim": False,
        "numerical_source": {
            "head": "ac75c8d348da91b38cbc6855b5fba51cd3089663",
            "tree": "6284882bd06e8db379495675aba7a4f153fb4afa",
            "kokkos": "6739bc623081648af9e752b616d9671527922cbf",
            "v9_patch_sha256": "ce50eda70904d6fd13512ba8dbf83cd6ab4d01d7ea1a3e820b36dda63eac436b",
            "v14_patch_sha256": "d4e0db8efdac39317a33dce18a9df4c86b86b30d177e96174fb615304ab8c328",
            "final_commit": "0b398079cb2e33bbb8dbb485078b13d83ecb71b8",
            "final_tree": "5b4301d83effce38c435b8f265ac67777a85fa54",
            "branch": "codex/brill-chi-stage3-mechanism-20260818",
        },
        "phase1": {
            "job_id": 57239599,
            "gpu": "NVIDIA A100-SXM4-40GB",
            "remote_root": "/pscratch/sd/h/hzhu/axisymmetric-cartoon-chi-stage3-ac75c8d3-v9-20260818",
            "root_manifest_sha256": "104fc41525880c39b76e4e8d08e04e1cca5fae7ad3177f1c03c9703f4e1a7dbb",
            "detached_manifest_sha256": "e284a4a4797523aebf0ad774fde25e9e507867603a6717ee0e1f3f9a9dcd46e4",
            "executable_sha256": "aab5704fa8684aea5cbdb5a1dfcbd89cc0b7b243d3f9ac4310dfe81a6d266d28",
            "cache_sha256": "95531979a24734bc2450eb38962fc29f9bdf7134c0583d55c6cb9553f8033035",
            "native_status": 1,
        },
        "control": {
            "job_id": 57245427,
            "gpu": "NVIDIA A100-SXM4-40GB",
            "remote_root": "/pscratch/sd/h/hzhu/axisymmetric-cartoon-chi-stage3-ac75c8d3-v14-20260818",
            "root_manifest_sha256": "560d8208d24641ee58b66a37f779bc205619cbd7e0da8663f0597089c9c40140",
            "detached_manifest_sha256": "ed8da8202fc57654d1f3133f9b5cf575901eeea9a17b68fcd17ac695a398c619",
            "executable_sha256": "223318d7a31ac9a978b36bbb25505a68c1ee3e4c5e0c6c929fb35e9a4300c621",
            "cache_sha256": "6b93a1d3a35af2eb21bdc5f3537818db491b538d76972d7e8ccd5cc50fe8c570",
            "native_status": 1,
            "target_tree_exact_match": True,
        },
        "inputs": {
            "restart_sha256": "2e2e8f7febd0d4fbb204f172df149f9295de6aa66097ef3c9f19048aa29a20e9",
            "authority_history_sha256": "d0e1289757bd8f5b6510ca8a7e8b8c5c42bec54f5f08480f607abc866af57555",
            "shadow_requests_sha256": "6d28a3743cc84dc3a111869f86ae8bc764e3c0db55a53196ff6b5461050ad483",
            "brill_coefficients_sha256": "ff0993c390513c15d6aa65857a0a3c710f2e2c3faf5717d9d63245203ccf2d6b",
            "pilot_input_sha256": "edced480bbd934347aa80152dda4c164c4b6fd59c2a7abe764ac990983004791",
        },
        "files": files,
    }
    finite(manifest)
    (ROOT / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"STRICT_EVIDENCE_MANIFEST_PASS files={len(files)}")


if __name__ == "__main__":
    main()
