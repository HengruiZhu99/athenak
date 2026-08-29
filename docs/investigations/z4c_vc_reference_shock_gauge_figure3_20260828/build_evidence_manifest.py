#!/usr/bin/env python3
"""Build the strict, reproducible evidence index for this investigation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
OUT = ROOT / "EVIDENCE_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(REPO).as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def selected_artifacts() -> list[Path]:
    files: set[Path] = set()

    for name in (
        "REPORT.md",
        "REPORT.tex",
        "REPORT.pdf",
        "N256_REPRODUCTION.md",
        "N512_REPLAY.md",
        "CONVERGENCE.md",
        "AURORA_PVC_TESTS.md",
        "REFERENCE_GAUGE.md",
        "GAUGE_IMPLEMENTATION_AUDIT.md",
        "figure3_published_curves.csv",
        "brill_vc_reference_shock_gauge.athinput",
        "analyze_reference_n256.py",
        "analyze_reference_comparison.py",
        "analyze_reference_convergence.py",
        "analyze_reference_field_patch.py",
        "compare_reference_field_patch.py",
        "classify_reference_constraint_locations.py",
        "aurora_run_reference_replay.sh",
        "aurora_reference_n128_segment.pbs",
        "aurora_reference_n512_segment.pbs",
        "build_evidence_manifest.py",
    ):
        path = ROOT / name
        if path.is_file():
            files.add(path)

    for relative_dir in (
        "analysis/aurora_n256",
        "analysis/aurora_n256_n512/final",
        "analysis/aurora_n256_n512/field_patch/n256",
        "analysis/aurora_n256_n512/field_patch/n512",
        "analysis/aurora_n256_n512/field_patch/comparison",
        "analysis/aurora_n128_n256_n512/final",
        "evidence/aurora/authority",
        "evidence/aurora/qualification_8789659",
        "evidence/aurora/n256_reference_shock_seg000_retry1",
        "evidence/aurora/n128_reference_shock_seg000",
        "evidence/aurora/n128_reference_shock_seg001",
        "evidence/aurora/n512_reference_shock_seg000",
        "evidence/aurora/n512_reference_shock_seg001",
        "evidence/aurora/n512_reference_shock_seg002",
        "evidence/aurora/n512_reference_shock_seg003",
        "evidence/aurora/n512_reference_shock_seg004",
        "evidence/aurora/n512_reference_shock_seg005",
    ):
        directory = ROOT / relative_dir
        if directory.is_dir():
            files.update(path for path in directory.rglob("*") if path.is_file())

    return sorted(files)


manifest = {
    "schema": "athenak.z4c.reference_shock_gauge_figure3.evidence.v4",
    "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "repository": "https://github.com/HengruiZhu99/athenak",
    "branch": "codex/z4c-vc-reference-shock-gauge-figure3-20260828",
    "source_fix_commit": "f8303c6be7eb214fa1e91b646123ee0d434b3698",
    "source_fix_tree": "7a585ca487b12351b084eb425bb812775849b001",
    "verdict": (
        "AURORA_PVC_QUALIFIED; N128_N512_EXACT_N256_TREE_REPLAY; "
        "STRONG_RESOLUTION_IMPROVEMENT; N512_COVERS_FIGURE3_INTERVAL; "
        "CONSTRAINT_INVALID_AT_FIRST_PEAK; UNIFORM_CONVERGENCE_NOT_ESTABLISHED"
    ),
    "fixed_numerics": {
        "lapse": "shock-avoiding Bona-Masso kappa=1 with unit initial lapse",
        "shift": "prescribed zero shift",
        "telegraph_lapse": False,
        "discretization": "native vertex-centered Cartoon SO(2), O4, q6, RK4",
        "cfl": 0.15,
        "ko_dissipation_parameter": 0.5,
        "z4c_kappa1": 0.0,
        "z4c_kappa2": 0.0,
        "outer_boundary_M": 128.0,
    },
    "build_authority": {
        "athenak_commit": "f8303c6be7eb214fa1e91b646123ee0d434b3698",
        "athenak_executable_sha256": "aae7ccb8739fb4951221ad7be69ea0e220548b52d402086f57d7857fa2c97a13",
        "cmake_cache_sha256": "8da40bcb47564d9184119ca207f9847a33a3d1b5bd2930627d705cda8fb36386",
        "kokkos_commit": "6739bc623081648af9e752b616d9671527922cbf",
        "irisk_source_commit": "620acca67c2736d9add98ecae3ec76f0f2800b29",
        "irisk_library_sha256": "380a90d5b1d9762fe7f9076edcb27fb4a209f4cd8c070da376c36284a438c7a1",
        "brill_coefficients_sha256": "1b5f0efc3f080215ed7d7994194ba63ea123415bfd8e74c54ca1fd72680aea10",
        "input_sha256": "6c694cf871a3d694d745f0fb58b279b6cd07516463ac8ad54f1c91d2689c90ba",
        "pvc_qualification_job": 8789659,
        "pvc_qualification": "11 of 11 focused tests passed",
    },
    "hierarchy_authority": {
        "path": (
            "docs/investigations/z4c_vc_reference_shock_gauge_figure3_20260828/"
            "evidence/aurora/authority/n256_reference_shock_authority.jsonl"
        ),
        "sha256": "7055de601e6181e5ad7e1432b5c20a111b0ba67e0e8d5377c170ea80e7bedcde",
        "initial_leaves": 104,
        "event_1_leaves": 200,
        "event_1_checksum": "24316947a3a67cd8",
        "event_2_leaves": 212,
        "event_2_checksum": "cf0d2384b11c1d42",
        "terminal_physical_level": 5,
        "n128_exact_replay": True,
        "n512_exact_replay": True,
    },
    "execution": {
        "aurora_account": "CompactBinaryMerger",
        "queue": "debug-scaling",
        "nodes_per_segment": 2,
        "mpi_ranks_per_segment": 24,
        "n128_jobs": [8790272, 8790338],
        "n256_authority_job": 8789703,
        "n512_jobs": [8789895, 8789956, 8790025, 8790135, 8790202, 8790242],
        "n512_segment_0_limitation": (
            "Athena completed cleanly and wrote a usable restart, but PBS exit -29 occurred "
            "during a redundant second full artifact hash pass; later segments are sealed and exit zero."
        ),
        "active_jobs_at_handoff": 0,
    },
    "case_observations": {
        "n128": {
            "coordinate_time_final_M": 45.0,
            "central_proper_time_final_M": 19.332397517508756,
            "first_peak_tau_M": 10.313957475928007,
            "first_peak_log10_abs_kretschmann": 4.297648610475093,
            "max_C_squared_integral": 107.60843422037156,
        },
        "n256": {
            "coordinate_time_final_M": 30.0,
            "central_proper_time_final_M": 11.286306780061583,
            "first_peak_tau_M": 10.303327789519376,
            "first_peak_log10_abs_kretschmann": 5.013488501190193,
            "max_C_squared_integral": 48.233017130731284,
        },
        "n512": {
            "coordinate_time_final_M": 38.65233198686742,
            "central_proper_time_final_M": 14.982526976960017,
            "first_peak_tau_M": 10.30810787031197,
            "first_peak_log10_abs_kretschmann": 5.381115645566895,
            "deep_minimum_tau_M": 12.622799481368615,
            "deep_minimum_log10_abs_kretschmann": -6.0787503048865625,
            "rebound_tau_M": 13.216292031619886,
            "rebound_log10_abs_kretschmann": -2.818493659098744,
            "max_C_squared_integral": 4.099301297753094,
        },
    },
    "convergence_observations": {
        "common_proper_time_max_M": 11.286306780061583,
        "central_kretschmann_median_order": {
            "tau_0_8": 4.8629540656172345,
            "tau_8_10": 3.3427263758082746,
            "tau_10_11_286": 2.0959588228488637,
        },
        "central_lapse_median_order": {
            "tau_0_8": 3.934542276119108,
            "tau_8_10": 3.3585123851430208,
            "tau_10_11_286": 1.395423777943999,
        },
        "constraint_pair_orders": "positive but inconsistent between N128/N256 and N256/N512",
    },
    "diagnostic_observations": {
        "constraint_history_measure": (
            "2*pi*rho*sqrt(abs(det_gamma))*drho*dz with VC trapezoid weights and canonical shared-node ownership"
        ),
        "collapsed_y_normalization_artifact": False,
        "maxima_location": (
            "N256 and N512 C/H maxima are axis-adjacent and far from a coarse-fine interface; "
            "this is geometric classification, not causal attribution."
        ),
        "rho5_high_frequency": (
            "No measured rho near 5 high-k branch grows with resolution on the stitched 25-field patch."
        ),
    },
    "claims_supported": [
        "N128 and N512 replay the complete N256 LogicalLocation hierarchy and both event times exactly.",
        "Increasing cells per physical MeshBlock strongly improves the central curve and constraints on the same AMR tree.",
        "N512 directly covers the published first peak, deep minimum, and rebound without fitting.",
        "Early central curvature and lapse are compatible with O4 convergence before order degrades through collapse.",
        "Bulk or parent under-resolution is a major contributor to the N256 failure and is not ruled out.",
        "The Cartoon history norm already uses the physical axisymmetric ring measure.",
    ],
    "claims_not_supported": [
        "A constraint-qualified Figure-3 reproduction",
        "Uniform convergence through collapse",
        "Exclusion of axis or coarse-fine interface contributions",
        "A unique source-level defect or continuum formulation instability",
        "A production numerical correction",
        "A horizon conclusion",
    ],
    "natural_next_step": (
        "A bounded shared-RHS audit at the retained N512 peak state, separately accumulating active-axis, "
        "same-level seam, coarse-fine ghost/interface, and clean-interior contributions while holding the exact "
        "tree and all numerics fixed."
    ),
    "artifact_selection_note": (
        "Includes final reports, generators, final analyses, compact histories, scheduler/provenance records, "
        "and sealed segment evidence. Intermediate partial-analysis directories and the reviewer prompt are excluded."
    ),
    "artifacts": [artifact(path) for path in selected_artifacts()],
}

OUT.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
print(f"wrote {OUT} with {len(manifest['artifacts'])} artifacts")
