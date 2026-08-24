#!/usr/bin/env python3
"""Production-path regression for native-VC multi-family derefinement slots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--dimensions", choices=(2, 3), type=int, default=2)
    parser.add_argument("--family-count", type=int, default=2)
    parser.add_argument("--authority-map", action="store_true")
    parser.add_argument("--order", choices=(2, 4, 6), type=int)
    parser.add_argument("--transfer-order", choices=(4, 6, 8), type=int)
    parser.add_argument("--amplitude", type=float)
    parser.add_argument("--mixed-refine", action="store_true")
    parser.add_argument("--writer-lifecycle", action="store_true")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    shutil.rmtree(args.work_dir, ignore_errors=True)
    args.work_dir.mkdir(parents=True)
    audit_path = args.work_dir / "vc_derefine_slot_audit.json"
    environment = os.environ.copy()
    environment["ATHENA_Z4C_VC_DEREFINE_SLOT_AUDIT"] = str(audit_path)
    writer_path = args.work_dir / "vc_derefine_writer.jsonl"
    lifecycle_path = args.work_dir / "vc_amr_lifecycle.jsonl"
    if args.writer_lifecycle:
        environment["ATHENA_Z4C_VC_DEREFINE_WRITER_JSONL"] = str(writer_path)
        environment["ATHENA_Z4C_VC_AMR_LIFECYCLE"] = "all"
        environment["ATHENA_Z4C_VC_AMR_LIFECYCLE_JSONL"] = str(lifecycle_path)
    environment["OMP_NUM_THREADS"] = str(args.threads)
    environment.setdefault("OMP_PROC_BIND", "false")
    command = [str(args.athena.resolve()), "-i", str(args.input.resolve()),
               "-d", str(args.work_dir.resolve())]
    if args.order is not None:
        command.append(f"z4c/spatial_order={args.order}")
    if args.transfer_order is not None:
        command.append(
            f"z4c/vertex_prolongation_order={args.transfer_order}")
    if args.amplitude is not None:
        command.append(f"problem/amp={args.amplitude:.17g}")
    if args.mixed_refine:
        command.append("problem/exercise_mixed_amr=true")
    if args.writer_lifecycle:
        command.append("time/nlim=3")
    completed = subprocess.run(command, cwd=args.work_dir, env=environment,
                               text=True, capture_output=True, check=False)
    require(completed.returncode == 0,
            f"multi-family fixture failed ({completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    require(audit_path.is_file(), "A5/A6 slot audit was not emitted")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require(audit["schema"] == "athenak_vc_derefine_slot_audit_v1",
            "unexpected slot-audit schema")
    require(audit["rank"] == 0 and audit["first_old"] == 0 and
            audit["first_new"] == 0,
            "fixture did not remain a one-rank zero-based slot transaction")
    child_count = 1 << args.dimensions
    expected_net_deletion = ((args.family_count -
                              (1 if args.mixed_refine else 0)) *
                             (child_count - 1))
    require(audit["old_count"] - audit["new_count"] ==
            expected_net_deletion,
            "fixture did not create the intended multi-family hierarchy")
    require(audit["variables"] == 25,
            "slot audit did not cover all 25 evolved Z4c variables")

    families = audit["families"]
    require(len(families) == args.family_count,
            "transaction did not derefine the requested family count")
    if args.authority_map:
        require([(family["old_gid"], family["new_gid"], family["source_m"],
                  family["destination_m"]) for family in families] ==
                [(16, 16, 16, 16), (29, 26, 29, 26)],
                "synthetic transaction lost the authority event-3 slot map")
    require(any(family["destination_m"] < family["source_m"]
                for family in families[1:]),
            "fixture did not exercise a later family moving left")
    if args.dimensions == 3:
        require(any(family["source_m"] - family["destination_m"] >= 7
                    for family in families),
                "3D fixture did not move an octet parent left by at least seven slots")
    require(all(family["a5_staging_matches"] for family in families),
            "A5 did not stage every parent in its old lower-child slot; "
            f"later-family signatures={families[1:]}")
    require(audit["a5_modified_live_old_gids"] == [],
            "A5 overwrote a still-live old source; "
            f"to remain intact, got {audit['a5_modified_live_old_gids']}")
    require(all(family["a6_parent_matches"] for family in families),
            "A6 did not preserve the independently injected parents; "
            f"later-family signatures={families[1:]}")
    require(audit["a6_bad_unaffected_old_gids"] == [],
            "A6 corrupted an unaffected logical block; "
            f"to remain exact, got {audit['a6_bad_unaffected_old_gids']}")
    if args.writer_lifecycle:
        require(writer_path.is_file(), "per-parent writer JSONL was not emitted")
        writer_rows = [json.loads(line) for line in
                       writer_path.read_text(encoding="utf-8").splitlines()]
        required = {"A4", "A5", "A6", "A8", "A14", "A15", "A16",
                    "R0", "U0"}
        observed = {row["phase"] for row in writer_rows}
        require(required <= observed,
                f"writer lifecycle omitted checkpoints {sorted(required-observed)}")
        for checkpoint in required:
            rows = [row for row in writer_rows if row["phase"] == checkpoint]
            require(len(rows) == args.family_count,
                    f"{checkpoint} omitted a derefined parent")
            for row in rows:
                require(len(row["hashes"]["pre_a5_lower_child"]) == 25 and
                        len(row["hashes"]["independent_restriction_oracle"]) == 25 and
                        len(row["hashes"]["checkpoint_parent"]) == 25,
                        f"{checkpoint} omitted per-variable hashes")
        a16_rows = [row for row in writer_rows if row["phase"] == "A16"]
        require(all(not row["first_oracle_mismatch"]["found"]
                    for row in a16_rows),
                "the repaired parent was already wrong at A5/A6")
        require(all(all(entry["exact"] for entry in
                        row["post_a6_relocation_survivors"])
                    for row in a16_rows),
                "an unaffected relocation survivor changed at A6")
        require(lifecycle_path.is_file(), "global lifecycle JSONL was not emitted")
        lifecycle = [json.loads(line) for line in
                     lifecycle_path.read_text(encoding="utf-8").splitlines()]
        require({"R0", "U0"} <= {row["phase"] for row in lifecycle},
                "global lifecycle omitted first post-event RHS/update")
        require(all(len(row["variables"]) == 25 and
                    all(variable["nonfinite"] == 0 for variable in row["variables"])
                    for row in lifecycle if row["phase"] in {"R0", "U0"}),
                "R0/U0 lifecycle hashes are incomplete or nonfinite")
    print("PASS: native-VC multi-family derefinement preserves exact staging and "
          "logical relocation for all 25 variables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
