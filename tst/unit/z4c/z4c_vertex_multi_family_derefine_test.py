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
    args = parser.parse_args()

    shutil.rmtree(args.work_dir, ignore_errors=True)
    args.work_dir.mkdir(parents=True)
    audit_path = args.work_dir / "vc_derefine_slot_audit.json"
    environment = os.environ.copy()
    environment["ATHENA_Z4C_VC_DEREFINE_SLOT_AUDIT"] = str(audit_path)
    environment["OMP_NUM_THREADS"] = "16"
    environment.setdefault("OMP_PROC_BIND", "false")
    command = [str(args.athena.resolve()), "-i", str(args.input.resolve()),
               "-d", str(args.work_dir.resolve())]
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
    require(audit["old_count"] == 38 and audit["new_count"] == 32,
            "fixture did not create the intended two-family hierarchy")
    require(audit["variables"] == 25,
            "slot audit did not cover all 25 evolved Z4c variables")

    families = audit["families"]
    require(len(families) == 2, "transaction did not derefine exactly two families")
    require([(family["old_gid"], family["new_gid"], family["source_m"],
              family["destination_m"]) for family in families] ==
            [(16, 16, 16, 16), (29, 26, 29, 26)],
            "synthetic transaction lost the authority event-3 slot map")
    require(all(family["a5_staging_matches"] for family in families),
            "A5 did not stage every parent in its old lower-child slot; "
            f"old 29 -> new parent 26 signature={families[1]}")
    require(audit["a5_modified_live_old_gids"] == [],
            "A5 overwrote a still-live old source; expected old 26 -> new 23 "
            f"to remain intact, got {audit['a5_modified_live_old_gids']}")
    require(all(family["a6_parent_matches"] for family in families),
            "A6 did not preserve the independently injected parents; "
            f"old 29 -> new parent 26 signature={families[1]}")
    require(audit["a6_bad_unaffected_old_gids"] == [],
            "A6 corrupted an unaffected logical block; expected old 26 -> new 23 "
            f"to remain exact, got {audit['a6_bad_unaffected_old_gids']}")
    print("PASS: native-VC multi-family derefinement preserves exact staging and "
          "logical relocation for all 25 variables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
