#!/usr/bin/env python3
"""Fail-closed static/evidence gate for the common-tree O4 campaign."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_input(path: Path) -> dict[str, dict[str, str]]:
    blocks: dict[str, dict[str, str]] = {}
    block = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("<") and line.endswith(">"):
            block = line[1:-1]
            blocks.setdefault(block, {})
            continue
        require(block and "=" in line, f"malformed input line: {raw}")
        key, value = (item.strip() for item in line.split("=", 1))
        require(key not in blocks[block], f"duplicate {block}/{key}")
        blocks[block][key] = value
    return blocks


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    values = parse_input(ROOT / "brill_o4_common_tree.athinput")
    exact = {
        ("mesh", "nghost"): "4",
        ("mesh", "nx1"): "128",
        ("mesh", "nx2"): "256",
        ("mesh", "nx3"): "1",
        ("mesh", "x1min"): "0.0",
        ("mesh", "x1max"): "16.0",
        ("mesh", "x2min"): "-16.0",
        ("mesh", "x2max"): "16.0",
        ("meshblock", "nx1"): "32",
        ("meshblock", "nx2"): "32",
        ("mesh_refinement", "num_levels"): "21",
        ("mesh_refinement", "ncycle_check"): "1",
        ("mesh_refinement", "refinement_interval"): "1",
        ("z4c_amr", "method"): "dchi",
        ("z4c_amr", "dchi_max"): "0.01",
        ("z4c_amr", "dchi_derefine_factor"): "0.25",
        ("z4c_amr", "max_ref_lev"): "20",
        ("time", "integrator"): "rk4",
        ("time", "cfl_number"): "0.15",
        ("time", "tlim"): "50.0",
        ("z4c", "symmetry"): "cartoon_so2",
        ("z4c", "spatial_order"): "4",
        ("z4c", "amr_transfer"): "high_order",
        ("z4c", "history_kretschmann"): "true",
        ("z4c", "diss"): "0.02",
        ("z4c", "floor_chi"): "false",
        ("z4c", "telegraph_lapse"): "true",
        ("z4c", "telegraph_damping_prescription"): "max_domain_abs_K",
        ("z4c", "telegraph_tau"): "1.0",
        ("z4c", "telegraph_kappa"): "1.0",
        ("z4c", "shift_Gamma"): "1.0",
        ("z4c", "shift_eta"): "2.0",
        ("z4c", "shift_advect"): "1.0",
        ("z4c", "damp_kappa1"): "0.0",
        ("z4c", "damp_kappa2"): "0.0",
        ("problem", "irisk_adm_import_mode"): "direct_global_coefficients",
        ("problem", "brill_direct_initial_lapse"): "precollapsed_psi_minus_2",
    }
    for (block, key), expected in exact.items():
        require(values.get(block, {}).get(key) == expected,
                f"unexpected {block}/{key}")

    runner = (ROOT / "aurora_run_segment.pbs").read_text(encoding="utf-8")
    for token in (
        "n128:replay:1:64:128:16:16:16384",
        "n256:record:1:128:256:32:32:16384",
        "n512:replay:4:256:512:64:64:4096",
        "time/tlim=\"${RUN_TLIM}\" time/nlim=-1",
        "output2/dcycle=512 output3/dcycle=128 output4/dcycle=128",
        "output5/dcycle=0 output6/dcycle=128 output7/dcycle=0 output8/dcycle=0",
        "sha256sum -c SHA256SUMS",
    ):
        require(token in runner, f"production runner lacks {token!r}")

    evidence = ROOT / "evidence" / "qualification"
    result = json.loads((evidence / "qualification_result.json").read_text())
    require(result["result"] == "pass", "qualification did not pass")
    require(result["source"]["commit"] ==
            "16931a5f9830e7c8a75a9b72e93c4c7230cb6906",
            "qualification source drift")
    require(result["gates"]["maximum_absolute_event_time_ulp_difference"] == 0,
            "qualification replay time was not exact")
    state = json.loads((evidence / "run/state-extraction/z4c_state_failure.json").read_text())
    require(state["reason"] == "nonpositive_chi" and state["chi"] == -1,
            "PVC failure extraction evidence is not the intentional fixture")
    authority = jsonl(evidence / "run/short-authority.jsonl")
    require(authority[0]["root_blocks"] == [4, 8, 1] and
            authority[0]["cells_per_meshblock"] == [32, 32, 1],
            "short authority geometry drift")
    for label in ("n128_prefix", "n512"):
        ledger = jsonl(evidence / f"run/{label}/short_{label.split('_')[0]}.amr_history_replay.jsonl")
        require(ledger and all(row["exact_match"] and row["ulp_difference"] == 0
                               for row in ledger),
                f"{label} replay is not exact")
    for label, cells in (("n128_prefix", [16, 16, 1]),
                         ("n512", [64, 64, 1])):
        shadow = next((evidence / "run" / label).glob("*.amr_native_shadow.rank0000.jsonl"))
        rows = jsonl(shadow)
        require(rows and all(row["cells_per_meshblock"] == cells for row in rows),
                f"{label} shadow geometry drift")
    print("BRILL_O4_COMMON_TREE_CAMPAIGN_CONTRACT_PASS")


if __name__ == "__main__":
    main()
