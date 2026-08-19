#!/usr/bin/env python3
"""Fail-closed source contract for authenticated append-only replay extension."""

import argparse
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    source = (args.source_root / "src/mesh/amr_history.cpp").read_text(
        encoding="utf-8")

    environment = 'std::getenv("ATHENA_AMR_HISTORY_EXTENSION_FILE")'
    branch_base = 'std::getenv("ATHENA_AMR_HISTORY_BRANCH_BASE_EVENT")'
    parameter = 'DoesParameterExist("mesh_refinement", "amr_history_extension_file")'
    source_compatibility = (
        'DoesParameterExist("mesh_refinement", "amr_history_compatible_source_id")')
    require(environment in source, "replay extension environment input is absent")
    require(branch_base in source, "authenticated branch base input is absent")
    require(parameter in source, "replay extension parameter input is absent")
    require(source_compatibility in source,
            "explicit replay source compatibility input is absent")
    require('if (!replay()) Fatal("amr_history_compatible_source_id is replay-only")'
            in source, "source compatibility is not replay-only")
    require('header_.source_id != compatible_source_id_' in source,
            "recorded source identity is not checked exactly")
    require('candidate.source_id = header_.source_id;' in source,
            "strict compatibility check is not narrowly rebound")
    require('if (!replay()) Fatal("amr_history_extension_file is replay-only")' in source,
            "extension is not replay-only")
    require('parameter_path != environment_path' in source,
            "parameter/environment mismatch is not rejected")
    require('LoadRestartCarrier();' in source and 'LoadAppendOnlyExtension();' in source,
            "restart carrier and extension loads are missing")
    require(source.index('LoadRestartCarrier();') < source.index('LoadAppendOnlyExtension();'),
            "extension is loaded before the authenticated restart carrier")
    require('AppendOnlyExtension(events_, extension, &error)' in source,
            "append-only prefix validation is missing")
    require('AuthenticatedBranch(events_, extension, base, &error)' in source,
            "authenticated branch-prefix validation is missing")
    require('base < last_applied_event_ || base + 1 < next_event_' in source,
            "branch can alter already applied replay events")
    print("AMR_HISTORY_EXTENSION_STATIC_PASS")


if __name__ == "__main__":
    main()
