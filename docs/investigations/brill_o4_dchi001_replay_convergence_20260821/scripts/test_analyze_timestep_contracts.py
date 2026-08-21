#!/usr/bin/env python3
"""Focused replay-clip classification test."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import tempfile


SCRIPT = Path(__file__).with_name("analyze_timestep_contracts.py")
spec = importlib.util.spec_from_file_location("timestep_analysis", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "contract.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "cycle", "time", "dt_spatial", "dt_source", "dt_final"))
        writer.writeheader()
        writer.writerow({"cycle": 0, "time": 0, "dt_spatial": .1,
                         "dt_source": 1, "dt_final": .015})
        writer.writerow({"cycle": 1, "time": .015, "dt_spatial": .1,
                         "dt_source": 1, "dt_final": 1e-6})
        writer.writerow({"cycle": 2, "time": .015001, "dt_spatial": 1e-4,
                         "dt_source": 1, "dt_final": 1.5e-5})
    result = module.analyze(path, .15)
    assert result["clip_count"] == 1
    assert result["landmarks"]["1e-05"]["dt_final"]["cycle"] == 1
    assert result["landmarks"]["1e-05"]["dt_final"]["event_or_external_clip"] is True
    assert result["landmarks"]["1e-05"]["unclipped_z4c_candidate"] is None

print("TIMESTEP_CONTRACT_ANALYZER_TEST_PASS")
