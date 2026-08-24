#!/usr/bin/env python3
"""Run the reviewed common-vertex analysis at the native-authority milestones."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


SCRIPT = Path(
    os.environ.get(
        "NATIVE_AUTHORITY_FIELD_SCRIPT",
        str(
            Path(__file__).resolve().parents[1]
            / "z4c_vc_figure3_convergence_20260823"
            / "analyze_fields.py"
        ),
    )
)
spec = importlib.util.spec_from_file_location("native_authority_fields", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.TARGET_TIMES = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0,
                       4.0, 5.0, 6.0, 6.5)
module.main()
