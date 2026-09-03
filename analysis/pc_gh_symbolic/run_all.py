#!/usr/bin/env python3
"""Run the currently established PC-GH symbolic gates."""

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "verify_puncture_regular_55.py",
    "verify_regularization.py",
    "verify_q_projection.py",
    "verify_flat_algebra_randomized.py",
    "verify_conformal_ricci.py",
    "verify_primary_projections.py",
    "verify_gradient_rhs.py",
    "verify_z4c_mp_gauge.py",
    "analyze_z4c_mp_principal.py",
    "verify_4d_component_oracle.py",
    "verify_fo_gh_map.py",
    "generate_gauge_a0_table.py",
    "audit_gauge_a0_cancellation.py",
    "audit_bowen_york_cancellation.py",
    "verify_ko_symbol.py",
    "verify_reduction_constraint_growth.py",
    "verify_source_policy.py",
]


def main():
    for script in SCRIPTS:
        print(f"== {script} ==", flush=True)
        subprocess.run([sys.executable, str(HERE / script)], check=True)
    print("PASS: all currently established PC-GH symbolic gates")


if __name__ == "__main__":
    main()
