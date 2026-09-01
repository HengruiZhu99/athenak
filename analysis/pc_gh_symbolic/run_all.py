#!/usr/bin/env python3
"""Run the currently established PC-GH symbolic gates."""

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "verify_regularization.py",
    "verify_q_projection.py",
    "verify_conformal_ricci.py",
    "verify_primary_projections.py",
    "verify_gradient_rhs.py",
    "verify_4d_component_oracle.py",
]


def main():
    for script in SCRIPTS:
        print(f"== {script} ==", flush=True)
        subprocess.run([sys.executable, str(HERE / script)], check=True)
    print("PASS: all currently established PC-GH symbolic gates")


if __name__ == "__main__":
    main()
