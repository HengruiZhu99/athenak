#!/usr/bin/env python3
"""Record and plot zero-step counterexamples; never labels them evolution results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sympy

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    with (run / "configuration-map.csv").open() as stream:
        rows = [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]
    assert len(rows) == 6
    assert all(row["residual"] != 0 for row in rows[:5])
    assert rows[-1]["residual"] == 0
    assert "PASS: all currently established PC-GH symbolic gates" in (
        run / "full-symbolic.log").read_text()
    x = np.array([row["w"] for row in rows[:4]])
    y = np.array([abs(row["residual"]) for row in rows[:4]])
    power = float(np.polyfit(np.log(x), np.log(y), 1)[0])
    assert abs(power + 1) < 1.e-12
    plt.rcParams.update({"font.size": 11, "svg.hashsalt": "pc-gh-gamma2-map-audit",
                         "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ax.loglog(x, y, "o-", color="#245e91", label="Measured production − FO-GH RHS")
    ax.loglog(x, 1/(64*x), "--", color="#a34f23", label=r"Exact magnitude $1/(64w)$")
    ax.set(xlabel=r"Positive conformal variable $w$",
           ylabel=r"$|\partial_t\rho_{\rm PC}-\partial_t\rho_{\rm FO}|$",
           title="Zero-step configuration-map counterexample")
    ax.grid(True, which="both", alpha=0.18)
    ax.legend(frameon=False, fontsize=9)
    fig.text(0.53, 0.16, r"$\rho=1,\ \beta^x=1/4,\ L_x=1/8$"+"\n"
             "Independent bounded reduction error; no evolution",
             ha="center", fontsize=9)
    fig.savefig(run / "configuration-map.png", dpi=180)
    fig.savefig(run / "configuration-map.svg", metadata={"Date": None})
    plt.close(fig)
    summary = {
        "classification": "failed dynamical scheme: exact-retrofit derivation gate rejected",
        "production_equations_modified": False,
        "evolution_tests_performed": False,
        "della_run_submitted": False,
        "hybrid_fallback_established": False,
        "counterexamples": rows,
        "measured_rhs_discrepancy_power_in_w": power,
        "source_only_regular_rate": "lambda=rho^2*w^4*f, f bounded",
        "limitations": [
            "The source-only rate does not repair the off-constraint baseline.",
            "Constant-lambda singular source powers are symbolic, not evolved-field fits.",
            "The old moving-gauge symbol calculation assumes zero shift and flat frozen fields.",
            "No puncture stability, exterior convergence, AMR-pulse, or hybrid claim."]}
    (run / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    paths = [*sorted((ROOT / "src/pc_gh").glob("*.*")),
             *sorted((ROOT / "analysis/pc_gh_gamma2").glob("*.*")),
             ROOT / "analysis/pc_gh_symbolic/verify_fo_gh_gamma2.py",
             ROOT / "analysis/pc_gh_symbolic/verify_fo_gh_map.py",
             ROOT / "analysis/pc_gh_symbolic/run_all.py",
             ROOT / "docs/pc_gh_gamma2_audit.md",
             ROOT / "docs/pc_gh_derivation.md",
             ROOT / "docs/pc_gh_regularization_audit.md",
             ROOT / "docs/pc_gh_qualification_log.md",
             ROOT / "build-gamma2-map-audit/CMakeCache.txt",
             ROOT / "build-gamma2-map-audit/src/athena",
             *sorted(p for p in run.glob("*.log") if p.name != "summarize.log"),
             run / "configuration-map.csv",
             run / "configuration-map.png", run / "configuration-map.svg",
             run / "summary.json", run / "commands.json",
             run / "CMakeCache.txt", run / "compiler-flags.txt",
             run / "link-command.txt", run / "goal-objective.md"]
    record = {"base_commit": "5811268b",
              "head_at_audit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "python": sys.version, "platform": platform.platform(),
              "versions": {"sympy": sympy.__version__, "numpy": np.__version__,
                           "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
              "files": {str(p.relative_to(ROOT)): {
                  "bytes": p.stat().st_size,
                  "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in paths if p.is_file()}}
    (run / "provenance.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({"classification": summary["classification"],
                      "measured_rhs_discrepancy_power_in_w": power}))


if __name__ == "__main__":
    main()
