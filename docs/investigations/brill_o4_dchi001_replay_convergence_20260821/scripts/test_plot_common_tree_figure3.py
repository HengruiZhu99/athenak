#!/usr/bin/env python3
"""Focused smoke test for the authenticated Figure-3 overlay."""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "plot_common_tree_figure3.py"


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    histories = []
    for case, scale in (("n128", 1.0), ("n256", 1.1), ("n512", 1.2)):
        path = root / f"{case}.hst"
        path.write_text(
            "# [1]=time [2]=axisTau [3]=axisKret\n"
            f"0 0 {scale}\n1 1 {2 * scale}\n", encoding="utf-8")
        histories.extend((f"--{case}", str(path)))
    reference = root / "reference.csv"
    with reference.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("series", "tau", "log10_abs_I"))
        writer.writeheader()
        for series in ("bamps", "prague", "sphGR"):
            writer.writerow({"series": series, "tau": 0, "log10_abs_I": -3})
    secondary = root / "secondary.csv"
    secondary.write_text("axisTau,axisKret\n0,1\n1,3\n", encoding="utf-8")
    output = root / "output"
    subprocess.run([sys.executable, str(SCRIPT), *histories,
                    "--reference", str(reference), "--secondary", str(secondary),
                    "--output", str(output)], check=True)
    assert (output / "fig3_o4_common_tree_n128_n256_n512.png").stat().st_size > 0
    with (output / "figure3_plotted_data.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 11 and {row["series"] for row in rows} == {
        "bamps", "prague", "sphGR", "n128", "n256", "n512", "n256_dchi002_prior"}

print("COMMON_TREE_FIGURE3_OVERLAY_TEST_PASS")
