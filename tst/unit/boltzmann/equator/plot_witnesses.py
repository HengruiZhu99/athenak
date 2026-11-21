"""Plot the two null expansions of each earliest verified archived witness."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from witness import LocalPolynomial, NativePatch, measure, read_binary


def plot(frames, report, output):
    theta = np.r_[0., np.linspace(1.e-5, np.pi-1.e-5, 4097), np.pi]
    angles = np.rad2deg(theta)
    datasets = [
        ("j09508", "j09508-shallow-surface-checks.json", "0.9508", "#175779"),
        ("j10054", "j10054-surface-checks.json", "1.0054", "#b85a24"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.65), sharex=True)
    for run, name, spin, color in datasets:
        fits = json.loads((report/name).read_text())
        first = min((fit for fit in fits if fit["all_negative"]), key=lambda f:f["time"])
        geom = NativePatch(read_binary(frames/run/first["file"]))
        curves = []
        for interp in (geom, LocalPolynomial(geom, 3), LocalPolynomial(geom, 5)):
            h, plus, minus, _, _ = measure(interp, first["coeff"], theta)
            curves.append([h*plus, h*minus])
        curves = np.array(curves)
        if np.max(curves) >= 0.:
            raise ValueError(f"archived witness failed strict trapping check for {run}")
        np.testing.assert_allclose(np.max(curves, axis=2), list(first["checks"].values()), atol=5.e-9)
        for component, ax in enumerate(axes):
            values = curves[:, component]
            ax.fill_between(angles, values.min(axis=0), values.max(axis=0), color=color, alpha=.28)
            ax.plot(angles, values[0], color=color, lw=1.2, label=spin)
    for ax, label in zip(axes, [r"Outgoing expansion  $h\Theta_+$", r"Ingoing expansion  $h\Theta_-$"]):
        ax.axhline(0, color=".35", lw=.8)
        ax.set(xlim=(0, 180), xticks=[0, 45, 90, 135, 180], xlabel="Polar angle (degrees)", ylabel=label)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(-3,-3), useMathText=True)
    axes[1].legend(title=r"Initial $J_{\rm ADM}/M_{\rm ADM}^2$", loc="upper center", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("extracted_frames", type=Path, help="parent of j09508 and j10054")
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    report = Path(__file__).resolve().parents[4]/"reports/equatorial-reflection"
    plot(args.extracted_frames, report, args.output)
