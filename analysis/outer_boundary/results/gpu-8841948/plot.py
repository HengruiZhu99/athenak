"""Reproduce the original-production GPU comparison, without Aurora access.

Run with Python/NumPy/Matplotlib in this directory. No pickle is loaded.
Checks use all stored samples, not downsampled plot values.
"""
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
data = np.load(HERE / "histories.npz", allow_pickle=False)
expected = json.loads((HERE / "results.json").read_text())


def table(case, name):
    key = case + "_" + name
    return dict(zip(data[key + "_names"], data[key].T))


series = {}
for case in ("baseline", "sponge"):
    h, c = table(case, "user"), table(case, "z4c_user")
    series[case] = {
        "Theta_max": (h["time"], h["Theta-max"]),
        "Theta_RMS": (c["time"], np.sqrt(c["Theta-norm"] / c["Volume"])),
        "alpha_res_max": (h["time"], h["alpha-res"]),
    }
    for arrays in (h, c, table(case, "mhd")):
        assert all(np.isfinite(a).all() for a in arrays.values())
    for label, reference in expected["cases"][case]["window_fits"].items():
        lo, hi = map(float, label.split("-"))
        for name, (t, y) in series[case].items():
            mask = (t >= lo) & (t <= hi) & (y > 0)
            slope = np.polyfit(t[mask], np.log(y[mask]), 1)[0]
            assert abs(slope - reference[name]["gamma_per_M"]) < 1e-12

fig, axes = plt.subplots(1, 3, figsize=(12, 3.3),
                         layout="constrained", facecolor="white")
for ax, name, label in zip(
        axes, ("Theta_max", "Theta_RMS", "alpha_res_max"),
        (r"$\max|\Theta|$", r"$\sqrt{\int\Theta^2dV/\int dV}$",
         r"$\max|\delta\alpha|$")):
    for case in ("baseline", "sponge"):
        ax.semilogy(*series[case][name], label=case)
    ax.axvline(2361, color=".6", ls=":", lw=.8)
    ax.set_xlabel(r"$t/M$")
    ax.set_ylabel(label)
    ax.grid(alpha=.15)
axes[0].legend(frameon=False)
fig.savefig(HERE / "gpu-sponge-comparison.png", dpi=180)
plt.close(fig)

profile = json.loads((HERE / "spatial-profile.json").read_text())
rows = profile["active_face_distance_profile"]
distance = np.array([row["nearest_face_distance_M"] for row in rows])
integral = np.array([row["Theta_squared_integral"] for row in rows])
volume = np.array([row["proper_volume"] for row in rows])
c = table("sponge", "z4c_user")
assert np.isclose(integral.sum(), c["Theta-norm"][-1], rtol=1e-13, atol=0)
assert np.isclose(volume.sum(), c["Volume"][-1], rtol=1e-13, atol=0)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2),
                         layout="constrained", facecolor="white")
axes[0].semilogy(distance, [row["Theta_max"] for row in rows], "o-")
axes[0].set_ylabel(r"$\max |\Theta|$ in distance bin")
axes[1].plot(distance, integral / integral.sum(), "o-")
axes[1].set_ylabel(r"Fraction of $\int\Theta^2dV$")
for ax in axes:
    ax.axvline(256, color=".5", ls="--", lw=1)
    ax.set_xlabel("Distance to nearest physical face / M")
    ax.grid(alpha=.15)
fig.savefig(HERE / "sponge-final-face-profile.png", dpi=180)
plt.close(fig)
print("PASS: window slopes, finite history arrays and final spatial integrals")
