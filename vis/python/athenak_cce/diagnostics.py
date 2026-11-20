"""Compute Bondi charges and fit a late superrest cut to SpECTRE CCE data."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import scri
from scipy.integrate import cumulative_trapezoid


def read_cce(path):
    """Use scri's SpEC to Moreschi–Boyle conversion, retaining the CCE time origin."""
    with h5py.File(path) as f:
        groups = [key for key in f if key.startswith("SpectreR") and key.endswith(".cce")]
        if len(groups) != 1:
            raise ValueError("expected one SpECTRE radius group per reduction file")
        group = f[groups[0]]
        times = group["Strain"][:, 0]
        for key in ("News", "Psi0", "Psi1", "Psi2", "Psi3", "Psi4", "Strain"):
            if not np.array_equal(times, group[key][:, 0]):
                raise ValueError(f"{key} and strain cuts differ")
            if not np.isfinite(group[key][:]).all():
                raise ValueError(f"nonfinite {key}")
        if not np.all(np.diff(times) > 0):
            raise ValueError("CCE time must be strictly increasing")
        values = group["News"][:, 1:].copy().view(complex)
    abd = scri.SpEC.file_io.create_abd_from_h5(
        file_format="spectrecce_v1", file_name=str(path)
    )
    if not np.allclose(abd.t - abd.t[0], times - times[0], rtol=0, atol=1e-10):
        raise ValueError("scri and CCE clocks disagree")
    abd.t = times
    news = scri.WaveformModes(
        t=times,
        data=values[:, 4:],
        ell_min=2,
        ell_max=abd.ell_max,
        dataType=scri.hdot,
        frameType=scri.Inertial,
        r_is_scaled_out=True,
        m_is_scaled_out=True,
    )
    return abd, news


def summarize_cut(abd, cut, half_width):
    t = abd.t
    if cut - half_width < t[0] or cut + half_width > t[-1] or half_width <= 0:
        raise ValueError("requested fitting window outside complete Bondi cuts")
    window = abd.interpolate(np.linspace(cut - half_width, cut + half_width, 41))
    p = window.bondi_four_momentum()
    mass = window.bondi_rest_mass()
    if np.any(p[:, 0] <= 0) or not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("remnant window lacks a future timelike Bondi four-momentum")
    psi = np.asarray(window.supermomentum("Moreschi"))[20]
    return {
        "speed": float(np.linalg.norm(p[20, 1:]) / p[20, 0]),
        "center_M0": float(np.linalg.norm(window.bondi_CoM_charge()[20]) / p[20, 0]),
        "proper_supermomentum": float(np.linalg.norm(psi[4:]) / abs(psi[0])),
        "mass_M0": float(mass[20]),
        "mass_range_M0": float(np.ptp(mass)),
    }


def save_diagnostics(path, abd, news):
    """Store both independently extracted news and the strain-derivative check."""
    # The two transformation algorithms can trim different incomplete end cuts.
    t = abd.t[(abd.t >= news.t[0]) & (abd.t <= news.t[-1])]
    a = abd if np.array_equal(t, abd.t) else abd.interpolate(t)
    n = news.interpolate(t)
    h = a.h
    energy, momentum, angular, boost = scri.flux.poincare_fluxes(h, n)
    four_p = a.bondi_four_momentum()
    radiated = cumulative_trapezoid(
        np.column_stack((energy, momentum)), t, axis=0, initial=0
    )
    data = {
        "u_M0": t,
        "ell_max": a.ell_max,
        "strain": h.data,
        "news": n.data,
        "news_minus_dh": n.data - h.data_dot,
        "energy_flux": energy,
        "momentum_flux": momentum,
        "angular_momentum_flux": angular,
        "boost_flux": boost,
        "radiated_four_momentum": radiated,
        "four_momentum_balance": four_p - four_p[0] + radiated,
        "psi4_plus_dnews": 2 * np.asarray(a.psi4)[:, 4:] + n.data_dot,
        "bondi_mass": a.bondi_rest_mass(),
        "bondi_four_momentum": four_p,
        "spin": a.bondi_dimensionless_spin(),
        "angular_momentum": a.bondi_angular_momentum(),
        "boost": a.bondi_boost_charge(),
        "center_of_mass_charge": a.bondi_CoM_charge(),
        "mass_aspect": np.asarray(a.mass_aspect()),
        "moreschi": np.asarray(a.supermomentum("Moreschi")),
        "constraint_norms": np.asarray(a.bondi_violation_norms),
    }
    for i in range(5):
        data[f"psi{i}"] = np.asarray(getattr(a, f"psi{i}"))
    if not all(np.isfinite(x).all() for x in data.values()):
        raise ValueError("nonfinite Bondi diagnostic")
    np.savez_compressed(path, **data)


def render(path, t, modes, degree, m, frame, window=None):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    column = degree * degree + degree + m
    if degree < 2 or abs(m) > degree or column >= modes.shape[1]:
        raise ValueError("invalid psi4 harmonic")
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    z = modes[:, column]
    ax.plot(t, z.real, color="#135773", label="Real")
    ax.plot(t, z.imag, color="#b55536", ls="--", label="Imaginary")
    if window:
        ax.axvspan(*window, color="#758078", alpha=0.13, label="Frame-fit window")
    ax.set(
        xlabel=r"Bondi retarded time $u/M_0$",
        ylabel=r"$M_0\,\Psi^0_{4,\ell m}$ (SpECTRE)",
        title=f"({degree}, {m}) curvature waveform · {frame}",
    )
    ax.axhline(0, color="0.65", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parameters(transform):
    return {
        "boost_velocity": transform.boost_velocity,
        "supertranslation": transform.supertranslation,
        "frame_rotation": transform.frame_rotation.components,
    }


def fit_frame(abd, cut, half_width, keep_axes):
    # Explicitly restrict the data used to fit the center-of-mass drift. scri's
    # general wrapper adds interpolation padding that can encompass a short run.
    first, last = np.searchsorted(abd.t, [cut - half_width, cut + half_width])
    fit = abd.interpolate(np.union1d(abd.t[first:last], [cut]))
    if len(fit.t) < 8:
        raise ValueError("at least eight complete cuts are needed in the fit window")
    if not keep_axes:
        _, transform, _ = fit.map_to_superrest_frame(t_0=cut, padding_time=half_width)
        return transform

    from scri.asymptotic_bondi_data.map_to_superrest_frame import (
        com_transformation_to_map_to_superrest_frame,
        supertranslation_to_map_to_superrest_frame,
    )
    from scri.bms_transformations import BMSTransformation
    from spherical_functions import constant_as_ell_0_mode

    # The nonspinning solution has no preferred spin direction. Use the two
    # charge solvers directly so a zero spin never enters a rotation objective.
    origin = BMSTransformation(supertranslation=[constant_as_ell_0_mode(cut)])
    fit.t = fit.t - cut
    total = BMSTransformation()
    order = ["supertranslation", "frame_rotation", "boost_velocity"]
    current = fit
    # Re-evaluate the charge on the transformed fields after each correction.
    # Iterating a frozen mass aspect can generate unphysical trial momenta.
    for _ in range(3):
        current = current.interpolate(np.union1d(current.t, [0.0]))
        translation, _ = supertranslation_to_map_to_superrest_frame(
            current, N_itr_max=1, ell_max=abd.ell_max
        )
        total = (translation * total).reorder(order)
        current = fit.transform(**parameters(total))
        center, _ = com_transformation_to_map_to_superrest_frame(current)
        total = (center * total).reorder(order)
        current = fit.transform(**parameters(total))
    return (origin.inverse() * total * origin).reorder(order)


def analyze(source, dest, cut, half_width, ell, m, tolerance, keep_axes, spacing=None):
    dest.mkdir(parents=True, exist_ok=False)
    abd, news = read_cce(source)
    if spacing is not None:
        if spacing <= 0 or not np.isfinite(spacing):
            raise ValueError("analysis spacing must be positive")
        steps = int(np.ceil((abd.t[-1] - abd.t[0]) / spacing))
        grid = np.linspace(abd.t[0], abd.t[-1], steps + 1)
        abd, news = abd.interpolate(grid), news.interpolate(grid)
    render(
        dest / "extraction-frame.png",
        abd.t,
        2 * np.asarray(abd.psi4),
        ell,
        m,
        "CCE extraction frame",
    )
    save_diagnostics(dest / "extraction-frame.npz", abd, news)
    before = summarize_cut(abd, cut, half_width)
    transform = fit_frame(abd, cut, half_width, keep_axes)
    params = parameters(transform)
    mapped = abd.transform(**params)
    after = summarize_cut(mapped, cut, half_width)
    transform.to_file(str(dest / "transformation.h5"))
    keys = ("speed", "center_M0", "proper_supermomentum")
    passed = max(after[k] for k in keys) <= tolerance
    n = news.transform(**params)
    filename = "superrest" if passed else "frame-fit"
    save_diagnostics(dest / (filename + ".npz"), mapped, n)
    render(
        dest / (filename + ".png"),
        mapped.t,
        2 * np.asarray(mapped.psi4),
        ell,
        m,
        "superrest cut" if passed else "frame fit above tolerance",
        (cut - half_width, cut + half_width),
    )
    report = {
        "source": str(source.resolve()),
        "cut": cut,
        "analysis_spacing_M0": spacing,
        "half_width": half_width,
        "before": before,
        "after": after,
        "tolerance": tolerance,
        "frame_fit_passed": passed,
        "boost": transform.boost_velocity.tolist(),
        "keep_axes": keep_axes,
        "scri_version": scri.__version__,
    }
    (dest / "frame.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--cut", type=float, required=True)
    p.add_argument("--half-width", type=float, required=True)
    p.add_argument(
        "--keep-axes", action="store_true", help="retain axes for a nonspinning remnant"
    )
    p.add_argument("--tolerance", type=float, default=1e-3)
    p.add_argument("--spacing", type=float, help="maximum analysis spacing in M0")
    p.add_argument("--mode", type=int, nargs=2, default=(2, 0))
    a = p.parse_args()
    print(
        json.dumps(
            analyze(
                a.source,
                a.output,
                a.cut,
                a.half_width,
                *a.mode,
                a.tolerance,
                a.keep_axes,
                a.spacing,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
