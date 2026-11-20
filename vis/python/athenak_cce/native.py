"""Differentiate native Bondi news on its saved cuts in the initial extraction frame."""

import argparse
from pathlib import Path

import numpy as np
import scri

from .diagnostics import render


def curvature(path):
    """Return M0 R Psi4 = -dN/d(u/M0) in the SpECTRE convention."""
    with np.load(path) as archive:
        t = archive["bondi_time_over_M0"]
        labels = archive["ell_m"]
        values = archive["news_modes"]
    lmax = int(labels[-1, 0])
    expected = [(ell, m) for ell in range(2, lmax + 1) for m in range(-ell, ell + 1)]
    if not np.array_equal(labels, expected) or not np.all(np.diff(t) > 0):
        raise ValueError("invalid spin -2 modes or nonmonotonic Bondi cuts")
    if not np.isfinite(values).all() or not np.isfinite(t).all():
        raise ValueError("nonfinite waveform")
    news = scri.WaveformModes(
        t=t,
        data=values,
        ell_min=2,
        ell_max=lmax,
        dataType=scri.hdot,
        frameType=scri.Inertial,
        r_is_scaled_out=True,
        m_is_scaled_out=True,
    )
    return t, -news.data_dot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("news", type=Path)
    p.add_argument("plot", type=Path)
    p.add_argument("--mode", type=int, nargs=2, default=(2, 0))
    args = p.parse_args()
    t, psi4 = curvature(args.news)
    render(
        args.plot,
        t,
        np.pad(psi4, ((0, 0), (4, 0))),
        *args.mode,
        "native news derivative, initial frame",
    )


if __name__ == "__main__":
    main()
